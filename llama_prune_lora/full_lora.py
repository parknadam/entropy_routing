#!/usr/bin/env python3
"""
Full-model LoRA training for LLaMA.

This keeps the same SQuAD/Trainer flow as `total_optimized_lora.py`, but adds
an explicit FULL(ABC) stage for training a LoRA adapter on the entire 7B model.

Two common modes:

1) Full HF base model -> attach LoRA to all layers directly
CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0 \
python -m llama_prune_lora.full_lora \
  --base_dir /acpl-ssd32/llama2-13b/baseline_full_single \
  --stage 3 \
  --out_adapters ./no_2048_fullmodel_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 4096 --lr 3e-4 --epochs 2 --bs 2 --grad_acc 32

2) Pruned A model + bundles -> restore B/C and train FULL(ABC)
CUDA_VISIBLE_DEVICES=6 DEVICE=cuda:0 \
python -m llama_prune_lora.full_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 3 \
  --out_adapters ./2048_full_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 2048 --lr 3e-4 --epochs 2 --bs 1 --grad_acc 32

Optional split-aware stages:
- `--stage 1` / `A`:   LoRA on A only
- `--stage 2` / `AB`:  LoRA on A+B
- `--stage 3` / `FULL` / `ABC`: LoRA on A+B+C (full model)
"""

import argparse
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

try:
    import numpy as np
except ImportError:
    np = None

try:
    from .total_optimized_lora import (
        PassLayer,
        _assert_bundles,
        _attach,
        _enable_lora_only,
        _ensure_original_layout,
        _layers,
        _load_bundle_indices,
        _load_qa_dataset,
        _rehydrate,
        train_adapter,
    )
except ImportError:
    from llama_prune_lora.total_optimized_lora import (
        PassLayer,
        _assert_bundles,
        _attach,
        _enable_lora_only,
        _ensure_original_layout,
        _layers,
        _load_bundle_indices,
        _load_qa_dataset,
        _rehydrate,
        train_adapter,
    )


def _stage_arg(raw):
    stage = str(raw).strip().upper()
    mapping = {
        "1": "A",
        "A": "A",
        "2": "AB",
        "AB": "AB",
        "3": "FULL",
        "ABC": "FULL",
        "FULL": "FULL",
    }
    if stage not in mapping:
        raise argparse.ArgumentTypeError("stage must be one of: 1, 2, 3, A, AB, FULL, ABC")
    return mapping[stage]


def _set_seed(seed: int):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument(
        "--bundles_dir",
        default=None,
        help="Optional pruning bundles root. Needed for A/AB stages and for FULL when base_dir is pruned.",
    )
    p.add_argument("--stage", type=_stage_arg, default="FULL")
    p.add_argument("--out_adapters", required=True)
    p.add_argument("--original_num_layers", type=int, default=None)

    p.add_argument("--qa_dataset", default="squad")
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--max_eval_samples", type=int, default=8000)
    p.add_argument("--seq_len", type=int, default=1024)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad_acc", type=int, default=32)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _short(indices, limit=8):
    if len(indices) <= limit:
        return str(indices)
    return f"{indices[:limit]}..."


def _verify_layers(layers, expected_real, expected_pass, tag):
    bad_real = [i for i in expected_real if not isinstance(layers[i], LlamaDecoderLayer)]
    if bad_real:
        raise RuntimeError(f"{tag} real layer mismatch: {bad_real}")

    bad_pass = [i for i in expected_pass if not isinstance(layers[i], PassLayer)]
    if bad_pass:
        raise RuntimeError(f"{tag} pass layer mismatch: {bad_pass}")


def _load_layout_info(base_dir, bundles_dir=None):
    info = {"B": [], "C": [], "L_full": None}

    manifest_path = os.path.join(base_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        info["L_full"] = manifest.get("counts", {}).get("L_full")
        stages = manifest.get("stages", {})
        info["B"] = sorted(int(x) for x in stages.get("B", {}).get("removed_layers", []))
        info["C"] = sorted(int(x) for x in stages.get("C", {}).get("removed_layers", []))

    log_path = os.path.join(base_dir, "prune_log.json")
    if os.path.isfile(log_path):
        with open(log_path) as f:
            prune_log = json.load(f)
        if not info["B"]:
            info["B"] = sorted(int(x) for x in prune_log.get("split", {}).get("B", []))
        if not info["C"]:
            info["C"] = sorted(int(x) for x in prune_log.get("split", {}).get("C", []))

    if bundles_dir:
        if not info["B"]:
            info["B"] = _load_bundle_indices(os.path.join(bundles_dir, "B"))
        if not info["C"]:
            info["C"] = _load_bundle_indices(os.path.join(bundles_dir, "C"))

    return info


def main():
    args = parse_args()
    _set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True, local_files_only=True)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    device = torch.device(os.environ.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_dir,
        torch_dtype=dtype,
        device_map=None,
        local_files_only=True,
    )
    model.to(device)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:
        pass

    loaded_L = len(_layers(model))

    info = _load_layout_info(args.base_dir, args.bundles_dir)

    B_idx = sorted(int(x) for x in info.get("B", []))
    C_idx = sorted(int(x) for x in info.get("C", []))
    original_N = args.original_num_layers or info.get("L_full") or model.config.num_hidden_layers
    removed_all = sorted(set(B_idx + C_idx))
    A_idx = sorted(set(range(original_N)) - set(removed_all))
    AB_idx = sorted(set(A_idx + B_idx))
    FULL_idx = list(range(original_N))

    print(f"\n[Index] stage={args.stage}, original={original_N}, loaded={loaded_L}")
    print(f"  A({len(A_idx)}): {_short(A_idx)}")
    print(f"  B({len(B_idx)}): {_short(B_idx)}")
    print(f"  C({len(C_idx)}): {_short(C_idx)}")

    if args.stage in ("A", "AB") and not removed_all:
        raise ValueError(
            "Stage A/AB needs pruning split metadata. Use --bundles_dir with a pruned layout, "
            "or run --stage 3/FULL for a plain full model."
        )

    if args.stage == "FULL" and removed_all and not args.bundles_dir:
        raise ValueError("FULL stage on a pruned layout requires --bundles_dir to restore B/C layers.")
    if args.stage == "AB" and B_idx and not args.bundles_dir:
        raise ValueError("Stage AB requires --bundles_dir to restore B layers.")

    model, _ = _ensure_original_layout(model, removed_all, original_N)
    layers = _layers(model)

    print("\n[Loading] Datasets")
    train_ds = _load_qa_dataset(tok, args.qa_dataset, "train", args.max_samples, args.seq_len, seed=args.seed)
    eval_ds = _load_qa_dataset(
        tok,
        args.qa_dataset,
        "validation",
        args.max_eval_samples,
        args.seq_len,
        seed=args.seed,
    )
    print(f"  train={len(train_ds)}, eval={len(eval_ds)}")

    if args.stage == "A":
        print(f"\n{'='*60}\nSTAGE A: LoRA on A only\n{'='*60}")
        _verify_layers(layers, A_idx, B_idx + C_idx, "stage A")

        model = _attach(model, "stageA", target_layers=A_idx)
        model.set_adapter("stageA")
        _enable_lora_only(model, A_idx, "stageA")

        out_dir = os.path.join(args.out_adapters, "A_lora", "stageA")
        train_adapter(model, tok, out_dir, train_ds, eval_ds, args, "stageA")

    elif args.stage == "AB":
        print(f"\n{'='*60}\nSTAGE AB: LoRA on A+B\n{'='*60}")
        B_bundle_dir = os.path.join(args.bundles_dir, "B")
        _assert_bundles(B_bundle_dir, B_idx)
        _rehydrate(model, B_bundle_dir, B_idx)
        _verify_layers(layers, AB_idx, C_idx, "stage AB")

        model = _attach(model, "stageAB", target_layers=AB_idx)
        model.set_adapter("stageAB")
        _enable_lora_only(model, AB_idx, "stageAB")

        out_dir = os.path.join(args.out_adapters, "AB_lora", "stageAB")
        train_adapter(model, tok, out_dir, train_ds, eval_ds, args, "stageAB")

    else:
        print(f"\n{'='*60}\nSTAGE FULL(ABC): LoRA on all layers\n{'='*60}")
        if B_idx:
            B_bundle_dir = os.path.join(args.bundles_dir, "B")
            _assert_bundles(B_bundle_dir, B_idx)
            _rehydrate(model, B_bundle_dir, B_idx)
        if C_idx:
            C_bundle_dir = os.path.join(args.bundles_dir, "C")
            _assert_bundles(C_bundle_dir, C_idx)
            _rehydrate(model, C_bundle_dir, C_idx)
        _verify_layers(layers, FULL_idx, [], "stage FULL")

        model = _attach(model, "stageABC", target_layers=FULL_idx)
        model.set_adapter("stageABC")
        _enable_lora_only(model, FULL_idx, "stageABC")

        out_dir = os.path.join(args.out_adapters, "ABC_lora", "stageABC")
        train_adapter(model, tok, out_dir, train_ds, eval_ds, args, "stageABC")

    print("\n[Done] Training completed")


if __name__ == "__main__":
    main()
