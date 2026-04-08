"""
레이어 드랍만 하는 코드 (Gemma / LLaMA / OPT 공용)

# Gemma 7B IT
CUDA_VISIBLE_DEVICES=4 DEVICE=cuda:0 \
python -m gemma_prune_lora.pruning.layeronly_drop \
  --model google/gemma-7b-it \
  --device cuda:0 \
  --drop_frac 0.20 \
  --keep_last_layer \
  --nsamples 64 \
  --seqlen 1024 \
  --max_batches 32 \
  --save_dir ./20_gemma_7b_results/pruning/A \
  --save_removed_dir ./20_gemma_7b_results/pruning/bundles
"""



import argparse
import json
import math
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .data import get_loaders
from .simdrop import choose_block_to_drop, drop_consecutive_layers
from .bundler import export_two_bundles
from .model_utils import detect_arch, get_layers, get_num_layers, get_layer_prefix


def _resolve_embed_device(model, fallback_device: str):
    embed_key = "model.embed_tokens"
    if (
        hasattr(model, "hf_device_map")
        and isinstance(model.hf_device_map, dict)
        and embed_key in model.hf_device_map
    ):
        dev = model.hf_device_map[embed_key]
        return torch.device(dev) if not isinstance(dev, torch.device) else dev
    return torch.device(fallback_device)


def _load_model(model_name: str, seqlen: int, device_str: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
        attn_implementation="eager",
    ).to(device_str)
    model.seqlen = seqlen
    model.config.use_cache = False
    model.eval()
    return model, tok


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def _count_params_of_layers(m, idxs, arch):
    layers_ = get_layers(m, arch)
    s = 0
    for i_ in idxs:
        for p_ in layers_[i_].parameters(recurse=True):
            s += p_.numel()
    return s


def build_layers_map(model, arch: str, out_path: str):
    layers = get_layers(model, arch)
    sd_keys = list(model.state_dict().keys())
    prefix = get_layer_prefix(arch)
    L = len(layers)

    layer_map = {str(i): [] for i in range(L)}
    non_layer_keys = []

    for k in sd_keys:
        if k.startswith(prefix):
            lid = int(k[len(prefix):].split(".", 1)[0])
            layer_map[str(lid)].append(k)
        else:
            non_layer_keys.append(k)

    payload = {
        "num_layers": L,
        "layer_prefix": prefix,
        "layers": layer_map,
        "non_layer_keys": non_layer_keys,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_selection_retry_schedule(requested_seqlen: int, min_seqlen: int):
    requested_seqlen = max(1, int(requested_seqlen))
    min_seqlen = max(1, min(int(min_seqlen), requested_seqlen))

    schedule = []
    cur = requested_seqlen
    while True:
        schedule.append(cur)
        if cur <= min_seqlen:
            break
        nxt = max(min_seqlen, cur // 2)
        if nxt == cur:
            break
        cur = nxt
    return schedule


def _should_retry_selection(exc: Exception) -> bool:
    msg = str(exc).lower()
    retry_markers = (
        "no complete activations were captured",
        "no eligible start index",
        "out of memory",
        "incomplete activation capture",
    )
    return any(marker in msg for marker in retry_markers)


def _format_selection_failure(L_full: int, n: int, attempts):
    attempt_summary = ", ".join(f"{seqlen}:{msg}" for seqlen, msg in attempts)
    return (
        "Block selection failed after exhausting the selection sequence-length retries. "
        f"L={L_full}, n={n}. Attempts: {attempt_summary}. "
        "This usually means calibration forwards could not complete within the current GPU memory budget. "
        "Try freeing GPU memory, using a less-occupied GPU, or rerunning with a smaller "
        "--selection_seqlen / --seqlen."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="google/gemma-7b-it")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--drop_frac", type=float, default=0.30)
    ap.add_argument("--keep_last_layer", action="store_true")
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--selection_seqlen", type=int, default=None)
    ap.add_argument("--min_selection_seqlen", type=int, default=128)
    ap.add_argument("--max_batches", type=int, default=64)
    ap.add_argument("--save_dir", type=str, default="./A")
    ap.add_argument("--save_removed_dir", type=str, default="./bundles")
    ap.add_argument("--split_policy", type=str, choices=["half", "ratio"], default="half")
    ap.add_argument("--split_ratio", type=float, default=0.5)
    ap.add_argument("--prune_log", type=str, default="prune_log.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # ── [0] 모델/토크나이저 로드 ──
    model, tokenizer = _load_model(args.model, args.seqlen, args.device)
    arch = detect_arch(args.model)
    print(f"[arch] detected: {arch}")
    embed_dev = _resolve_embed_device(model, args.device)

    # original_config 저장
    orig_dir = os.path.join(args.save_dir, "original_config")
    os.makedirs(orig_dir, exist_ok=True)
    try:
        model.config.to_json_file(os.path.join(orig_dir, "config.json"))
    except Exception:
        pass
    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.to_json_file(os.path.join(orig_dir, "generation_config.json"))
    except Exception:
        pass
    tokenizer.save_pretrained(orig_dir)
    with open(os.path.join(orig_dir, "source.json"), "w", encoding="utf-8") as f:
        json.dump({"base_model": args.model}, f, ensure_ascii=False, indent=2)

    # ── [1] 캘리브레이션 로더 ──
    print("[1/3] Calibration loader (C4)")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        tokenizer=tokenizer,
    )

    # ── [2] 블록 선택 & 번들 저장 ──
    print("[2/3] Angular-distance block selection & export bundles")
    L_full = get_num_layers(model, arch)
    n = math.floor(args.drop_frac * L_full)

    orig_cfg_dir = os.path.join(args.save_dir, "original_config")

    if n <= 0:
        print("→ drop_frac <= 0, nothing to drop. Just saving A.")
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.save_dir)

        kept_idx = list(range(L_full))
        removed_indices, B_idx, C_idx = [], [], []

        manifest = {
            "version": "1.0",
            "base_model": args.model,
            "arch": arch,
            "counts": {
                "L_full": int(L_full),
                "A_kept": len(kept_idx),
                "removed_total": 0,
                "B": 0,
                "C": 0,
            },
            "selection": {
                "method": "angular_distance",
                "block": {"start": None, "n": 0},
                "angular_distance": None,
                "keep_last_layer": args.keep_last_layer,
            },
            "stages": {
                "A": {"kept_layers": kept_idx, "dropped_layers": removed_indices},
                "B": {"removed_layers": B_idx},
                "C": {"removed_layers": C_idx},
            },
            "artifacts": {
                "A": {"dir": os.path.abspath(args.save_dir)},
                "B": {"dir": os.path.abspath(os.path.join(args.save_removed_dir, "B"))},
                "C": {"dir": os.path.abspath(os.path.join(args.save_removed_dir, "C"))},
                "original_config": {"dir": os.path.abspath(orig_cfg_dir)},
            },
        }
        with open(os.path.join(args.save_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print("[3/3] Done.")
        return

    requested_selection_seqlen = args.seqlen
    if args.selection_seqlen is not None:
        requested_selection_seqlen = min(args.selection_seqlen, args.seqlen)
    retry_schedule = _build_selection_retry_schedule(
        requested_selection_seqlen,
        args.min_selection_seqlen,
    )

    best_ell, best_d, _L_old = None, None, None
    selection_failures = []
    if requested_selection_seqlen != args.seqlen:
        print(f"  [info] Using selection_seqlen={requested_selection_seqlen} for block search")

    for attempt_idx, selection_seqlen in enumerate(retry_schedule, start=1):
        if torch.cuda.is_available() and ("cuda" in str(embed_dev)):
            torch.cuda.empty_cache()
        try:
            if len(retry_schedule) > 1:
                print(
                    f"  [try {attempt_idx}/{len(retry_schedule)}] "
                    f"selection_seqlen={selection_seqlen}"
                )
            best_ell, best_d, _L_old = choose_block_to_drop(
                model,
                dataloader,
                embed_dev,
                n=n,
                arch=arch,
                keep_last_layer=args.keep_last_layer,
                max_batches=args.max_batches,
                input_seqlen=selection_seqlen,
            )
            break
        except RuntimeError as e:
            if not _should_retry_selection(e):
                raise
            selection_failures.append((selection_seqlen, str(e)))
            if attempt_idx == len(retry_schedule):
                raise RuntimeError(
                    _format_selection_failure(L_full, n, selection_failures)
                ) from e
            next_selection_seqlen = retry_schedule[attempt_idx]
            print(
                f"  [warn] Block selection failed at selection_seqlen={selection_seqlen}: {e}"
            )
            print(
                f"  [retry] Retrying block selection with selection_seqlen={next_selection_seqlen}"
            )

    if best_ell is None:
        raise RuntimeError(
            f"choose_block_to_drop returned None. "
            f"Activation capture may have failed. Check model/data compatibility."
        )

    if args.keep_last_layer and best_ell + n > L_full - 1:
        best_ell = max(0, L_full - 1 - n)

    removed_indices = list(range(best_ell, best_ell + n))
    if n > 0 and args.keep_last_layer:
        assert removed_indices[-1] < (L_full - 1), "keep_last_layer 위반"

    P_total = _count_params(model)
    P_drop = _count_params_of_layers(model, removed_indices, arch)

    # B/C 번들 저장
    os.makedirs(args.save_removed_dir, exist_ok=True)
    B_idx, C_idx = export_two_bundles(
        model=model,
        removed_indices=removed_indices,
        out_root=args.save_removed_dir,
        arch=arch,
        config=model.config,
        split_policy=args.split_policy,
        split_ratio=args.split_ratio,
    )
    print(f"→ Saved bundles: B({len(B_idx)}), C({len(C_idx)}) → {args.save_removed_dir}")

    # 드랍 적용
    model, _ = drop_consecutive_layers(model, best_ell, n, arch=arch)
    model = model.to(embed_dev)
    if torch.cuda.is_available() and ("cuda" in str(embed_dev)):
        torch.cuda.empty_cache()

    new_depth = get_num_layers(model, arch)
    print(f"→ Depth: {L_full} → {new_depth} (d_min={best_d:.4f}, n={n}, start={best_ell})")

    # ── [3] 저장 ──
    print("[3/3] Saving A & manifest")
    os.makedirs(args.save_dir, exist_ok=True)

    model.save_pretrained(args.save_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.save_dir)
    build_layers_map(model, arch, os.path.join(args.save_dir, "layers_map.json"))

    kept_idx = [i for i in range(L_full) if i not in removed_indices]

    prune_log = {
        "model": args.model,
        "arch": arch,
        "seed": args.seed,
        "seqlen": args.seqlen,
        "drop_frac": args.drop_frac,
        "keep_last_layer": args.keep_last_layer,
        "selected_block": {
            "start": best_ell,
            "n": n,
            "indices": removed_indices,
            "angular_distance": float(best_d),
        },
        "split": {
            "policy": args.split_policy,
            "ratio": args.split_ratio,
            "B": B_idx,
            "C": C_idx,
        },
        "params": {"P_total": int(P_total), "P_drop": int(P_drop)},
    }
    with open(os.path.join(args.save_dir, args.prune_log), "w", encoding="utf-8") as f:
        json.dump(prune_log, f, ensure_ascii=False, indent=2)

    manifest = {
        "version": "1.0",
        "base_model": args.model,
        "arch": arch,
        "counts": {
            "L_full": int(L_full),
            "A_kept": len(kept_idx),
            "removed_total": len(removed_indices),
            "B": len(B_idx),
            "C": len(C_idx),
        },
        "selection": {
            "method": "angular_distance",
            "block": {"start": int(best_ell), "n": int(n)},
            "angular_distance": float(best_d),
            "keep_last_layer": args.keep_last_layer,
        },
        "stages": {
            "A": {"kept_layers": kept_idx, "dropped_layers": removed_indices},
            "B": {"removed_layers": B_idx},
            "C": {"removed_layers": C_idx},
        },
        "artifacts": {
            "A": {"dir": os.path.abspath(args.save_dir)},
            "B": {"dir": os.path.abspath(os.path.join(args.save_removed_dir, "B"))},
            "C": {"dir": os.path.abspath(os.path.join(args.save_removed_dir, "C"))},
            "original_config": {"dir": os.path.abspath(orig_cfg_dir)},
        },
    }
    with open(os.path.join(args.save_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print(f"→ A saved to: {args.save_dir}")
    print(f"→ B/C bundles: {args.save_removed_dir}")


if __name__ == "__main__":
    main()
