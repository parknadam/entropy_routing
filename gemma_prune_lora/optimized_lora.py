#!/usr/bin/env python3
"""
Progressive 3-stage LoRA training for Gemma pruned layouts.

Stage 1: load pruned A -> restore original skeleton -> B/C = PassLayer -> LoRA on A
Stage 2: load A_merged -> restore original skeleton -> restore B + C = PassLayer -> LoRA on B
Stage 3: load A_merged -> restore original skeleton -> restore B_merged + C -> LoRA on C

Example:
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 \
python -m gemma_prune_lora.optimized_lora \
  --base_dir ./ori_20_gemma_7b_results/pruning/A \
  --bundles_dir ./ori_20_gemma_7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./gemma_7b_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-4 --epochs 2 --bs 1 --grad_acc 32
"""

import argparse
import inspect
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List

import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

try:
    from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
except Exception:
    GemmaDecoderLayer = None

try:
    from .pruning.model_utils import detect_layer_return_tuple as _shared_detect_layer_return_tuple
except ImportError:
    try:
        from gemma_prune_lora.pruning.model_utils import (
            detect_layer_return_tuple as _shared_detect_layer_return_tuple,
        )
    except ImportError:
        _shared_detect_layer_return_tuple = None


def detect_layer_return_tuple(model) -> bool:
    """
    Gemma helper가 구버전이라 shared utility가 없어도 optimized_lora 단독으로 동작하도록
    로컬 fallback을 둡니다.
    """
    try:
        core = model.model if hasattr(model, "model") else model
        src = inspect.getsource(core.forward)
        if "hidden_states = decoder_layer" in src and "layer_outputs[0]" not in src:
            return False
        if "layer_outputs[0]" in src or "layer_outputs = decoder_layer" in src:
            return True
    except Exception:
        pass

    try:
        sig = inspect.signature(GemmaDecoderLayer.forward)
        ret_ann = sig.return_annotation
        if ret_ann is torch.Tensor or str(ret_ann) == "torch.Tensor":
            return False
    except Exception:
        pass

    if _shared_detect_layer_return_tuple is not None:
        try:
            return bool(_shared_detect_layer_return_tuple(model))
        except Exception:
            pass
    return False


KST = timezone(timedelta(hours=9))
CANON_PATH = "model.layers"
_BUNDLE_LAYER_FILE_RE = re.compile(r"^layer_(\d+)\.safetensors$")


class EpochSaveCallback(TrainerCallback):
    def __init__(self, out_dir, adapter_name):
        self.out_dir = out_dir
        self.adapter_name = adapter_name

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch_n = int(state.epoch)
        save_path = os.path.join(self.out_dir, f"epoch_{epoch_n}")
        os.makedirs(save_path, exist_ok=True)
        if isinstance(model, PeftModel):
            try:
                model.save_pretrained(save_path, selected_adapters=[self.adapter_name])
            except TypeError:
                model.save_pretrained(save_path)
        metrics = {"epoch": epoch_n, "global_step": state.global_step}
        recent = [log for log in (state.log_history or []) if "loss" in log]
        if recent:
            metrics["last_train_loss"] = recent[-1]["loss"]
        evals = [log for log in (state.log_history or []) if "eval_loss" in log]
        if evals:
            metrics["last_eval_loss"] = evals[-1]["eval_loss"]
        with open(os.path.join(save_path, "epoch_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[checkpoint] epoch {epoch_n} saved -> {save_path}")


class _GemmaTrainPassLayer(nn.Module):
    """
    Gemma training path 전용 tensor-only pass layer.
    현재 transformers Gemma decoder contract와 동일하게 hidden_states tensor만 반환합니다.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.return_tuple = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        return hidden_states


def _write_readme(out_dir, start_time, end_time=None, args=None, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    readme = os.path.join(out_dir, "README.md")

    cmd = " ".join(sys.argv)
    elapsed = ""
    if end_time and start_time:
        dt = end_time - start_time
        h, rem = divmod(int(dt), 3600)
        m, s = divmod(rem, 60)
        elapsed = f"{h}h {m}m {s}s"

    lines = []
    if not os.path.isfile(readme):
        lines.append("# LoRA Training Log\n")

    lines.append(f"\n## Run @ {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')}\n")
    lines.append(f"- **Command**: `{cmd}`")
    lines.append(
        f"- **Start**: {datetime.fromtimestamp(start_time, KST).strftime('%Y-%m-%d %H:%M:%S KST')}"
    )
    if end_time:
        lines.append(
            f"- **End**: {datetime.fromtimestamp(end_time, KST).strftime('%Y-%m-%d %H:%M:%S KST')}"
        )
        lines.append(f"- **Elapsed**: {elapsed}")
    if args:
        lines.append(f"- **Stage**: {args.stage}")
        lines.append(f"- **LR**: {args.lr}, **Epochs**: {args.epochs}, **BS**: {args.bs}x{args.grad_acc}")
        lines.append(f"- **Seq len**: {args.seq_len}, **Dataset**: {args.qa_dataset}")
    if extra:
        for key, value in extra.items():
            lines.append(f"- **{key}**: {value}")
    lines.append("")

    with open(readme, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[readme] logged -> {readme}")


def _resolve(root, dotted):
    parent = root
    for seg in dotted.split(".")[:-1]:
        parent = getattr(parent, seg)
    last = dotted.split(".")[-1]
    return parent, last, getattr(parent, last)


def _canonicalize(model):
    for path in (
        "model.layers",
        "model.model.layers",
        "base_model.model.layers",
        "base_model.model.model.layers",
    ):
        try:
            parent, name, cur = _resolve(model, path)
            if not hasattr(cur, "__len__"):
                continue
            if not isinstance(cur, (list, nn.ModuleList)):
                cur = nn.ModuleList(list(cur))
                setattr(parent, name, cur)
            try:
                cp, _, _ = _resolve(model, CANON_PATH.replace(".layers", ""))
                setattr(cp, "layers", cur)
                model._clp = CANON_PATH
            except Exception:
                model._clp = path
            model._cl = cur
            return cur
        except Exception:
            continue
    raise AttributeError("decoder layers not found")


def _layers(model):
    if not hasattr(model, "_cl"):
        _canonicalize(model)
    return model._cl


def _invalidate(model):
    for attr in ("_cl", "_clp"):
        if hasattr(model, attr):
            delattr(model, attr)


def _prefix(model, idx):
    if not hasattr(model, "_clp"):
        _canonicalize(model)
    return f"{model._clp}.{idx}."


def _short(indices, limit=8):
    if len(indices) <= limit:
        return str(indices)
    return f"{indices[:limit]}..."


def _is_decoder_layer(module):
    return GemmaDecoderLayer is not None and isinstance(module, GemmaDecoderLayer)


def _make_pass_layer(hidden_size, return_tuple, device=None, dtype=None):
    """
    optimized_lora는 Gemma 전용 학습 경로이므로 tensor-only pass layer를 강제합니다.
    외부 pruning.identity.PassLayer 버전 차이에 영향받지 않도록 로컬 구현을 사용합니다.
    """
    del return_tuple
    layer = _GemmaTrainPassLayer(hidden_size)

    if device is not None:
        try:
            layer = layer.to(device=device, dtype=dtype)
        except TypeError:
            layer = layer.to(device=device)
    return layer


def _ensure_original_layout(model, removed_indices, original_num_layers):
    layers = _layers(model)
    current_num_layers = len(layers)
    removed = sorted(set(int(i) for i in removed_indices))
    kept = sorted(set(range(original_num_layers)) - set(removed))
    hidden_size = int(model.config.hidden_size)
    # Gemma current HF runtime expects decoder layers to return a tensor, not a tuple.
    return_tuple = False

    try:
        ref_param = next(model.parameters())
        device, dtype = ref_param.device, ref_param.dtype
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.float32

    if current_num_layers == original_num_layers:
        print(f"[layout] sparse skeleton: {current_num_layers}L, PassLayer at {removed}")
        for idx in removed:
            old = layers[idx]
            layers[idx] = _make_pass_layer(
                hidden_size,
                return_tuple,
                device=device,
                dtype=dtype,
            )
            del old
        return model, kept

    if current_num_layers != len(kept):
        raise ValueError(
            f"layer mismatch: loaded={current_num_layers}, expected compact={len(kept)} or sparse={original_num_layers}"
        )

    print(f"[layout] compact: {current_num_layers}L -> {original_num_layers}L expand")
    old_layers = [layers[i] for i in range(current_num_layers)]
    new_layers = [None] * original_num_layers
    for packed_idx, original_idx in enumerate(kept):
        new_layers[original_idx] = old_layers[packed_idx]
    for idx in removed:
        new_layers[idx] = _make_pass_layer(
            hidden_size,
            return_tuple,
            device=device,
            dtype=dtype,
        )

    if any(layer is None for layer in new_layers):
        raise RuntimeError("expanded layout contains empty layer slots")

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = nn.ModuleList(new_layers)
    elif hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        model.model.model.layers = nn.ModuleList(new_layers)
    else:
        raise RuntimeError("cannot find Gemma decoder layers path")

    model.config.num_hidden_layers = original_num_layers
    _invalidate(model)
    print(f"  real: {_short(kept)} ({len(kept)}), pass: {removed} ({len(removed)})")
    return model, kept


def _pick_bundle_layer_file(bundle_dir, idx):
    for fname in (f"layer_{int(idx):03d}.safetensors", f"layer_{int(idx)}.safetensors"):
        path = os.path.join(bundle_dir, fname)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"layer file missing: idx={idx} in {bundle_dir}")


def _extract_layer_state_dict(raw_state_dict, idx):
    prefixes = (
        f"model.layers.{idx}.",
        f"model.model.layers.{idx}.",
        f"layers.{idx}.",
        f"base_model.model.model.layers.{idx}.",
    )
    for prefix in prefixes:
        out = {key[len(prefix):]: value for key, value in raw_state_dict.items() if key.startswith(prefix)}
        if out:
            return out
    return raw_state_dict


def _load_bundle_indices(bundle_dir):
    meta_path = os.path.join(bundle_dir, "bundle_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        indices = meta.get("indices", []) or meta.get("layer_indices", [])
        if indices:
            return sorted(int(i) for i in indices)
    if not os.path.isdir(bundle_dir):
        return []
    out = []
    for fname in sorted(os.listdir(bundle_dir)):
        match = _BUNDLE_LAYER_FILE_RE.match(fname)
        if match:
            out.append(int(match.group(1)))
    return sorted(set(out))


def _assert_bundles(bundle_dir, indices):
    missing = []
    for idx in indices:
        try:
            path = _pick_bundle_layer_file(bundle_dir, idx)
            if os.path.getsize(path) == 0:
                missing.append(idx)
        except FileNotFoundError:
            missing.append(idx)
    if missing:
        raise FileNotFoundError(f"[bundles] missing/empty: {missing} in {bundle_dir}")
    print(f"[bundles-ok] {len(indices)} files in {bundle_dir}")


def _rehydrate(model, bundle_dir, indices):
    if GemmaDecoderLayer is None:
        raise RuntimeError("GemmaDecoderLayer import failed. Check transformers version.")

    layers = _layers(model)
    try:
        ref_param = next(model.parameters())
        device, dtype = ref_param.device, ref_param.dtype
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.float32

    for idx in indices:
        try:
            new_layer = GemmaDecoderLayer(model.config, int(idx))
        except TypeError:
            new_layer = GemmaDecoderLayer(model.config)
        new_layer = new_layer.to(device=device, dtype=dtype)

        raw_state_dict = load_file(_pick_bundle_layer_file(bundle_dir, int(idx)))
        layer_state_dict = {
            key: value.to(device=device, dtype=dtype)
            for key, value in _extract_layer_state_dict(raw_state_dict, int(idx)).items()
        }
        try:
            new_layer.load_state_dict(layer_state_dict, strict=True)
        except RuntimeError:
            new_layer.load_state_dict(layer_state_dict, strict=False)
        old = layers[int(idx)]
        layers[int(idx)] = new_layer
        del old
        print(f"[rehydrate] layer {idx} restored")


def _attach(model, name, target_layers=None, r=8, alpha=16, dropout=0.05):
    kwargs = dict(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if target_layers is not None:
        kwargs["layers_to_transform"] = target_layers
    config = LoraConfig(**kwargs)
    if isinstance(model, PeftModel):
        if name not in getattr(model, "peft_config", {}):
            model.add_adapter(name, config)
        return model
    return get_peft_model(model, config, adapter_name=name)


def _enable_lora_only(model, indices, adapter_name):
    for param in model.parameters():
        param.requires_grad = False
    patterns = [_prefix(model, idx) for idx in indices]
    enabled_params = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns) and "lora_" in name.lower():
            param.requires_grad = True
            enabled_params += param.numel()
    if enabled_params == 0:
        raise RuntimeError(f"No LoRA params on layers {indices} for '{adapter_name}'")
    print(f"[trainable] {adapter_name}: {enabled_params:,} params on {len(indices)} layers")


def _build_msgs(context, question, dataset_name):
    system = "You are a helpful QA assistant."
    if dataset_name == "squad_v2":
        system += " If the answer is not in the context, say 'unanswerable'."
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "Answer the question using the context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                "Answer:"
            ),
        },
    ]


def _load_qa_dataset(tokenizer, dataset_name, split, max_samples, seq_len):
    dataset_map = {"squad": "rajpurkar/squad", "squad_v2": "rajpurkar/squad_v2"}
    dataset = load_dataset(dataset_map.get(dataset_name, dataset_name), split=split)
    if max_samples:
        dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    has_chat = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None

    def _to_list(x):
        if hasattr(x, "input_ids"):
            x = x.input_ids
        elif isinstance(x, dict):
            x = x.get("input_ids", x)
        if hasattr(x, "tolist"):
            x = x.tolist()
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
            x = x[0]
        return list(x) if x else []

    def process(example):
        context = example.get("context", "")
        question = example.get("question", "")
        answer = example.get("answers", {}).get("text", [""])[0] or (
            "unanswerable" if dataset_name == "squad_v2" else ""
        )
        messages = _build_msgs(context, question, dataset_name)

        if has_chat:
            prompt_ids = _to_list(
                tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            )
        else:
            prompt_text = (
                f"{messages[0]['content']}\n\n"
                f"{messages[1]['content']}"
            )
            prompt_ids = _to_list(
                tokenizer(
                    prompt_text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=seq_len - 64,
                )["input_ids"]
            )

        answer_ids = _to_list(tokenizer(" " + answer, add_special_tokens=False)["input_ids"])
        if eos_id is not None:
            answer_ids += [eos_id]
        if not answer_ids:
            return {"__drop__": 1}

        full = prompt_ids + answer_ids
        prompt_len = len(prompt_ids)
        if len(full) > seq_len:
            cut = len(full) - seq_len
            full = full[cut:]
            prompt_len = max(0, prompt_len - cut)
        pad_n = seq_len - len(full)
        input_ids = [pad_id] * pad_n + full
        attention_mask = [0] * pad_n + [1] * len(full)
        labels = input_ids[:]
        for idx in range(pad_n + prompt_len):
            if idx < len(labels):
                labels[idx] = -100
        if pad_n + prompt_len >= seq_len:
            return {"__drop__": 1}
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "__drop__": 0,
        }

    dataset = dataset.map(process, remove_columns=dataset.column_names, num_proc=4)
    dataset = dataset.filter(lambda x: x["__drop__"] == 0)
    if "__drop__" in dataset.column_names:
        dataset = dataset.remove_columns("__drop__")
    return dataset


def _load_index_info(base_dir, bundles_dir, stage, b_merged_dir=None):
    info = {"B": [], "C": [], "L_full": None}

    manifest_path = os.path.join(base_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        info["L_full"] = manifest.get("counts", {}).get("L_full")
        stages = manifest.get("stages", {})
        info["B"] = sorted(int(x) for x in stages.get("B", {}).get("removed_layers", []))
        info["C"] = sorted(int(x) for x in stages.get("C", {}).get("removed_layers", []))

    log_path = os.path.join(base_dir, "prune_log.json")
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            prune_log = json.load(f)
        if not info["B"]:
            info["B"] = sorted(int(x) for x in prune_log.get("split", {}).get("B", []))
        if not info["C"]:
            info["C"] = sorted(int(x) for x in prune_log.get("split", {}).get("C", []))

    if not info["B"] and b_merged_dir:
        info["B"] = _load_bundle_indices(b_merged_dir)

    if not info["B"] and stage < 3 and bundles_dir:
        info["B"] = _load_bundle_indices(os.path.join(bundles_dir, "B"))

    if not info["C"] and bundles_dir:
        c_dir = bundles_dir if stage == 3 else os.path.join(bundles_dir, "C")
        info["C"] = _load_bundle_indices(c_dir)

    return info


def train_adapter(model, tokenizer, out_dir, train_ds, eval_ds, args, adapter_name):
    os.makedirs(out_dir, exist_ok=True)
    num_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"\n[train] {adapter_name}: {num_trainable:,} trainable -> {out_dir}")
    if num_trainable == 0:
        raise RuntimeError("No trainable params!")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    common = dict(
        output_dir=out_dir,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=use_bf16,
        fp16=not use_bf16,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=args.save_total_limit,
    )
    try:
        training_args = TrainingArguments(
            **common,
            eval_strategy="steps" if args.eval_steps > 0 else "no",
            eval_steps=args.eval_steps if args.eval_steps > 0 else None,
            save_strategy="steps" if args.save_steps > 0 else "no",
            save_steps=args.save_steps if args.save_steps > 0 else None,
        )
    except TypeError:
        training_args = TrainingArguments(
            **common,
            evaluation_strategy="steps" if args.eval_steps > 0 else "no",
            eval_steps=args.eval_steps if args.eval_steps > 0 else None,
            save_strategy="steps" if args.save_steps > 0 else "no",
            save_steps=args.save_steps if args.save_steps > 0 else None,
        )

    epoch_callback = EpochSaveCallback(out_dir, adapter_name)
    t0 = time.time()
    Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        processing_class=tokenizer,
        callbacks=[epoch_callback],
    ).train()
    train_elapsed = time.time() - t0

    if isinstance(model, PeftModel):
        try:
            model.save_pretrained(out_dir, selected_adapters=[adapter_name])
        except TypeError:
            model.save_pretrained(out_dir)
    return train_elapsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--bundles_dir", required=True)
    parser.add_argument("--b_merged_dir", default=None, help="Stage 3: B_merged bundle dir")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--out_adapters", required=True)
    parser.add_argument("--original_num_layers", type=int, default=None)

    parser.add_argument("--qa_dataset", default="squad")
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--max_eval_samples", type=int, default=8000)
    parser.add_argument("--seq_len", type=int, default=1024)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=32)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    loaded_num_layers = len(_layers(model))
    info = _load_index_info(args.base_dir, args.bundles_dir, args.stage, args.b_merged_dir)
    b_indices = info["B"]
    c_indices = info["C"]
    original_num_layers = args.original_num_layers or info["L_full"] or model.config.num_hidden_layers
    removed_all = sorted(set(b_indices + c_indices))
    a_indices = sorted(set(range(original_num_layers)) - set(removed_all))

    print(f"\n[Index] original={original_num_layers}, loaded={loaded_num_layers}")
    print(f"  A({len(a_indices)}): {_short(a_indices)}")
    print(f"  B({len(b_indices)}): {b_indices}")
    print(f"  C({len(c_indices)}): {c_indices}")

    model, _ = _ensure_original_layout(model, removed_all, original_num_layers)
    layers = _layers(model)

    print("\n[Loading] Datasets")
    train_ds = _load_qa_dataset(tokenizer, args.qa_dataset, "train", args.max_samples, args.seq_len)
    eval_ds = _load_qa_dataset(tokenizer, args.qa_dataset, "validation", args.max_eval_samples, args.seq_len)
    print(f"  train={len(train_ds)}, eval={len(eval_ds)}")

    readme_extra = {
        "A_layers": len(a_indices),
        "B_layers": len(b_indices),
        "C_layers": len(c_indices),
        "original_N": original_num_layers,
        "loaded_L": loaded_num_layers,
    }

    if args.stage == 1:
        print(f"\n{'=' * 60}\nSTAGE 1: A-LoRA (A=real, B+C=PassLayer)\n{'=' * 60}")
        bad = [idx for idx in a_indices if not _is_decoder_layer(layers[idx])]
        if bad:
            raise RuntimeError(f"A layer mismatch: {bad}")

        model = _attach(model, "stageA", target_layers=a_indices)
        model.set_adapter("stageA")
        _enable_lora_only(model, a_indices, "stageA")

        out_dir = os.path.join(args.out_adapters, "A_lora", "stageA")
        train_elapsed = train_adapter(model, tokenizer, out_dir, train_ds, eval_ds, args, "stageA")
        readme_extra["train_time"] = f"{train_elapsed / 60:.1f} min"
        _write_readme(out_dir, start_time, time.time(), args, readme_extra)
        print(
            f"\n[Next] Merge: merge_adapter.py --base_model {args.base_dir} "
            f"--adapter_path {out_dir} --output_dir ./merged_models/A_merged"
        )

    elif args.stage == 2:
        print(f"\n{'=' * 60}\nSTAGE 2: B-LoRA (A=merged, B=restored, C=PassLayer)\n{'=' * 60}")
        b_bundle_dir = os.path.join(args.bundles_dir, "B")
        _assert_bundles(b_bundle_dir, b_indices)
        _rehydrate(model, b_bundle_dir, b_indices)

        bad = [idx for idx in b_indices if not _is_decoder_layer(layers[idx])]
        if bad:
            raise RuntimeError(f"B restore mismatch: {bad}")

        model = _attach(model, "stageB", target_layers=b_indices)
        model.set_adapter("stageB")
        _enable_lora_only(model, b_indices, "stageB")

        out_dir = os.path.join(args.out_adapters, "B_lora", "stageB")
        train_elapsed = train_adapter(model, tokenizer, out_dir, train_ds, eval_ds, args, "stageB")
        readme_extra["train_time"] = f"{train_elapsed / 60:.1f} min"
        _write_readme(out_dir, start_time, time.time(), args, readme_extra)
        print("\n[Next] Merge B adapter with B bundle -> B_merged")

    else:
        print(f"\n{'=' * 60}\nSTAGE 3: C-LoRA (A=merged, B=merged, C=restored)\n{'=' * 60}")
        if not args.b_merged_dir:
            raise ValueError("Stage 3 requires --b_merged_dir")

        b_merged_indices = _load_bundle_indices(args.b_merged_dir) or b_indices
        _assert_bundles(args.b_merged_dir, b_merged_indices)
        _rehydrate(model, args.b_merged_dir, b_merged_indices)

        c_bundle_dir = args.bundles_dir
        _assert_bundles(c_bundle_dir, c_indices)
        _rehydrate(model, c_bundle_dir, c_indices)

        bad = [idx for idx in c_indices if not _is_decoder_layer(layers[idx])]
        if bad:
            raise RuntimeError(f"C restore mismatch: {bad}")

        model = _attach(model, "stageC", target_layers=c_indices)
        model.set_adapter("stageC")
        _enable_lora_only(model, c_indices, "stageC")

        out_dir = os.path.join(args.out_adapters, "C_lora", "stageC")
        train_elapsed = train_adapter(model, tokenizer, out_dir, train_ds, eval_ds, args, "stageC")
        readme_extra["train_time"] = f"{train_elapsed / 60:.1f} min"
        _write_readme(out_dir, start_time, time.time(), args, readme_extra)
        print("\n[Next] Merge C adapter with C bundle -> C_merged")

    print("\n[Done] Training completed")


if __name__ == "__main__":
    main()
