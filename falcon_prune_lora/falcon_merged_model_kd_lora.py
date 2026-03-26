#!/usr/bin/env python3
"""
Falcon Progressive KD-LoRA 학습 코드 (LLaMA 로직 통일 버전)

핵심:
  LLaMA kd-lora 코드와 동일한 로직 사용:
  - 32층 skeleton 유지, 삭제된 위치에 FalconPassLayer 설치
  - _collapse_sparse_loaded_layers 제거
  - _ensure_original_layout으로 sparse/compact 양방향 대응
  - layers_to_transform으로 대상 레이어에만 LoRA 적용

Stage 1: A 로드 → 32층 유지 → B,C=PassLayer → A에만 LoRA
Stage 2: A_merged 로드 → 32층 유지 → B 복원 + C=PassLayer → B에만 LoRA
Stage 3: A_merged 로드 → 32층 유지 → B_merged+C 복원 → C에만 LoRA

사용법:
# Stage 1
<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=0,1 DEVICE=cuda:0 \
=======
CUDA_VISIBLE_DEVICES=2,3 DEVICE=cuda:0 \
>>>>>>> ea86042 (수정)
python -m falcon_prune_lora.falcon_merged_model_kd_lora \
  --base_dir ./falcon_results/pruning/A \
  --bundles_dir ./falcon_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./new_falcon_kd_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-4 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model tiiuae/falcon-7b-instruct \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0

# Stage 2 (A_merged 모델 사용)
CUDA_VISIBLE_DEVICES=0,1 DEVICE=cuda:0 \
python -m falcon_prune_lora.falcon_merged_model_kd_lora \
  --base_dir ./merged_models_falcon/A_merged \
  --bundles_dir ./falcon_results/pruning/bundles \
  --stage 2 \
  --out_adapters ./new_falcon_kd_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-5 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model tiiuae/falcon-7b-instruct \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0

# Stage 3
CUDA_VISIBLE_DEVICES=0,1 DEVICE=cuda:0 \
python -m falcon_prune_lora.falcon_merged_model_kd_lora \
  --base_dir ./merged_models_falcon/A_merged \
  --b_merged_dir ./merged_models_falcon/B_merged \
  --bundles_dir ./falcon_results/pruning/bundles/C \
  --stage 3 \
  --out_adapters ./new_falcon_kd_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-5 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model tiiuae/falcon-7b-instruct \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0
"""

import os, json, re, math, argparse
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, default_data_collator,
    BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
try:
    from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
except Exception:
    FalconDecoderLayer = None

# ============================================================
# Falcon 레이어 관리
# ============================================================
def _clean_auto_map(model_dir: str):
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        if "auto_map" in cfg:
            bak_path = cfg_path + ".bak"
            if not os.path.isfile(bak_path):
                import shutil
                shutil.copy2(cfg_path, bak_path)
            del cfg["auto_map"]
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"[config] Removed auto_map from {cfg_path}")
    except Exception as e:
        print(f"[warn] Failed to clean auto_map: {e}")


def _get_layer_container(model):
    """Falcon: model.transformer.h ModuleList 반환"""
    for attr_chain in [
        ("transformer", "h"),
        ("base_model", "model", "transformer", "h"),
        ("base_model", "transformer", "h"),
        ("model", "transformer", "h"),
    ]:
        obj = model
        try:
            for a in attr_chain:
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Falcon layer container not found (transformer.h)")


def _invalidate_layer_cache(model):
    """PeftModel 등에서 캐시 무효화 (필요시)"""
    pass  # Falcon은 _canonical_layers 캐시 미사용


def _layer_name_prefix(model, i: int):
    """Falcon 레이어의 named_parameter prefix 탐색"""
    for name, _ in model.named_parameters():
        if f".h.{i}." in name:
            idx = name.index(f".h.{i}.")
            return name[:idx] + f".h.{i}."
    return f"transformer.h.{i}."


# ============================================================
# FalconPassLayer
# ============================================================
class FalconPassLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden_states, alibi=None, attention_mask=None,
                position_ids=None, layer_past=None, head_mask=None,
                use_cache=False, output_attentions=False, **kwargs):
        if use_cache:
            return (hidden_states, layer_past)
        return (hidden_states,)


# ============================================================
# 원본 레이아웃 보장 (sparse/compact 양방향 대응) — LLaMA 로직 통일
# ============================================================
def _ensure_original_layout(model, removed_indices, original_num_layers):
    """
    모델을 원본 레이어 수(original_num_layers)로 맞추고,
    removed 위치에 FalconPassLayer를 설치.

    Case A (sparse): config=32로 로드, 21~28 MISSING → PassLayer 교체만
    Case B (compact): 물리적 축소 24층 → 32층 확장 + PassLayer 삽입

    Returns: (model, kept_indices)
    """
    layers = _get_layer_container(model)
    current_num = len(layers)
    removed_set = set(int(i) for i in removed_indices)
    kept = sorted(set(range(original_num_layers)) - removed_set)
    dev = next(model.parameters()).device
    hidden_size = model.config.hidden_size

    # ── Case A: sparse (이미 원본 크기) ──
    if current_num == original_num_layers:
        print(f"[layout] sparse mode: {current_num}층, "
              f"installing PassLayer at {sorted(removed_indices)}")
        for idx in removed_indices:
            layers[int(idx)] = FalconPassLayer(hidden_size).to(dev)
        return model, kept

    # ── Case B: compact (축소된 상태) ──
    expected_compact = len(kept)
    if current_num != expected_compact:
        raise ValueError(
            f"레이어 수 불일치: 모델 {current_num}층, "
            f"예상 compact={expected_compact} 또는 sparse={original_num_layers}")

    print(f"[layout] compact mode: {current_num}층 → {original_num_layers}층 expand")

    current_layers = [layers[i] for i in range(current_num)]
    new_layers = [None] * original_num_layers

    for pruned_idx, orig_idx in enumerate(kept):
        new_layers[orig_idx] = current_layers[pruned_idx]

    for idx in removed_indices:
        new_layers[int(idx)] = FalconPassLayer(hidden_size).to(dev)

    assert all(l is not None for l in new_layers), "확장 후 None 레이어 존재"

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        model.transformer.h = nn.ModuleList(new_layers)
    else:
        raise RuntimeError("model.transformer.h 경로를 찾을 수 없음")

    model.config.num_hidden_layers = original_num_layers

    print(f"  실제 레이어: {kept[:5]}{'...' if len(kept) > 5 else ''} ({len(kept)}개)")
    print(f"  PassLayer: {sorted(removed_indices)} ({len(removed_indices)}개)")
    return model, kept


# ============================================================
# 번들 관리
# ============================================================
def _load_bundle_indices(bundle_dir: str) -> list:
    meta_path = os.path.join(bundle_dir, "bundle_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return sorted(meta.get("indices", []))
    indices = []
    for fname in os.listdir(bundle_dir):
        m = re.match(r"layer_(\d+)\.safetensors", fname)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def _pick_layer_file(bundle_dir: str, idx: int) -> str:
    for fmt in [f"layer_{idx:03d}.safetensors", f"layer_{idx}.safetensors"]:
        p = os.path.join(bundle_dir, fmt)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Layer file not found for idx={idx} in {bundle_dir}")


def _extract_layer_sd(raw_sd: dict, idx: int):
    for pref in [f"transformer.h.{idx}.", f"h.{idx}."]:
        out = {k[len(pref):]: v for k, v in raw_sd.items() if k.startswith(pref)}
        if out:
            return out
    return raw_sd


def _assert_bundle_files_exist(bundle_dir: str, indices: list):
    missing = []
    for i in indices:
        try:
            f = _pick_layer_file(bundle_dir, int(i))
            if os.path.getsize(f) == 0:
                missing.append(i)
        except FileNotFoundError:
            missing.append(i)
    if missing:
        raise FileNotFoundError(f"[bundles] missing/empty: {missing} in {bundle_dir}")
    print(f"[bundles-ok] {len(indices)} files in {bundle_dir}")


def _rehydrate_layers(model, bundle_dir: str, indices: List[int]):
    """PassLayer → FalconDecoderLayer 복원"""
    if FalconDecoderLayer is None:
        raise RuntimeError("FalconDecoderLayer import 실패")
    layers = _get_layer_container(model)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    for i in indices:
        i = int(i)
        try:
            new_layer = FalconDecoderLayer(model.config, layer_idx=i).to(device=device, dtype=dtype)
        except TypeError:
            new_layer = FalconDecoderLayer(model.config).to(device=device, dtype=dtype)

        raw_sd = load_file(_pick_layer_file(bundle_dir, i))
        sd = _extract_layer_sd(raw_sd, i)
        sd = {k: v.to(device=device, dtype=dtype) for k, v in sd.items()}

        try:
            new_layer.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"[warn] layer {i}: {e} -> non-strict")
            new_layer.load_state_dict(sd, strict=False)

        layers[i] = new_layer
        print(f"[rehydrate] layer {i} restored")


# ============================================================
# 인덱스 정보 로드 — LLaMA 로직 통일
# ============================================================
def _load_index_info(base_dir: str, bundles_dir: str, stage: int,
                     b_merged_dir: str = None) -> dict:
    info = {"B": [], "C": [], "L_full": None}

    # manifest.json 우선
    man_path = os.path.join(base_dir, "manifest.json")
    if os.path.isfile(man_path):
        man = json.load(open(man_path))
        info["L_full"] = man.get("counts", {}).get("L_full")
        stages = man.get("stages", {})
        info["B"] = sorted(int(x) for x in stages.get("B", {}).get("removed_layers", []))
        info["C"] = sorted(int(x) for x in stages.get("C", {}).get("removed_layers", []))

    # prune_log.json fallback
    log_path = os.path.join(base_dir, "prune_log.json")
    if os.path.isfile(log_path):
        log = json.load(open(log_path))
        if not info["B"]:
            info["B"] = sorted(log.get("split", {}).get("B", []))
        if not info["C"]:
            info["C"] = sorted(log.get("split", {}).get("C", []))

    # bundle_meta.json fallback
    if not info["B"] and stage < 3:
        info["B"] = _load_bundle_indices(os.path.join(bundles_dir, "B"))
    if not info["C"]:
        c_dir = bundles_dir if stage == 3 else os.path.join(bundles_dir, "C")
        info["C"] = _load_bundle_indices(c_dir)

    if not info["L_full"]:
        all_idx = info["B"] + info["C"]
        if all_idx:
            info["L_full"] = max(all_idx) + 1

    return info


# ============================================================
# LoRA 어댑터
# ============================================================
def _detect_falcon_lora_targets(model) -> list:
    cfg = model.config
    new_arch = getattr(cfg, "new_decoder_architecture", False)
    if new_arch:
        targets = ["q_proj", "k_proj", "v_proj", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    else:
        targets = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    param_names = {n for n, _ in model.named_parameters()}
    verified = [t for t in targets if any(t in n for n in param_names)]
    if not verified:
        verified = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    print(f"[LoRA] target_modules: {verified}")
    return verified


def _attach_new_adapter(model, name: str, target_layers=None, r=8, alpha=16, dropout=0.05):
    target_modules = _detect_falcon_lora_targets(model)
    cfg_kwargs = dict(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules,
        bias="none", task_type="CAUSAL_LM",
    )
    if target_layers is not None:
        cfg_kwargs["layers_to_transform"] = target_layers

    cfg = LoraConfig(**cfg_kwargs)
    if isinstance(model, PeftModel):
        if name not in getattr(model, "peft_config", {}):
            model.add_adapter(name, cfg)
            print(f"[adapter] added '{name}'")
        return model
    return get_peft_model(model, cfg, adapter_name=name)


def _enable_only_lora_on_indices(model, indices: List[int], adapter_name: str):
    for p in model.parameters():
        p.requires_grad = False
    layer_patterns = [_layer_name_prefix(model, i) for i in indices]
    enabled = 0
    for name, param in model.named_parameters():
        if any(pat in name for pat in layer_patterns) and "lora_" in name.lower():
            param.requires_grad = True
            enabled += param.numel()
    if enabled == 0:
        print(f"[ERROR] No LoRA params on layers {indices}")
        for n, _ in model.named_parameters():
            if "lora_" in n.lower():
                print(f"  {n}")
        raise RuntimeError("No LoRA params enabled")
    print(f"[trainable] {adapter_name}: {enabled:,} params on {len(indices)} layers")
    return enabled


# ============================================================
# Teacher
# ============================================================
def load_teacher(args):
    if not args.use_kd or not args.teacher_model:
        return None
    print(f"\n[Teacher] Loading {args.teacher_model} on {args.teacher_device}")
    kwargs = {"torch_dtype": torch.float16, "device_map": {"": args.teacher_device}}
    if args.teacher_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, **kwargs)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


# ============================================================
# Dataset
# ============================================================
def _load_qa_sft_dataset(tokenizer, dataset_name, split, max_samples, seq_len):
    ds = load_dataset("squad" if dataset_name == "squad" else dataset_name, split=split)
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id

    def to_list(x):
        return x.tolist() if hasattr(x, "tolist") else list(x)

    def process(example):
        ctx = example.get("context", "")
        q = example.get("question", "")
        ans_list = example.get("answers", {}).get("text", [])
        ans = ans_list[0] if ans_list else ""
        if not ans:
            return {"__drop__": 1}

        sys_msg = "You are a helpful QA assistant."
        user_msg = f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer:"
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]

        try:
            prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_text = f"### System:\n{sys_msg}\n### User:\n{user_msg}\n### Assistant:\n"

        prompt_ids = to_list(tokenizer(prompt_text, add_special_tokens=True)["input_ids"])
        ans_ids = to_list(tokenizer(" " + ans, add_special_tokens=False)["input_ids"])
        if eos_id:
            ans_ids += [eos_id]
        if not ans_ids:
            return {"__drop__": 1}

        full = prompt_ids + ans_ids
        prompt_len = len(prompt_ids)
        if len(full) > seq_len:
            cut = len(full) - seq_len
            full = full[cut:]
            prompt_len = max(0, prompt_len - cut)

        pad_len = seq_len - len(full)
        input_ids = [pad_id] * pad_len + full
        attention_mask = [0] * pad_len + [1] * len(full)
        labels = input_ids[:]
        for i in range(pad_len + prompt_len):
            if i < len(labels):
                labels[i] = -100
        if pad_len + prompt_len >= seq_len:
            return {"__drop__": 1}

        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "__drop__": 0}

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["__drop__"] == 0)
    if "__drop__" in ds.column_names:
        ds = ds.remove_columns("__drop__")
    return ds


# ============================================================
# KD Trainer
# ============================================================
class KDTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, kd_alpha=0.1, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.kd_alpha = kd_alpha
        self.T = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        s_out = model(**inputs)
        s_logits_full = s_out.logits

        if self.teacher is None:
            loss = s_out.loss
            return (loss, s_out) if return_outputs else loss

        with torch.no_grad():
            t_dev = next(self.teacher.parameters()).device
            t_inputs = {k: v.to(t_dev) for k, v in inputs.items() if k != "labels"}
            t_out = self.teacher(**t_inputs)
            t_logits_full = t_out.logits.to(s_logits_full.device)

        s_logits = s_logits_full[:, :-1, :].contiguous()
        t_logits = t_logits_full[:, :-1, :].contiguous()
        labels_s = labels[:, 1:].contiguous()
        attn = inputs.get("attention_mask")
        attn_s = attn[:, 1:].contiguous() if attn is not None else torch.ones_like(labels_s)

        mask = (labels_s != -100) & (attn_s == 1)
        tok_n = mask.sum().item()
        if tok_n == 0:
            loss = s_logits_full.sum() * 0.0
            return (loss, s_out) if return_outputs else loss

        s = s_logits[mask].float().clamp(-50, 50)
        t = t_logits[mask].float().clamp(-50, 50)
        y = labels_s[mask]
        T = self.T

        soft_loss = F.kl_div(
            F.log_softmax(s / T, dim=-1), F.log_softmax(t / T, dim=-1),
            reduction="batchmean", log_target=True) * (T * T)
        hard_loss = F.cross_entropy(s, y)

        if not (torch.isfinite(soft_loss) and torch.isfinite(hard_loss)):
            loss = hard_loss if torch.isfinite(hard_loss) else s_logits_full.sum() * 0.0
        else:
            loss = self.kd_alpha * soft_loss + (1 - self.kd_alpha) * hard_loss

        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            self.log({"tok_n": float(tok_n),
                      "kd_soft": float(soft_loss) if torch.isfinite(soft_loss) else float("nan"),
                      "kd_hard": float(hard_loss),
                      "kd_ppl": float(math.exp(min(hard_loss.item(), 20)))})

        return (loss, s_out) if return_outputs else loss


# ============================================================
# 학습
# ============================================================
def train_adapter(model, out_dir, train_ds, eval_ds, args, adapter_name,
                  use_kd=False, teacher=None):
    os.makedirs(out_dir, exist_ok=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[train] {adapter_name}: {trainable:,} trainable params → {out_dir}")
    if trainable == 0:
        raise RuntimeError("No trainable parameters!")

    common = dict(
        output_dir=out_dir, per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_acc, learning_rate=args.lr,
        num_train_epochs=args.epochs, logging_strategy="steps",
        logging_steps=args.logging_steps, logging_first_step=True,
        max_grad_norm=args.max_grad_norm, warmup_ratio=args.warmup_ratio,
        remove_unused_columns=False, report_to="none",
        save_total_limit=args.save_total_limit)
    dtype_map = {"bf16": {"bf16": True, "fp16": False}, "fp16": {"fp16": True, "bf16": False}}
    common.update(dtype_map.get(args.dtype, {}))

    try:
        targs = TrainingArguments(**common,
            eval_strategy="steps" if args.eval_steps > 0 else "no",
            eval_steps=args.eval_steps if args.eval_steps > 0 else None,
            save_strategy="steps" if args.save_steps > 0 else "no",
            save_steps=args.save_steps if args.save_steps > 0 else None)
    except TypeError:
        targs = TrainingArguments(**common,
            evaluation_strategy="steps" if args.eval_steps > 0 else "no",
            eval_steps=args.eval_steps if args.eval_steps > 0 else None,
            save_strategy="steps" if args.save_steps > 0 else "no",
            save_steps=args.save_steps if args.save_steps > 0 else None)

    cls = KDTrainer if (use_kd and teacher) else Trainer
    kw = dict(model=model, args=targs, train_dataset=train_ds,
              eval_dataset=eval_ds, data_collator=default_data_collator)
    if cls is KDTrainer:
        kw.update(teacher_model=teacher, kd_alpha=args.kd_alpha, temperature=args.kd_T)

    cls(**kw).train()
    if isinstance(model, PeftModel):
        try:
            model.save_pretrained(out_dir, safe_serialization=True)
            print(f"[Saved] Adapter to {out_dir}")
        except Exception as e:
            print(f"[Error] Save failed: {e}")


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--bundles_dir", required=True)
    p.add_argument("--b_merged_dir", default=None, help="Stage 3: B_merged 번들 디렉토리")
    p.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--out_adapters", required=True)
    p.add_argument("--original_num_layers", type=int, default=None)

    p.add_argument("--qa_dataset", default="squad")
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--max_eval_samples", type=int, default=8000)
    p.add_argument("--seq_len", type=int, default=1024)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad_acc", type=int, default=32)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--save_total_limit", type=int, default=2)

    p.add_argument("--use_kd", action="store_true")
    p.add_argument("--teacher_model", default="tiiuae/falcon-7b-instruct")
    p.add_argument("--teacher_4bit", action="store_true")
    p.add_argument("--teacher_device", default="cuda:1")
    p.add_argument("--kd_alpha", type=float, default=0.1)
    p.add_argument("--kd_T", type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Tokenizer ──
    tok = AutoTokenizer.from_pretrained(args.base_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Student 로드 ──
    print(f"\n[Loading] Student from {args.base_dir}")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    _clean_auto_map(args.base_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=dtype_map[args.dtype],
        device_map=None, attn_implementation="eager")

    device = torch.device(os.environ.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:
        pass

    loaded_L = len(_get_layer_container(model))

    # ── 인덱스 정보 (LLaMA 로직 통일) ──
    info = _load_index_info(args.base_dir, args.bundles_dir, args.stage, args.b_merged_dir)
    B_idx, C_idx = info["B"], info["C"]

    original_N = (args.original_num_layers
                  or info["L_full"]
                  or model.config.num_hidden_layers)

    removed_all = sorted(set(B_idx + C_idx))
    A_idx = sorted(set(range(original_N)) - set(removed_all))

    print(f"\n[Index] original={original_N}, loaded={loaded_L}")
    print(f"  A: {A_idx[:5]}{'...' if len(A_idx) > 5 else ''} ({len(A_idx)}개)")
    print(f"  B: {B_idx} ({len(B_idx)}개)")
    print(f"  C: {C_idx} ({len(C_idx)}개)")

    # ── 원본 레이아웃 보장 (sparse/compact 자동 대응) ──
    model, kept = _ensure_original_layout(model, removed_all, original_N)
    layers = _get_layer_container(model)

    # ── Datasets ──
    print("\n[Loading] Datasets")
    train_ds = _load_qa_sft_dataset(tok, args.qa_dataset, "train", args.max_samples, args.seq_len)
    eval_ds = _load_qa_sft_dataset(tok, args.qa_dataset, "validation", args.max_eval_samples, args.seq_len)

    # ── Teacher ──
    teacher = load_teacher(args)

    # ================================================================
    # Stage 1: A 레이어에만 LoRA (B,C = PassLayer)
    # ================================================================
    if args.stage == 1:
        print("\n" + "=" * 60)
        print("STAGE 1: A-LoRA (A=real, B+C=PassLayer)")
        print("=" * 60)

        bad = [i for i in A_idx if (FalconDecoderLayer and not isinstance(layers[i], FalconDecoderLayer))]
        if bad:
            raise RuntimeError(f"A 위치에 비정상 레이어: {bad}")

        model = _attach_new_adapter(model, "stageA", target_layers=A_idx)
        model.set_adapter("stageA")
        _enable_only_lora_on_indices(model, A_idx, "stageA")

        out = os.path.join(args.out_adapters, "A_lora", "stageA")
        train_adapter(model, out, train_ds, eval_ds, args, "stageA", args.use_kd, teacher)

    # ================================================================
    # Stage 2: B 레이어에만 LoRA
    # ================================================================
    elif args.stage == 2:
        print("\n" + "=" * 60)
        print("STAGE 2: B-LoRA (A=merged, B=restored, C=PassLayer)")
        print("=" * 60)

        B_bundle_dir = os.path.join(args.bundles_dir, "B")
        _assert_bundle_files_exist(B_bundle_dir, B_idx)
        _rehydrate_layers(model, B_bundle_dir, B_idx)

        bad = [i for i in B_idx if (FalconDecoderLayer and not isinstance(layers[i], FalconDecoderLayer))]
        if bad:
            raise RuntimeError(f"B 복원 실패: {bad}")

        model = _attach_new_adapter(model, "stageB", target_layers=B_idx)
        model.set_adapter("stageB")
        _enable_only_lora_on_indices(model, B_idx, "stageB")

        out = os.path.join(args.out_adapters, "B_lora", "stageB")
        train_adapter(model, out, train_ds, eval_ds, args, "stageB", args.use_kd, teacher)

    # ================================================================
    # Stage 3: C 레이어에만 LoRA
    # ================================================================
    elif args.stage == 3:
        print("\n" + "=" * 60)
        print("STAGE 3: C-LoRA (A=merged, B=merged, C=restored)")
        print("=" * 60)

        if not args.b_merged_dir:
            raise ValueError("Stage 3 requires --b_merged_dir")

        B_merged_indices = _load_bundle_indices(args.b_merged_dir)
        if not B_merged_indices:
            B_merged_indices = B_idx
        _assert_bundle_files_exist(args.b_merged_dir, B_merged_indices)
        _rehydrate_layers(model, args.b_merged_dir, B_merged_indices)

        C_bundle_dir = args.bundles_dir
        _assert_bundle_files_exist(C_bundle_dir, C_idx)
        _rehydrate_layers(model, C_bundle_dir, C_idx)

        bad = [i for i in C_idx if (FalconDecoderLayer and not isinstance(layers[i], FalconDecoderLayer))]
        if bad:
            raise RuntimeError(f"C 복원 실패: {bad}")

        model = _attach_new_adapter(model, "stageC", target_layers=C_idx)
        model.set_adapter("stageC")
        _enable_only_lora_on_indices(model, C_idx, "stageC")

        out = os.path.join(args.out_adapters, "C_lora", "stageC")
        train_adapter(model, out, train_ds, eval_ds, args, "stageC", args.use_kd, teacher)

    print("\n[Done] Training completed")


if __name__ == "__main__":
    main()