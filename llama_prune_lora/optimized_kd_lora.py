#!/usr/bin/env python3
"""
LLaMA 구조용 최적화 KD-LoRA 학습 코드 (sparse/compact 양방향 대응)

핵심 수정:
  drop_consecutive_layers()는 PassLayer로 '교체'하며 삭제하지 않음.
  save_pretrained()는 config.num_hidden_layers=32를 유지한 채 저장.
  → from_pretrained() 시 32레이어 생성, dropped 위치는 MISSING(랜덤 초기화).
  → _ensure_original_layout()이 sparse/compact 양쪽 모두 처리.

Stage 1: A 로드 → 원본 레이아웃 보장 → B,C=PassLayer → A에만 LoRA
Stage 2: A_merged 로드 → 원본 레이아웃 보장 → B 복원 + C=PassLayer → B에만 LoRA
Stage 3: A_merged 로드 → 원본 레이아웃 보장 → B_merged+C 복원 → C에만 LoRA

사용법:
# Stage 1
CUDA_VISIBLE_DEVICES=2,5 DEVICE=cuda:0 \
python -m llama_prune_lora.optimized_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./llama_kd_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-4 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0

# Stage 2 (A_merged 기준)
CUDA_VISIBLE_DEVICES=2,4 DEVICE=cuda:0 \
python -m llama_prune_lora.optimized_kd_lora \
  --base_dir ./merged_models_llama_7b/A_merged \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 2 \
  --out_adapters ./llama_kd_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-5 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0

# Stage 3 (A_merged + B_merged)
CUDA_VISIBLE_DEVICES=0,1 DEVICE=cuda:0 \
python -m llama_prune_lora.optimized_kd_lora \
  --base_dir ./merged_models_llama_7b/A_merged \
  --b_merged_dir ./merged_models_llama_7b/B_merged \
  --bundles_dir ./7b_results/pruning/bundles/C \
  --stage 3 \
  --out_adapters ./llama_kd_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-5 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0
"""

import os, json, re, math, inspect
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, default_data_collator
)
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# ============================================================
# 레이어 관리
# ============================================================
CANON_PATH = "model.layers"


def _resolve_attr_path(root, dotted: str):
    parent = root
    for seg in dotted.split(".")[:-1]:
        parent = getattr(parent, seg)
    last = dotted.split(".")[-1]
    return parent, last, getattr(parent, last)


def _canonicalize_layers(model):
    candidates = [
        "model.layers", "model.model.layers",
        "base_model.model.layers", "base_model.model.model.layers",
        "model.decoder.layers", "model.model.decoder.layers",
    ]
    for path in candidates:
        try:
            parent, name, cur = _resolve_attr_path(model, path)
            if hasattr(cur, "__len__") and hasattr(cur, "__getitem__"):
                if not isinstance(cur, (list, nn.ModuleList)):
                    cur = nn.ModuleList(list(cur))
                    setattr(parent, name, cur)
                try:
                    canon_parent, _, _ = _resolve_attr_path(model, CANON_PATH.replace(".layers", ""))
                    setattr(canon_parent, "layers", cur)
                    model._canonical_layers_path = CANON_PATH
                except Exception:
                    model._canonical_layers_path = path
                model._canonical_layers = cur
                return cur
        except Exception:
            continue
    raise AttributeError("decoder layers not found")


def _get_layer_container(model):
    if not hasattr(model, "_canonical_layers"):
        _canonicalize_layers(model)
    return model._canonical_layers


def _invalidate_layer_cache(model):
    for attr in ("_canonical_layers", "_canonical_layers_path"):
        if hasattr(model, attr):
            delattr(model, attr)


def _layer_name_prefix(model, i: int):
    if not hasattr(model, "_canonical_layers_path"):
        _canonicalize_layers(model)
    return f"{model._canonical_layers_path}.{i}."


# ============================================================
# PassLayer
# ============================================================
def _decoder_layer_returns_tuple() -> bool:
    try:
        sig = inspect.signature(LlamaDecoderLayer.forward)
    except Exception:
        return True
    return "output_attentions" in sig.parameters


class LlamaPassLayer(nn.Module):
    def __init__(self, return_tuple: bool = True):
        super().__init__()
        self.return_tuple = return_tuple

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        if not self.return_tuple:
            return hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (None,)
        return outputs


# ============================================================
# 원본 레이아웃 보장 (sparse/compact 양방향 대응)
# ============================================================
def _ensure_original_layout(model, removed_indices, original_num_layers):
    """
    모델을 원본 레이어 수(original_num_layers)로 맞추고,
    removed 위치에 PassLayer를 설치.

    두 가지 케이스 자동 처리:
      (a) sparse: from_pretrained가 config대로 32레이어 생성,
          dropped 위치는 MISSING(랜덤 초기화) → PassLayer로 교체만
      (b) compact: 물리적으로 축소된 24레이어 모델 →
          원본 크기로 확장 후 PassLayer 삽입

    Returns: (model, kept_indices)
    """
    layers = _get_layer_container(model)
    current_num = len(layers)
    removed_set = set(int(i) for i in removed_indices)
    kept = sorted(set(range(original_num_layers)) - removed_set)
    use_tuple = _decoder_layer_returns_tuple()
    dev = next(model.parameters()).device

    # ── Case A: sparse (이미 원본 크기) ──
    if current_num == original_num_layers:
        print(f"[layout] sparse mode: {current_num}층, "
              f"installing PassLayer at {sorted(removed_indices)}")
        for idx in removed_indices:
            layers[int(idx)] = LlamaPassLayer(return_tuple=use_tuple).to(dev)
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

    # A 레이어를 원본 위치에 배치
    for pruned_idx, orig_idx in enumerate(kept):
        new_layers[orig_idx] = current_layers[pruned_idx]

    # removed 위치에 PassLayer
    for idx in removed_indices:
        new_layers[int(idx)] = LlamaPassLayer(return_tuple=use_tuple).to(dev)

    assert all(l is not None for l in new_layers), "확장 후 None 레이어 존재"

    # ModuleList 교체
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model.model.layers = nn.ModuleList(new_layers)
    elif hasattr(model, 'model') and hasattr(model.model, 'model'):
        model.model.model.layers = nn.ModuleList(new_layers)
    else:
        raise RuntimeError("model.model.layers 경로를 찾을 수 없음")

    model.config.num_hidden_layers = original_num_layers
    _invalidate_layer_cache(model)

    print(f"  실제 레이어: {kept[:5]}{'...' if len(kept) > 5 else ''} ({len(kept)}개)")
    print(f"  PassLayer: {sorted(removed_indices)} ({len(removed_indices)}개)")
    return model, kept


# ============================================================
# 번들 관리
# ============================================================
def _extract_layer_sd(raw_sd: dict, idx: int):
    for pref in [f"model.layers.{idx}.", f"model.model.layers.{idx}.", f"layers.{idx}."]:
        out = {k[len(pref):]: v for k, v in raw_sd.items() if k.startswith(pref)}
        if out:
            return out
    return raw_sd


def _pick_layer_file(bundle_dir: str, idx: int) -> str:
    for fmt in [f"layer_{int(idx):03d}.safetensors", f"layer_{int(idx)}.safetensors"]:
        p = os.path.join(bundle_dir, fmt)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"layer file missing for {idx} in {bundle_dir}")


def _load_bundle_indices(bundle_dir: str) -> list:
    meta = os.path.join(bundle_dir, "bundle_meta.json")
    if os.path.isfile(meta):
        with open(meta) as f:
            return sorted(json.load(f).get("indices", []))
    indices = []
    for fname in os.listdir(bundle_dir):
        m = re.match(r"layer_(\d+)\.safetensors", fname)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


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
    """원본 인덱스 위치에 번들 레이어 복원 (PassLayer → LlamaDecoderLayer)."""
    layers = _get_layer_container(model)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    for i in indices:
        try:
            new_layer = LlamaDecoderLayer(model.config, layer_idx=int(i))
        except TypeError:
            new_layer = LlamaDecoderLayer(model.config)
        new_layer = new_layer.to(device=device, dtype=dtype)

        raw_sd = load_file(_pick_layer_file(bundle_dir, int(i)))
        sd = _extract_layer_sd(raw_sd, int(i))
        sd = {k: v.to(device=device, dtype=dtype) for k, v in sd.items()}

        try:
            new_layer.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"[warn] layer {i}: {e} -> non-strict")
            new_layer.load_state_dict(sd, strict=False)

        layers[int(i)] = new_layer
        print(f"[rehydrate] layer {i} restored")


# ============================================================
# LoRA 어댑터
# ============================================================
def _attach_new_adapter(model, name: str, target_layers=None, r=8, alpha=16, dropout=0.05):
    cfg_kwargs = dict(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
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
# 데이터셋
# ============================================================
def _build_chat_messages(ctx: str, q: str, dataset: str):
    sys = "You are a helpful QA assistant."
    if dataset == "squad_v2":
        sys += " If the answer is not in the context, say 'unanswerable'."
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Answer the question using the context.\n\nContext:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:"},
    ]


def _load_qa_sft_dataset(tokenizer, qa_dataset, split, max_samples, seq_len):
    DATASET_MAP = {"squad": "rajpurkar/squad", "squad_v2": "rajpurkar/squad_v2"}
    ds = load_dataset(DATASET_MAP.get(qa_dataset, qa_dataset), split=split)
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    has_chat = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None

    def to_list(x):
        if hasattr(x, "input_ids"):
            x = x.input_ids
        elif isinstance(x, dict):
            x = x.get("input_ids", x)
        if hasattr(x, "tolist"):
            x = x.tolist()
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
            x = x[0]
        return list(x) if x else []

    def process(ex):
        ctx, q = ex.get("context", ""), ex.get("question", "")
        ans = (ex.get("answers", {}).get("text", [""])[0] or
               ("unanswerable" if qa_dataset == "squad_v2" else ""))
        messages = _build_chat_messages(ctx, q, qa_dataset)

        if has_chat:
            prompt_ids = to_list(tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True))
        else:
            sys_msg, user_msg = messages[0]["content"], messages[1]["content"]
            prompt = f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{user_msg} [/INST] "
            prompt_ids = to_list(tokenizer(prompt, add_special_tokens=False,
                                           truncation=True, max_length=seq_len - 64)["input_ids"])

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
        if teacher_model is None:
            raise ValueError("KDTrainer requires teacher_model")
        self.teacher_model = teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.kd_alpha = float(kd_alpha)
        self.T = float(temperature)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        s_out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        s_logits_full = s_out.logits

        with torch.no_grad():
            t_out = self.teacher_model(
                input_ids=inputs["input_ids"].to(self.teacher_model.device),
                attention_mask=inputs["attention_mask"].to(self.teacher_model.device))
            t_logits_full = t_out.logits.to(s_logits_full.device)

        s_logits = s_logits_full[:, :-1, :].contiguous()
        t_logits = t_logits_full[:, :-1, :].contiguous()
        labels_s = labels[:, 1:].contiguous()
        attn_s = inputs["attention_mask"][:, 1:].contiguous()

        mask = (labels_s != -100) & (attn_s == 1)
        tok_n = mask.sum().item()
        if tok_n == 0:
            z = s_logits_full.sum() * 0.0
            return (z, s_out) if return_outputs else z

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
# 학습 / 티처
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
            model.save_pretrained(out_dir, selected_adapters=[adapter_name])
        except TypeError:
            model.save_pretrained(out_dir)


def load_teacher(args):
    if not args.use_kd:
        return None
    print(f"[Teacher] {args.teacher_model} on {args.teacher_device}")
    if args.teacher_4bit:
        from transformers import BitsAndBytesConfig
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={"": args.teacher_device})
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, torch_dtype=torch.bfloat16,
            device_map={"": args.teacher_device})
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


# ============================================================
# Main
# ============================================================
def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--bundles_dir", required=True)
    p.add_argument("--b_merged_dir", default=None, help="Stage 3: B_merged 번들 디렉토리")
    p.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--out_adapters", required=True)
    p.add_argument("--original_num_layers", type=int, default=None,
                   help="원본 레이어 수 (기본: manifest/prune_log에서 자동 결정)")

    p.add_argument("--qa_dataset", default="squad")
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--max_eval_samples", type=int, default=8000)
    p.add_argument("--seq_len", type=int, default=1024)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad_acc", type=int, default=32)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--save_total_limit", type=int, default=2)

    p.add_argument("--use_kd", action="store_true")
    p.add_argument("--teacher_model", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--teacher_device", default="cuda:1")
    p.add_argument("--teacher_4bit", action="store_true")
    p.add_argument("--kd_alpha", type=float, default=0.1)
    p.add_argument("--kd_T", type=float, default=2.0)
    return p.parse_args()


def _load_index_info(base_dir: str, bundles_dir: str, stage: int,
                     b_merged_dir: str = None) -> dict:
    """
    manifest.json / prune_log.json에서 B/C 인덱스와 원본 레이어 수 결정.
    Returns: {"B": [...], "C": [...], "L_full": int}
    """
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

    # L_full 추론
    if not info["L_full"]:
        all_idx = info["B"] + info["C"]
        if all_idx:
            info["L_full"] = max(all_idx) + 1
            # 연속 블록 제거이므로 마지막 kept 레이어 뒤에 더 있을 수 있음
            # config에서 읽는 게 더 정확
    return info


def main():
    args = parse_args()

    # ── Tokenizer ──
    tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True, local_files_only=True)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # ── Student 로드 ──
    print(f"\n[Loading] Student from {args.base_dir}")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=dtype_map[args.dtype],
        device_map=None, local_files_only=True)

    device = torch.device(os.environ.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:
        pass

    loaded_L = len(_get_layer_container(model))

    # ── 인덱스 정보 ──
    info = _load_index_info(args.base_dir, args.bundles_dir, args.stage, args.b_merged_dir)
    B_idx, C_idx = info["B"], info["C"]

    # original_num_layers 결정: CLI > manifest > config
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

        bad = [i for i in A_idx if not isinstance(layers[i], LlamaDecoderLayer)]
        if bad:
            raise RuntimeError(f"A 위치에 비정상 레이어: {bad}")

        # model = _attach_new_adapter(model, "stageA", r=8, alpha=16, dropout=0.05)
        model = _attach_new_adapter(model, "stageA", target_layers=A_idx)

        model.set_adapter("stageA")
        _enable_only_lora_on_indices(model, A_idx, "stageA")

        out = os.path.join(args.out_adapters, "A_lora", "stageA")
        train_adapter(model, out, train_ds, eval_ds, args, "stageA", args.use_kd, teacher)

        print(f"\n[Next] Merge: llama_merge_adapter.py --base_model {args.base_dir} "
              f"--adapter_path {out} --output_dir ./merged_models_llama_7b/A_merged")

    # ================================================================
    # Stage 2: B 레이어에만 LoRA (A=merged, B=복원, C=PassLayer)
    # ================================================================
    elif args.stage == 2:
        print("\n" + "=" * 60)
        print("STAGE 2: B-LoRA (A=merged, B=restored, C=PassLayer)")
        print("=" * 60)

        B_bundle_dir = os.path.join(args.bundles_dir, "B")
        _assert_bundle_files_exist(B_bundle_dir, B_idx)
        _rehydrate_layers(model, B_bundle_dir, B_idx)

        bad = [i for i in B_idx if not isinstance(layers[i], LlamaDecoderLayer)]
        if bad:
            raise RuntimeError(f"B 복원 실패: {bad}")

        #model = _attach_new_adapter(model, "stageB", r=8, alpha=16, dropout=0.05)
        model = _attach_new_adapter(model, "stageB", target_layers=B_idx)
        model.set_adapter("stageB")
        _enable_only_lora_on_indices(model, B_idx, "stageB")

        out = os.path.join(args.out_adapters, "B_lora", "stageB")
        train_adapter(model, out, train_ds, eval_ds, args, "stageB", args.use_kd, teacher)

        print(f"\n[Next] Merge B adapter with B bundle layers")

    # ================================================================
    # Stage 3: C 레이어에만 LoRA (A=merged, B=merged, C=복원)
    # ================================================================
    elif args.stage == 3:
        print("\n" + "=" * 60)
        print("STAGE 3: C-LoRA (A=merged, B=merged, C=restored)")
        print("=" * 60)

        if not args.b_merged_dir:
            raise ValueError("Stage 3 requires --b_merged_dir")

        # B_merged 복원
        B_merged_indices = _load_bundle_indices(args.b_merged_dir)
        if not B_merged_indices:
            B_merged_indices = B_idx
        _assert_bundle_files_exist(args.b_merged_dir, B_merged_indices)
        _rehydrate_layers(model, args.b_merged_dir, B_merged_indices)

        # C 복원
        C_bundle_dir = args.bundles_dir
        _assert_bundle_files_exist(C_bundle_dir, C_idx)
        _rehydrate_layers(model, C_bundle_dir, C_idx)

        bad = [i for i in C_idx if not isinstance(layers[i], LlamaDecoderLayer)]
        if bad:
            raise RuntimeError(f"C 복원 실패: {bad}")

        #model = _attach_new_adapter(model, "stageC", r=8, alpha=16, dropout=0.05)
        model = _attach_new_adapter(model, "stageC", target_layers=C_idx)
        model.set_adapter("stageC")
        _enable_only_lora_on_indices(model, C_idx, "stageC")

        out = os.path.join(args.out_adapters, "C_lora", "stageC")
        train_adapter(model, out, train_ds, eval_ds, args, "stageC", args.use_kd, teacher)

        print(f"\n[Next] Merge C adapter with C bundle layers")

    print("\n[Done] Training completed")


if __name__ == "__main__":
    main()