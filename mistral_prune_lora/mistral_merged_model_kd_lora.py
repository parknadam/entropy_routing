#!/usr/bin/env python3
"""
Progressive KD-LoRA 학습 코드 (인덱스 리매핑 수정 버전)

핵심 수정:
  layeronly_drop.py는 레이어를 물리적으로 제거하여 모델을 축소함 (32→24층).
  이로 인해 A_merged 모델의 레이어 인덱스와 bundle_meta.json의 원본 인덱스가 불일치.
  
  수정 방법:
  - Stage 2/3에서 A_merged(24층)를 원본 레이어 수(32층)로 확장
  - 제거된 위치에 PassLayer 삽입 후, bundle에서 실제 레이어 복원
  - LoRA는 layers_to_transform으로 대상 레이어에만 정확히 적용

Stage 1: Pruned model (24층) → A adapter (전체 24층 학습)
Stage 2: A_merged (24→32 확장) + B 복원 + C=PassLayer → B adapter
Stage 3: A_merged (24→32 확장) + B_merged 복원 + C 복원 → C adapter

사용법:
# Stage 1
CUDA_VISIBLE_DEVICES=0,1 DEVICE=cuda:0 \
python -m mistral_prune_lora.mistral_merged_model_kd_lora \
  --base_dir ./25_mistral_results/pruning/A \
  --bundles_dir ./25_mistral_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./mistral_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-4 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model mistralai/Mistral-7B-Instruct-v0.2 \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0

# Stage 2 (A_merged 모델 사용)
CUDA_VISIBLE_DEVICES=0,1 DEVICE=cuda:0 \
python -m mistral_prune_lora.mistral_merged_model_kd_lora \
  --base_dir ./merged_models_mistral_7b/A_merged \
  --bundles_dir ./25_mistral_results/pruning/bundles \
  --stage 2 \
  --out_adapters ./mistral_kd_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-5 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model mistralai/Mistral-7B-Instruct-v0.2 \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0

# Stage 3
CUDA_VISIBLE_DEVICES=0,1 DEVICE=cuda:0 \
python -m mistral_prune_lora.mistral_merged_model_kd_lora \
  --base_dir ./merged_models_mistral_7b/A_merged \
  --b_merged_dir ./merged_models_mistral_7b/B_merged \
  --bundles_dir ./25_mistral_results/pruning/bundles/C \
  --stage 3 \
  --out_adapters ./mistral_kd_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-5 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model mistralai/Mistral-7B-Instruct-v0.2 \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0
"""

import os, json, re, math, argparse, inspect
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
from safetensors import safe_open
try:
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
except Exception:
    MistralDecoderLayer = None

# ============================================================
# 유틸리티: 레이어 관리
# ============================================================
CANON_PATH = "model.layers"  # MistralForCausalLM.model.layers
_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")


def _resolve_attr_path(root, dotted: str):
    """점으로 구분된 경로를 따라 객체 탐색"""
    parent = root
    for seg in dotted.split(".")[:-1]:
        parent = getattr(parent, seg)
    last = dotted.split(".")[-1]
    return parent, last


def _get_layer_container(model):
    """model.model.layers ModuleList 반환"""
    for path in ["model.layers", "model.model.layers"]:
        try:
            parts = path.split(".")
            obj = model
            for p in parts:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Cannot find layer container")


def _load_bundle_indices(bundle_dir: str) -> list:
    """bundle_meta.json에서 레이어 인덱스 로드"""
    meta_path = os.path.join(bundle_dir, "bundle_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return sorted(meta.get("indices", []))

    # fallback: safetensors 파일명에서 추출
    indices = []
    for fname in os.listdir(bundle_dir):
        m = re.match(r"layer_(\d+)\.safetensors", fname)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def _get_present_layer_indices_from_safetensors(model_dir: str) -> list:
    """
    model.safetensors에서 실제로 존재하는 model.layers.{i} 인덱스 목록 추출.
    파일이 없거나 파싱 실패 시 빈 리스트 반환.
    """
    st_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.isfile(st_path):
        return []

    present = set()
    try:
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if not k.startswith("model.layers."):
                    continue
                parts = k.split(".")
                if len(parts) < 3:
                    continue
                try:
                    idx = int(parts[2])
                except ValueError:
                    continue
                present.add(idx)
    except Exception as e:
        print(f"[warn] safetensors layer index scan failed: {e}")
        return []

    return sorted(present)


def _collapse_sparse_loaded_layers(model, model_dir: str):
    """
    config.num_hidden_layers와 실제 safetensors의 레이어 인덱스가 불일치할 때 보정.
    예) config=32, present=[0..20,29,30,31] → ModuleList를 24층으로 압축.
    """
    layers = _get_layer_container(model)
    current_num = len(layers)
    present = _get_present_layer_indices_from_safetensors(model_dir)

    if not present:
        return model

    # 이미 연속 인덱스(0..N-1)면 정상 모델로 간주
    if present == list(range(len(present))) and len(present) == current_num:
        return model

    # sparse 케이스만 처리
    if max(present) >= current_num:
        print(
            f"[warn] present layer index out of range: max={max(present)} current={current_num}. "
            "skip sparse collapse."
        )
        return model

    if len(present) >= current_num:
        return model

    print(
        f"[sparse-load] Detected sparse layer indices in {model_dir}: "
        f"{len(present)} present over configured {current_num}."
    )
    print(f"[sparse-load] Present indices: {present}")

    compact_layers = [layers[i] for i in present]

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = nn.ModuleList(compact_layers)
    elif hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        model.model.model.layers = nn.ModuleList(compact_layers)
    else:
        raise RuntimeError("model.model.layers 경로를 찾을 수 없음")

    model.config.num_hidden_layers = len(compact_layers)
    print(f"[sparse-load] Collapsed layers: {current_num} -> {len(compact_layers)}")
    return model


# ============================================================
# PassLayer: 프루닝된 위치에 identity 역할
# ============================================================
def _decoder_layer_returns_tuple() -> bool:
    """
    transformers 버전별 MistralDecoderLayer 출력 형식 판별.
    - 구버전(4.x 계열): tuple 반환
    - 신버전(5.x 계열): torch.Tensor 반환
    """
    if MistralDecoderLayer is None:
        return True

    try:
        sig = inspect.signature(MistralDecoderLayer.forward)
    except Exception:
        return True

    # output_attentions/use_cache 인자가 있으면 legacy(tuple) API일 가능성이 높다.
    if "output_attentions" in sig.parameters:
        return True

    ret = sig.return_annotation
    if ret is inspect.Signature.empty:
        return True
    return ret is not torch.Tensor


class MistralPassLayer(nn.Module):
    """프루닝 위치를 통과시키는 identity layer (transformers 버전 호환)."""
    def __init__(self, return_tuple: bool):
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
# 모델 확장: 프루닝된 모델을 원본 레이어 수로 복원
# ============================================================
def _expand_to_original_layout(model, removed_indices, original_num_layers):
    """
    프루닝으로 축소된 모델(N-k층)을 원본 레이어 수(N층)로 확장.
    제거된 위치에 MistralPassLayer를 삽입하고,
    기존 레이어를 원래의 인덱스 위치로 재배치.

    예시: 원본 32층, removed=[16..23]
      A_merged: 24층 (indices 0-23)
        - A_merged[0..15]  → 원본 0..15
        - A_merged[16..23] → 원본 24..31
      확장 후: 32층
        - [0..15]  = A_merged[0..15]  (원본 A 레이어)
        - [16..23] = PassLayer         (B/C 위치)
        - [24..31] = A_merged[16..23] (원본 A 레이어)
    """
    layers = _get_layer_container(model)
    current_num = len(layers)
    removed_set = set(int(i) for i in removed_indices)
    kept_original_indices = sorted(set(range(original_num_layers)) - removed_set)

    expected_current = len(kept_original_indices)
    if current_num != expected_current:
        raise ValueError(
            f"레이어 수 불일치: 모델 {current_num}층, "
            f"예상 {expected_current}층 (원본 {original_num_layers} - 제거 {len(removed_indices)})"
        )

    # 현재 레이어 추출
    current_layers = [layers[i] for i in range(current_num)]

    # 원본 크기로 새 배열 구성
    new_layers = [None] * original_num_layers

    # A 레이어를 원래 위치에 배치
    for pruned_idx, orig_idx in enumerate(kept_original_indices):
        new_layers[orig_idx] = current_layers[pruned_idx]

    # 제거된 위치에 PassLayer 삽입
    dev = next(model.parameters()).device
    legacy_tuple_output = _decoder_layer_returns_tuple()
    for idx in removed_indices:
        new_layers[int(idx)] = MistralPassLayer(return_tuple=legacy_tuple_output).to(dev)

    # 검증
    none_positions = [i for i, l in enumerate(new_layers) if l is None]
    if none_positions:
        raise RuntimeError(f"확장 후 None 레이어 존재: {none_positions}")

    # ModuleList 교체
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model.model.layers = nn.ModuleList(new_layers)
    elif hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
        model.model.model.layers = nn.ModuleList(new_layers)
    else:
        raise RuntimeError("model.model.layers 경로를 찾을 수 없음")

    # config 업데이트 (forward에서 layer 수 체크하는 경우 대비)
    model.config.num_hidden_layers = original_num_layers

    print(f"[expand] {current_num}층 → {original_num_layers}층 확장 완료")
    print(f"  A 위치 (유지): {kept_original_indices[:5]}{'...' if len(kept_original_indices) > 5 else ''} ({len(kept_original_indices)}개)")
    print(f"  PassLayer 위치: {sorted(removed_indices)} ({len(removed_indices)}개)")

    return model, kept_original_indices


# ============================================================
# 번들 레이어 복원
# ============================================================
def _extract_layer_sd(raw_sd: dict, idx: int):
    """safetensors에서 특정 레이어의 state_dict 추출 (prefix 제거)"""
    prefixes = [
        f"model.layers.{idx}.",
        f"model.model.layers.{idx}.",
        f"layers.{idx}.",
    ]
    for p in prefixes:
        out = {k[len(p):]: v for k, v in raw_sd.items() if k.startswith(p)}
        if len(out) > 0:
            return out
    return raw_sd


def _pick_layer_file(bundle_dir: str, idx: int) -> str:
    """레이어 파일 경로 찾기"""
    for fmt in [f"layer_{idx:03d}.safetensors", f"layer_{idx}.safetensors"]:
        p = os.path.join(bundle_dir, fmt)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Layer file not found for idx={idx} in {bundle_dir}")


def _rehydrate_layers(model, bundle_dir: str, indices):
    """Bundle에서 실제 레이어 가중치를 복원 (PassLayer → 실제 MistralDecoderLayer)"""
    if MistralDecoderLayer is None:
        raise RuntimeError("MistralDecoderLayer import 실패 (transformers 버전 확인 필요)")

    layers = _get_layer_container(model)

    dtype = getattr(model, "dtype", None)
    if dtype is None:
        dtype = next(model.parameters()).dtype

    for i in indices:
        i = int(i)

        # 디바이스 결정
        original_layer = layers[i]
        if hasattr(original_layer, "self_attn") and any(True for _ in original_layer.parameters()):
            dev = next(original_layer.parameters()).device
        else:
            dev = None
            if hasattr(model, "hf_device_map"):
                for key_fmt in [f"model.layers.{i}", f"model.model.layers.{i}", f"layers.{i}"]:
                    dev = model.hf_device_map.get(key_fmt)
                    if dev is not None:
                        break
            if dev is None:
                dev = next(model.parameters()).device

        print(f"[rehydrate] layer {i} -> device {dev}")

        try:
            new_layer = MistralDecoderLayer(model.config, layer_idx=i).to(device=dev, dtype=dtype)
        except TypeError:
            new_layer = MistralDecoderLayer(model.config).to(device=dev, dtype=dtype)

        f = _pick_layer_file(bundle_dir, i)
        raw = load_file(f)
        sd = _extract_layer_sd(raw, i)
        sd = {k: v.to(device=dev, dtype=dtype) for k, v in sd.items()}

        try:
            new_layer.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"[warn] layer {i}: {e} -> non-strict")
            new_layer.load_state_dict(sd, strict=False)

        layers[i] = new_layer
        print(f"  ✓ restored from {os.path.basename(f)} on {dev}")


# ============================================================
# LoRA 어댑터 관리
# ============================================================
def _attach_new_adapter(model, adapter_name: str, r=8, alpha=16, dropout=0.05,
                        target_layers=None):
    """
    새 LoRA 어댑터 추가.
    target_layers: 지정하면 해당 레이어에만 LoRA 적용 (layers_to_transform).
                   None이면 모든 레이어에 적용.
    """
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
        layers_to_transform=target_layers,
    )
    return get_peft_model(model, cfg, adapter_name=adapter_name)


def _count_trainable(model):
    """학습 가능 파라미터 수"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Teacher 모델
# ============================================================
def load_teacher(args):
    if not args.use_kd or not args.teacher_model:
        return None

    print(f"\n[Teacher] Loading {args.teacher_model} on {args.teacher_device}")
    kwargs = {
        "torch_dtype": torch.float16,
        "device_map": args.teacher_device,
        "trust_remote_code": True
    }
    if args.teacher_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, **kwargs)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


# ============================================================
# Dataset
# ============================================================
def _load_qa_sft_dataset(tokenizer, dataset_name, split, max_samples, seq_len):
    if dataset_name == "squad":
        ds = load_dataset("squad", split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id

    def to_list(x):
        return x if isinstance(x, list) else x.tolist()

    def process(ex):
        if dataset_name == "squad":
            prompt = (
                f"Answer the following question based on the context.\n\n"
                f"Context: {ex['context']}\n\n"
                f"Question: {ex['question']}\n\n"
                f"Answer:"
            )
            ans = ex["answers"]["text"][0] if ex["answers"]["text"] else "unknown"
        else:
            prompt = ex.get("question", "")
            ans = ex.get("answer", "")

        prompt_ids = to_list(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        ans_ids = to_list(tokenizer(" " + ans, add_special_tokens=False)["input_ids"])
        if eos_id:
            ans_ids = ans_ids + [eos_id]

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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "__drop__": 0,
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["__drop__"] == 0)
    if "__drop__" in ds.column_names:
        ds = ds.remove_columns("__drop__")
    return ds


# ============================================================
# KD Trainer (causal shift 적용)
# ============================================================
class KDTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, kd_alpha=0.1, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.kd_alpha = kd_alpha
        self.T = temperature

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]

        s_out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        s_logits_full = s_out.logits

        # Teacher 없으면 순수 CE
        if self.teacher is None:
            s_logits = s_logits_full[:, :-1, :].contiguous().view(-1, s_logits_full.size(-1))
            labels_flat = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(s_logits, labels_flat, ignore_index=-100)
            return (loss, s_out) if return_outputs else loss

        # KD: Teacher forward
        teacher_device = next(self.teacher.parameters()).device
        with torch.no_grad():
            t_out = self.teacher(
                input_ids=inputs["input_ids"].to(teacher_device),
                attention_mask=inputs["attention_mask"].to(teacher_device),
            )
            t_logits_full = t_out.logits.to(s_logits_full.device)

        # Causal shift: logits[:, :-1] → labels[:, 1:]
        s_logits = s_logits_full[:, :-1, :].contiguous()
        t_logits = t_logits_full[:, :-1, :].contiguous()
        labels_shifted = labels[:, 1:].contiguous()
        attn_shifted = inputs["attention_mask"][:, 1:].contiguous()

        mask = (labels_shifted != -100) & (attn_shifted == 1)
        tok_n = mask.sum().item()

        if tok_n == 0:
            loss = s_logits_full.sum() * 0.0
            return (loss, s_out) if return_outputs else loss

        s = s_logits[mask].float().clamp(-50, 50)
        t = t_logits[mask].float().clamp(-50, 50)
        y = labels_shifted[mask]

        T = self.T
        # 논문 식 (3): L_total = α·L_task + (1-α)·L_KD
        soft_loss = F.kl_div(
            F.log_softmax(s / T, dim=-1),
            F.log_softmax(t / T, dim=-1),
            reduction="batchmean",
            log_target=True,
        ) * (T * T)

        hard_loss = F.cross_entropy(s, y)

        if not (torch.isfinite(soft_loss) and torch.isfinite(hard_loss)):
            loss = hard_loss if torch.isfinite(hard_loss) else s_logits_full.sum() * 0.0
        else:
            loss = self.kd_alpha * soft_loss + (1 - self.kd_alpha) * hard_loss

        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "tok_n": float(tok_n),
                "kd_soft": float(soft_loss) if torch.isfinite(soft_loss) else float("nan"),
                "kd_hard": float(hard_loss),
                "kd_ppl": float(math.exp(min(hard_loss.item(), 20))),
            })

        return (loss, s_out) if return_outputs else loss


# ============================================================
# 학습
# ============================================================
def train_adapter(model, out_dir, train_ds, eval_ds, args, adapter_name, use_kd=False, teacher=None):
    os.makedirs(out_dir, exist_ok=True)

    trainable = _count_trainable(model)
    print(f"\n[Pre-train] Trainable params: {trainable:,}")
    if trainable == 0:
        raise RuntimeError("No trainable parameters!")

    common = {
        "output_dir": out_dir,
        "per_device_train_batch_size": args.bs,
        "gradient_accumulation_steps": args.grad_acc,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "remove_unused_columns": False,
        "report_to": "none",
        "save_total_limit": args.save_total_limit,
    }

    dtype_map = {"bf16": {"bf16": True, "fp16": False}, "fp16": {"fp16": True, "bf16": False}}
    common.update(dtype_map.get(args.dtype, {"fp16": False, "bf16": False}))

    try:
        targs = TrainingArguments(
            **common,
            eval_strategy="steps" if args.eval_steps > 0 else "no",
            eval_steps=args.eval_steps if args.eval_steps > 0 else None,
            save_strategy="steps" if args.save_steps > 0 else "no",
            save_steps=args.save_steps if args.save_steps > 0 else None,
        )
    except TypeError:
        targs = TrainingArguments(
            **common,
            evaluation_strategy="steps" if args.eval_steps > 0 else "no",
            eval_steps=args.eval_steps if args.eval_steps > 0 else None,
            save_strategy="steps" if args.save_steps > 0 else "no",
            save_steps=args.save_steps if args.save_steps > 0 else None,
        )

    trainer_cls = KDTrainer if (use_kd and teacher) else Trainer
    kwargs = {
        "model": model,
        "args": targs,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": default_data_collator,
    }
    if trainer_cls is KDTrainer:
        kwargs.update({
            "teacher_model": teacher,
            "kd_alpha": args.kd_alpha,
            "temperature": args.kd_T,
        })

    trainer = trainer_cls(**kwargs)
    trainer.train()

    if isinstance(model, PeftModel):
        try:
            model.save_pretrained(out_dir, safe_serialization=True)
            print(f"[Saved] Adapter to {out_dir}")
        except Exception as e:
            print(f"[Error] Failed to save adapter: {e}")
    else:
        try:
            trainer.save_model(out_dir)
            print(f"[Saved] Model to {out_dir}")
        except Exception as e:
            print(f"[Error] Failed to save model: {e}")


# ============================================================
# 모델 로드 헬퍼
# ============================================================
def _load_model(model_dir, dtype_str="bf16"):
    """모델 로드 + 기본 설정"""
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype_map[dtype_str],
            device_map={"": "cuda:0"},
            trust_remote_code=True
        )
        print(f"[Device] Model loaded on cuda:0 (single GPU)")
    except RuntimeError as e:
        print(f"[Warning] Single GPU 로드 실패: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype_map[dtype_str],
            device_map="auto",
            trust_remote_code=True
        )

    # config(32) + sparse weights(24) 형태로 저장된 A_merged를 자동 보정
    model = _collapse_sparse_loaded_layers(model, model_dir)

    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:
        pass

    return model


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, required=True,
                    help="Base model directory (A for stage 1, A_merged for stage 2/3)")
    p.add_argument("--bundles_dir", type=str, required=True,
                    help="Pruned bundles directory (B+C parent for stage 1/2, C bundle for stage 3)")
    p.add_argument("--b_merged_dir", type=str, default=None,
                    help="B_merged directory (for stage 3 only)")
    p.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--out_adapters", type=str, required=True)

    # Dataset
    p.add_argument("--qa_dataset", type=str, default="squad")
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--max_eval_samples", type=int, default=8000)
    p.add_argument("--seq_len", type=int, default=1024)

    # Training
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad_acc", type=int, default=32)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--save_total_limit", type=int, default=2)

    # KD
    p.add_argument("--use_kd", action="store_true")
    p.add_argument("--teacher_model", type=str, default=None)
    p.add_argument("--teacher_4bit", action="store_true")
    p.add_argument("--teacher_device", type=str, default="cuda:1")
    p.add_argument("--kd_alpha", type=float, default=0.1)
    p.add_argument("--kd_T", type=float, default=2.0)

    return p.parse_args()


def main():
    args = parse_args()

    # ============================================================
    # Bundle 메타데이터 로드
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Loading Bundle Metadata")
    print(f"{'='*60}")

    if args.stage == 3:
        if args.b_merged_dir is None:
            raise ValueError("--b_merged_dir is required for stage 3")

        B_bundle_dir = args.b_merged_dir
        C_bundle_dir = args.bundles_dir

        B_idx = _load_bundle_indices(B_bundle_dir)
        C_idx = _load_bundle_indices(C_bundle_dir)
        print(f"B_merged: {len(B_idx)} layers → {B_idx}")
        print(f"C bundle: {len(C_idx)} layers → {C_idx}")
    else:
        B_bundle_dir = os.path.join(args.bundles_dir, "B")
        C_bundle_dir = os.path.join(args.bundles_dir, "C")

        B_idx = _load_bundle_indices(B_bundle_dir)
        C_idx = _load_bundle_indices(C_bundle_dir)
        print(f"B bundle: {len(B_idx)} layers → {B_idx}")
        print(f"C bundle: {len(C_idx)} layers → {C_idx}")

    removed_indices = sorted(set(B_idx + C_idx))

    # 토크나이저
    tok = AutoTokenizer.from_pretrained(args.base_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 데이터셋
    print("\n[Loading] Datasets")
    train_ds = _load_qa_sft_dataset(tok, args.qa_dataset, "train", args.max_samples, args.seq_len)
    eval_ds = _load_qa_sft_dataset(tok, args.qa_dataset, "validation", args.max_eval_samples, args.seq_len)

    # Teacher
    teacher = load_teacher(args)

    # ========================================
    # Stage 1: Pruned A model → A adapter
    # ========================================
    if args.stage == 1:
        print("\n" + "="*60)
        print("STAGE 1: A-LoRA Training (on pruned model)")
        print("="*60)

        model = _load_model(args.base_dir, args.dtype)
        layers = _get_layer_container(model)
        L = len(layers)

        # ★ 핵심 수정: 프루닝된 모델의 모든 레이어가 A 그룹
        # B_idx/C_idx는 원본 인덱스이므로 프루닝된 모델에는 존재하지 않음
        A_idx = list(range(L))

        print(f"\n[Layer Distribution]")
        print(f"  Pruned model layers: {L}")
        print(f"  Training target: ALL {L} layers (indices 0..{L-1})")
        print(f"  (B/C layers were physically removed by layeronly_drop)")

        # layers_to_transform으로 전체 레이어에 LoRA 적용
        model = _attach_new_adapter(model, "stageA", r=8, alpha=16, dropout=0.05,
                                    target_layers=A_idx)
        print(f"  Trainable params: {_count_trainable(model):,}")

        adapter_out = os.path.join(args.out_adapters, "A_lora")
        train_adapter(model, adapter_out, train_ds, eval_ds, args, "stageA", args.use_kd, teacher)

        print(f"\n{'='*60}")
        print(f"✓ Stage 1 Complete")
        print(f"{'='*60}")
        print(f"Adapter saved: {adapter_out}")
        print(f"\n[Next Steps]")
        print(f"1. Merge: python mistral_merge_adapter.py \\")
        print(f"     --base_model {args.base_dir} \\")
        print(f"     --adapter_path {adapter_out} --output_dir ./merged_models/A_merged")
        print(f"2. Run stage 2 with --base_dir ./merged_models/A_merged")

    # ========================================
    # Stage 2: A_merged → 확장 → B 복원 → B adapter
    # ========================================
    elif args.stage == 2:
        print("\n" + "="*60)
        print("STAGE 2: B-LoRA Training (expand A_merged + rehydrate B)")
        print("="*60)

        # 1. A_merged 로드
        print("\n[Step 1] Loading A_merged model...")
        model = _load_model(args.base_dir, args.dtype)

        layers = _get_layer_container(model)
        current_L = len(layers)
        original_L = current_L + len(removed_indices)

        print(f"  A_merged layers: {current_L}")
        print(f"  Original layers: {original_L}")
        print(f"  Removed indices: {removed_indices}")

        # 2. ★ 핵심: 원본 레이어 구조로 확장 (PassLayer 삽입)
        print("\n[Step 2] Expanding to original layout...")
        model, A_orig_indices = _expand_to_original_layout(model, removed_indices, original_L)

        # 3. B 레이어 복원 (PassLayer → 실제 MistralDecoderLayer)
        print("\n[Step 3] Rehydrating B layers...")
        if B_idx:
            _rehydrate_layers(model, B_bundle_dir, B_idx)
            print(f"  ✓ {len(B_idx)} B layers restored")
        else:
            print("  ⚠ WARNING: No B layers!")

        # 4. C 위치는 PassLayer 유지 (forward에서 identity)
        print(f"\n[Layer State After Expansion]")
        print(f"  A layers (trained, frozen): {A_orig_indices[:5]}... ({len(A_orig_indices)}개)")
        print(f"  B layers (rehydrated, TRAINING TARGET): {B_idx}")
        print(f"  C layers (PassLayer, identity): {C_idx}")

        # 5. B 레이어에만 LoRA 적용
        print("\n[Step 4] Attaching LoRA to B layers only...")
        model = _attach_new_adapter(model, "stageB", r=8, alpha=16, dropout=0.05,
                                    target_layers=list(B_idx))
        print(f"  Trainable params: {_count_trainable(model):,}")

        # 6. 학습
        print("\n[Step 5] Training B adapter with KD-LoRA...")
        adapter_out = os.path.join(args.out_adapters, "B_lora")
        train_adapter(model, adapter_out, train_ds, eval_ds, args, "stageB", args.use_kd, teacher)

        print(f"\n{'='*60}")
        print(f"✓ Stage 2 Complete")
        print(f"{'='*60}")
        print(f"Adapter saved: {adapter_out}")
        print(f"\n[Next Steps]")
        print(f"1. Merge B adapter into B bundle → B_merged")
        print(f"2. Run stage 3 with:")
        print(f"   --base_dir {args.base_dir} \\")
        print(f"   --b_merged_dir ./merged_models/B_merged")

    # ========================================
    # Stage 3: A_merged → 확장 → B_merged + C 복원 → C adapter
    # ========================================
    elif args.stage == 3:
        print("\n" + "="*60)
        print("STAGE 3: C-LoRA Training (expand A_merged + B_merged + C)")
        print("="*60)

        # 1. A_merged 로드
        print("\n[Step 1] Loading A_merged model...")
        model = _load_model(args.base_dir, args.dtype)

        layers = _get_layer_container(model)
        current_L = len(layers)
        original_L = current_L + len(removed_indices)

        # 2. 원본 레이어 구조로 확장
        print("\n[Step 2] Expanding to original layout...")
        model, A_orig_indices = _expand_to_original_layout(model, removed_indices, original_L)

        # 3. B_merged 레이어 복원
        print("\n[Step 3] Rehydrating B_merged layers...")
        if B_idx:
            _rehydrate_layers(model, B_bundle_dir, B_idx)
            print(f"  ✓ {len(B_idx)} B_merged layers restored")

        # 4. C 레이어 복원
        print("\n[Step 4] Rehydrating C layers...")
        if C_idx:
            _rehydrate_layers(model, C_bundle_dir, C_idx)
            print(f"  ✓ {len(C_idx)} C layers restored")

        print(f"\n[Layer State After Expansion]")
        print(f"  A layers (trained, frozen): {len(A_orig_indices)}개")
        print(f"  B layers (B_merged, frozen): {B_idx}")
        print(f"  C layers (rehydrated, TRAINING TARGET): {C_idx}")

        # 5. C 레이어에만 LoRA 적용
        print("\n[Step 5] Attaching LoRA to C layers only...")
        model = _attach_new_adapter(model, "stageC", r=8, alpha=16, dropout=0.05,
                                    target_layers=list(C_idx))
        print(f"  Trainable params: {_count_trainable(model):,}")

        # 6. 학습
        print("\n[Step 6] Training C adapter with KD-LoRA...")
        adapter_out = os.path.join(args.out_adapters, "C_lora")
        train_adapter(model, adapter_out, train_ds, eval_ds, args, "stageC", args.use_kd, teacher)

        print(f"\n{'='*60}")
        print(f"✓ Stage 3 Complete - FINAL")
        print(f"{'='*60}")
        print(f"Adapter saved: {adapter_out}")

    print("\n[Done] Training completed")


if __name__ == "__main__":
    main()
