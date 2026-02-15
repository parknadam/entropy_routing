#!/usr/bin/env python3
"""
Progressive KD-LoRA 학습 코드 (어댑터 생성만)

Stage 1: Base model + A 그룹 → A adapter 생성
Stage 2: A_merged model + B 그룹 → B adapter 생성  
Stage 3: AB_merged model + C 그룹 → C adapter 생성

사용법:
# Stage 1
CUDA_VISIBLE_DEVICES=3,4 DEVICE=cuda:0 \
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
  --base_dir ./merged_models/A_merged \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 2 \
  --out_adapters ./kd_lora_results/adapters \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-4 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0

# Stage 3 실행
CUDA_VISIBLE_DEVICES=0,1 DEVICE=cuda:0 \
python -m mistral_prune_lora.mistral_merged_model_kd_lora \
  --base_dir ./merged_models/A_merged \
  --b_merged_dir ./merged_models/B_merged \
  --bundles_dir ./7b_results/pruning/bundles/C \
  --stage 3 \
  --out_adapters ./kd_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-4 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model meta-llama/Llama-2-7b-chat-hf \
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
    """모델에서 레이어 컨테이너 가져오기"""
    candidates = (
        "model.layers",
        "model.model.layers",
        "base_model.model.layers",
        "base_model.model.model.layers",
    )
    for path in candidates:
        try:
            parent, attr = _resolve_attr_path(model, path)
            layers = getattr(parent, attr)
            if isinstance(layers, nn.ModuleList):
                return layers
        except Exception:
            continue
    raise RuntimeError("Mistral layers 경로를 찾지 못했습니다. (예: model.layers)")


def _layer_name_prefix(model, i: int) -> str:
    """레이어 i에 해당하는 파라미터 이름 prefix 반환"""
    # PeftModel 래핑 여부에 따라 prefix가 달라질 수 있으므로
    # 실제 파라미터 이름에서 패턴 매칭으로 처리
    return f"layers.{i}."


def _load_bundle_indices(bundle_dir: str) -> List[int]:
    """Bundle 메타데이터에서 레이어 인덱스 로드"""
    meta_path = os.path.join(bundle_dir, "bundle_meta.json")

    if not os.path.exists(meta_path):
        print(f"    ✗ bundle_meta.json not found in {bundle_dir}")
        return []

    with open(meta_path, "r") as f:
        meta = json.load(f)

    if "indices" not in meta:
        print(f"    ✗ 'indices' key not found in bundle_meta.json")
        print(f"       Available keys: {list(meta.keys())}")
        return []

    indices = meta["indices"]
    if not isinstance(indices, list):
        print(f"    ✗ 'indices' is not a list: {type(indices)}")
        return []

    return sorted(indices)


def _extract_layer_sd(raw_sd: dict, idx: int) -> dict:
    """번들에 저장된 key prefix 케이스들 대응"""
    prefixes = [
        f"model.layers.{idx}.",
        f"model.model.layers.{idx}.",
        f"layers.{idx}.",
    ]
    for p in prefixes:
        out = {k[len(p):]: v for k, v in raw_sd.items() if k.startswith(p)}
        if len(out) > 0:
            return out
    # 이미 prefix 없이 저장된 경우면 그대로
    return raw_sd


def _pick_layer_file(bundle_dir: str, idx: int) -> str:
    """레이어 파일 경로 찾기 (layer_021.safetensors 또는 layer_21.safetensors)"""
    p = os.path.join(bundle_dir, f"layer_{idx:03d}.safetensors")
    if os.path.exists(p):
        return p
    p2 = os.path.join(bundle_dir, f"layer_{idx}.safetensors")
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Layer file not found for idx={idx}: tried {p} and {p2}")


def _rehydrate_layers(model, bundle_dir: str, indices):
    """Pruned bundle에서 레이어 복원"""
    if MistralDecoderLayer is None:
        raise RuntimeError("MistralDecoderLayer import 실패 (transformers 버전 확인 필요)")

    layers = _get_layer_container(model)

    dtype = getattr(model, "dtype", None)
    if dtype is None:
        dtype = next(model.parameters()).dtype

    for i in indices:
        i = int(i)

        # 원래 레이어의 디바이스를 그대로 사용
        original_layer = layers[i]
        if hasattr(original_layer, "self_attn"):
            dev = next(original_layer.parameters()).device
        else:
            dev = None
            if hasattr(model, "hf_device_map"):
                dev = model.hf_device_map.get(f"model.layers.{i}", None)
                if dev is None:
                    dev = model.hf_device_map.get(f"model.model.layers.{i}", None)
                if dev is None:
                    dev = model.hf_device_map.get(f"layers.{i}", None)
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
def _attach_new_adapter(model, adapter_name: str, r=8, alpha=16, dropout=0.05):
    """새 LoRA 어댑터 추가"""
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )
    return get_peft_model(model, cfg, adapter_name=adapter_name)


def enable_only_lora_params_on_layers(model, indices, adapter_name=None):
    """특정 레이어의 LoRA 파라미터만 활성화"""
    indices = set(int(i) for i in indices)

    # 전부 freeze
    for _, p in model.named_parameters():
        p.requires_grad = False

    enabled = 0
    # 특정 레이어의 LoRA 파라미터만 unfreeze
    for n, p in model.named_parameters():
        m = _LAYER_RE.search(n)
        if not m:
            continue

        layer_idx = int(m.group(1))
        if layer_idx not in indices:
            continue

        if "lora_" not in n:
            continue

        p.requires_grad = True
        enabled += p.numel()

    if enabled == 0:
        print(f"[ERROR] No LoRA params for adapter '{adapter_name}' on layers {sorted(indices)}")
        print("Available LoRA params:")
        for n, _ in model.named_parameters():
            if "lora_" in n.lower():
                print(f"  {n}")
        raise RuntimeError(f"No LoRA params enabled")

    print(f"[trainable] {adapter_name}: {enabled:,} params on layers {sorted(indices)}")
    return enabled


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

        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        prompt_ids = to_list(prompt_ids)

        ans_ids = tokenizer(" " + ans, add_special_tokens=False)["input_ids"]
        ans_ids = to_list(ans_ids)

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
        """
        Causal LM용 KD loss 계산.

        핵심: logits[:, i, :]는 position i+1의 토큰을 예측하므로,
        logits[:, :-1, :] ↔ labels[:, 1:] 로 shift 후 비교해야 한다.
        """
        labels = inputs["labels"]

        s_out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        s_logits_full = s_out.logits

        # ── Teacher 없이 순수 CE만 쓰는 경우 ──
        if self.teacher is None:
            # Causal shift 적용
            s_logits = s_logits_full[:, :-1, :].contiguous().view(-1, s_logits_full.size(-1))
            labels_flat = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(s_logits, labels_flat, ignore_index=-100)
            return (loss, s_out) if return_outputs else loss

        # ── KD: Teacher forward ──
        teacher_device = next(self.teacher.parameters()).device
        with torch.no_grad():
            t_out = self.teacher(
                input_ids=inputs["input_ids"].to(teacher_device),
                attention_mask=inputs["attention_mask"].to(teacher_device),
            )
            t_logits_full = t_out.logits.to(s_logits_full.device)

        # ── Causal shift ──
        # logits[:, :-1] 는 position 1~N 의 토큰을 예측
        # labels[:, 1:]  는 position 1~N 의 정답 토큰
        s_logits = s_logits_full[:, :-1, :].contiguous()
        t_logits = t_logits_full[:, :-1, :].contiguous()
        labels_shifted = labels[:, 1:].contiguous()
        attn_shifted = inputs["attention_mask"][:, 1:].contiguous()

        # 실제 학습 대상 토큰만 선택 (padding, prompt 제외)
        mask = (labels_shifted != -100) & (attn_shifted == 1)
        tok_n = mask.sum().item()

        if tok_n == 0:
            loss = s_logits_full.sum() * 0.0
            return (loss, s_out) if return_outputs else loss

        s = s_logits[mask].float().clamp(-50, 50)
        t = t_logits[mask].float().clamp(-50, 50)
        y = labels_shifted[mask]

        # Soft loss: student ↔ teacher 분포 매칭
        T = self.T
        soft_loss = F.kl_div(
            F.log_softmax(s / T, dim=-1),
            F.log_softmax(t / T, dim=-1),
            reduction="batchmean",
            log_target=True,
        ) * (T * T)

        # Hard loss: student ↔ ground truth
        hard_loss = F.cross_entropy(s, y)

        # NaN 방어
        if not (torch.isfinite(soft_loss) and torch.isfinite(hard_loss)):
            loss = hard_loss if torch.isfinite(hard_loss) else s_logits_full.sum() * 0.0
        else:
            loss = self.kd_alpha * soft_loss + (1 - self.kd_alpha) * hard_loss

        # 로깅
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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, required=True,
                    help="Base model directory (A_merged for stage 2/3)")
    p.add_argument("--bundles_dir", type=str, required=True,
                    help="Pruned bundles directory (for stage 1,2) or C bundle dir (for stage 3)")
    p.add_argument("--b_merged_dir", type=str, default=None,
                    help="B_merged directory (for stage 3 only)")
    p.add_argument("--stage", type=int, required=True, choices=[1, 2, 3],
                    help="Training stage")
    p.add_argument("--out_adapters", type=str, required=True,
                    help="Output directory for adapters")

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

    # Bundle 메타데이터에서 레이어 인덱스 로드
    print(f"\n{'='*60}")
    print(f"Loading Bundle Metadata")
    print(f"{'='*60}")

    if args.stage == 3:
        # Stage 3: B_merged와 C bundle 경로 분리
        if args.b_merged_dir is None:
            raise ValueError("--b_merged_dir is required for stage 3")

        print(f"A_merged directory: {args.base_dir}")
        print(f"B_merged directory: {args.b_merged_dir}")
        print(f"C bundle directory: {args.bundles_dir}")

        B_bundle_dir = args.b_merged_dir
        C_bundle_dir = args.bundles_dir

        print(f"\n[B Bundle (merged)]")
        B_idx = _load_bundle_indices(B_bundle_dir)
        print(f"  → {len(B_idx)} layers: {B_idx}")

        print(f"\n[C Bundle]")
        C_idx = _load_bundle_indices(C_bundle_dir)
        print(f"  → {len(C_idx)} layers: {C_idx}")
    else:
        # Stage 1, 2: 기존 로직
        print(f"Bundles directory: {args.bundles_dir}")

        B_bundle_dir = os.path.join(args.bundles_dir, "B")
        C_bundle_dir = os.path.join(args.bundles_dir, "C")

        print(f"\n[B Bundle]")
        B_idx = _load_bundle_indices(B_bundle_dir)
        print(f"  → {len(B_idx)} layers: {B_idx}")

        print(f"\n[C Bundle]")
        C_idx = _load_bundle_indices(C_bundle_dir)
        print(f"  → {len(C_idx)} layers: {C_idx}")

    if not B_idx and not C_idx:
        print("\n⚠ WARNING: Both B and C bundles are empty!")
        print("  This may indicate a problem with bundle_meta.json files")

    # 토크나이저
    tok = AutoTokenizer.from_pretrained(args.base_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("\n[Loading] Datasets")
    train_ds = _load_qa_sft_dataset(tok, args.qa_dataset, "train", args.max_samples, args.seq_len)
    eval_ds = _load_qa_sft_dataset(tok, args.qa_dataset, "validation", args.max_eval_samples, args.seq_len)

    # Teacher
    teacher = load_teacher(args)

    # ========================================
    # Stage 1: A 그룹 → A adapter 생성
    # ========================================
    if args.stage == 1:
        print("\n" + "="*60)
        print("STAGE 1: A-LoRA Training")
        print("="*60)

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_dir,
                torch_dtype=dtype_map[args.dtype],
                device_map={"": "cuda:0"},
                trust_remote_code=True
            )
            print("[Device] Model loaded on cuda:0 (single GPU)")
        except RuntimeError as e:
            print(f"[Warning] Could not load model on single GPU: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_dir,
                torch_dtype=dtype_map[args.dtype],
                device_map="auto",
                trust_remote_code=True
            )

        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        except Exception:
            pass

        layers = _get_layer_container(model)
        L = len(layers)
        A_idx = [i for i in range(L) if i not in set(B_idx) | set(C_idx)]

        print(f"\n[Layer Distribution]")
        print(f"  Total layers: {L}")
        print(f"  A layers: {len(A_idx)} → {A_idx[:5]}{'...' if len(A_idx) > 5 else ''}")
        print(f"  B layers: {len(B_idx)} → {B_idx}")
        print(f"  C layers: {len(C_idx)} → {C_idx}")

        model = _attach_new_adapter(model, "stageA", r=8, alpha=16, dropout=0.05)
        model.set_adapter("stageA")
        enable_only_lora_params_on_layers(model, A_idx, adapter_name="stageA")

        adapter_out = os.path.join(args.out_adapters, "A_lora")
        train_adapter(model, adapter_out, train_ds, eval_ds, args, "stageA", args.use_kd, teacher)

        print(f"\n{'='*60}")
        print(f"✓ Stage 1 Complete")
        print(f"{'='*60}")
        print(f"Adapter saved: {adapter_out}")
        print(f"\n[Next Steps]")
        print(f"1. Merge adapter:")
        print(f"   python merge_adapter.py --base_model {args.base_dir} \\")
        print(f"     --adapter_path {adapter_out} --output_dir ./merged_models/A_merged")
        print(f"2. Run stage 2 with --base_dir ./merged_models/A_merged")

    # ========================================
    # Stage 2: A merged + B 그룹 → B adapter 생성
    # ========================================
    elif args.stage == 2:
        print("\n" + "="*60)
        print("STAGE 2: B-LoRA Training (on A-merged model)")
        print("="*60)

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_dir,
                torch_dtype=dtype_map[args.dtype],
                device_map={"": "cuda:0"},
                trust_remote_code=True
            )
            print("[Device] Model loaded on cuda:0 (single GPU)")
        except RuntimeError as e:
            print(f"[Warning] Could not load model on single GPU: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_dir,
                torch_dtype=dtype_map[args.dtype],
                device_map="auto",
                trust_remote_code=True
            )

        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        except Exception:
            pass

        layers = _get_layer_container(model)
        L = len(layers)

        # B 레이어 복원
        if B_idx:
            _rehydrate_layers(model, B_bundle_dir, B_idx)
        else:
            print("\n⚠ WARNING: No B layers to rehydrate!")

        print(f"\n[Layer Distribution]")
        print(f"  Total layers: {L}")
        print(f"  B layers: {len(B_idx)} → training ONLY on these")
        print(f"  A layers (already trained): {L - len(B_idx) - len(C_idx)}")
        print(f"  C layers (excluded): {len(C_idx)}")

        # B 어댑터 추가 - B 레이어에만
        model = _attach_new_adapter(model, "stageB", r=8, alpha=16, dropout=0.05)
        model.set_adapter("stageB")
        enable_only_lora_params_on_layers(model, B_idx, adapter_name="stageB")

        adapter_out = os.path.join(args.out_adapters, "B_lora")
        train_adapter(model, adapter_out, train_ds, eval_ds, args, "stageB", args.use_kd, teacher)

        print(f"\n{'='*60}")
        print(f"✓ Stage 2 Complete")
        print(f"{'='*60}")
        print(f"Adapter saved: {adapter_out}")
        print(f"\n[Next Steps]")
        print(f"1. Merge B adapter into model")
        print(f"2. Run stage 3 with:")
        print(f"   --base_dir ./merged_models/A_merged \\")
        print(f"   --b_merged_dir ./merged_models/B_merged")

    # ========================================
    # Stage 3: A_merged + B_merged bundle + C 그룹 → C adapter 생성
    # ========================================
    elif args.stage == 3:
        print("\n" + "="*60)
        print("STAGE 3: C-LoRA Training (A_merged + B_merged + C layers)")
        print("="*60)

        # 1. A_merged 모델 로드
        print("\n[Step 1] Loading A_merged model...")
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_dir,
                torch_dtype=dtype_map[args.dtype],
                device_map={"": "cuda:0"},
                trust_remote_code=True
            )
            print("[Device] Model loaded on cuda:0 (single GPU)")
        except RuntimeError as e:
            print(f"[Warning] Could not load model on single GPU: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_dir,
                torch_dtype=dtype_map[args.dtype],
                device_map="auto",
                trust_remote_code=True
            )

        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        except Exception:
            pass

        layers = _get_layer_container(model)
        L = len(layers)

        # 2. B_merged 레이어 복원
        print("\n[Step 2] Rehydrating B_merged layers...")
        if B_idx:
            _rehydrate_layers(model, B_bundle_dir, B_idx)
            print(f"  ✓ Loaded {len(B_idx)} B layers from bundle")
        else:
            print("  ⚠ WARNING: No B layers to rehydrate!")

        # 3. C 레이어 복원
        print("\n[Step 3] Rehydrating C layers...")
        if C_idx:
            _rehydrate_layers(model, C_bundle_dir, C_idx)
            print(f"  ✓ Loaded {len(C_idx)} C layers from bundle")
        else:
            print("  ⚠ WARNING: No C layers to rehydrate!")

        print(f"\n[Layer Distribution]")
        print(f"  Total layers: {L}")
        print(f"  A layers (from A_merged): {L - len(B_idx) - len(C_idx)}")
        print(f"  B layers (from B_merged bundle): {len(B_idx)} → {B_idx}")
        print(f"  C layers (from C bundle): {len(C_idx)} → {C_idx}")
        print(f"  Training target: C layers ONLY")

        # 4. C 어댑터 추가 - C 레이어에만
        print("\n[Step 4] Adding LoRA adapter for C layers...")
        model = _attach_new_adapter(model, "stageC", r=8, alpha=16, dropout=0.05)
        model.set_adapter("stageC")
        enable_only_lora_params_on_layers(model, C_idx, adapter_name="stageC")

        # 5. KD-LoRA 훈련
        print("\n[Step 5] Training C adapter with KD-LoRA...")
        adapter_out = os.path.join(args.out_adapters, "C_lora")
        train_adapter(model, adapter_out, train_ds, eval_ds, args, "stageC", args.use_kd, teacher)

        print(f"\n{'='*60}")
        print(f"✓ Stage 3 Complete - FINAL")
        print(f"{'='*60}")
        print(f"Adapter saved: {adapter_out}")
        print(f"\n[Model Composition]")
        print(f"  • A layers: from {args.base_dir}")
        print(f"  • B layers: from {B_bundle_dir}")
        print(f"  • C adapter: {adapter_out}")

    print("\n[Done] Training completed")


if __name__ == "__main__":
    main()