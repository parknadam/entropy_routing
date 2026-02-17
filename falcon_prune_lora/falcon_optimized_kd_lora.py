#!/usr/bin/env python3
"""
Falcon 구조용 최적화 KD-LoRA 학습 코드

layeronly_drop.py로 프루닝된 Falcon 모델(A)에 LoRA를 붙여 KD 학습.
Stage 1: A 모델의 남은 레이어에 LoRA → stageA
Stage 2: B 번들 복원 후 AB 레이어에 LoRA → stageAB

사용법:
# Stage 1
CUDA_VISIBLE_DEVICES=0,1 DEVICE=cuda:0 \
python -m falcon_prune_lora.falcon_optimized_kd_lora \
  --base_dir ./falcon_results/pruning/A \
  --bundles_dir ./falcon_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./falcon_kd_lora_results/adapters \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-4 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model tiiuae/falcon-7b-instruct \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0

# Stage 2 (A merged 모델 기준)
python -m falcon_prune_lora.falcon_optimized_kd_lora \
  --base_dir ./merged_models_falcon/A_merged \
  --bundles_dir ./falcon_results/pruning/bundles \
  --stage 2 \
  --out_adapters ./falcon_kd_lora_results/adapters \
  --qa_dataset squad \
  --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-4 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model tiiuae/falcon-7b-instruct \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0
"""

import os, json, re, math
from typing import List
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
try:
    from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
except Exception:
    FalconDecoderLayer = None

# ============================================================
# Falcon 레이어 관리
# ============================================================
FALCON_LAYER_PREFIX = "transformer.h"
_LAYER_RE = re.compile(r"\bh\.(\d+)\.")


def _clean_auto_map(model_dir: str):
    """
    로컬 저장 모델의 config.json에서 auto_map 필드 제거.
    trust_remote_code=True 없이 로드할 수 있게 함.
    """
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        if "auto_map" in cfg:
            del cfg["auto_map"]
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"[config] Removed auto_map from {cfg_path}")
    except Exception as e:
        print(f"[warn] Failed to clean auto_map: {e}")


def _get_layer_container(model):
    """Falcon: model.transformer.h ModuleList 반환"""
    # 일반 FalconForCausalLM
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # PeftModel wrapping
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model") and hasattr(base.model, "transformer"):
            return base.model.transformer.h
        if hasattr(base, "transformer"):
            return base.transformer.h
    # model.model.transformer.h (다중 래핑)
    if hasattr(model, "model"):
        if hasattr(model.model, "transformer") and hasattr(model.model.transformer, "h"):
            return model.model.transformer.h
    raise RuntimeError("Falcon layer container not found (transformer.h)")


def _layer_name_prefix(model, i: int):
    """Falcon 레이어의 named_parameter prefix"""
    # PeftModel wrapping 대응
    for name, _ in model.named_parameters():
        if f".h.{i}." in name:
            # "base_model.model.transformer.h.{i}." 또는 "transformer.h.{i}."
            idx = name.index(f".h.{i}.")
            return name[:idx] + f".h.{i}."
    return f"transformer.h.{i}."


# ============================================================
# 번들 검증 및 레이어 복원
# ============================================================
def _extract_layer_sd(raw_sd: dict, idx: int):
    """safetensors에서 특정 레이어의 state_dict 추출 (prefix 제거)"""
    prefixes = [
        f"transformer.h.{idx}.",
        f"h.{idx}.",
    ]
    for pref in prefixes:
        out = {k[len(pref):]: v for k, v in raw_sd.items() if k.startswith(pref)}
        if out:
            return out
    # prefix 없는 경우 (bundle 자체가 layer-only)
    return raw_sd


def _pick_layer_file(bundle_dir: str, idx: int) -> str:
    p3 = os.path.join(bundle_dir, f"layer_{int(idx):03d}.safetensors")
    if os.path.isfile(p3):
        return p3
    p = os.path.join(bundle_dir, f"layer_{int(idx)}.safetensors")
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"layer file missing for {idx}: {p3} / {p}")


def _assert_bundle_files_exist(bundles_dir: str, group: str, indices: list):
    group_dir = os.path.join(bundles_dir, group)
    if not os.path.isdir(group_dir):
        raise FileNotFoundError(f"[bundles] group dir not found: {group_dir}")

    missing = []
    for i in indices:
        try:
            f = _pick_layer_file(group_dir, int(i))
        except FileNotFoundError:
            missing.append(i)
            continue
        if os.path.getsize(f) == 0:
            missing.append(i)

    if missing:
        raise FileNotFoundError(f"[bundles] missing/empty files for layers: {missing}")
    print(f"[bundles-ok] all {len(indices)} files present in {group_dir}")


def _rehydrate_layers(model, bundle_dir: str, indices: List[int]):
    """Bundle에서 실제 FalconDecoderLayer 가중치 복원"""
    if FalconDecoderLayer is None:
        raise RuntimeError("FalconDecoderLayer import 실패 (transformers 버전 확인 필요)")

    layers = _get_layer_container(model)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    for i in indices:
        try:
            new_layer = FalconDecoderLayer(model.config, layer_idx=int(i)).to(device=device, dtype=dtype)
        except TypeError:
            new_layer = FalconDecoderLayer(model.config).to(device=device, dtype=dtype)

        f = _pick_layer_file(bundle_dir, int(i))
        raw_sd = load_file(f)
        sd = _extract_layer_sd(raw_sd, int(i))
        sd = {k: v.to(device=device, dtype=dtype) for k, v in sd.items()}

        try:
            new_layer.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"[warn] layer {i}: {e} -> non-strict")
            new_layer.load_state_dict(sd, strict=False)

        layers[int(i)] = new_layer
        print(f"[rehydrate] layer {i} restored")


def _reapply_passlayers_from_manifest(model, base_dir: str):
    """manifest.json 기반으로 PassLayer 재적용"""
    man_path = os.path.join(base_dir, "manifest.json")
    if not os.path.isfile(man_path):
        return model

    try:
        man = json.load(open(man_path))
        removed = None
        stages = man.get("stages", {}) or {}
        A_drop = (stages.get("A", {}) or {}).get("dropped_layers", [])
        B_rem = (stages.get("B", {}) or {}).get("removed_layers", [])
        C_rem = (stages.get("C", {}) or {}).get("removed_layers", [])
        removed = A_drop or sorted(set(B_rem + C_rem))

        if not removed:
            return model
        removed = sorted(set(int(i) for i in removed))
    except Exception:
        return model

    class FalconSafePass(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden_size = hidden

        def forward(self, hidden_states, alibi=None, attention_mask=None,
                    position_ids=None, layer_past=None, head_mask=None,
                    use_cache=False, output_attentions=False, **kwargs):
            if use_cache:
                return (hidden_states, layer_past)
            return (hidden_states,)

    layers = _get_layer_container(model)
    for i in removed:
        if 0 <= i < len(layers):
            layers[i] = FalconSafePass(model.config.hidden_size)

    print(f"[reapply] installed PassLayer on: {removed}")
    return model


# ============================================================
# LoRA 어댑터 관리
# ============================================================
def _attach_new_adapter(model, name: str, r=8, alpha=16, dropout=0.05):
    """Falcon용 LoRA 어댑터 부착"""
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM",
        # Falcon attention/MLP 모듈명
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    )

    if isinstance(model, PeftModel):
        if name not in getattr(model, "peft_config", {}):
            model.add_adapter(name, cfg)
            print(f"[adapter] added '{name}'")
        return model
    return get_peft_model(model, cfg, adapter_name=name)


def _enable_only_lora_on_indices(model, indices: List[int], adapter_name: str):
    """지정 레이어의 LoRA 파라미터만 학습 가능하게 설정"""
    for p in model.parameters():
        p.requires_grad = False

    layer_patterns = [_layer_name_prefix(model, i) for i in indices]
    enabled = 0

    for name, param in model.named_parameters():
        if any(pat in name for pat in layer_patterns) and "lora_" in name.lower():
            param.requires_grad = True
            enabled += param.numel()

    if enabled == 0:
        print(f"[ERROR] No LoRA params for adapter '{adapter_name}' on layers {indices}")
        print("Available LoRA params:")
        for n, _ in model.named_parameters():
            if "lora_" in n.lower():
                print(f"  {n}")
        raise RuntimeError(f"No LoRA params enabled")

    print(f"[trainable] {adapter_name}: {enabled:,} params on layers {indices}")
    return enabled


# ============================================================
# 데이터셋
# ============================================================
def _build_chat_messages(ctx: str, q: str, dataset: str):
    sys = "You are a helpful QA assistant."
    if dataset == "squad_v2":
        sys += " If the answer is not in the context, say 'unanswerable'."
    user_msg = f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer:"
    return [{"role": "system", "content": sys}, {"role": "user", "content": user_msg}]


def _load_qa_sft_dataset(tokenizer, dataset_name, split, max_samples, seq_len):
    if dataset_name == "squad":
        ds = load_dataset("squad", split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id

    def to_list(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        if isinstance(x, list):
            return x
        return list(x)

    def process(example):
        ctx = example.get("context", "")
        q = example.get("question", "")
        ans_list = example.get("answers", {}).get("text", [])
        ans = ans_list[0] if ans_list else ""

        if not ans:
            return {"__drop__": 1}

        msgs = _build_chat_messages(ctx, q, dataset_name)

        try:
            prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_text = f"### System:\n{msgs[0]['content']}\n### User:\n{msgs[1]['content']}\n### Assistant:\n"

        prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
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
        if attn is not None:
            attn_s = attn[:, 1:].contiguous()
        else:
            attn_s = torch.ones_like(labels_s)

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
            F.log_softmax(s / T, dim=-1),
            F.log_softmax(t / T, dim=-1),
            reduction="batchmean",
            log_target=True
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
        except Exception:
            try:
                model.save_pretrained(out_dir, selected_adapters=[adapter_name])
            except TypeError:
                model.save_pretrained(out_dir)


def load_teacher(args):
    if not args.use_kd:
        return None

    print(f"[Teacher] Loading {args.teacher_model} on {args.teacher_device}")

    try:
        # ★ trust_remote_code 사용하지 않음 (Hub의 구버전 커스텀 코드가 최신 PyTorch와 비호환)
        #   transformers>=4.33은 Falcon을 네이티브 지원
        if args.teacher_4bit:
            from transformers import BitsAndBytesConfig
            teacher = AutoModelForCausalLM.from_pretrained(
                args.teacher_model,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                device_map={"": args.teacher_device},
            )
        else:
            teacher = AutoModelForCausalLM.from_pretrained(
                args.teacher_model,
                torch_dtype=torch.bfloat16,
                device_map={"": args.teacher_device},
            )
    except Exception as e:
        print(f"[Teacher] Error: {e}")
        raise

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
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--out_adapters", required=True)

    p.add_argument("--qa_dataset", default="squad")
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--max_eval_samples", type=int, default=8000)
    p.add_argument("--seq_len", type=int, default=1024)

    p.add_argument("--lr", type=float, default=2e-4)
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
    p.add_argument("--teacher_model", default="tiiuae/falcon-7b-instruct")
    p.add_argument("--teacher_device", default="cuda:1")
    p.add_argument("--teacher_4bit", action="store_true")
    p.add_argument("--kd_alpha", type=float, default=0.1)
    p.add_argument("--kd_T", type=float, default=2.0)

    return p.parse_args()


def main():
    args = parse_args()

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True, local_files_only=True)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Student
    print(f"\n[Loading] Student from {args.base_dir}")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    # ★ 로컬 저장 모델은 trust_remote_code 불필요 (transformers>=4.33 네이티브 Falcon 지원)
    #   trust_remote_code=True 사용 시 configuration_falcon.py 를 찾아 OSError 발생
    load_kwargs = dict(
        torch_dtype=dtype_map[args.dtype],
        device_map=None,
        local_files_only=True,
        attn_implementation="eager",
    )
    # config.json에 auto_map이 남아있으면 제거
    _clean_auto_map(args.base_dir)
    model = AutoModelForCausalLM.from_pretrained(args.base_dir, **load_kwargs)

    device = torch.device(os.environ.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.config.use_cache = False

    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:
        pass

    model = _reapply_passlayers_from_manifest(model, args.base_dir)

    # Indices
    with open(os.path.join(args.base_dir, "prune_log.json")) as f:
        log = json.load(f)
    B_idx, C_idx = log["split"]["B"], log["split"]["C"]

    # Datasets
    print("\n[Loading] Datasets")
    train_ds = _load_qa_sft_dataset(tok, args.qa_dataset, "train", args.max_samples, args.seq_len)
    eval_ds = _load_qa_sft_dataset(tok, args.qa_dataset, "validation", args.max_eval_samples, args.seq_len)

    # Teacher
    teacher = load_teacher(args)

    layers = _get_layer_container(model)
    L = len(layers)

    if args.stage == 1:
        print("\n" + "=" * 60)
        print("STAGE 1: A-LoRA")
        print("=" * 60)

        A_idx = [i for i in range(L) if i not in set(B_idx) | set(C_idx)]
        print(f"A layers: {A_idx}")

        model = _attach_new_adapter(model, "stageA", r=8, alpha=16, dropout=0.05)
        model.set_adapter("stageA")
        _enable_only_lora_on_indices(model, A_idx, "stageA")

        train_adapter(model, os.path.join(args.out_adapters, "A_lora", "stageA"),
                      train_ds, eval_ds, args, "stageA", args.use_kd, teacher)

    elif args.stage == 2:
        print("\n" + "=" * 60)
        print("STAGE 2: AB-LoRA")
        print("=" * 60)

        AB_idx = [i for i in range(L) if i not in set(C_idx)]
        print(f"AB layers: {AB_idx}")

        _assert_bundle_files_exist(args.bundles_dir, "B", B_idx)
        _rehydrate_layers(model, os.path.join(args.bundles_dir, "B"), B_idx)

        bad = [i for i in AB_idx if (FalconDecoderLayer is not None
               and not isinstance(layers[i], FalconDecoderLayer))]
        if bad:
            raise RuntimeError(f"Non-real layers: {bad}")

        model = _attach_new_adapter(model, "stageAB", r=8, alpha=16, dropout=0.05)
        model.set_adapter("stageAB")
        _enable_only_lora_on_indices(model, AB_idx, "stageAB")

        train_adapter(model, os.path.join(args.out_adapters, "AB_lora", "stageAB"),
                      train_ds, eval_ds, args, "stageAB", args.use_kd, teacher)

    print("\n[Done] Training completed")


if __name__ == "__main__":
    main()