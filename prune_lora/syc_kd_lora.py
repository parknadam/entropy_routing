#!/usr/bin/env python3
"""
최적화된 KD-LoRA 학습 코드

사용법:
python -m prune_lora.syc_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./syc_kd_lora_results/adapters \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0
"""

import os, json, re, math
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, default_data_collator
)
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# ============================================================
# 유틸리티: 레이어 관리
# ============================================================
CANON_PATH = "model.model.layers"
_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")

def _resolve_attr_path(root, dotted: str):
    parent = root
    segs = dotted.split(".")
    for seg in segs[:-1]:
        parent = getattr(parent, seg)
    last = segs[-1]
    val = getattr(parent, last)
    return parent, last, val

def _canonicalize_layers(model):
    candidates = [
        "model.layers", "model.decoder.layers",
        "model.model.layers", "model.model.decoder.layers",
        "base_model.model.layers", "base_model.model.decoder.layers",
        "base_model.model.model.layers", "base_model.model.model.decoder.layers",
    ]
    found = None
    found_parent = None
    found_name = None
    found_path = None

    for path in candidates:
        try:
            parent, name, cur = _resolve_attr_path(model, path)
        except Exception:
            continue
        if hasattr(cur, "__len__") and hasattr(cur, "__getitem__"):
            found, found_parent, found_name, found_path = cur, parent, name, path
            break

    if found is None:
        raise AttributeError(f"decoder layers not found (checked: {', '.join(candidates)})")

    if not isinstance(found, (list, nn.ModuleList)):
        new_cur = nn.ModuleList(list(found))
        setattr(found_parent, found_name, new_cur)
        found = new_cur

    try:
        canon_parent, _, _ = _resolve_attr_path(model, CANON_PATH.replace(".layers", ""))
        setattr(canon_parent, "layers", found)
        model._canonical_layers_path = CANON_PATH
    except Exception:
        model._canonical_layers_path = found_path

    model._canonical_layers = found
    return found

def _get_layer_container(model):
    if not hasattr(model, "_canonical_layers"):
        _canonicalize_layers(model)
    return model._canonical_layers

def _layer_name_prefix(model, i: int):
    if not hasattr(model, "_canonical_layers_path"):
        _canonicalize_layers(model)
    return f"{model._canonical_layers_path}.{i}."

# ============================================================
# 번들 검증 및 레이어 복원
# ============================================================
def _assert_bundle_files_exist(bundles_dir: str, group: str, indices: list):
    group_dir = os.path.join(bundles_dir, group)
    if not os.path.isdir(group_dir):
        raise FileNotFoundError(f"[bundles] group dir not found: {group_dir}")

    missing = []
    for i in indices:
        fname = os.path.join(group_dir, f"layer_{int(i):03d}.safetensors")
        if not os.path.isfile(fname) or os.path.getsize(fname) == 0:
            missing.append(i)

    if missing:
        raise FileNotFoundError(f"[bundles] missing/empty files for layers: {missing}")

    print(f"[bundles-ok] all {len(indices)} files present in {group_dir}")

def _rehydrate_layers(model, bundle_dir: str, indices: List[int]):
    layers = _get_layer_container(model)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    for i in indices:
        new_layer = LlamaDecoderLayer(model.config, layer_idx=int(i)).to(device=device, dtype=dtype)
        f = os.path.join(bundle_dir, f"layer_{int(i):03d}.safetensors")
        if not os.path.isfile(f):
            raise FileNotFoundError(f"bundle miss: {f}")
        
        sd = load_file(f)
        sd = {k: v.to(device=device, dtype=dtype) for k, v in sd.items()}
        
        try:
            new_layer.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"[warn] strict load failed for {i}: {e} -> non-strict")
            new_layer.load_state_dict(sd, strict=False)

        layers[int(i)] = new_layer
        print(f"[rehydrate] layer {i} restored")

def _reapply_passlayers_from_manifest(model, base_dir: str):
    man_path = os.path.join(base_dir, "manifest.json")
    if not os.path.isfile(man_path):
        print("[reapply] manifest.json not found -> skip")
        return model

    try:
        man = json.load(open(man_path, "r"))
    except Exception as e:
        print(f"[reapply] failed to read manifest: {e} -> skip")
        return model

    removed = (man.get("simdrop", {}) or {}).get("removed_layers")
    if not removed:
        removed = man.get("removed_layers")
    if not removed:
        stages = man.get("stages", {}) or {}
        A_drop = (stages.get("A", {}) or {}).get("dropped_layers", []) or []
        B_rem = (stages.get("B", {}) or {}).get("removed_layers", []) or []
        C_rem = (stages.get("C", {}) or {}).get("removed_layers", []) or []
        removed = A_drop or sorted(set(B_rem + C_rem))

    if not removed:
        print("[reapply] removed_layers empty -> skip")
        return model

    try:
        removed = sorted(set(int(i) for i in removed))
    except Exception:
        print("[reapply] removed_layers has non-int -> skip")
        return model

    try:
        from prune_lora.pruning.identity import LlamaPassLayer as _Inner
        class _Wrapper(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.inner = _Inner(hidden)
            def forward(self, hidden_states, *a, **kw):
                out = self.inner(hidden_states, *a, **kw)
                return out[0] if isinstance(out, tuple) else out
        def _make(h): 
            return _Wrapper(h)
        print("[reapply] using project LlamaPassLayer")
    except Exception:
        class SafePass(nn.Module):
            def __init__(self, hidden):
                super().__init__()
            def forward(self, x, *a, **kw):
                return x
        def _make(h):
            return SafePass(h)
        print("[reapply] using SafePassLayer")

    layers = _get_layer_container(model)
    L = len(layers)
    hidden = model.config.hidden_size

    for i in removed:
        if 0 <= i < L:
            layers[i] = _make(hidden)

    print(f"[reapply] installed PassLayer on: {removed}")
    return model

# ============================================================
# LoRA 어댑터 관리
# ============================================================
def _attach_new_adapter(
    model, name: str,
    target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"),
    r=8, alpha=16, dropout=0.05
):
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )

    if isinstance(model, PeftModel):
        if name not in getattr(model, "peft_config", {}):
            model.add_adapter(name, cfg)
            print(f"[adapter] added adapter '{name}'")
        return model
    else:
        return get_peft_model(model, cfg, adapter_name=name)

def _enable_only_lora_on_indices(model, indices: List[int], adapter_name: str):
    """
    핵심 수정: 지정된 레이어의 LoRA 파라미터만 활성화
    """
    # 1) 모든 파라미터 비활성화
    for p in model.parameters():
        p.requires_grad = False

    # 2) 대상 레이어의 LoRA 파라미터만 활성화
    layer_patterns = [_layer_name_prefix(model, i) for i in indices]
    enabled_count = 0
    enabled_params = []

    for name, param in model.named_parameters():
        # 레이어 인덱스 매칭
        is_target_layer = any(pat in name for pat in layer_patterns)
        is_lora_param = "lora_" in name.lower()
        
        if is_target_layer and is_lora_param:
            param.requires_grad = True
            enabled_count += param.numel()
            enabled_params.append(name)

    if enabled_count == 0:
        print("[ERROR] No LoRA parameters were enabled!")
        print(f"Target adapter: {adapter_name}")
        print(f"Target layers: {indices}")
        print("\nAvailable LoRA parameters:")
        for name, _ in model.named_parameters():
            if "lora_" in name.lower():
                print(f"  - {name}")
        raise RuntimeError(f"No LoRA params enabled for adapter='{adapter_name}'")

    print(f"[trainable] adapter={adapter_name} layers={indices}")
    print(f"[trainable] enabled {enabled_count:,} parameters ({len(enabled_params)} tensors)")
    
    return enabled_count

# ============================================================
# 데이터셋 준비
# ============================================================
def _build_chat_messages(context: str, question: str, qa_dataset: str, unans_token="unanswerable"):
    sys = "You are a helpful QA assistant."
    if qa_dataset == "squad_v2":
        sys += f" If the answer is not in the context, say '{unans_token}'."

    user = (
        "Answer the question using the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]

def _encode_chat_prompt_ids(tokenizer, messages):
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]
    return prompt_ids

def _load_qa_sft_dataset(
    tokenizer,
    qa_dataset="squad",
    split="train",
    max_samples=5000,
    seq_len=1024,
    unans_token="unanswerable",
    add_eos=True,
):
    #ds = load_dataset(qa_dataset, split=split)
    DATASET_ID = {"squad": "rajpurkar/squad", "squad_v2": "rajpurkar/squad_v2"}
    repo = DATASET_ID.get(qa_dataset, qa_dataset)
    ds = load_dataset(repo, split=split)

    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    def _to_list_ids(x):
        # apply_chat_template가 dict/BatchEncoding을 주는 케이스
        if isinstance(x, dict):
            x = x.get("input_ids", x)
        if isinstance(x, BatchEncoding):
            x = x["input_ids"]

        # tensor/ndarray → list
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().tolist()
        elif isinstance(x, np.ndarray):
            x = x.tolist()

        # batched 형태(list[list[int]])면 첫 요소만
        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
            x = x[0]

        # 최종적으로 list[int]
        return list(map(int, x))
        

    def to_ex(ex):
        ctx = ex.get("context", "")
        q = ex.get("question", "")
        ans_list = ex.get("answers", {}).get("text", [])
        target = (ans_list[0] if ans_list else ("unanswerable" if qa_dataset == "squad_v2" else ""))

        messages = _build_chat_messages(ctx, q, qa_dataset, unans_token=unans_token)
        prompt_ids = _encode_chat_prompt_ids(tokenizer, messages)
        prompt_ids = _to_list_ids(prompt_ids)

        ans_text = (" " + target) if target else ""
        ans_ids = tokenizer(ans_text, add_special_tokens=False)["input_ids"]
        ans_ids = _to_list_ids(ans_ids)

        if add_eos and eos_id is not None:
            ans_ids = ans_ids + [eos_id]

        if len(ans_ids) < 1:
            return {"__drop__": 1}

        full_ids = prompt_ids + ans_ids
        prompt_len = len(prompt_ids)

        if len(full_ids) > seq_len:
            cut = len(full_ids) - seq_len
            full_ids = full_ids[cut:]
            prompt_len = max(0, prompt_len - cut)

        pad_len = seq_len - len(full_ids)
        input_ids = ([pad_id] * pad_len) + full_ids
        attention_mask = ([0] * pad_len) + ([1] * len(full_ids))

        labels = input_ids.copy()
        for i in range(pad_len):
            labels[i] = -100

        prompt_start = pad_len
        prompt_end = pad_len + prompt_len
        for i in range(prompt_start, min(prompt_end, seq_len)):
            labels[i] = -100

        if prompt_end >= seq_len:
            return {"__drop__": 1}

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "__drop__": 0,
        }

    ds = ds.map(to_ex, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["__drop__"] == 0)
    if "__drop__" in ds.column_names:
        ds = ds.remove_columns(["__drop__"])
    return ds

# ============================================================
# KD Trainer (핵심 수정)
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

        # Student forward
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        s_logits_full = student_outputs.logits

        # Teacher forward
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"].to(self.teacher_model.device),
                attention_mask=inputs["attention_mask"].to(self.teacher_model.device),
            )
            t_logits_full = teacher_outputs.logits.to(s_logits_full.device)

        # Causal shift
        s_logits = s_logits_full[:, :-1, :].contiguous()
        t_logits = t_logits_full[:, :-1, :].contiguous()
        labels_s = labels[:, 1:].contiguous()
        attn_s = inputs["attention_mask"][:, 1:].contiguous()

        # Supervised token mask
        mask = (labels_s != -100) & (attn_s == 1)
        tok_n = int(mask.sum().item())

        if tok_n == 0:
            if self.args.logging_steps > 0 and (self.state.global_step % self.args.logging_steps == 0):
                self.log({"tok_n": 0.0, "skip_batch": 1.0})
            loss = s_logits_full.sum() * 0.0
            return (loss, student_outputs) if return_outputs else loss

        # Gather supervised tokens
        s = s_logits[mask].float()  # [N, V]
        t = t_logits[mask].float()  # [N, V]
        y = labels_s[mask]          # [N]

        # Clamp for stability
        s = s.clamp(-50, 50)
        t = t.clamp(-50, 50)

        # KD loss
        T = self.T
        log_s = F.log_softmax(s / T, dim=-1)
        log_t = F.log_softmax(t / T, dim=-1)
        
        soft_loss = F.kl_div(log_s, log_t, reduction="batchmean", log_target=True) * (T * T)
        hard_loss = F.cross_entropy(s, y)

        # NaN guard
        if not torch.isfinite(soft_loss) or not torch.isfinite(hard_loss):
            if self.args.logging_steps > 0 and (self.state.global_step % self.args.logging_steps == 0):
                self.log({"tok_n": float(tok_n), "nan_guard": 1.0})
            loss = hard_loss if torch.isfinite(hard_loss) else s_logits_full.sum() * 0.0
        else:
            a = self.kd_alpha
            loss = a * soft_loss + (1.0 - a) * hard_loss

        # Logging
        if self.args.logging_steps > 0 and (self.state.global_step % self.args.logging_steps == 0):
            self.log({
                "tok_n": float(tok_n),
                "kd_soft": float(soft_loss.detach().cpu()) if torch.isfinite(soft_loss) else float("nan"),
                "kd_hard": float(hard_loss.detach().cpu()),
                "kd_ppl": float(math.exp(hard_loss.detach().cpu())),
            })

        return (loss, student_outputs) if return_outputs else loss

# ============================================================
# Training wrapper
# ============================================================
def _make_training_args(out_dir: str, args):
    common = dict(
        output_dir=out_dir,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=args.save_total_limit,
    )

    if args.dtype == "bf16":
        common.update(dict(bf16=True, fp16=False))
    elif args.dtype == "fp16":
        common.update(dict(fp16=True, bf16=False))
    else:
        common.update(dict(fp16=False, bf16=False))

    try:
        return TrainingArguments(
            **common,
            eval_strategy=("steps" if args.eval_steps > 0 else "no"),
            eval_steps=(args.eval_steps if args.eval_steps > 0 else None),
            save_strategy=("steps" if args.save_steps > 0 else "no"),
            save_steps=(args.save_steps if args.save_steps > 0 else None),
        )
    except TypeError:
        return TrainingArguments(
            **common,
            evaluation_strategy=("steps" if args.eval_steps > 0 else "no"),
            eval_steps=(args.eval_steps if args.eval_steps > 0 else None),
            save_strategy=("steps" if args.save_steps > 0 else "no"),
            save_steps=(args.save_steps if args.save_steps > 0 else None),
        )

def train_adapter(model, tokenizer, out_dir: str, train_ds, eval_ds, args, adapter_name: str, use_kd: bool = False, teacher_model=None):
    os.makedirs(out_dir, exist_ok=True)
    targs = _make_training_args(out_dir, args)

    # 핵심: trainable parameters 확인
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total_trainable = sum(x[1] for x in trainable)
    print(f"\n[Pre-train check] Trainable parameters: {total_trainable:,}")
    
    if total_trainable == 0:
        raise RuntimeError("No trainable parameters! Check LoRA attachment and layer enablement.")

    trainer_cls = KDTrainer if (use_kd and teacher_model is not None) else Trainer

    """
    kwargs = dict(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    if trainer_cls is KDTrainer:
        kwargs.update(dict(
            teacher_model=teacher_model,
            kd_alpha=args.kd_alpha,
            temperature=args.kd_T,
        ))

    trainer = trainer_cls(**kwargs)
    """
    # 수정 후 (직접 인자로 전달)
    trainer_kwargs = {
        "model": model,
        "args": targs,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": default_data_collator,
        "tokenizer": tokenizer, # 명시적으로 포함
    }

    if trainer_cls is KDTrainer:
        trainer_kwargs.update({
            "teacher_model": teacher_model,
            "kd_alpha": args.kd_alpha,
            "temperature": args.kd_T,
        })

    # **kwargs 대신 명시적 인자 혹은 정리된 dict 전달
    trainer = trainer_cls(**trainer_kwargs)


    # 학습 전 optimizer 확인
    print(f"[Optimizer] Learning rate: {trainer.optimizer.defaults.get('lr', 'N/A')}")
    print(f"[Optimizer] Param groups: {len(trainer.optimizer.param_groups)}")
    for i, pg in enumerate(trainer.optimizer.param_groups):
        print(f"  Group {i}: {len(pg['params'])} params, lr={pg.get('lr', 'N/A')}")

    trainer.train()

    if isinstance(model, PeftModel):
        try:
            model.save_pretrained(out_dir, selected_adapters=[adapter_name])
        except TypeError:
            model.save_pretrained(out_dir)

# ============================================================
# Teacher 로드
# ============================================================
def load_teacher(args):
    if not args.use_kd:
        return None

    print(f"[KD] Loading teacher: {args.teacher_model} on {args.teacher_device}")

    common = dict(
        device_map={"": args.teacher_device},  # 예: "cuda:1"
        local_files_only=False,
    )

    if args.teacher_4bit:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # 설치 확인용

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            teacher = AutoModelForCausalLM.from_pretrained(
                args.teacher_model,
                quantization_config=bnb_cfg,
                **common,
            )
        except Exception as e:
            print(f"[KD] 4bit requested but unavailable ({e}). Falling back to bf16.")
            teacher = AutoModelForCausalLM.from_pretrained(
                args.teacher_model,
                torch_dtype=torch.bfloat16,
                **common,
            )
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.bfloat16,
            **common,
        )

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher

# ============================================================
# Main
# ============================================================
def parse_args():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--bundles_dir", type=str, required=True)
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--out_adapters", type=str, required=True)

    ap.add_argument("--qa_dataset", type=str, default="squad") 
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--max_eval_samples", type=int, default=8000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--unans_token", type=str, default="unanswerable")

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad_acc", type=int, default=32)
    ap.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=0)
    ap.add_argument("--save_total_limit", type=int, default=2)

    ap.add_argument("--use_kd", action="store_true")
    ap.add_argument("--teacher_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--teacher_device", type=str, default="cuda:1")
    ap.add_argument("--teacher_4bit", action="store_true")
    ap.add_argument("--kd_alpha", type=float, default=0.1)
    ap.add_argument("--kd_T", type=float, default=2.0)

    return ap.parse_args()

def main():
    args = parse_args()

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Student model
    print(f"\n[Loading] Student model from {args.base_dir}")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.base_dir,
        torch_dtype=dtype_map[args.dtype],
        device_map=None,
        local_files_only=True,
    )
    device = torch.device(os.environ.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.config.use_cache = False

    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:
        pass

    model = _reapply_passlayers_from_manifest(model, args.base_dir)

    # Load indices
    with open(os.path.join(args.base_dir, "prune_log.json"), "r") as f:
        log = json.load(f)
    B_idx, C_idx = log["split"]["B"], log["split"]["C"]

    # Datasets
    print("\n[Loading] Datasets")
    train_ds = _load_qa_sft_dataset(tok, args.qa_dataset, "train", args.max_samples, args.seq_len, args.unans_token)
    eval_ds = _load_qa_sft_dataset(tok, args.qa_dataset, "validation", args.max_eval_samples, args.seq_len, args.unans_token)

    # Teacher
    teacher = load_teacher(args)

    layers = _get_layer_container(model)
    L = len(layers)

    if args.stage == 1:
        print("\n" + "="*60)
        print("STAGE 1: A-LoRA Training")
        print("="*60)
        
        removed = set(B_idx) | set(C_idx)
        A_idx = [i for i in range(L) if i not in removed]
        print(f"A layers: {A_idx}")

        model = _attach_new_adapter(model, "stageA", r=8, alpha=16, dropout=0.05)
        model.set_adapter("stageA")
        
        enabled = _enable_only_lora_on_indices(model, A_idx, "stageA")
        print(f"Enabled {enabled:,} LoRA parameters for stage A")

        out_dir = os.path.join(args.out_adapters, "A_lora", "stageA")
        train_adapter(model, tok, out_dir, train_ds, eval_ds, args, "stageA", args.use_kd, teacher)

    elif args.stage == 2:
        print("\n" + "="*60)
        print("STAGE 2: AB-LoRA Training")
        print("="*60)
        
        AB_idx = [i for i in range(L) if i not in set(C_idx)]
        print(f"AB layers: {AB_idx}")
        print(f"Restoring B layers: {B_idx}")

        _assert_bundle_files_exist(args.bundles_dir, "B", B_idx)
        _rehydrate_layers(model, os.path.join(args.bundles_dir, "B"), B_idx)

        bad = [i for i in AB_idx if not isinstance(layers[i], LlamaDecoderLayer)]
        if bad:
            raise RuntimeError(f"[check] AB indices not real LlamaDecoderLayer: {bad}")

        model = _attach_new_adapter(model, "stageAB", r=8, alpha=16, dropout=0.05)
        model.set_adapter("stageAB")
        
        enabled = _enable_only_lora_on_indices(model, AB_idx, "stageAB")
        print(f"Enabled {enabled:,} LoRA parameters for stage AB")

        out_dir = os.path.join(args.out_adapters, "AB_lora", "stageAB")
        train_adapter(model, tok, out_dir, train_ds, eval_ds, args, "stageAB", args.use_kd, teacher)

    print("\n[Done] Training completed successfully")

if __name__ == "__main__":
    main()