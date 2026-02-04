#!/usr/bin/env python3
"""
최적화된 KD-LoRA 학습 코드 (버그 수정 완료)

사용법:
CUDA_VISIBLE_DEVICES=0,4 DEVICE=cuda:0 \
python -m prune_lora.optimized_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./kd_lora_results/adapters \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-4 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0

# 5th 실험
python -m prune_lora.optimized_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./kd_result3/adapters \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 \
  --lr 3e-4 \
  --epochs 2 \
  --bs 1 \
  --grad_acc 32 \
  --use_kd \
  --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit \
  --teacher_device cuda:1 \
  --kd_alpha 0.2 \
  --kd_T 2.0
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
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# ============================================================
# 유틸리티: 레이어 관리
# ============================================================
CANON_PATH = "model.model.layers"
_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")

def _resolve_attr_path(root, dotted: str):
    parent = root
    for seg in dotted.split(".")[:-1]:
        parent = getattr(parent, seg)
    last = dotted.split(".")[-1]
    return parent, last, getattr(parent, last)

def _canonicalize_layers(model):
    candidates = [
        "model.layers", "model.decoder.layers",
        "model.model.layers", "model.model.decoder.layers",
        "base_model.model.layers", "base_model.model.decoder.layers",
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
                except:
                    model._canonical_layers_path = path
                
                model._canonical_layers = cur
                return cur
        except:
            continue
    
    raise AttributeError("decoder layers not found")

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
    
    missing = [i for i in indices if not os.path.isfile(
        os.path.join(group_dir, f"layer_{int(i):03d}.safetensors")
    ) or os.path.getsize(os.path.join(group_dir, f"layer_{int(i):03d}.safetensors")) == 0]
    
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
        sd = {k: v.to(device=device, dtype=dtype) for k, v in load_file(f).items()}
        
        try:
            new_layer.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"[warn] layer {i}: {e} -> non-strict")
            new_layer.load_state_dict(sd, strict=False)
        
        layers[int(i)] = new_layer
        print(f"[rehydrate] layer {i} restored")

def _reapply_passlayers_from_manifest(model, base_dir: str):
    man_path = os.path.join(base_dir, "manifest.json")
    if not os.path.isfile(man_path):
        return model
    
    try:
        man = json.load(open(man_path))
        removed = (man.get("simdrop", {}) or {}).get("removed_layers")
        if not removed:
            removed = man.get("removed_layers")
        if not removed:
            stages = man.get("stages", {}) or {}
            A_drop = (stages.get("A", {}) or {}).get("dropped_layers", [])
            B_rem = (stages.get("B", {}) or {}).get("removed_layers", [])
            C_rem = (stages.get("C", {}) or {}).get("removed_layers", [])
            removed = A_drop or sorted(set(B_rem + C_rem))
        
        if not removed:
            return model
        removed = sorted(set(int(i) for i in removed))
    except:
        return model
    
    class SafePass(nn.Module):
        def __init__(self, hidden):
            super().__init__()
        def forward(self, x, *a, **kw):
            return x
    
    layers = _get_layer_container(model)
    for i in removed:
        if 0 <= i < len(layers):
            layers[i] = SafePass(model.config.hidden_size)
    
    print(f"[reapply] installed PassLayer on: {removed}")
    return model

# ============================================================
# LoRA 어댑터 관리
# ============================================================
def _attach_new_adapter(model, name: str, r=8, alpha=16, dropout=0.05):
#def _attach_new_adapter(model, name: str, r=16, alpha=32, dropout=0.05):

    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    
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
    
    def to_list(x):
        """BatchEncoding/dict/tensor/ndarray를 list[int]로 변환"""
        if hasattr(x, "input_ids"):  # BatchEncoding
            x = x.input_ids
        elif isinstance(x, dict):
            x = x.get("input_ids", x)
        
        if hasattr(x, "tolist"):  # tensor/ndarray
            x = x.tolist()
        
        # batched 형태면 첫 요소만
        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
            x = x[0]
        
        return list(x) if x else []
    
    def process(ex):
        ctx = ex.get("context", "")
        q = ex.get("question", "")
        ans = (ex.get("answers", {}).get("text", [""])[0] or 
               ("unanswerable" if qa_dataset == "squad_v2" else ""))
        
        messages = _build_chat_messages(ctx, q, qa_dataset)
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
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
                attention_mask=inputs["attention_mask"].to(self.teacher_model.device),
            )
            t_logits_full = t_out.logits.to(s_logits_full.device)
        
        # Causal shift
        s_logits = s_logits_full[:, :-1, :].contiguous()
        t_logits = t_logits_full[:, :-1, :].contiguous()
        labels_s = labels[:, 1:].contiguous()
        attn_s = inputs["attention_mask"][:, 1:].contiguous()
        
        mask = (labels_s != -100) & (attn_s == 1)
        tok_n = mask.sum().item()
        
        if tok_n == 0:
            return (s_logits_full.sum() * 0.0, s_out) if return_outputs else s_logits_full.sum() * 0.0
        
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
                "kd_ppl": float(math.exp(hard_loss)),
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
            save_steps=args.save_steps if args.eval_steps > 0 else None,
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
        kwargs.update({"teacher_model": teacher, "kd_alpha": args.kd_alpha, "temperature": args.kd_T})
    
    trainer = trainer_cls(**kwargs)
    
    trainer.train()
    
    if isinstance(model, PeftModel):
        try:
            model.save_pretrained(out_dir, selected_adapters=[adapter_name])
        except TypeError:
            model.save_pretrained(out_dir)

def load_teacher(args):
    if not args.use_kd:
        return None
    
    print(f"[Teacher] Loading {args.teacher_model} on {args.teacher_device}")
    
    try:
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
    p.add_argument("--teacher_model", default="meta-llama/Llama-2-7b-chat-hf")
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
    except:
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
        print("\n" + "="*60)
        print("STAGE 1: A-LoRA")
        print("="*60)
        
        A_idx = [i for i in range(L) if i not in set(B_idx) | set(C_idx)]
        print(f"A layers: {A_idx}")
        
        model = _attach_new_adapter(model, "stageA", r=8, alpha=16, dropout=0.05)
        model.set_adapter("stageA")
        _enable_only_lora_on_indices(model, A_idx, "stageA")
        
        train_adapter(model, os.path.join(args.out_adapters, "A_lora", "stageA"), 
                     train_ds, eval_ds, args, "stageA", args.use_kd, teacher)
    
    elif args.stage == 2:
        print("\n" + "="*60)
        print("STAGE 2: AB-LoRA")
        print("="*60)
        
        AB_idx = [i for i in range(L) if i not in set(C_idx)]
        print(f"AB layers: {AB_idx}")
        
        _assert_bundle_files_exist(args.bundles_dir, "B", B_idx)
        _rehydrate_layers(model, os.path.join(args.bundles_dir, "B"), B_idx)
        
        bad = [i for i in AB_idx if not isinstance(layers[i], LlamaDecoderLayer)]
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