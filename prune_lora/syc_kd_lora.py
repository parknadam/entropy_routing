# lora와 공정한 비교를 위해 기본적인 세팅 - lora 마스킹 범위 등을 동일하게 둠
# Trainer만 KDTrainer로 변경

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Progressive QA LoRA (A / AB) + optional KD-LoRA with identical LoRA setup.

Example (LoRA):
python -m prune_lora.total_progressive_qa_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --epochs 1 --bs 1 --grad_acc 32

Example (KD-LoRA, same settings + KD only):
python -m prune_lora.total_progressive_qa_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./kd_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model meta-llama/Llama-2-7b-chat-hf --teacher_4bit \
  --kd_alpha 0.1 --kd_T 2.0 --teacher_device cuda:1

python -m prune_lora.syc_kd_lora \
  --base_dir /dev/shm/7b_results/pruning/A \
  --bundles_dir /dev/shm/7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters /dev/shm/syc_kd_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --epochs 1 --bs 1 --grad_acc 32 \
  --dtype bf16 --logging_steps 20 --eval_steps 200 \
  --use_kd --teacher_model meta-llama/Llama-2-7b-chat-hf --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0

"""

import os
import re
import json
import argparse
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from peft import (
    LoraConfig,
    TaskType,
    PeftModel,
    get_peft_model,
)
from peft.utils import get_peft_model_state_dict

from safetensors.torch import load_file


# ============================================================
# Utilities: Layers / Bundles / PassLayer re-apply
# ============================================================

CANON_PATH = "model.model.layers"
_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")  # key contains ...layers.21....

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
        "model.layers",
        "model.decoder.layers",
        "model.model.layers",
        "model.model.decoder.layers",
        "base_model.model.layers",
        "base_model.model.decoder.layers",
        "base_model.model.model.layers",
        "base_model.model.model.decoder.layers",
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

def _assert_bundle_files_exist(bundles_dir: str, group: str, indices: list):
    group_dir = os.path.join(bundles_dir, group)
    if not os.path.isdir(group_dir):
        raise FileNotFoundError(f"[bundles] group dir not found: {group_dir}")

    missing = []
    zero_size = []
    for i in indices:
        fname = os.path.join(group_dir, f"layer_{int(i):03d}.safetensors")
        if not os.path.isfile(fname):
            missing.append(i)
        else:
            try:
                if os.path.getsize(fname) == 0:
                    zero_size.append(i)
            except OSError:
                zero_size.append(i)

    if missing or zero_size:
        msg = []
        if missing:
            msg.append(f"missing files for layers: {missing}")
        if zero_size:
            msg.append(f"zero-size files for layers: {zero_size}")
        raise FileNotFoundError(f"[bundles] problems in {group_dir}: " + "; ".join(msg))

    print(f"[bundles-ok] all {len(indices)} files present in {group_dir}")

def _reapply_passlayers_from_manifest(model, base_dir: str):
    """
    A 디렉토리 manifest.json을 보고 제거된 레이어 인덱스에 PassLayer를 다시 깔아
    '논리적으로 32 레이어 구조'를 유지하게 함.
    """
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

    # 프로젝트 커스텀 PassLayer 있으면 사용, 없으면 SafePass
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

    hidden = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden is None:
        try:
            hidden = model.config.hidden_size
        except Exception:
            print("[reapply] hidden_size not found -> skip")
            return model

    for i in removed:
        if 0 <= i < L:
            layers[i] = _make(hidden)
        else:
            print(f"[reapply] index {i} out of range -> skip")

    print(f"[reapply] installed PassLayer on: {removed}")
    return model


# ============================================================
# Rehydrate layers for AB stage
# ============================================================

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def _rehydrate_layers(model, bundle_dir: str, indices: List[int]):
    layers = _get_layer_container(model)
    dtype = next(model.parameters()).dtype
    tgt = next(model.parameters()).device  # 단일 디바이스로 고정

    for i in indices:
        new_layer = LlamaDecoderLayer(model.config, layer_idx=int(i)).to(device=tgt, dtype=dtype)
        f = os.path.join(bundle_dir, f"layer_{int(i):03d}.safetensors")
        if not os.path.isfile(f):
            raise FileNotFoundError(f"bundle miss: {f}")
        sd = load_file(f)
        sd = {k: v.to(device=tgt, dtype=dtype) for k, v in sd.items()}

        try:
            new_layer.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"[warn] strict load failed for {i}: {e} -> non-strict")
            new_layer.load_state_dict(sd, strict=False)

        layers[int(i)] = new_layer
        print(f"[rehydrate] layer {i} restored on {tgt}")


# ============================================================
# Adapter attach / enable helpers
# ============================================================

def _freeze_all(model):
    for _, p in model.named_parameters():
        p.requires_grad = False

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

def _enable_only_lora_on_indices_for_adapter_by_name(model, indices: List[int], adapter_name: str, keep_layernorm=False):
    for _, p in model.named_parameters():
        p.requires_grad = False

    enabled = 0
    layer_patterns = [_layer_name_prefix(model, i) for i in indices]

    for pname, p in model.named_parameters():
        # 대상 레이어 + LoRA 파라미터만
        if any(pat in pname for pat in layer_patterns) and ("lora_" in pname.lower()):
            # adapter_name이 이름에 안 박히는 구현도 있어서 여기서는 강제 필터는 약하게
            p.requires_grad = True
            enabled += p.numel()
            continue

        if keep_layernorm:
            if any(pat in pname for pat in layer_patterns) and ("layernorm" in pname.lower() or ".ln_" in pname.lower() or ".norm" in pname.lower()):
                p.requires_grad = True

    if enabled == 0:
        raise RuntimeError(f"No LoRA params enabled for adapter='{adapter_name}' on layers={indices}.")
    print(f"[trainable] adapter={adapter_name} layers={indices} -> enabled params count {enabled}")


# ============================================================
# Dataset: QA SFT (chat_template 기반, answer-only labels)
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
    ds = load_dataset(qa_dataset, split=split)
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    def to_ex(ex):
        ctx = ex.get("context", "")
        q = ex.get("question", "")
        ans_list = ex.get("answers", {}).get("text", [])
        target = (ans_list[0] if ans_list else ("unanswerable" if qa_dataset == "squad_v2" else ""))

        messages = _build_chat_messages(ctx, q, qa_dataset, unans_token=unans_token)
        prompt_ids = _encode_chat_prompt_ids(tokenizer, messages)

        ans_text = (" " + target) if target else ""
        ans_ids = tokenizer(ans_text, add_special_tokens=False)["input_ids"]

        if add_eos and eos_id is not None:
            ans_ids = ans_ids + [eos_id]

        if len(ans_ids) < 1:
            return {"__drop__": 1}

        full_ids = prompt_ids + ans_ids
        prompt_len = len(prompt_ids)

        # 길면 왼쪽 잘라내서 최신 토큰 유지 (answer가 남도록)
        if len(full_ids) > seq_len:
            cut = len(full_ids) - seq_len
            full_ids = full_ids[cut:]
            prompt_len = max(0, prompt_len - cut)

        pad_len = seq_len - len(full_ids)

        # left padding 고정
        input_ids = ([pad_id] * pad_len) + full_ids
        attention_mask = ([0] * pad_len) + ([1] * len(full_ids))

        labels = input_ids.copy()

        # pad 마스킹
        for i in range(pad_len):
            labels[i] = -100

        # prompt 마스킹
        prompt_start = pad_len
        prompt_end = pad_len + prompt_len
        for i in range(prompt_start, min(prompt_end, seq_len)):
            labels[i] = -100

        # answer가 완전히 잘렸으면 drop
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
# KD Trainer (same LoRA setup, only loss differs)
# ============================================================

class KDTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, kd_alpha=0.1, temperature=2.0, clamp_logits=False, **kwargs):
        super().__init__(*args, **kwargs)
        if teacher_model is None:
            raise ValueError("KDTrainer requires teacher_model")
        self.teacher_model = teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        self.kd_alpha = float(kd_alpha)
        self.T = float(temperature)
        self.clamp_logits = bool(clamp_logits)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

        # student forward (logits only)
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        s_logits = student_outputs.logits  # [B,L,V]

        with torch.no_grad():
            t_inputs = {
                "input_ids": inputs["input_ids"].to(self.teacher_model.device),
                "attention_mask": inputs["attention_mask"].to(self.teacher_model.device),
            }
            t_logits = self.teacher_model(**t_inputs).logits.to(s_logits.device)

        # causal shift
        s_logits = s_logits[:, :-1, :].contiguous()
        t_logits = t_logits[:, :-1, :].contiguous()
        labels_s = labels[:, 1:].contiguous()
        attn_s   = inputs["attention_mask"][:, 1:].contiguous()

        mask = (labels_s != -100) & (attn_s == 1)
        tok_n = int(mask.sum().item())

        if tok_n == 0:
            if getattr(self.args, "logging_steps", 0) and (self.state.global_step % self.args.logging_steps == 0):
                self.log({"tok_n": 0.0, "skip_batch_no_supervised_tokens": 1.0})
            loss = s_logits.sum() * 0.0
            return (loss, student_outputs) if return_outputs else loss

        s = s_logits[mask]      # [N,V]
        t = t_logits[mask]      # [N,V]
        y = labels_s[mask]      # [N]

        # fp32 stabilize
        T = self.T
        s_fp32 = s.float()
        t_fp32 = t.float()

        if self.clamp_logits:
            s_fp32 = s_fp32.clamp(-50, 50)
            t_fp32 = t_fp32.clamp(-50, 50)

        # KD soft (log_target=True for stability)
        log_s = F.log_softmax(s_fp32 / T, dim=-1)
        log_t = F.log_softmax(t_fp32 / T, dim=-1)
        soft_loss = F.kl_div(log_s, log_t, reduction="batchmean", log_target=True) * (T * T)

        # hard CE (answer-only)
        hard_loss = F.cross_entropy(s_fp32, y)

        # NaN/Inf guard: KD가 터지면 hard만으로 fallback
        if (not torch.isfinite(soft_loss)) or (not torch.isfinite(hard_loss)):
            if getattr(self.args, "logging_steps", 0) and (self.state.global_step % self.args.logging_steps == 0):
                self.log({"tok_n": float(tok_n), "nan_guard_triggered": 1.0})
            loss = hard_loss
        else:
            a = self.kd_alpha
            loss = a * soft_loss + (1.0 - a) * hard_loss

        if getattr(self.args, "logging_steps", 0) and (self.state.global_step % self.args.logging_steps == 0):
            hard = float(hard_loss.detach().cpu())
            soft = float(soft_loss.detach().cpu()) if torch.isfinite(soft_loss) else float("nan")
            self.log({
                "tok_n": float(tok_n),
                "kd_soft": soft,
                "kd_hard": hard,
                "kd_ppl": float(torch.exp(torch.tensor(hard))),
            })

        return (loss, student_outputs) if return_outputs else loss


# ============================================================
# Adapter export (optional)
# ============================================================

def export_adapter_pt_and_recipe(model, out_dir, adapter_name, *, base_dir, bundles_dir, stage, trained_indices, tokenizer_dir=None):
    os.makedirs(out_dir, exist_ok=True)

    state = get_peft_model_state_dict(model, adapter_name=adapter_name)

    def _filter_state_dict_by_layers(state_dict, keep_layers: set[int]):
        out = {}
        for k, v in state_dict.items():
            m = _LAYER_RE.search(k)
            if m and int(m.group(1)) in keep_layers:
                out[k] = v
        return out

    keep = set(int(i) for i in trained_indices)
    slim_state = _filter_state_dict_by_layers(state, keep)
    if not slim_state:
        print(f"[warn] slim_state empty for adapter={adapter_name}. fallback to full adapter state.")
        slim_state = state

    pt_path = os.path.join(out_dir, f"{adapter_name}.pt")
    torch.save(slim_state, pt_path)

    raw_cfg = None
    try:
        if hasattr(model, "peft_config"):
            pc = model.peft_config
            if isinstance(pc, dict):
                raw_cfg = pc.get(adapter_name, None) or pc.get("default", None) or pc
            else:
                raw_cfg = pc
    except Exception:
        raw_cfg = None

    cfg = raw_cfg.to_dict() if hasattr(raw_cfg, "to_dict") else (raw_cfg if isinstance(raw_cfg, dict) else None)

    recipe = {
        "stage": stage,
        "adapter_name": adapter_name,
        "adapter_pt": os.path.abspath(pt_path),
        "base_dir": os.path.abspath(base_dir),
        "bundles_dir": os.path.abspath(bundles_dir) if bundles_dir else None,
        "tokenizer_dir": os.path.abspath(tokenizer_dir) if tokenizer_dir else base_dir,
        "trained_layer_indices": sorted(list(map(int, trained_indices))),
        "peft_config": cfg,
    }
    with open(os.path.join(out_dir, f"{adapter_name}_recipe.json"), "w", encoding="utf-8") as f:
        json.dump(recipe, f, ensure_ascii=False, indent=2)

    print(f"[export] adapter={adapter_name} -> {pt_path}")


# ============================================================
# Training wrapper (LoRA or KD-LoRA)
# ============================================================

def _make_training_args(out_dir: str, args):
    """
    transformers 버전차로 eval_strategy/evaluation_strategy 둘 다 대응.
    """
    common = dict(
        output_dir=out_dir,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,

        # logs
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,

        # 안정성
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        remove_unused_columns=False,
        report_to="none",  # tensorboard 의존성 제거
        save_total_limit=args.save_total_limit,
    )

    # precision
    if args.dtype == "bf16":
        common.update(dict(bf16=True, fp16=False))
    elif args.dtype == "fp16":
        common.update(dict(fp16=True, bf16=False))
    else:
        common.update(dict(fp16=False, bf16=False))

    # evaluation/save strategies
    # 최신: eval_strategy, 구버전: evaluation_strategy
    try:
        ta = TrainingArguments(
            **common,
            eval_strategy=("steps" if args.eval_steps > 0 else "no"),
            eval_steps=(args.eval_steps if args.eval_steps > 0 else None),
            save_strategy=("steps" if args.save_steps > 0 else "no"),
            save_steps=(args.save_steps if args.save_steps > 0 else None),
        )
        return ta
    except TypeError:
        ta = TrainingArguments(
            **common,
            evaluation_strategy=("steps" if args.eval_steps > 0 else "no"),
            eval_steps=(args.eval_steps if args.eval_steps > 0 else None),
            save_strategy=("steps" if args.save_steps > 0 else "no"),
            save_steps=(args.save_steps if args.save_steps > 0 else None),
        )
        return ta

def train_adapter(
    model,
    tokenizer,
    out_dir: str,
    train_ds,
    eval_ds,
    args,
    adapter_name: str,
    use_kd: bool = False,
    teacher_model=None,
):
    os.makedirs(out_dir, exist_ok=True)

    targs = _make_training_args(out_dir, args)

    trainer_cls = KDTrainer if (use_kd and teacher_model is not None) else Trainer

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
            clamp_logits=args.kd_clamp_logits,
        ))

    trainer = trainer_cls(**kwargs)
    trainer.train()

    # 최종 어댑터 저장
    if isinstance(model, PeftModel):
        try:
            model.save_pretrained(out_dir, selected_adapters=[adapter_name])
        except TypeError:
            model.save_pretrained(out_dir)
    else:
        print("[warn] model is not PeftModel; adapter save may be skipped")


# ============================================================
# Main
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_dir", type=str, required=True, help="프루닝된 A 모델 디렉토리")
    ap.add_argument("--bundles_dir", type=str, required=True, help="bundles/B, bundles/C 루트")
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True, help="1:A-LoRA, 2:AB-LoRA")
    ap.add_argument("--out_adapters", type=str, required=True)

    # dataset
    ap.add_argument("--qa_dataset", type=str, choices=["squad", "squad_v2"], default="squad")
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--max_eval_samples", type=int, default=8000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--unans_token", type=str, default="unanswerable")

    # train hparams
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad_acc", type=int, default=32)
    ap.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # logging/eval/save
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--eval_steps", type=int, default=200)      # 0이면 eval 안 함
    ap.add_argument("--save_steps", type=int, default=0)        # 0이면 저장 안 함 (adapter만 저장)
    ap.add_argument("--save_total_limit", type=int, default=2)

    # KD options
    ap.add_argument("--use_kd", action="store_true", help="KD-LoRA 사용")
    ap.add_argument("--teacher_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--teacher_device", type=str, default="cuda:0")
    ap.add_argument("--teacher_4bit", action="store_true")
    ap.add_argument("--kd_alpha", type=float, default=0.1)
    ap.add_argument("--kd_T", type=float, default=2.0)
    ap.add_argument("--kd_clamp_logits", action="store_true", help="KD 안정화용 logits clamp(-50,50)")

    # export pt+recipe (optional)
    ap.add_argument("--export_recipe", action="store_true")

    return ap.parse_args()


def load_teacher(args):
    if not args.use_kd:
        return None

    print(f"[KD] Loading teacher: {args.teacher_model} on {args.teacher_device} 4bit={args.teacher_4bit}")
    teacher = None

    # teacher dtype는 bf16로 두는 게 안정적 (4bit면 내부적으로 처리)
    try:
        if args.teacher_4bit:
            teacher = AutoModelForCausalLM.from_pretrained(
                args.teacher_model,
                device_map={"": args.teacher_device},
                load_in_4bit=True,
                local_files_only=False,
            )
        else:
            teacher = AutoModelForCausalLM.from_pretrained(
                args.teacher_model,
                torch_dtype=torch.bfloat16,
                device_map={"": args.teacher_device},
                local_files_only=False,
            )
    except Exception as e:
        print(f"[KD] teacher load fallback due to: {e}")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.bfloat16,
            device_map={"": args.teacher_device},
            local_files_only=False,
        )

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def main():
    args = parse_args()

    # tokenizer: base_dir 기준 (A 모델 tokenizer)
    tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # student model load (단일 GPU trainer 안정성 위해 device_map=None 권장)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_dir,
        torch_dtype=(torch.bfloat16 if args.dtype == "bf16" else (torch.float16 if args.dtype == "fp16" else torch.float32)),
        device_map=None,
        local_files_only=True,
    )
    device = torch.device(os.environ.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.config.use_cache = False

    # checkpointing 안정화
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:
        pass

    # keep 32 logical layers
    model = _reapply_passlayers_from_manifest(model, args.base_dir)

    # load prune_log.json for B/C indices
    with open(os.path.join(args.base_dir, "prune_log.json"), "r", encoding="utf-8") as f:
        log = json.load(f)
    B_idx, C_idx = log["split"]["B"], log["split"]["C"]

    # datasets
    train_ds = _load_qa_sft_dataset(
        tok,
        qa_dataset=args.qa_dataset,
        split="train",
        max_samples=args.max_samples,
        seq_len=args.seq_len,
        unans_token=args.unans_token,
    )
    eval_ds = _load_qa_sft_dataset(
        tok,
        qa_dataset=args.qa_dataset,
        split="validation",
        max_samples=args.max_eval_samples,
        seq_len=args.seq_len,
        unans_token=args.unans_token,
    )

    # teacher (optional)
    teacher = load_teacher(args)

    layers = _get_layer_container(model)
    L = len(layers)

    if args.stage == 1:
        removed = set(B_idx) | set(C_idx)
        all_idx = list(range(L))
        A_idx = [i for i in all_idx if i not in removed]

        model = _attach_new_adapter(
            model, "stageA",
            target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"),
            r=8, alpha=16, dropout=0.05
        )
        model.set_adapter("stageA")
        _enable_only_lora_on_indices_for_adapter_by_name(model, A_idx, "stageA", keep_layernorm=False)

        out_dir = os.path.join(args.out_adapters, "A_lora", "stageA")
        print(f"[Start] Stage1 {'KD-LoRA' if args.use_kd else 'LoRA'} -> {out_dir}")
        train_adapter(
            model, tok, out_dir,
            train_ds=train_ds, eval_ds=eval_ds,
            args=args, adapter_name="stageA",
            use_kd=args.use_kd, teacher_model=teacher,
        )

        if args.export_recipe:
            export_adapter_pt_and_recipe(
                model, os.path.join(args.out_adapters, "A_lora"),
                "stageA",
                base_dir=args.base_dir,
                bundles_dir=args.bundles_dir,
                stage="A",
                trained_indices=A_idx,
                tokenizer_dir=args.base_dir,
            )

    elif args.stage == 2:
        # AB = A∪B (C는 PassLayer)
        AB_idx = [i for i in range(L) if i not in set(C_idx)]

        # restore B layers from bundles
        _assert_bundle_files_exist(args.bundles_dir, "B", B_idx)
        _rehydrate_layers(model, os.path.join(args.bundles_dir, "B"), B_idx)

        # sanity: AB idx should be real decoder layers (not PassLayer)
        layers = _get_layer_container(model)
        bad = [i for i in AB_idx if not isinstance(layers[i], LlamaDecoderLayer)]
        if bad:
            raise RuntimeError(f"[check] AB indices not real LlamaDecoderLayer: {bad}")

        model = _attach_new_adapter(
            model, "stageAB",
            target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"),
            r=8, alpha=16, dropout=0.05
        )
        model.set_adapter("stageAB")
        _enable_only_lora_on_indices_for_adapter_by_name(model, AB_idx, "stageAB", keep_layernorm=False)

        out_dir = os.path.join(args.out_adapters, "AB_lora", "stageAB")
        print(f"[Start] Stage2 {'KD-LoRA' if args.use_kd else 'LoRA'} -> {out_dir}")
        train_adapter(
            model, tok, out_dir,
            train_ds=train_ds, eval_ds=eval_ds,
            args=args, adapter_name="stageAB",
            use_kd=args.use_kd, teacher_model=teacher,
        )

        if args.export_recipe:
            export_adapter_pt_and_recipe(
                model, os.path.join(args.out_adapters, "AB_lora"),
                "stageAB",
                base_dir=args.base_dir,
                bundles_dir=args.bundles_dir,
                stage="AB",
                trained_indices=AB_idx,
                tokenizer_dir=args.base_dir,
            )

    print("[Done]")


if __name__ == "__main__":
    main()
