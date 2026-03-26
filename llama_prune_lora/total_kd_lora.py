#!/usr/bin/env python3
"""
LLaMA 구조용 2-Stage KD-LoRA 학습 코드

Stage 1: base 로드 → B,C=PassLayer → A에만 LoRA
Stage 2: base 로드 → B 복원, C=PassLayer → A+B에 LoRA

사용법:
# Stage 1
CUDA_VISIBLE_DEVICES=2,5 DEVICE=cuda:0 \
python -m llama_prune_lora.total_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./llama_kd_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-4 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0

# Stage 2 (A+B LoRA)
CUDA_VISIBLE_DEVICES=6,1 DEVICE=cuda:0 \
python -m llama_prune_lora.total_kd_lora \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 2 \
  --out_adapters ./llama_kd_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-5 --epochs 1 --bs 1 --grad_acc 32 \
  --use_kd --teacher_model meta-llama/Llama-2-7b-chat-hf \
  --teacher_4bit --teacher_device cuda:1 \
  --kd_alpha 0.1 --kd_T 2.0
"""

import os, json, re, math, inspect
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
# 레이어 유틸
# ============================================================
CANON_PATH = "model.layers"


def _resolve_attr_path(root, dotted: str):
    parent = root
    for seg in dotted.split(".")[:-1]:
        parent = getattr(parent, seg)
    last = dotted.split(".")[-1]
    return parent, last, getattr(parent, last)


def _canonicalize_layers(model):
    for path in ["model.layers", "model.model.layers",
                  "base_model.model.layers", "base_model.model.model.layers"]:
        try:
            parent, name, cur = _resolve_attr_path(model, path)
            if hasattr(cur, "__len__"):
                if not isinstance(cur, nn.ModuleList):
                    cur = nn.ModuleList(list(cur))
                    setattr(parent, name, cur)
                model._canonical_layers_path = path
                model._canonical_layers = cur
                return cur
        except Exception:
            continue
    raise AttributeError("decoder layers not found")


def _get_layers(model):
    if not hasattr(model, "_canonical_layers"):
        _canonicalize_layers(model)
    return model._canonical_layers


def _invalidate_cache(model):
    for a in ("_canonical_layers", "_canonical_layers_path"):
        if hasattr(model, a):
            delattr(model, a)


def _layer_prefix(model, i: int):
    if not hasattr(model, "_canonical_layers_path"):
        _canonicalize_layers(model)
    return f"{model._canonical_layers_path}.{i}."


# ============================================================
# PassLayer
# ============================================================
class LlamaPassLayer(nn.Module):
    def __init__(self):
        super().__init__()
        sig = inspect.signature(LlamaDecoderLayer.forward)
        self._tuple = "output_attentions" in sig.parameters

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kw):
        if not self._tuple:
            return hidden_states
        out = (hidden_states,)
        if output_attentions: out += (None,)
        if use_cache: out += (None,)
        return out


# ============================================================
# 원본 레이아웃 보장 (sparse/compact 자동)
# ============================================================
def _ensure_original_layout(model, removed_indices, orig_N):
    layers = _get_layers(model)
    cur_N = len(layers)
    removed = set(int(i) for i in removed_indices)
    kept = sorted(set(range(orig_N)) - removed)
    dev = next(model.parameters()).device

    if cur_N == orig_N:  # sparse
        print(f"[layout] sparse: PassLayer at {sorted(removed)}")
        for i in removed:
            layers[i] = LlamaPassLayer().to(dev)
        return model, kept

    if cur_N != len(kept):  # compact 검증
        raise ValueError(f"레이어 수 불일치: {cur_N} (expected {len(kept)} or {orig_N})")

    # compact → expand
    print(f"[layout] compact: {cur_N} → {orig_N}")
    old = [layers[i] for i in range(cur_N)]
    new = [None] * orig_N
    for pi, oi in enumerate(kept):
        new[oi] = old[pi]
    for i in removed:
        new[i] = LlamaPassLayer().to(dev)

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model.model.layers = nn.ModuleList(new)
    elif hasattr(model, 'model') and hasattr(model.model, 'model'):
        model.model.model.layers = nn.ModuleList(new)
    else:
        raise RuntimeError("layers 경로 못 찾음")

    model.config.num_hidden_layers = orig_N
    _invalidate_cache(model)
    return model, kept


# ============================================================
# 번들 관리
# ============================================================
def _pick_layer_file(bdir, idx):
    for fmt in [f"layer_{int(idx):03d}.safetensors", f"layer_{int(idx)}.safetensors"]:
        p = os.path.join(bdir, fmt)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"layer {idx} missing in {bdir}")


def _load_bundle_indices(bdir):
    meta = os.path.join(bdir, "bundle_meta.json")
    if os.path.isfile(meta):
        return sorted(json.load(open(meta)).get("indices", []))
    return sorted(int(re.match(r"layer_(\d+)", f).group(1))
                  for f in os.listdir(bdir) if re.match(r"layer_\d+\.safetensors", f))


def _extract_layer_sd(raw, idx):
    for pref in [f"model.layers.{idx}.", f"model.model.layers.{idx}.", f"layers.{idx}."]:
        out = {k[len(pref):]: v for k, v in raw.items() if k.startswith(pref)}
        if out:
            return out
    return raw


def _rehydrate_layers(model, bdir, indices):
    layers = _get_layers(model)
    dtype = next(model.parameters()).dtype
    dev = next(model.parameters()).device
    for i in indices:
        try:
            nl = LlamaDecoderLayer(model.config, layer_idx=int(i))
        except TypeError:
            nl = LlamaDecoderLayer(model.config)
        nl = nl.to(device=dev, dtype=dtype)
        sd = _extract_layer_sd(load_file(_pick_layer_file(bdir, int(i))), int(i))
        sd = {k: v.to(device=dev, dtype=dtype) for k, v in sd.items()}
        try:
            nl.load_state_dict(sd, strict=True)
        except RuntimeError:
            nl.load_state_dict(sd, strict=False)
        layers[int(i)] = nl
        print(f"[rehydrate] layer {i}")


def _assert_bundles(bdir, indices):
    missing = [i for i in indices
               if not os.path.isfile(_pick_layer_file(bdir, int(i)))
               if True]  # _pick raises; catch below
    # 실제 체크
    bad = []
    for i in indices:
        try:
            _pick_layer_file(bdir, int(i))
        except FileNotFoundError:
            bad.append(i)
    if bad:
        raise FileNotFoundError(f"missing bundles: {bad} in {bdir}")
    print(f"[bundles-ok] {len(indices)} in {bdir}")


# ============================================================
# LoRA
# ============================================================
def _attach_lora(model, name, target_layers, r=8, alpha=16, dropout=0.05):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        layers_to_transform=target_layers,
    )
    if isinstance(model, PeftModel):
        if name not in getattr(model, "peft_config", {}):
            model.add_adapter(name, cfg)
        return model
    return get_peft_model(model, cfg, adapter_name=name)


def _enable_lora_only(model, indices, adapter_name):
    for p in model.parameters():
        p.requires_grad = False
    pats = [_layer_prefix(model, i) for i in indices]
    n = 0
    for name, p in model.named_parameters():
        if any(pat in name for pat in pats) and "lora_" in name.lower():
            p.requires_grad = True
            n += p.numel()
    if n == 0:
        raise RuntimeError(f"No LoRA params on {indices}")
    print(f"[trainable] {adapter_name}: {n:,} params on {len(indices)} layers")
    return n


# ============================================================
# 데이터셋
# ============================================================
def _load_qa_dataset(tok, qa_dataset, split, max_samples, seq_len):
    MAP = {"squad": "rajpurkar/squad", "squad_v2": "rajpurkar/squad_v2"}
    ds = load_dataset(MAP.get(qa_dataset, qa_dataset), split=split)
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    pad_id = tok.pad_token_id or tok.eos_token_id
    eos_id = tok.eos_token_id
    has_chat = hasattr(tok, "apply_chat_template") and tok.chat_template is not None

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
        ctx, q = ex.get("context",""), ex.get("question","")
        ans = (ex.get("answers",{}).get("text",[""])[0] or
               ("unanswerable" if qa_dataset=="squad_v2" else ""))
        sys_c = "You are a helpful QA assistant."
        if qa_dataset == "squad_v2":
            sys_c += " If the answer is not in the context, say 'unanswerable'."
        msgs = [{"role":"system","content":sys_c},
                {"role":"user","content":f"Answer the question using the context.\n\nContext:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:"}]

        EMPTY = {"input_ids":[pad_id]*seq_len, "attention_mask":[0]*seq_len,
                "labels":[-100]*seq_len, "__drop__":1}

        if has_chat:
            pid = to_list(tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True))
        else:
            prompt = f"<s>[INST] <<SYS>>\n{sys_c}\n<</SYS>>\n\n{msgs[1]['content']} [/INST] "
            pid = to_list(tok(prompt, add_special_tokens=False,
                              truncation=True, max_length=seq_len-64))

        aid = to_list(tok(" "+ans, add_special_tokens=False))
        if eos_id: aid += [eos_id]
        if not aid: return EMPTY

        full = pid + aid
        plen = len(pid)
        if len(full) > seq_len:
            cut = len(full)-seq_len; full = full[cut:]; plen = max(0, plen-cut)
        padN = seq_len - len(full)
        ids = [pad_id]*padN + full
        attn = [0]*padN + [1]*len(full)
        labels = ids[:]
        for i in range(padN+plen):
            if i < len(labels): labels[i] = -100
        if padN+plen >= seq_len: return EMPTY
        return {"input_ids":ids, "attention_mask":attn, "labels":labels, "__drop__":0}

    ds = ds.map(process, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["__drop__"]==0).remove_columns("__drop__")
    return ds


# ============================================================
# KD Trainer
# ============================================================
class KDTrainer(Trainer):
    def __init__(self, *a, teacher_model=None, kd_alpha=0.1, temperature=2.0, **kw):
        super().__init__(*a, **kw)
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters(): p.requires_grad = False
        self.alpha = float(kd_alpha)
        self.T = float(temperature)

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs["labels"]
        s_out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        sl = s_out.logits

        with torch.no_grad():
            tl = self.teacher(
                input_ids=inputs["input_ids"].to(self.teacher.device),
                attention_mask=inputs["attention_mask"].to(self.teacher.device)
            ).logits.to(sl.device)

        s, t = sl[:,:-1,:].contiguous(), tl[:,:-1,:].contiguous()
        lab = labels[:,1:].contiguous()
        mask = (lab != -100) & (inputs["attention_mask"][:,1:].contiguous() == 1)
        n = mask.sum().item()
        if n == 0:
            z = sl.sum()*0.0
            return (z, s_out) if return_outputs else z

        sm, tm, ym = s[mask].float().clamp(-50,50), t[mask].float().clamp(-50,50), lab[mask]
        T = self.T
        soft = F.kl_div(F.log_softmax(sm/T,-1), F.log_softmax(tm/T,-1),
                        reduction="batchmean", log_target=True) * T*T
        hard = F.cross_entropy(sm, ym)
        loss = (self.alpha*soft + (1-self.alpha)*hard
                if torch.isfinite(soft) and torch.isfinite(hard)
                else hard if torch.isfinite(hard) else sl.sum()*0.0)

        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            self.log({"tok_n":float(n), "kd_soft":float(soft), "kd_hard":float(hard),
                      "kd_ppl":float(math.exp(min(hard.item(),20)))})
        return (loss, s_out) if return_outputs else loss


# ============================================================
# 학습 헬퍼
# ============================================================
def _train(model, out_dir, train_ds, eval_ds, args, name, teacher=None):
    os.makedirs(out_dir, exist_ok=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[train] {name}: {trainable:,} params → {out_dir}")
    if trainable == 0: raise RuntimeError("No trainable params!")

    common = dict(
        output_dir=out_dir, per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_acc, learning_rate=args.lr,
        num_train_epochs=args.epochs, logging_strategy="steps",
        logging_steps=args.logging_steps, logging_first_step=True,
        max_grad_norm=args.max_grad_norm, warmup_ratio=args.warmup_ratio,
        remove_unused_columns=False, report_to="none",
        save_total_limit=args.save_total_limit)
    if args.dtype == "bf16": common.update(bf16=True, fp16=False)
    elif args.dtype == "fp16": common.update(fp16=True, bf16=False)

    try:
        ta = TrainingArguments(**common,
            eval_strategy="steps" if args.eval_steps>0 else "no",
            eval_steps=args.eval_steps if args.eval_steps>0 else None,
            save_strategy="steps" if args.save_steps>0 else "no",
            save_steps=args.save_steps if args.save_steps>0 else None)
    except TypeError:
        ta = TrainingArguments(**common,
            evaluation_strategy="steps" if args.eval_steps>0 else "no",
            eval_steps=args.eval_steps if args.eval_steps>0 else None,
            save_strategy="steps" if args.save_steps>0 else "no",
            save_steps=args.save_steps if args.save_steps>0 else None)

    cls = KDTrainer if (args.use_kd and teacher) else Trainer
    kw = dict(model=model, args=ta, train_dataset=train_ds,
              eval_dataset=eval_ds, data_collator=default_data_collator)
    if cls is KDTrainer:
        kw.update(teacher_model=teacher, kd_alpha=args.kd_alpha, temperature=args.kd_T)
    cls(**kw).train()

    if isinstance(model, PeftModel):
        try: model.save_pretrained(out_dir, selected_adapters=[name])
        except TypeError: model.save_pretrained(out_dir)


def _load_teacher(args):
    if not args.use_kd: return None
    print(f"[Teacher] {args.teacher_model} on {args.teacher_device}")
    if args.teacher_4bit:
        from transformers import BitsAndBytesConfig
        t = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={"":args.teacher_device})
    else:
        t = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, torch_dtype=torch.bfloat16,
            device_map={"":args.teacher_device})
    t.eval()
    for p in t.parameters(): p.requires_grad = False
    return t


# ============================================================
# 인덱스 로드
# ============================================================
def _load_index_info(base_dir, bundles_dir):
    info = {"B":[], "C":[], "L_full":None}
    for path in [os.path.join(base_dir,"manifest.json")]:
        if os.path.isfile(path):
            m = json.load(open(path))
            info["L_full"] = m.get("counts",{}).get("L_full")
            st = m.get("stages",{})
            info["B"] = sorted(int(x) for x in st.get("B",{}).get("removed_layers",[]))
            info["C"] = sorted(int(x) for x in st.get("C",{}).get("removed_layers",[]))

    log = os.path.join(base_dir, "prune_log.json")
    if os.path.isfile(log):
        lg = json.load(open(log))
        if not info["B"]: info["B"] = sorted(lg.get("split",{}).get("B",[]))
        if not info["C"]: info["C"] = sorted(lg.get("split",{}).get("C",[]))

    if not info["B"]: info["B"] = _load_bundle_indices(os.path.join(bundles_dir,"B"))
    if not info["C"]: info["C"] = _load_bundle_indices(os.path.join(bundles_dir,"C"))
    return info


# ============================================================
# Main
# ============================================================
def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--bundles_dir", required=True)
    p.add_argument("--stage", type=int, choices=[1,2], required=True)
    p.add_argument("--out_adapters", required=True)
    p.add_argument("--original_num_layers", type=int, default=None)

    p.add_argument("--qa_dataset", default="squad")
    p.add_argument("--max_samples", type=int, default=20000)
    p.add_argument("--max_eval_samples", type=int, default=8000)
    p.add_argument("--seq_len", type=int, default=1024)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad_acc", type=int, default=32)
    p.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
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

    tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True, local_files_only=True)
    if not tok.pad_token: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # ── Student ──
    print(f"\n[Load] Student: {args.base_dir}")
    dtypes = {"bf16":torch.bfloat16, "fp16":torch.float16, "fp32":torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=dtypes[args.dtype],
        device_map=None, local_files_only=True)
    dev = torch.device(os.environ.get("DEVICE","cuda:0") if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.config.use_cache = False
    try: model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    except: pass

    # ── 인덱스 ──
    info = _load_index_info(args.base_dir, args.bundles_dir)
    B_idx, C_idx = info["B"], info["C"]
    orig_N = args.original_num_layers or info["L_full"] or model.config.num_hidden_layers
    removed_all = sorted(set(B_idx + C_idx))
    A_idx = sorted(set(range(orig_N)) - set(removed_all))

    print(f"\n[Index] orig={orig_N}, A={len(A_idx)}, B={len(B_idx)}, C={len(C_idx)}")
    print(f"  A: {A_idx[:5]}{'...' if len(A_idx)>5 else ''}")
    print(f"  B: {B_idx}")
    print(f"  C: {C_idx}")

    # ── Datasets ──
    print("\n[Load] Datasets")
    train_ds = _load_qa_dataset(tok, args.qa_dataset, "train", args.max_samples, args.seq_len)
    eval_ds = _load_qa_dataset(tok, args.qa_dataset, "validation", args.max_eval_samples, args.seq_len)
    teacher = _load_teacher(args)

    # ================================================================
    # Stage 1: A만 LoRA (B+C = PassLayer)
    # ================================================================
    if args.stage == 1:
        print("\n" + "="*60)
        print("STAGE 1: A-LoRA  (B+C = PassLayer)")
        print("="*60)

        model, _ = _ensure_original_layout(model, removed_all, orig_N)
        layers = _get_layers(model)

        bad_a = [i for i in A_idx if not isinstance(layers[i], LlamaDecoderLayer)]
        if bad_a: raise RuntimeError(f"A 위치 비정상: {bad_a}")
        bad_bc = [i for i in (B_idx+C_idx) if not isinstance(layers[i], LlamaPassLayer)]
        if bad_bc: raise RuntimeError(f"B+C가 PassLayer가 아님: {bad_bc}")

        print(f"\n[Layer Verify] 총 {orig_N}층")
        print(f"  A (LlamaDecoderLayer): {len(A_idx)}개 ✓")
        print(f"  B+C (PassLayer):       {len(B_idx)+len(C_idx)}개 ✓")

        model = _attach_lora(model, "stageA", target_layers=A_idx)
        model.set_adapter("stageA")
        _enable_lora_only(model, A_idx, "stageA")

        out = os.path.join(args.out_adapters, "A_lora", "stageA")
        _train(model, out, train_ds, eval_ds, args, "stageA", teacher)

    # ================================================================
    # Stage 2: A+B LoRA (C = PassLayer)
    # ================================================================
    elif args.stage == 2:
        print("\n" + "="*60)
        print("STAGE 2: AB-LoRA  (A+B real, C = PassLayer)")
        print("="*60)

        # 먼저 B+C 전부 PassLayer로 세팅
        model, _ = _ensure_original_layout(model, removed_all, orig_N)
        layers = _get_layers(model)

        # B 번들 복원 → PassLayer → LlamaDecoderLayer
        B_bdir = os.path.join(args.bundles_dir, "B")
        _assert_bundles(B_bdir, B_idx)
        _rehydrate_layers(model, B_bdir, B_idx)

        # ── 레이어 상태 검증 ──
        bad_b = [i for i in B_idx if not isinstance(layers[i], LlamaDecoderLayer)]
        if bad_b: raise RuntimeError(f"B 복원 실패: {bad_b}")
        bad_a = [i for i in A_idx if not isinstance(layers[i], LlamaDecoderLayer)]
        if bad_a: raise RuntimeError(f"A 위치 비정상: {bad_a}")
        bad_c = [i for i in C_idx if not isinstance(layers[i], LlamaPassLayer)]
        if bad_c: raise RuntimeError(f"C가 PassLayer가 아님: {bad_c}")

        print(f"\n[Layer Verify] 총 {orig_N}층")
        print(f"  A (LlamaDecoderLayer): {A_idx[:5]}{'...' if len(A_idx)>5 else ''} ({len(A_idx)}개) ✓")
        print(f"  B (LlamaDecoderLayer): {B_idx} ({len(B_idx)}개) ✓")
        print(f"  C (PassLayer):         {C_idx} ({len(C_idx)}개) ✓")

        # A+B 전체에 LoRA
        AB_idx = sorted(A_idx + B_idx)
        model = _attach_lora(model, "stageAB", target_layers=AB_idx)
        model.set_adapter("stageAB")
        _enable_lora_only(model, AB_idx, "stageAB")

        out = os.path.join(args.out_adapters, "AB_lora", "stageAB")
        _train(model, out, train_ds, eval_ds, args, "stageAB", teacher)

    print("\n[Done]")


if __name__ == "__main__":
    main()