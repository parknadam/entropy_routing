#!/usr/bin/env python3
"""
Progressive 3-Stage LoRA Training (skeleton-preserving, no KD)

Stage 1: A 로드 → 원본 레이아웃 보장 → B,C=PassLayer → A에만 LoRA
Stage 2: A_merged 로드 → 원본 레이아웃 보장 → B 복원 + C=PassLayer → B에만 LoRA
Stage 3: A_merged 로드 → 원본 레이아웃 보장 → B_merged+C 복원 → C에만 LoRA

Usage:
# Stage 1
<<<<<<< HEAD
DEVICE=cuda:0 python progressive_lora.py \
=======
CUDA_VISIBLE_DEVICES=4 DEVICE=cuda:0 \
python -m llama_prune_lora.optimized_lora \
>>>>>>> ea86042 (수정)
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 --out_adapters ./lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
<<<<<<< HEAD
  --seq_len 1024 --lr 3e-4 --epochs 1 --bs 1 --grad_acc 32

# Stage 2 (A_merged 기준)
DEVICE=cuda:0 python progressive_lora.py \
  --base_dir ./merged_models/A_merged \
=======
  --seq_len 1024 --lr 3e-4 --epochs 2 --bs 1 --grad_acc 32

# Stage 2 (A_merged 기준)
CUDA_VISIBLE_DEVICES=4 DEVICE=cuda:0 \
python -m llama_prune_lora.optimized_lora \
  --base_dir ./new_merged_models_llama_7b_lora/A_merged \
>>>>>>> ea86042 (수정)
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 2 --out_adapters ./lora_results/adapters \
  --seq_len 1024 --lr 3e-5 --epochs 1 --bs 1 --grad_acc 32

# Stage 3 (A_merged + B_merged)
DEVICE=cuda:0 python progressive_lora.py \
<<<<<<< HEAD
  --base_dir ./merged_models/A_merged \
  --b_merged_dir ./merged_models/B_merged \
=======
  --base_dir ./new_merged_models_llama_7b_lora/A_merged \
  --b_merged_dir ./new_merged_models_llama_7b_lora/B_merged \
>>>>>>> ea86042 (수정)
  --bundles_dir ./7b_results/pruning/bundles/C \
  --stage 3 --out_adapters ./lora_results/adapters \
  --seq_len 1024 --lr 3e-5 --epochs 1 --bs 1 --grad_acc 32
"""

import os, json, re, inspect, argparse
from typing import List
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, default_data_collator,
)
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# ============================================================
# Layer utilities
# ============================================================
CANON_PATH = "model.layers"

def _resolve(root, dotted):
    parent = root
    for seg in dotted.split(".")[:-1]:
        parent = getattr(parent, seg)
    last = dotted.split(".")[-1]
    return parent, last, getattr(parent, last)

def _canonicalize(model):
    for path in ["model.layers", "model.model.layers",
                  "base_model.model.layers", "base_model.model.model.layers"]:
        try:
            parent, name, cur = _resolve(model, path)
            if hasattr(cur, "__len__"):
                if not isinstance(cur, (list, nn.ModuleList)):
                    cur = nn.ModuleList(list(cur)); setattr(parent, name, cur)
                try:
                    cp, _, _ = _resolve(model, CANON_PATH.replace(".layers", ""))
                    setattr(cp, "layers", cur)
                    model._clp = CANON_PATH
                except: model._clp = path
                model._cl = cur
                return cur
        except: continue
    raise AttributeError("decoder layers not found")

def _layers(model):
    if not hasattr(model, "_cl"): _canonicalize(model)
    return model._cl

def _invalidate(model):
    for a in ("_cl", "_clp"): 
        if hasattr(model, a): delattr(model, a)

def _prefix(model, i):
    if not hasattr(model, "_clp"): _canonicalize(model)
    return f"{model._clp}.{i}."

# ============================================================
# PassLayer
# ============================================================
def _returns_tuple():
    try: return "output_attentions" in inspect.signature(LlamaDecoderLayer.forward).parameters
    except: return True

class PassLayer(nn.Module):
    def __init__(self, ret_tuple=True):
        super().__init__()
        self.ret_tuple = ret_tuple
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kw):
        if not self.ret_tuple: return hidden_states
        out = (hidden_states,)
        if output_attentions: out += (None,)
        if use_cache: out += (None,)
        return out

# ============================================================
# Layout: sparse/compact → original skeleton
# ============================================================
def _ensure_original_layout(model, removed_indices, original_N):
    """모델을 원본 레이어 수로 맞추고 removed 위치에 PassLayer 설치.
    sparse(이미 full size)와 compact(축소됨) 양쪽 대응."""
    layers = _layers(model)
    cur_N = len(layers)
    removed = set(int(i) for i in removed_indices)
    kept = sorted(set(range(original_N)) - removed)
    use_t = _returns_tuple()
    dev = next(model.parameters()).device

    # sparse: 이미 원본 크기
    if cur_N == original_N:
        print(f"[layout] sparse: {cur_N}L, PassLayer at {sorted(removed_indices)}")
        for i in removed_indices:
            layers[int(i)] = PassLayer(use_t).to(dev)
        return model, kept

    # compact: 축소된 상태 → 확장
    if cur_N != len(kept):
        raise ValueError(f"layer mismatch: {cur_N} vs expected compact={len(kept)} or sparse={original_N}")

    print(f"[layout] compact: {cur_N}L → {original_N}L expand")
    old = [layers[i] for i in range(cur_N)]
    new = [None] * original_N
    for pi, oi in enumerate(kept): new[oi] = old[pi]
    for i in removed_indices: new[int(i)] = PassLayer(use_t).to(dev)
    assert all(l is not None for l in new)

    # ModuleList 교체
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model.model.layers = nn.ModuleList(new)
    elif hasattr(model, 'model') and hasattr(model.model, 'model'):
        model.model.model.layers = nn.ModuleList(new)
    else:
        raise RuntimeError("cannot find layers path")
    model.config.num_hidden_layers = original_N
    _invalidate(model)
    print(f"  real: {kept[:5]}{'...' if len(kept)>5 else ''} ({len(kept)}), pass: {sorted(removed_indices)} ({len(removed_indices)})")
    return model, kept

# ============================================================
# Bundle management
# ============================================================
def _pick_file(bdir, idx):
    for fmt in [f"layer_{int(idx):03d}.safetensors", f"layer_{int(idx)}.safetensors"]:
        p = os.path.join(bdir, fmt)
        if os.path.isfile(p): return p
    raise FileNotFoundError(f"layer file missing: idx={idx} in {bdir}")

def _extract_sd(raw, idx):
    for pref in [f"model.layers.{idx}.", f"model.model.layers.{idx}.", f"layers.{idx}."]:
        out = {k[len(pref):]: v for k, v in raw.items() if k.startswith(pref)}
        if out: return out
    return raw

def _load_bundle_indices(bdir):
    meta = os.path.join(bdir, "bundle_meta.json")
    if os.path.isfile(meta):
        with open(meta) as f: return sorted(json.load(f).get("indices", []))
    return sorted(int(re.match(r"layer_(\d+)", fn).group(1))
                  for fn in os.listdir(bdir) if re.match(r"layer_\d+\.safetensors", fn))

def _assert_bundles(bdir, indices):
    missing = []
    for i in indices:
        try:
            f = _pick_file(bdir, i)
            if os.path.getsize(f) == 0: missing.append(i)
        except FileNotFoundError: missing.append(i)
    if missing: raise FileNotFoundError(f"[bundles] missing/empty: {missing} in {bdir}")
    print(f"[bundles-ok] {len(indices)} files in {bdir}")

def _rehydrate(model, bdir, indices):
    """PassLayer → LlamaDecoderLayer 복원"""
    layers = _layers(model)
    dtype, dev = next(model.parameters()).dtype, next(model.parameters()).device
    for i in indices:
        try: nl = LlamaDecoderLayer(model.config, layer_idx=int(i))
        except TypeError: nl = LlamaDecoderLayer(model.config)
        nl = nl.to(device=dev, dtype=dtype)
        raw = load_file(_pick_file(bdir, int(i)))
        sd = {k: v.to(device=dev, dtype=dtype) for k, v in _extract_sd(raw, int(i)).items()}
        try: nl.load_state_dict(sd, strict=True)
        except RuntimeError: nl.load_state_dict(sd, strict=False)
        layers[int(i)] = nl
        print(f"[rehydrate] layer {i} restored")

# ============================================================
# LoRA adapter
# ============================================================
def _attach(model, name, target_layers=None, r=8, alpha=16, dropout=0.05):
    kw = dict(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
              task_type="CAUSAL_LM",
              target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    if target_layers is not None:
        kw["layers_to_transform"] = target_layers
    cfg = LoraConfig(**kw)
    if isinstance(model, PeftModel):
        if name not in getattr(model, "peft_config", {}):
            model.add_adapter(name, cfg)
        return model
    return get_peft_model(model, cfg, adapter_name=name)

def _enable_lora_only(model, indices, adapter_name):
    for p in model.parameters(): p.requires_grad = False
    pats = [_prefix(model, i) for i in indices]
    n_en = 0
    for name, p in model.named_parameters():
        if any(pt in name for pt in pats) and "lora_" in name.lower():
            p.requires_grad = True; n_en += p.numel()
    if n_en == 0:
        raise RuntimeError(f"No LoRA params on layers {indices} for '{adapter_name}'")
    print(f"[trainable] {adapter_name}: {n_en:,} params on {len(indices)} layers")

# ============================================================
# Dataset
# ============================================================
def _build_msgs(ctx, q, ds_name):
    sys = "You are a helpful QA assistant."
    if ds_name == "squad_v2": sys += " If the answer is not in the context, say 'unanswerable'."
    return [{"role":"system","content":sys},
            {"role":"user","content":f"Answer the question using the context.\n\nContext:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:"}]

def _load_qa_dataset(tok, ds_name, split, max_samples, seq_len):
    DS_MAP = {"squad":"rajpurkar/squad","squad_v2":"rajpurkar/squad_v2"}
    ds = load_dataset(DS_MAP.get(ds_name, ds_name), split=split)
    if max_samples: ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    pad_id = tok.pad_token_id or tok.eos_token_id
    eos_id = tok.eos_token_id
    has_chat = hasattr(tok, "apply_chat_template") and tok.chat_template is not None

    def _to_list(x):
        if hasattr(x, "input_ids"): x = x.input_ids
        elif isinstance(x, dict): x = x.get("input_ids", x)
        if hasattr(x, "tolist"): x = x.tolist()
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)): x = x[0]
        return list(x) if x else []

    def proc(ex):
        ctx, q = ex.get("context",""), ex.get("question","")
        ans = ex.get("answers",{}).get("text",[""])[0] or ("unanswerable" if ds_name=="squad_v2" else "")
        msgs = _build_msgs(ctx, q, ds_name)

        if has_chat:
            p_ids = _to_list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        else:
            s, u = msgs[0]["content"], msgs[1]["content"]
            p_ids = _to_list(tok(f"<s>[INST] <<SYS>>\n{s}\n<</SYS>>\n\n{u} [/INST] ",
                                 add_special_tokens=False, truncation=True, max_length=seq_len-64)["input_ids"])

        a_ids = _to_list(tok(" "+ans, add_special_tokens=False)["input_ids"])
        if eos_id: a_ids += [eos_id]
        if not a_ids: return {"__drop__":1}

        full = p_ids + a_ids; plen = len(p_ids)
        if len(full) > seq_len:
            cut = len(full)-seq_len; full = full[cut:]; plen = max(0, plen-cut)
        pad_n = seq_len - len(full)
        ids = [pad_id]*pad_n + full
        mask = [0]*pad_n + [1]*len(full)
        labels = ids[:]
        for i in range(pad_n+plen):
            if i < len(labels): labels[i] = -100
        if pad_n+plen >= seq_len: return {"__drop__":1}
        return {"input_ids":ids,"attention_mask":mask,"labels":labels,"__drop__":0}

    ds = ds.map(proc, remove_columns=ds.column_names, num_proc=4)
    ds = ds.filter(lambda x: x["__drop__"]==0)
    if "__drop__" in ds.column_names: ds = ds.remove_columns("__drop__")
    return ds

# ============================================================
# Index info loader
# ============================================================
def _load_index_info(base_dir, bundles_dir, stage, b_merged_dir=None):
    info = {"B":[], "C":[], "L_full":None}
    for p in [os.path.join(base_dir, "manifest.json")]:
        if os.path.isfile(p):
            m = json.load(open(p))
            info["L_full"] = m.get("counts",{}).get("L_full")
            st = m.get("stages",{})
            info["B"] = sorted(int(x) for x in st.get("B",{}).get("removed_layers",[]))
            info["C"] = sorted(int(x) for x in st.get("C",{}).get("removed_layers",[]))
    log_p = os.path.join(base_dir, "prune_log.json")
    if os.path.isfile(log_p):
        log = json.load(open(log_p))
        if not info["B"]: info["B"] = sorted(log.get("split",{}).get("B",[]))
        if not info["C"]: info["C"] = sorted(log.get("split",{}).get("C",[]))
    if not info["B"] and stage < 3:
        info["B"] = _load_bundle_indices(os.path.join(bundles_dir, "B"))
    if not info["C"]:
        c_dir = bundles_dir if stage == 3 else os.path.join(bundles_dir, "C")
        info["C"] = _load_bundle_indices(c_dir)
    return info

# ============================================================
# Training
# ============================================================
def train_adapter(model, tok, out_dir, train_ds, eval_ds, args, adapter_name):
    os.makedirs(out_dir, exist_ok=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[train] {adapter_name}: {n_train:,} trainable → {out_dir}")
    if n_train == 0: raise RuntimeError("No trainable params!")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    common = dict(
        output_dir=out_dir,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=use_bf16, fp16=not use_bf16,
        dataloader_num_workers=4, dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps, logging_first_step=True,
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=args.save_total_limit,
    )
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

    Trainer(model=model, args=ta, train_dataset=train_ds, eval_dataset=eval_ds,
<<<<<<< HEAD
            data_collator=default_data_collator, tokenizer=tok).train()
=======
            data_collator=default_data_collator, processing_class=tok).train()
>>>>>>> ea86042 (수정)

    if isinstance(model, PeftModel):
        try: model.save_pretrained(out_dir, selected_adapters=[adapter_name])
        except TypeError: model.save_pretrained(out_dir)

# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--bundles_dir", required=True)
    p.add_argument("--b_merged_dir", default=None, help="Stage 3: B_merged bundle dir")
    p.add_argument("--stage", type=int, choices=[1,2,3], required=True)
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
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--save_total_limit", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True, local_files_only=True)
    if not tok.pad_token: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Model
    device = torch.device(os.environ.get("DEVICE","cuda:0") if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=dtype, device_map=None, local_files_only=True)
    model.to(device)
    model.config.use_cache = False
    try: model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    except: pass

    loaded_L = len(_layers(model))

    # Index info
    info = _load_index_info(args.base_dir, args.bundles_dir, args.stage, args.b_merged_dir)
    B_idx, C_idx = info["B"], info["C"]
    original_N = args.original_num_layers or info["L_full"] or model.config.num_hidden_layers
    removed_all = sorted(set(B_idx + C_idx))
    A_idx = sorted(set(range(original_N)) - set(removed_all))

    print(f"\n[Index] original={original_N}, loaded={loaded_L}")
    print(f"  A({len(A_idx)}): {A_idx[:5]}{'...' if len(A_idx)>5 else ''}")
    print(f"  B({len(B_idx)}): {B_idx}")
    print(f"  C({len(C_idx)}): {C_idx}")

    # Ensure original layout
    model, kept = _ensure_original_layout(model, removed_all, original_N)
    layers = _layers(model)

    # Datasets
    print("\n[Loading] Datasets")
    train_ds = _load_qa_dataset(tok, args.qa_dataset, "train", args.max_samples, args.seq_len)
    eval_ds = _load_qa_dataset(tok, args.qa_dataset, "validation", args.max_eval_samples, args.seq_len)
    print(f"  train={len(train_ds)}, eval={len(eval_ds)}")

    # ================================================================
    # Stage 1: A만 LoRA (B,C = PassLayer)
    # ================================================================
    if args.stage == 1:
        print(f"\n{'='*60}\nSTAGE 1: A-LoRA (A=real, B+C=PassLayer)\n{'='*60}")

        bad = [i for i in A_idx if not isinstance(layers[i], LlamaDecoderLayer)]
        if bad: raise RuntimeError(f"A 위치 비정상: {bad}")

        model = _attach(model, "stageA", target_layers=A_idx)
        model.set_adapter("stageA")
        _enable_lora_only(model, A_idx, "stageA")

        out = os.path.join(args.out_adapters, "A_lora", "stageA")
        train_adapter(model, tok, out, train_ds, eval_ds, args, "stageA")

        print(f"\n[Next] Merge: merge_adapter.py --base_model {args.base_dir} "
              f"--adapter_path {out} --output_dir ./merged_models/A_merged")

    # ================================================================
    # Stage 2: B만 LoRA (A=merged, B=복원, C=PassLayer)
    # ================================================================
    elif args.stage == 2:
        print(f"\n{'='*60}\nSTAGE 2: B-LoRA (A=merged, B=restored, C=PassLayer)\n{'='*60}")

        B_bdir = os.path.join(args.bundles_dir, "B")
        _assert_bundles(B_bdir, B_idx)
        _rehydrate(model, B_bdir, B_idx)

        bad = [i for i in B_idx if not isinstance(layers[i], LlamaDecoderLayer)]
        if bad: raise RuntimeError(f"B 복원 실패: {bad}")

        model = _attach(model, "stageB", target_layers=B_idx)
        model.set_adapter("stageB")
        _enable_lora_only(model, B_idx, "stageB")

        out = os.path.join(args.out_adapters, "B_lora", "stageB")
        train_adapter(model, tok, out, train_ds, eval_ds, args, "stageB")

        print(f"\n[Next] Merge B adapter with B bundle → B_merged")

    # ================================================================
    # Stage 3: C만 LoRA (A=merged, B=merged, C=복원)
    # ================================================================
    elif args.stage == 3:
        print(f"\n{'='*60}\nSTAGE 3: C-LoRA (A=merged, B=merged, C=restored)\n{'='*60}")

        if not args.b_merged_dir:
            raise ValueError("Stage 3 requires --b_merged_dir")

        # B_merged 복원
        bm_indices = _load_bundle_indices(args.b_merged_dir) or B_idx
        _assert_bundles(args.b_merged_dir, bm_indices)
        _rehydrate(model, args.b_merged_dir, bm_indices)

        # C 복원
        C_bdir = args.bundles_dir
        _assert_bundles(C_bdir, C_idx)
        _rehydrate(model, C_bdir, C_idx)

        bad = [i for i in C_idx if not isinstance(layers[i], LlamaDecoderLayer)]
        if bad: raise RuntimeError(f"C 복원 실패: {bad}")

        model = _attach(model, "stageC", target_layers=C_idx)
        model.set_adapter("stageC")
        _enable_lora_only(model, C_idx, "stageC")

        out = os.path.join(args.out_adapters, "C_lora", "stageC")
        train_adapter(model, tok, out, train_ds, eval_ds, args, "stageC")

        print(f"\n[Next] Merge C adapter with C bundle → C_merged")

    print("\n[Done] Training completed")


if __name__ == "__main__":
    main()