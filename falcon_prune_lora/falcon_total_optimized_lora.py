#!/usr/bin/env python3
"""
Falcon Progressive 2-Stage LoRA Training (skeleton-preserving, no KD)

Stage 1: A 로드 → 원본 레이아웃 보장 → B,C=PassLayer → A에만 LoRA
Stage 2: A 로드 → B 복원 + C=PassLayer → A,B 전체에 LoRA

Usage:
#7b
# Stage 1
CUDA_VISIBLE_DEVICES=2 DEVICE=cuda:0 \
python -m falcon_prune_lora.falcon_total_optimized_lora \
  --base_dir ./falcon_results/pruning/A \
  --bundles_dir ./falcon_results/pruning/bundles \
  --stage 1 --out_adapters ./falcon_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-4 --epochs 2 --bs 1 --grad_acc 32

# Stage 2
CUDA_VISIBLE_DEVICES=5 DEVICE=cuda:0 \
python -m falcon_prune_lora.falcon_total_optimized_lora \
  --base_dir ./falcon_results/pruning/A \
  --bundles_dir ./falcon_results/pruning/bundles \
  --stage 2 --out_adapters ./falcon_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-5 --epochs 2 --bs 1 --grad_acc 32
"""

import os, sys, json, re, argparse
from datetime import datetime
from typing import List
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, default_data_collator,
)
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file

try:
    from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
except Exception:
    FalconDecoderLayer = None

# ============================================================
# Falcon 레이어 유틸리티
# ============================================================
def _get_layers(model):
    for path in [
        lambda m: m.transformer.h,
        lambda m: m.base_model.model.transformer.h,
        lambda m: m.model.transformer.h,
    ]:
        try:
            layers = path(model)
            if hasattr(layers, "__len__"): return layers
        except (AttributeError, TypeError): continue
    raise AttributeError("Falcon decoder layers not found")


def _set_layers(model, new_layers):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        model.transformer.h = nn.ModuleList(new_layers)
    elif hasattr(model, "model") and hasattr(model.model, "transformer"):
        model.model.transformer.h = nn.ModuleList(new_layers)
    else:
        raise RuntimeError("cannot find transformer.h path")


def _layer_prefix(model, i):
    if isinstance(model, PeftModel):
        return f"base_model.model.transformer.h.{i}."
    return f"transformer.h.{i}."


# ============================================================
# FalconPassLayer
# ============================================================
class FalconPassLayer(nn.Module):
    def __init__(self, hidden_size=0):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden_states, alibi=None, attention_mask=None,
                position_ids=None, layer_past=None, head_mask=None,
                use_cache=False, output_attentions=False, **kw):
        if use_cache: return (hidden_states, layer_past)
        return (hidden_states,)


# ============================================================
# Layout: compact → original skeleton
# ============================================================
def _ensure_original_layout(model, removed_indices, original_N):
    layers = _get_layers(model)
    cur_N = len(layers)
    removed = set(int(i) for i in removed_indices)
    kept = sorted(set(range(original_N)) - removed)
    dev = next(model.parameters()).device
    hs = model.config.hidden_size

    if cur_N == original_N:
        print(f"[layout] sparse: {cur_N}L, PassLayer at {sorted(removed)}")
        for i in removed:
            layers[int(i)] = FalconPassLayer(hs).to(dev)
        return model, kept

    if cur_N != len(kept):
        raise ValueError(f"layer mismatch: {cur_N} vs compact={len(kept)} or sparse={original_N}")

    print(f"[layout] compact: {cur_N}L → {original_N}L expand")
    old = [layers[i] for i in range(cur_N)]
    new = [None] * original_N
    for pi, oi in enumerate(kept): new[oi] = old[pi]
    for i in removed: new[int(i)] = FalconPassLayer(hs).to(dev)
    assert all(l is not None for l in new)

    _set_layers(model, new)
    model.config.num_hidden_layers = original_N
    print(f"  real: {kept[:5]}{'...' if len(kept)>5 else ''} ({len(kept)}), "
          f"pass: {sorted(removed)} ({len(removed)})")
    return model, kept


# ============================================================
# Bundle 관리
# ============================================================
def _pick_file(bdir, idx):
    for fmt in [f"layer_{int(idx):03d}.safetensors", f"layer_{int(idx)}.safetensors"]:
        p = os.path.join(bdir, fmt)
        if os.path.isfile(p): return p
    raise FileNotFoundError(f"layer file missing: idx={idx} in {bdir}")


def _extract_sd(raw, idx):
    for pref in [f"transformer.h.{idx}.", f"h.{idx}."]:
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
    if FalconDecoderLayer is None:
        raise RuntimeError("FalconDecoderLayer import 실패")
    layers = _get_layers(model)
    dtype, dev = next(model.parameters()).dtype, next(model.parameters()).device
    for i in indices:
        i = int(i)
        try: nl = FalconDecoderLayer(model.config, layer_idx=i)
        except TypeError: nl = FalconDecoderLayer(model.config)
        nl = nl.to(device=dev, dtype=dtype)
        raw = load_file(_pick_file(bdir, i))
        sd = {k: v.to(device=dev, dtype=dtype) for k, v in _extract_sd(raw, i).items()}
        try: nl.load_state_dict(sd, strict=True)
        except RuntimeError: nl.load_state_dict(sd, strict=False)
        layers[i] = nl
        print(f"[rehydrate] layer {i} restored")


# ============================================================
# LoRA 어댑터
# ============================================================
def _detect_targets(model):
    new_arch = getattr(model.config, "new_decoder_architecture", False)
    if new_arch:
        candidates = ["q_proj", "k_proj", "v_proj", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    else:
        candidates = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    names = {n for n, _ in model.named_parameters()}
    verified = [t for t in candidates if any(t in n for n in names)]
    if not verified:
        verified = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    print(f"[LoRA] target_modules: {verified}")
    return verified


def _attach(model, name, target_layers=None, r=8, alpha=16, dropout=0.05):
    targets = _detect_targets(model)
    kw = dict(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
              task_type="CAUSAL_LM", target_modules=targets)
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
    pats = [_layer_prefix(model, i) for i in indices]
    n_en = 0
    for name, p in model.named_parameters():
        if any(pt in name for pt in pats) and "lora_" in name.lower():
            p.requires_grad = True; n_en += p.numel()
    if n_en == 0:
        raise RuntimeError(f"No LoRA params on layers {indices} for '{adapter_name}'")
    print(f"[trainable] {adapter_name}: {n_en:,} params on {len(indices)} layers")


# ============================================================
# Dataset  ★ 변경 1: 설명형 프롬프트 + 답변 보강
# ============================================================

# ---- [CHANGED] 설명형 시스템/유저 프롬프트 ----
def _build_msgs(ctx, q, ds_name):
    sys = ("You are a knowledgeable assistant. "
           "Provide a clear and detailed answer based on the given context. "
           "Explain your reasoning and include relevant details from the context.")
    if ds_name == "squad_v2":
        sys += " If the answer cannot be found in the context, explain why it is unanswerable."
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content":
         f"Read the following context and answer the question with a detailed explanation.\n\n"
         f"Context:\n{ctx}\n\nQuestion:\n{q}"},
    ]


# ---- [NEW] SQuAD 단답 → 설명형 답변으로 보강 ----
def _enrich_answer(ctx, ans, ds_name):
    """SQuAD의 짧은 span 답변을 context 문장을 활용해 설명형으로 확장"""
    if not ans or (ds_name == "squad_v2" and ans == "unanswerable"):
        return ("The answer cannot be determined from the provided context, "
                "as the relevant information is not explicitly stated.")

    # context에서 답을 포함하는 문장 추출
    sents = re.split(r'(?<=[.!?])\s+', ctx.strip())
    relevant = [s.strip() for s in sents if ans.lower() in s.lower()]

    if relevant:
        support = " ".join(relevant[:2])  # 최대 2문장
        return (f"Based on the context, {support} "
                f"Therefore, the answer is {ans}.")
    else:
        return (f"According to the provided context, the answer to this question "
                f"is {ans}.")


def _load_qa_dataset(tok, ds_name, split, max_samples, seq_len):
    DS_MAP = {"squad": "rajpurkar/squad", "squad_v2": "rajpurkar/squad_v2"}
    ds = load_dataset(DS_MAP.get(ds_name, ds_name), split=split)
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

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
        ctx, q = ex.get("context", ""), ex.get("question", "")
        raw_ans = ex.get("answers", {}).get("text", [""])[0] or \
                  ("unanswerable" if ds_name == "squad_v2" else "")
        msgs = _build_msgs(ctx, q, ds_name)

        # ---- [CHANGED] 단답 → 설명형 답변 ----
        ans = _enrich_answer(ctx, raw_ans, ds_name)

        if has_chat:
            p_ids = _to_list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        else:
            s, u = msgs[0]["content"], msgs[1]["content"]
            p_ids = _to_list(tok(f"System: {s}\nUser: {u}\nAssistant: ",
                                 add_special_tokens=True, truncation=True,
                                 max_length=seq_len - 128)["input_ids"])

        a_ids = _to_list(tok(" " + ans, add_special_tokens=False)["input_ids"])
        if eos_id: a_ids += [eos_id]
        if not a_ids: return {"__drop__": 1}

        full = p_ids + a_ids; plen = len(p_ids)
        if len(full) > seq_len:
            cut = len(full) - seq_len; full = full[cut:]; plen = max(0, plen - cut)
        pad_n = seq_len - len(full)
        ids = [pad_id] * pad_n + full
        mask = [0] * pad_n + [1] * len(full)
        labels = ids[:]
        for i in range(pad_n + plen):
            if i < len(labels): labels[i] = -100
        if pad_n + plen >= seq_len: return {"__drop__": 1}
        return {"input_ids": ids, "attention_mask": mask, "labels": labels, "__drop__": 0}

    ds = ds.map(proc, remove_columns=ds.column_names, num_proc=4)
    ds = ds.filter(lambda x: x["__drop__"] == 0)
    if "__drop__" in ds.column_names: ds = ds.remove_columns("__drop__")
    return ds


# ============================================================
# Index info loader (2-stage: stage 파라미터 불필요)
# ============================================================
def _load_index_info(base_dir, bundles_dir):
    info = {"B": [], "C": [], "L_full": None}
    manifest = os.path.join(base_dir, "manifest.json")
    if os.path.isfile(manifest):
        m = json.load(open(manifest))
        info["L_full"] = m.get("counts", {}).get("L_full")
        st = m.get("stages", {})
        info["B"] = sorted(int(x) for x in st.get("B", {}).get("removed_layers", []))
        info["C"] = sorted(int(x) for x in st.get("C", {}).get("removed_layers", []))
    log_p = os.path.join(base_dir, "prune_log.json")
    if os.path.isfile(log_p):
        log = json.load(open(log_p))
        if not info["B"]: info["B"] = sorted(log.get("split", {}).get("B", []))
        if not info["C"]: info["C"] = sorted(log.get("split", {}).get("C", []))
    if not info["B"]:
        info["B"] = _load_bundle_indices(os.path.join(bundles_dir, "B"))
    if not info["C"]:
        info["C"] = _load_bundle_indices(os.path.join(bundles_dir, "C"))
    return info


# ============================================================
# README 생성
# ============================================================
def _write_readme(out_dir, args, adapter_name, n_train, train_len, eval_len):
    env_vars = {k: os.environ.get(k, "") for k in ["CUDA_VISIBLE_DEVICES", "DEVICE"] if os.environ.get(k)}
    env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items())
    cmd = f"{env_prefix + ' ' if env_prefix else ''}python {' '.join(sys.argv)}"

    lines = [
        f"# {adapter_name} LoRA Adapter",
        f"",
        f"## Command",
        f"```bash",
        f"{cmd}",
        f"```",
        f"",
        f"## Training Info",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Adapter**: {adapter_name}",
        f"- **Trainable params**: {n_train:,}",
        f"- **Train samples**: {train_len:,}",
        f"- **Eval samples**: {eval_len:,}",
        f"",
        f"## Hyperparameters",
        f"| Param | Value |",
        f"|-------|-------|",
    ]
    for k in ["base_dir", "bundles_dir", "stage", "qa_dataset", "seq_len",
              "lr", "epochs", "bs", "grad_acc", "warmup_ratio", "max_grad_norm"]:
        lines.append(f"| {k} | `{getattr(args, k, '')}` |")
    lines.append("")

    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[readme] {readme_path}")


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
        learning_rate=args.lr, num_train_epochs=args.epochs,
        bf16=use_bf16, fp16=not use_bf16,
        dataloader_num_workers=4, dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        max_grad_norm=args.max_grad_norm, warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps, logging_first_step=True,
        remove_unused_columns=False, report_to="none",
        save_total_limit=args.save_total_limit,
    )
    try:
        ta = TrainingArguments(**common,
            eval_strategy="steps" if args.eval_steps > 0 else "no",
            eval_steps=args.eval_steps if args.eval_steps > 0 else None,
            save_strategy="steps" if args.save_steps > 0 else "no",
            save_steps=args.save_steps if args.save_steps > 0 else None)
    except TypeError:
        ta = TrainingArguments(**common,
            evaluation_strategy="steps" if args.eval_steps > 0 else "no",
            eval_steps=args.eval_steps if args.eval_steps > 0 else None,
            save_strategy="steps" if args.save_steps > 0 else "no",
            save_steps=args.save_steps if args.save_steps > 0 else None)

    Trainer(model=model, args=ta, train_dataset=train_ds, eval_dataset=eval_ds,
            data_collator=default_data_collator, processing_class=tok).train()

    if isinstance(model, PeftModel):
        try: model.save_pretrained(out_dir, selected_adapters=[adapter_name])
        except TypeError: model.save_pretrained(out_dir)

    _write_readme(out_dir, args, adapter_name, n_train, len(train_ds), len(eval_ds))


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--bundles_dir", required=True)
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
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
    device = torch.device(os.environ.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=dtype, device_map=None, local_files_only=True)
    model.to(device)
    model.config.use_cache = False
    try: model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    except: pass

    loaded_L = len(_get_layers(model))

    # Index info
    info = _load_index_info(args.base_dir, args.bundles_dir)
    B_idx, C_idx = info["B"], info["C"]
    original_N = args.original_num_layers or info["L_full"] or model.config.num_hidden_layers
    removed_all = sorted(set(B_idx + C_idx))
    A_idx = sorted(set(range(original_N)) - set(removed_all))
    AB_idx = sorted(A_idx + B_idx)

    print(f"\n[Index] original={original_N}, loaded={loaded_L}")
    print(f"  A({len(A_idx)}): {A_idx[:5]}{'...' if len(A_idx)>5 else ''}")
    print(f"  B({len(B_idx)}): {B_idx}")
    print(f"  C({len(C_idx)}): {C_idx}")

    # Ensure original layout
    model, kept = _ensure_original_layout(model, removed_all, original_N)
    layers = _get_layers(model)
    del kept

    # Datasets
    print("\n[Loading] Datasets")
    train_ds = _load_qa_dataset(tok, args.qa_dataset, "train", args.max_samples, args.seq_len)
    eval_ds = _load_qa_dataset(tok, args.qa_dataset, "validation", args.max_eval_samples, args.seq_len)
    print(f"  train={len(train_ds)}, eval={len(eval_ds)}")

    # ================================================================
    # Stage 1: A만 LoRA (B,C = PassLayer)
    # ================================================================
    if args.stage == 1:
        print(f"\n{'='*60}\nSTAGE 1: A-LoRA (A=real, B+C=FalconPassLayer)\n{'='*60}")

        if FalconDecoderLayer:
            bad_a = [i for i in A_idx if not isinstance(layers[i], FalconDecoderLayer)]
            if bad_a: raise RuntimeError(f"A 위치 비정상: {bad_a}")
        bad_bc = [i for i in (B_idx + C_idx) if not isinstance(layers[i], FalconPassLayer)]
        if bad_bc: raise RuntimeError(f"B+C가 PassLayer가 아님: {bad_bc}")

        print(f"\n[Layer Verify] 총 {original_N}층")
        print(f"  A (FalconDecoderLayer): {len(A_idx)}개 ✓")
        print(f"  B+C (FalconPassLayer):  {len(B_idx)+len(C_idx)}개 ✓")

        model = _attach(model, "stageA", target_layers=A_idx)
        model.set_adapter("stageA")
        _enable_lora_only(model, A_idx, "stageA")

        out = os.path.join(args.out_adapters, "A_lora", "stageA")
        train_adapter(model, tok, out, train_ds, eval_ds, args, "stageA")
        print("\n[Next] Stage 2 can reuse the same --base_dir and restore only B from bundles.")

    # ================================================================
    # Stage 2: A+B에 LoRA (A=real, B=복원, C=PassLayer)
    # ================================================================
    elif args.stage == 2:
        print(f"\n{'='*60}\nSTAGE 2: AB-LoRA (A=real, B=restored, C=FalconPassLayer)\n{'='*60}")

        B_bdir = os.path.join(args.bundles_dir, "B")
        _assert_bundles(B_bdir, B_idx)
        _rehydrate(model, B_bdir, B_idx)

        if FalconDecoderLayer:
            bad_a = [i for i in A_idx if not isinstance(layers[i], FalconDecoderLayer)]
            if bad_a: raise RuntimeError(f"A 위치 비정상: {bad_a}")
            bad_b = [i for i in B_idx if not isinstance(layers[i], FalconDecoderLayer)]
            if bad_b: raise RuntimeError(f"B 복원 실패: {bad_b}")
        bad_c = [i for i in C_idx if not isinstance(layers[i], FalconPassLayer)]
        if bad_c: raise RuntimeError(f"C가 PassLayer가 아님: {bad_c}")

        print(f"\n[Layer Verify] 총 {original_N}층")
        print(f"  A (FalconDecoderLayer): {A_idx[:5]}{'...' if len(A_idx)>5 else ''} ({len(A_idx)}개) ✓")
        print(f"  B (FalconDecoderLayer): {B_idx} ({len(B_idx)}개) ✓")
        print(f"  C (FalconPassLayer):    {C_idx} ({len(C_idx)}개) ✓")

        model = _attach(model, "stageAB", target_layers=AB_idx)
        model.set_adapter("stageAB")
        _enable_lora_only(model, AB_idx, "stageAB")

        out = os.path.join(args.out_adapters, "AB_lora", "stageAB")
        train_adapter(model, tok, out, train_ds, eval_ds, args, "stageAB")

    print("\n[Done] Training completed")


if __name__ == "__main__":
    main()