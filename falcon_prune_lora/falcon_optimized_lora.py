#!/usr/bin/env python3
"""
Falcon Progressive 3-Stage LoRA Training (skeleton-preserving, no KD)

Stage 1: A 로드 → 원본 레이아웃 보장 → B,C=PassLayer → A에만 LoRA
Stage 2: A_merged 로드 → 원본 레이아웃 보장 → B 복원 + C=PassLayer → B에만 LoRA
Stage 3: A_merged 로드 → 원본 레이아웃 보장 → B_merged+C 복원 → C에만 LoRA

Usage:
# Stage 1 - 2048
CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0 \
python -m falcon_prune_lora.falcon_optimized_lora \
  --base_dir ./falcon_results/pruning/A \
  --bundles_dir ./falcon_results/pruning/bundles \
  --stage 1 --out_adapters ./2048_falcon_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 2048 --lr 3e-4 --epochs 2 --bs 1 --grad_acc 32

# Stage 2 (A_merged 기준)
CUDA_VISIBLE_DEVICES=5 DEVICE=cuda:0 \
python -m falcon_prune_lora.falcon_optimized_lora \
  --base_dir ./merged_2048loraresults_falcon7b/A_merged \
  --bundles_dir ./falcon_results/pruning/bundles \
  --stage 2 --out_adapters ./2048_falcon_lora_results/adapters \
  --seq_len 2048 --lr 3e-5 --epochs 1 --bs 1 --grad_acc 32

# Stage 3 (A_merged + B_merged)
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 \
python -m falcon_prune_lora.falcon_optimized_lora \
  --base_dir ./merged_2048loraresults_falcon7b/A_merged \
  --b_merged_dir ./merged_2048loraresults_falcon7b/B_merged \
  --bundles_dir ./falcon_results/pruning/bundles/C \
  --stage 3 --out_adapters ./2048_falcon_lora_results/adapters \
  --seq_len 2048 --lr 3e-5 --epochs 1 --bs 1 --grad_acc 32
"""
#!/usr/bin/env python3
"""
Falcon Progressive 3-Stage LoRA Training (skeleton-preserving, no KD)

Stage 1: A 로드 → 원본 레이아웃 보장 → B,C=PassLayer → A에만 LoRA
Stage 2: A_merged 로드 → 원본 레이아웃 보장 → B 복원 + C=PassLayer → B에만 LoRA
Stage 3: A_merged 로드 → 원본 레이아웃 보장 → B_merged+C 복원 → C에만 LoRA

Usage:
# Stage 1
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 \
python -m falcon_prune_lora.falcon_optimized_lora \
  --base_dir ./falcon_results/pruning/A \
  --bundles_dir ./falcon_results/pruning/bundles \
  --stage 1 --out_adapters ./falcon_lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 \
  --seq_len 1024 --lr 3e-4 --epochs 2 --bs 1 --grad_acc 32

# Stage 2 (A_merged 기준)
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 \
python -m falcon_prune_lora.falcon_optimized_lora \
  --base_dir ./merged_models_falcon/A_merged \
  --bundles_dir ./falcon_results/pruning/bundles \
  --stage 2 --out_adapters ./falcon_lora_results/adapters \
  --seq_len 1024 --lr 3e-5 --epochs 1 --bs 1 --grad_acc 32

# Stage 3 (A_merged + B_merged)
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 \
python -m falcon_prune_lora.falcon_optimized_lora \
  --base_dir ./merged_models_falcon/A_merged \
  --b_merged_dir ./merged_models_falcon/B_merged \
  --bundles_dir ./falcon_results/pruning/bundles/C \
  --stage 3 --out_adapters ./falcon_lora_results/adapters \
  --seq_len 1024 --lr 3e-5 --epochs 1 --bs 1 --grad_acc 32
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
    """Falcon 디코더 레이어 컨테이너 반환 (다양한 wrapper 대응)"""
    for path in [
        lambda m: m.transformer.h,
        lambda m: m.base_model.model.transformer.h,
        lambda m: m.model.transformer.h,
    ]:
        try:
            layers = path(model)
            if hasattr(layers, "__len__"):
                return layers
        except (AttributeError, TypeError):
            continue
    raise AttributeError("Falcon decoder layers not found")


def _set_layers(model, new_layers):
    """Falcon 레이어 컨테이너 교체"""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        model.transformer.h = nn.ModuleList(new_layers)
    elif hasattr(model, "model") and hasattr(model.model, "transformer"):
        model.model.transformer.h = nn.ModuleList(new_layers)
    else:
        raise RuntimeError("cannot find transformer.h path")


def _layer_prefix(model, i):
    """PeftModel 래핑 여부에 따른 레이어 prefix"""
    if isinstance(model, PeftModel):
        return f"base_model.model.transformer.h.{i}."
    return f"transformer.h.{i}."


# ============================================================
# FalconPassLayer
# ============================================================
class FalconPassLayer(nn.Module):
    """Falcon decoder-compatible identity layer"""
    def __init__(self, hidden_size=0):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden_states, alibi=None, attention_mask=None,
                position_ids=None, layer_past=None, head_mask=None,
                use_cache=False, output_attentions=False, **kw):
        if use_cache:
            return (hidden_states, layer_past)
        return (hidden_states,)


# ============================================================
# Layout: compact → original skeleton
# ============================================================
def _ensure_original_layout(model, removed_indices, original_N):
    """모델을 원본 레이어 수로 맞추고 removed 위치에 FalconPassLayer 삽입"""
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
    for pi, oi in enumerate(kept):
        new[oi] = old[pi]
    for i in removed:
        new[int(i)] = FalconPassLayer(hs).to(dev)
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
        with open(meta) as f:
            return sorted(json.load(f).get("indices", []))
    return sorted(int(re.match(r"layer_(\d+)", fn).group(1))
                  for fn in os.listdir(bdir)
                  if re.match(r"layer_\d+\.safetensors", fn))


def _assert_bundles(bdir, indices):
    missing = []
    for i in indices:
        try:
            f = _pick_file(bdir, i)
            if os.path.getsize(f) == 0: missing.append(i)
        except FileNotFoundError:
            missing.append(i)
    if missing:
        raise FileNotFoundError(f"[bundles] missing/empty: {missing} in {bdir}")
    print(f"[bundles-ok] {len(indices)} files in {bdir}")


def _rehydrate(model, bdir, indices):
    """FalconPassLayer → FalconDecoderLayer 복원"""
    if FalconDecoderLayer is None:
        raise RuntimeError("FalconDecoderLayer import 실패")
    layers = _get_layers(model)
    dtype, dev = next(model.parameters()).dtype, next(model.parameters()).device
    for i in indices:
        i = int(i)
        try:
            nl = FalconDecoderLayer(model.config, layer_idx=i)
        except TypeError:
            nl = FalconDecoderLayer(model.config)
        nl = nl.to(device=dev, dtype=dtype)
        raw = load_file(_pick_file(bdir, i))
        sd = {k: v.to(device=dev, dtype=dtype) for k, v in _extract_sd(raw, i).items()}
        try:
            nl.load_state_dict(sd, strict=True)
        except RuntimeError:
            nl.load_state_dict(sd, strict=False)
        layers[i] = nl
        print(f"[rehydrate] layer {i} restored")


# ============================================================
# LoRA 어댑터
# ============================================================
def _detect_targets(model):
    """Falcon 아키텍처 LoRA target_modules 자동 감지"""
    new_arch = getattr(model.config, "new_decoder_architecture", False)
    if new_arch:
        candidates = ["q_proj", "k_proj", "v_proj", "dense",
                       "dense_h_to_4h", "dense_4h_to_h"]
    else:
        candidates = ["query_key_value", "dense",
                       "dense_h_to_4h", "dense_4h_to_h"]
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
    for p in model.parameters():
        p.requires_grad = False
    pats = [_layer_prefix(model, i) for i in indices]
    n_en = 0
    for name, p in model.named_parameters():
        if any(pt in name for pt in pats) and "lora_" in name.lower():
            p.requires_grad = True
            n_en += p.numel()
    if n_en == 0:
        raise RuntimeError(f"No LoRA params on layers {indices} for '{adapter_name}'")
    print(f"[trainable] {adapter_name}: {n_en:,} params on {len(indices)} layers")


# ============================================================
# Dataset
# ============================================================
def _build_msgs(ctx, q, ds_name):
    sys = "You are a helpful QA assistant."
    if ds_name == "squad_v2":
        sys += " If the answer is not in the context, say 'unanswerable'."
    return [{"role": "system", "content": sys},
            {"role": "user", "content":
             f"Answer the question using the context.\n\nContext:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:"}]


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
        ans = ex.get("answers", {}).get("text", [""])[0] or \
              ("unanswerable" if ds_name == "squad_v2" else "")
        msgs = _build_msgs(ctx, q, ds_name)

        if has_chat:
            p_ids = _to_list(tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True))
        else:
            s, u = msgs[0]["content"], msgs[1]["content"]
            p_ids = _to_list(tok(
                f"System: {s}\nUser: {u}\nAssistant: ",
                add_special_tokens=True, truncation=True,
                max_length=seq_len - 64)["input_ids"])

        a_ids = _to_list(tok(" " + ans, add_special_tokens=False)["input_ids"])
        if eos_id: a_ids += [eos_id]
        if not a_ids: return {"__drop__": 1}

        full = p_ids + a_ids
        plen = len(p_ids)
        if len(full) > seq_len:
            cut = len(full) - seq_len
            full = full[cut:]
            plen = max(0, plen - cut)
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
    if "__drop__" in ds.column_names:
        ds = ds.remove_columns("__drop__")
    return ds


# ============================================================
# Index info loader
# ============================================================
def _load_index_info(base_dir, bundles_dir, stage, b_merged_dir=None):
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
    if not info["B"] and stage < 3:
        info["B"] = _load_bundle_indices(os.path.join(bundles_dir, "B"))
    if not info["C"]:
        c_dir = bundles_dir if stage == 3 else os.path.join(bundles_dir, "C")
        info["C"] = _load_bundle_indices(c_dir)
    return info


# ============================================================
# README 생성
# ============================================================
def _write_readme(out_dir, args, adapter_name, start_time):
    """학습 메타정보를 README.md로 기록"""
    end_time = datetime.now()
    cmd = " ".join(sys.argv)
    readme_path = os.path.join(out_dir, "README.md")

    content = f"""# {adapter_name} — LoRA Adapter

## Training Info
- **Created**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Finished**: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Duration**: {str(end_time - start_time).split('.')[0]}

## Command
```bash
{cmd}
```

## Hyperparameters
| Param | Value |
|-------|-------|
| stage | {args.stage} |
| lr | {args.lr} |
| epochs | {args.epochs} |
| batch_size | {args.bs} |
| grad_acc | {args.grad_acc} |
| seq_len | {args.seq_len} |
| warmup_ratio | {args.warmup_ratio} |
| max_grad_norm | {args.max_grad_norm} |
| dataset | {args.qa_dataset} |
| max_samples | {args.max_samples} |

## Paths
- base_dir: `{args.base_dir}`
- bundles_dir: `{args.bundles_dir}`
- b_merged_dir: `{args.b_merged_dir or 'N/A'}`
"""
    with open(readme_path, "w") as f:
        f.write(content)
    print(f"[readme] saved → {readme_path}")


# ============================================================
# Training
# ============================================================
def train_adapter(model, tok, out_dir, train_ds, eval_ds, args, adapter_name):
    os.makedirs(out_dir, exist_ok=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[train] {adapter_name}: {n_train:,} trainable → {out_dir}")
    if n_train == 0: raise RuntimeError("No trainable params!")

    start_time = datetime.now()
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

    # ── epoch 단위 저장 + eval ──
    epoch_save = dict(
        save_strategy="epoch",
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
    )

    try:
        ta = TrainingArguments(**common, **epoch_save)
    except TypeError:
        # 구버전 transformers: evaluation_strategy
        epoch_save["evaluation_strategy"] = epoch_save.pop("eval_strategy")
        ta = TrainingArguments(**common, **epoch_save)

    Trainer(model=model, args=ta, train_dataset=train_ds, eval_dataset=eval_ds,
            data_collator=default_data_collator, processing_class=tok).train()

    if isinstance(model, PeftModel):
        try: model.save_pretrained(out_dir, selected_adapters=[adapter_name])
        except TypeError: model.save_pretrained(out_dir)

    # ── README 기록 ──
    _write_readme(out_dir, args, adapter_name, start_time)


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--bundles_dir", required=True)
    p.add_argument("--b_merged_dir", default=None)
    p.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
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
    device = torch.device(
        os.environ.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (torch.cuda.is_available()
                               and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=dtype, device_map=None, local_files_only=True)
    model.to(device)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except: pass

    loaded_L = len(_get_layers(model))

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
    layers = _get_layers(model)

    # Datasets
    print("\n[Loading] Datasets")
    train_ds = _load_qa_dataset(tok, args.qa_dataset, "train",
                                args.max_samples, args.seq_len)
    eval_ds = _load_qa_dataset(tok, args.qa_dataset, "validation",
                               args.max_eval_samples, args.seq_len)
    print(f"  train={len(train_ds)}, eval={len(eval_ds)}")

    # ================================================================
    # Stage 1: A만 LoRA
    # ================================================================
    if args.stage == 1:
        print(f"\n{'='*60}\nSTAGE 1: A-LoRA (A=real, B+C=PassLayer)\n{'='*60}")
        if FalconDecoderLayer:
            bad = [i for i in A_idx if not isinstance(layers[i], FalconDecoderLayer)]
            if bad: raise RuntimeError(f"A 위치 비정상: {bad}")

        model = _attach(model, "stageA", target_layers=A_idx)
        model.set_adapter("stageA")
        _enable_lora_only(model, A_idx, "stageA")

        out = os.path.join(args.out_adapters, "A_lora", "stageA")
        train_adapter(model, tok, out, train_ds, eval_ds, args, "stageA")
        print(f"\n[Next] Merge: merge_adapter.py --base_model {args.base_dir} "
              f"--adapter_path {out} --output_dir ./merged_models_falcon/A_merged")

    # ================================================================
    # Stage 2: B만 LoRA
    # ================================================================
    elif args.stage == 2:
        print(f"\n{'='*60}\nSTAGE 2: B-LoRA (A=merged, B=restored, C=PassLayer)\n{'='*60}")
        B_bdir = os.path.join(args.bundles_dir, "B")
        _assert_bundles(B_bdir, B_idx)
        _rehydrate(model, B_bdir, B_idx)

        if FalconDecoderLayer:
            bad = [i for i in B_idx if not isinstance(layers[i], FalconDecoderLayer)]
            if bad: raise RuntimeError(f"B 복원 실패: {bad}")

        model = _attach(model, "stageB", target_layers=B_idx)
        model.set_adapter("stageB")
        _enable_lora_only(model, B_idx, "stageB")

        out = os.path.join(args.out_adapters, "B_lora", "stageB")
        train_adapter(model, tok, out, train_ds, eval_ds, args, "stageB")
        print(f"\n[Next] Merge B adapter with B bundle → B_merged")

    # ================================================================
    # Stage 3: C만 LoRA
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

        if FalconDecoderLayer:
            bad = [i for i in C_idx if not isinstance(layers[i], FalconDecoderLayer)]
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