#!/usr/bin/env python3
"""
최적화된 LoRA 학습 코드
주요 개선사항:
1. DataLoader 병렬화 (num_workers, pin_memory)
2. Mixed precision 최적화 (bf16 사용)
3. Gradient checkpointing 개선
4. 효율적인 배치 처리
5. 토크나이저 최적화

# Stage1
python -m prune_lora.optimized_lora.py \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./lora_results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 --seq_len 1024 --epochs 1 --bs 1 --grad_acc 32
 

# Stage2
python Code.PruningAndLoRA.total_progressive_qa_lora.py \
  --base_dir ~/Code/results/pruning/A \
  --bundles_dir ~/Code/results/pruning/bundles \
  --stage 2 \
  --out_adapters ~/Code/results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 --seq_len 512 --epochs 1 --bs 4 --grad_acc 8

"""

import os, json, torch
import re
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
from typing import List
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    default_data_collator, DataCollatorWithPadding
)
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, PeftModel
from datetime import datetime, timezone as _tz
from peft.utils import get_peft_model_state_dict
import matplotlib.pyplot as plt

UTC = _tz.utc

def export_adapter_pt_and_recipe(model, out_dir, adapter_name, *, base_dir, bundles_dir, stage, trained_indices, tokenizer_dir=None):
    """어댑터 가중치 및 레시피 저장"""
    os.makedirs(out_dir, exist_ok=True)
    
    state = get_peft_model_state_dict(model, adapter_name=adapter_name)
    _LAYER_RE = re.compile(r"\blayers\.(\d+)\.")

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
        print(f"[warn] slim_state is empty for adapter={adapter_name}. Falling back to full state.")
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

    cfg = _peft_cfg_to_dict(raw_cfg) if raw_cfg is not None else None

    recipe = {
        "saved_at": datetime.now(UTC).isoformat(),
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

    print(f"[export] adapter={adapter_name} → {pt_path}")

def _peft_cfg_to_dict(cfg):
    if isinstance(cfg, dict):
        return cfg
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    out = {}
    for k, v in getattr(cfg, "__dict__", {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            if all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
                out[k] = list(v)
    return out

CANON_PATH = "model.model.layers"

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
        "model.layers", "model.decoder.layers", "model.model.layers",
        "model.model.decoder.layers", "base_model.model.layers",
        "base_model.model.decoder.layers", "base_model.model.model.layers",
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
        raise AttributeError("decoder layers not found")

    if not isinstance(found, (list, nn.ModuleList)):
        try:
            new_cur = nn.ModuleList(list(found))
        except Exception as e:
            raise TypeError(f"layers container is immutable: {e}")
        setattr(found_parent, found_name, new_cur)
        found = new_cur

    try:
        canon_parent, canon_last, _ = _resolve_attr_path(model, CANON_PATH.replace(".layers", ""))
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

def _find_embed_device(model):
    paths = ["model.embed_tokens","model.model.embed_tokens","base_model.model.model.embed_tokens"]
    for p in paths:
        cur = model
        try:
            for seg in p.split("."):
                cur = getattr(cur, seg)
            return next(cur.parameters()).device
        except Exception:
            pass
    return next(model.parameters()).device

def _load_prev_adapters(model, adapters_root: str, names: List[str]):
    from peft import PeftModel
    name2dir = {
        "stageA": "A_lora", "stageB": "B_lora", "stageC": "C_lora",
        "A": "A_lora", "B": "B_lora", "C": "C_lora",
    }
    print("[debug] adapters_root =", os.path.abspath(adapters_root))
    for i, nm in enumerate(names):
        top = os.path.join(adapters_root, name2dir.get(nm, nm))
        inner = os.path.join(top, nm)
        flat = os.path.join(adapters_root, nm)
        candidates = [inner, top, flat]
        chosen = None
        for adir in candidates:
            if os.path.isdir(adir):
                chosen = adir
                break
        if not chosen:
            raise FileNotFoundError(f"[adapter-load] cannot find {nm} under {adapters_root}")
        if i == 0 and not isinstance(model, PeftModel):
            model = PeftModel.from_pretrained(model, chosen, adapter_name=nm, local_files_only=True)
        else:
            model.load_adapter(chosen, adapter_name=nm, local_files_only=True)
        print(f"[adapter-load] loaded {nm} from {chosen}")

    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = False
    return model

def _reapply_passlayers_from_manifest(model, base_dir: str):
    man_path = os.path.join(base_dir, "manifest.json")
    print("[reapply] looking for:", man_path)
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
        print("[reapply] non-integer indices in removed_layers -> skip")
        return model

    try:
        from lib.identity import LlamaPassLayer as _Inner
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

    try:
        layers = _get_layer_container(model)
    except Exception as e:
        print("[reapply] cannot locate layers:", e, "-> skip")
        return model

    L = len(layers)
    hidden = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden is None:
        try:
            hidden = model.config.hidden_size
        except Exception:
            print("[reapply] hidden_size not found -> skip")
            return model

    for i in removed:
        if 0 <= int(i) < L:
            try: 
                layers[int(i)] = _make(hidden)
            except TypeError as te: 
                print("[reapply] layer container may be immutable ->", te)
                return model
        else:
            print(f"[reapply] index {i} out of range -> skip this one")

    print(f"[reapply] installed PassLayer on: {sorted(map(int, removed))}")
    return model

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def _rehydrate_layers(model, bundle_dir: str, indices: List[int]):
    layers = _get_layer_container(model)
    dtype = next(model.parameters()).dtype
    tgt = next(model.parameters()).device
    for i in indices:
        new_layer = LlamaDecoderLayer(model.config, layer_idx=i).to(device=tgt, dtype=dtype)
        f = os.path.join(bundle_dir, f"layer_{i:03d}.safetensors")
        if not os.path.isfile(f):
            raise FileNotFoundError(f"bundle miss: {f}")
        sd = load_file(f)
        sd = {k: v.to(device=tgt, dtype=dtype) for k, v in sd.items()}
        try:
            new_layer.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"[warn] strict load failed for {i}: {e} -> non-strict")
            new_layer.load_state_dict(sd, strict=False)
        layers[i] = new_layer
        print(f"[rehydrate] layer {i} restored on {tgt}")

def _freeze_all(model):
    for _, p in model.named_parameters():
        p.requires_grad = False

def _attach_new_adapter(
    model, name: str,
    target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"),
    r=8, alpha=16, dropout=0.05
):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )

    if isinstance(model, PeftModel):
        if name not in getattr(model, "peft_config", {}):
            model.add_adapter(name, cfg)
            print("어댑터 부착")
        return model
    else:
        return get_peft_model(model, cfg, adapter_name=name)

def _enable_only_lora_on_indices_for_adapter_by_name(model, indices: List[int], adapter_name: str, keep_layernorm=False):
    for n, p in model.named_parameters():
        p.requires_grad = False

    enabled = 0
    layer_patterns = [_layer_name_prefix(model, i) for i in indices]
    for pname, p in model.named_parameters():
        if any(pat in pname for pat in layer_patterns) and ("lora_" in pname.lower() or "lora" in pname.lower()):
            if adapter_name is None or adapter_name.lower() in pname.lower():
                p.requires_grad = True
                enabled += p.numel()
            continue
        if keep_layernorm:
            if any(pat in pname for pat in layer_patterns) and ("layernorm" in pname.lower() or ".ln_" in pname.lower() or ".norm" in pname.lower()):
                p.requires_grad = True
                continue

    if enabled == 0:
        raise RuntimeError(f"No LoRA params enabled for adapter='{adapter_name}' on layers={indices}.")
    print(f"[trainable] adapter={adapter_name} layers={indices} -> enabled params count {enabled}")

# ============== 최적화된 데이터셋 로딩 ==============
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
        add_generation_prompt=True
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
    add_eos=True
):
    """최적화된 데이터셋 로딩 - 배치 처리 및 캐싱"""
    ds = load_dataset(qa_dataset, split=split)
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def to_ex(ex):
        ctx = ex.get("context", "")
        q = ex.get("question", "")
        ans_list = ex.get("answers", {}).get("text", [])
        target = (ans_list[0] if ans_list else ("unanswerable" if qa_dataset == "squad_v2" else ""))

        messages = _build_chat_messages(ctx, q, qa_dataset, unans_token=unans_token)
        prompt_ids = _encode_chat_prompt_ids(tokenizer, messages)

        ans_text = (" " + target) if target else ""
        ans_ids = tokenizer(ans_text, add_special_tokens=False)["input_ids"]

        if add_eos and tokenizer.eos_token_id is not None:
            ans_ids = ans_ids + [tokenizer.eos_token_id]

        if len(ans_ids) < 1:
            return {"__drop__": 1}

        full_ids = prompt_ids + ans_ids
        prompt_len = len(prompt_ids)

        if len(full_ids) > seq_len:
            cut = len(full_ids) - seq_len
            full_ids = full_ids[cut:]
            prompt_len = max(0, prompt_len - cut)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
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

    # 배치 처리로 맵핑 최적화
    ds = ds.map(to_ex, remove_columns=ds.column_names, num_proc=4, batched=False)
    ds = ds.filter(lambda x: x["__drop__"] == 0)
    if "__drop__" in ds.column_names:
        ds = ds.remove_columns(["__drop__"])
    
    return ds

def plot_loss(log_history, out_dir):
    """학습 로그 시각화"""
    train_logs = [log for log in log_history if 'loss' in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log]

    if not eval_logs:
        print("[plot] No evaluation logs found, skipping plot.")
        return

    train_steps = [log['step'] for log in train_logs]
    train_losses = [log['loss'] for log in train_logs]

    eval_steps_for_plot = [log['step'] for log in eval_logs]
    eval_losses = [log['eval_loss'] for log in eval_logs]

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label='Training Loss', alpha=0.7)
    plt.plot(eval_steps_for_plot, eval_losses, label='Validation Loss', marker='o', linestyle='--')

    plt.title('Training and Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(out_dir, "loss_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[plot] Loss plot saved to {save_path}")

# ============== 최적화된 학습 함수 ==============
def train_lora(model, tokenizer, out_dir: str, train_ds, eval_ds=None, lr=2e-4, epochs=1, bs=4, grad_acc=8, fp16=True, adapter_name=None):
    """
    최적화된 LoRA 학습
    - BF16 사용 (A100에서 더 효율적)
    - DataLoader 병렬화
    - 효율적인 배치 처리
    """
    os.makedirs(out_dir, exist_ok=True)

    has_eval = eval_ds is not None

    # BF16 사용 가능 여부 확인
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,  # eval도 동일한 배치 사이즈
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        num_train_epochs=epochs,
        
        # ============ 최적화 설정 ============
        # Mixed Precision
        bf16=use_bf16,  # BF16이 FP16보다 A100에서 빠름
        fp16=not use_bf16,  # BF16 불가능하면 FP16
        fp16_full_eval=not use_bf16,
        bf16_full_eval=use_bf16,
        
        # DataLoader 최적화
        dataloader_num_workers=4,  # 병렬 데이터 로딩
        dataloader_pin_memory=True,  # GPU 전송 최적화
        dataloader_prefetch_factor=2,  # 프리페칭
        
        # Gradient Checkpointing
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 더 효율적
        
        # 메모리 최적화
        optim="adamw_torch_fused",  # Fused optimizer (더 빠름)
        max_grad_norm=1.0,
        
        # 로깅
        logging_steps=20,
        logging_first_step=True,
        
        # 평가 설정
        eval_accumulation_steps=None,  # 메모리 효율성
        prediction_loss_only=True,
        
        # 기타
        warmup_ratio=0.1,
        report_to="none",
        
        # 체크포인트 (필요시 활성화)
        # save_strategy="no",
        # save_total_limit=1,
    )

    # 커스텀 데이터 콜레이터 (동적 패딩)
    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if has_eval else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print(f"[train] Starting training with:")
    print(f"  - Precision: {'BF16' if use_bf16 else 'FP16'}")
    print(f"  - Batch size: {bs} x {grad_acc} = {bs * grad_acc} effective")
    print(f"  - DataLoader workers: 4")
    print(f"  - Optimizer: adamw_torch_fused")
    
    trainer.train()

    if has_eval:
        metrics = trainer.evaluate(eval_dataset=eval_ds)
        print("[eval/end] metrics:", metrics)

    plot_loss(trainer.state.log_history, out_dir)

    if isinstance(model, PeftModel):
        try:
            model.save_pretrained(out_dir, selected_adapters=[adapter_name] if adapter_name else None)
        except TypeError:
            model.save_pretrained(out_dir)
    else:
        print("[warn] model is not PeftModel; adapter save may be skipped")

# ============== 메인 함수 ==============
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--bundles_dir", type=str, required=True)
    ap.add_argument("--stage", type=int, choices=[1,2,3], required=True)
    ap.add_argument("--out_adapters", type=str, default="./adapters")

    ap.add_argument("--qa_dataset", type=str, choices=["squad","squad_v2"], default="squad")
    ap.add_argument("--max_samples", type=int, default=5000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--unans_token", type=str, default="unanswerable")

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_acc", type=int, default=8)
    ap.add_argument("--max_eval_samples", type=int, default=2000)
    
    args = ap.parse_args()

    # ============ 토크나이저 최적화 ============
    tok = AutoTokenizer.from_pretrained(
        args.base_dir, 
        use_fast=True,  # Fast tokenizer 사용
        local_files_only=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # ============ 모델 로딩 최적화 ============
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # BF16 사용 가능 여부에 따라 dtype 선택
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("[model] Using BF16")
    else:
        dtype = torch.float16
        print("[model] Using FP16")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, 
        torch_dtype=dtype,
        device_map=None,  # 단일 GPU
        local_files_only=True
    )
    model.to(device)
    model.config.use_cache = False

    # Gradient checkpointing 활성화
    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    model = _reapply_passlayers_from_manifest(model, args.base_dir)

    with open(os.path.join(args.base_dir, "prune_log.json"), "r", encoding="utf-8") as f:
        log = json.load(f)
    B_idx, C_idx = log["split"]["B"], log["split"]["C"]

    # ============ 데이터셋 로딩 (병렬 처리) ============
    print("[dataset] Loading training data...")
    train_ds = _load_qa_sft_dataset(
        tok, qa_dataset=args.qa_dataset, split="train",
        max_samples=args.max_samples, seq_len=args.seq_len
    )
    print(f"[dataset] Training samples: {len(train_ds)}")
    
    print("[dataset] Loading validation data...")
    eval_ds = _load_qa_sft_dataset(
        tok, qa_dataset=args.qa_dataset, split="validation",
        max_samples=args.max_eval_samples, seq_len=args.seq_len
    )
    print(f"[dataset] Validation samples: {len(eval_ds)}")

    # ============ Stage별 처리 ============
    if args.stage == 1:
        removed = set(B_idx) | set(C_idx)
        all_idx = list(range(len(_get_layer_container(model))))
        A_idx = [i for i in all_idx if i not in removed]

        model = _attach_new_adapter(model, "stageA")
        model.set_adapter("stageA")

        _enable_only_lora_on_indices_for_adapter_by_name(model, A_idx, "stageA", keep_layernorm=False)

        out_dir = os.path.join(args.out_adapters, "A_lora")
        train_lora(model, tok, out_dir, train_ds, eval_ds, args.lr, args.epochs, args.bs, args.grad_acc, adapter_name="stageA")
        
        export_adapter_pt_and_recipe(
            model, out_dir, "stageA",
            base_dir=args.base_dir, bundles_dir=args.bundles_dir, stage="A",
            trained_indices=A_idx, tokenizer_dir=args.base_dir
        )

    elif args.stage == 2:
        layers = _get_layer_container(model)
        L = len(layers)
        AB_idx = [i for i in range(L) if i not in C_idx]
        
        _assert_bundle_files_exist(args.bundles_dir, "B", B_idx)
        _rehydrate_layers(model, os.path.join(args.bundles_dir, "B"), B_idx)
        print(f"[rehydrate] Stage B layers ({len(B_idx)}) restored.")

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

        trainable = [(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
        print("[post-fix] total trainable params:", sum(x[1] for x in trainable))

        bad_keys = []
        for n, p in model.named_parameters():
            if ("lora_" in n.lower() or "lora" in n.lower()) and p.numel() > 0:
                if not torch.isfinite(p).all():
                    bad_keys.append(n)
        if bad_keys:
            print("[FATAL] NaN/Inf in LoRA params:", bad_keys[:20])
            raise RuntimeError("Detected NaN/Inf in LoRA parameters")
        print("[OK] LoRA finite & non-LoRA frozen for stageAB")

        out_dir = os.path.join(args.out_adapters, "AB_lora")
        train_lora(model, tok, out_dir, train_ds, eval_ds, lr=args.lr, epochs=args.epochs, bs=args.bs, grad_acc=args.grad_acc, fp16=True, adapter_name="stageAB")
        
        export_adapter_pt_and_recipe(
            model, out_dir, "stageAB",
            base_dir=args.base_dir, bundles_dir=args.bundles_dir, stage="AB",
            trained_indices=AB_idx, tokenizer_dir=args.base_dir
        )

if __name__ == "__main__":
    main()