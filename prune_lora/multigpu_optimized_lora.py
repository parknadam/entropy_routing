#!/usr/bin/env python3
"""
#멀티 gpu lora 학습 코드
Multi-GPU 최적화 LoRA 학습 (DDP)
GPU 2개를 모두 활용하여 학습 속도를 2배 가속


# GPU 2개로 실행 (가장 간단!)
CUDA_VISIBLE_DEVICES=0,1 python prune_lora/multigpu_optimized_lora.py \
  --base_dir ./7b_results/pruning/A \
  --bundles_dir ./7b_results/pruning/bundles \
  --stage 1 \
  --out_adapters ./lora_results/adapters \
  --qa_dataset squad \
  --max_samples 20000 \
  --max_eval_samples 8000 \
  --seq_len 1024 \
  --epochs 1 \
  --bs 4 \
  --grad_acc 4
"""


#!/usr/bin/env python3
"""
간단한 Multi-GPU 학습 (Transformers Trainer가 자동 처리)
CUDA_VISIBLE_DEVICES로 GPU 선택, Trainer가 DDP 자동 설정
"""

import os, json, torch
import re
import torch.nn as nn
from typing import List
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    default_data_collator
)
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, PeftModel
from datetime import datetime, timezone as _tz
from peft.utils import get_peft_model_state_dict
import matplotlib.pyplot as plt

UTC = _tz.utc

def export_adapter_pt_and_recipe(model, out_dir, adapter_name, *, base_dir, bundles_dir, stage, trained_indices, tokenizer_dir=None):
    """어댑터 저장"""
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

def _reapply_passlayers_from_manifest(model, base_dir: str):
    man_path = os.path.join(base_dir, "manifest.json")
    if not os.path.isfile(man_path):
        return model
    try:
        man = json.load(open(man_path, "r"))
    except Exception:
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
        return model

    try:
        removed = sorted(set(int(i) for i in removed))
    except Exception:
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
    except Exception:
        class SafePass(nn.Module):
            def __init__(self, hidden):
                super().__init__()
            def forward(self, x, *a, **kw):
               return x
        def _make(h): 
            return SafePass(h)

    try:
        layers = _get_layer_container(model)
    except Exception:
        return model

    L = len(layers)
    hidden = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden is None:
        return model

    for i in removed:
        if 0 <= int(i) < L:
            try: 
                layers[int(i)] = _make(hidden)
            except TypeError:
                return model

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
        except RuntimeError:
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
        raise RuntimeError(f"No LoRA params enabled for adapter='{adapter_name}'.")
    print(f"[trainable] adapter={adapter_name} layers={indices} -> {enabled} params")

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
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    elif isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]
    if not isinstance(prompt_ids, list):
        prompt_ids = list(prompt_ids)
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

    ds = ds.map(to_ex, remove_columns=ds.column_names, num_proc=4, batched=False)
    ds = ds.filter(lambda x: x["__drop__"] == 0)
    if "__drop__" in ds.column_names:
        ds = ds.remove_columns(["__drop__"])
    
    return ds

def plot_loss(log_history, out_dir):
    train_logs = [log for log in log_history if 'loss' in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log]

    if not eval_logs:
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

def train_lora(model, tokenizer, out_dir: str, train_ds, eval_ds=None, lr=2e-4, epochs=1, bs=4, grad_acc=8, fp16=True, adapter_name=None):
    """
    간단한 Multi-GPU 학습
    Trainer가 자동으로 DDP 처리
    """
    os.makedirs(out_dir, exist_ok=True)
    has_eval = eval_ds is not None

    # BF16 지원 확인
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    # local_rank 자동 감지 (DDP 사용 시)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        num_train_epochs=epochs,
        
        # Mixed Precision
        bf16=use_bf16,
        fp16=not use_bf16,
        fp16_full_eval=not use_bf16,
        bf16_full_eval=use_bf16,
        
        # DataLoader 최적화
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        
        # Gradient Checkpointing
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        
        # 로깅
        logging_steps=20,
        logging_first_step=True,
        
        # 평가
        eval_accumulation_steps=None,
        prediction_loss_only=True,
        
        warmup_ratio=0.1,
        report_to="none",
    )

    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if has_eval else None,
        data_collator=data_collator,
        #tokenizer=tokenizer,
    )
    
    # GPU 정보 출력 (메인 프로세스만)
    if local_rank <= 0:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"\n{'='*60}")
        print(f"[Training Configuration]")
        print(f"{'='*60}")
        print(f"  GPUs: {n_gpus}")
        print(f"  Precision: {'BF16' if use_bf16 else 'FP16'}")
        print(f"  Per-device batch: {bs}")
        print(f"  Gradient accumulation: {grad_acc}")
        print(f"  Effective batch: {bs * grad_acc * max(1, n_gpus)}")
        print(f"{'='*60}\n")
    
    trainer.train()

    if has_eval and local_rank <= 0:
        metrics = trainer.evaluate(eval_dataset=eval_ds)
        print("[eval/end] metrics:", metrics)

    if local_rank <= 0:
        plot_loss(trainer.state.log_history, out_dir)

    if isinstance(model, PeftModel):
        try:
            model.save_pretrained(out_dir, selected_adapters=[adapter_name] if adapter_name else None)
        except TypeError:
            model.save_pretrained(out_dir)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--bundles_dir", type=str, required=True)
    ap.add_argument("--stage", type=int, choices=[1,2], required=True)
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

    # 토크나이저
    tok = AutoTokenizer.from_pretrained(
        args.base_dir, 
        use_fast=True,
        local_files_only=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # 모델 로딩 (Trainer가 자동으로 GPU 분산)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("[model] Using BF16")
    else:
        dtype = torch.float16
        print("[model] Using FP16")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, 
        torch_dtype=dtype,
        local_files_only=True
    )
    
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass

    model = _reapply_passlayers_from_manifest(model, args.base_dir)

    with open(os.path.join(args.base_dir, "prune_log.json"), "r", encoding="utf-8") as f:
        log = json.load(f)
    B_idx, C_idx = log["split"]["B"], log["split"]["C"]

    # 데이터셋
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

        out_dir = os.path.join(args.out_adapters, "AB_lora")
        train_lora(model, tok, out_dir, train_ds, eval_ds, lr=args.lr, epochs=args.epochs, bs=args.bs, grad_acc=args.grad_acc, fp16=True, adapter_name="stageAB")
        
        export_adapter_pt_and_recipe(
            model, out_dir, "stageAB",
            base_dir=args.base_dir, bundles_dir=args.bundles_dir, stage="AB",
            trained_indices=AB_idx, tokenizer_dir=args.base_dir
        )

if __name__ == "__main__":
    main()