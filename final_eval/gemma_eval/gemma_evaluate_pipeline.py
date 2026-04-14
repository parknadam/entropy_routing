#!/usr/bin/env python3
"""
Gemma Progressive LoRA – Pipeline Evaluator (ASPLOS-ready)

메인 평가 구성:
  Stage1: A_merged + B,C=GemmaPassLayer
  Stage2: A_merged + B_merged + C=GemmaPassLayer
  Stage3: A_merged + B_merged + C_merged
  Dense:  original full model (baseline)

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m final_eval.gemma_eval.gemma_evaluate_pipeline \
    --config ./final_eval/gemma_eval/gamma7b_eval_config.json --output_dir ./eval_results/eval_results_gemma_it
"""

import os, sys, json, re, inspect, argparse, gc, math, platform
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

try:
    from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
except Exception:
    GemmaDecoderLayer = None

KST = timezone(timedelta(hours=9))
CANON_PATH = "model.layers"

# ════════════════════════════════════════════════════════════════
# Environment info collector
# ════════════════════════════════════════════════════════════════

def collect_env_info(dtype_str: str) -> dict:
    """Collect reproducibility-critical environment information."""
    info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
        "transformers_version": None,
        "peft_version": None,
        "lm_eval_version": None,
        "gpu_name": None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_memory_gb": None,
        "dtype": dtype_str,
        "os": f"{platform.system()} {platform.release()}",
        "timestamp": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
    }
    try:
        import transformers; info["transformers_version"] = transformers.__version__
    except ImportError: pass
    try:
        import peft; info["peft_version"] = peft.__version__
    except ImportError: pass
    try:
        import lm_eval; info["lm_eval_version"] = getattr(lm_eval, "__version__", "unknown")
    except ImportError: pass
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1024**3, 1
        )
    return info

# ════════════════════════════════════════════════════════════════
# Active parameter / sparsity counter
# ════════════════════════════════════════════════════════════════

def count_active_params(model) -> dict:
    """Count active (non-GemmaPassLayer) parameters and compute sparsity."""
    total_params = sum(p.numel() for p in model.parameters())

    layers = _get_layers(model)
    n_pass = 0
    n_real = 0
    for layer in layers:
        if isinstance(layer, GemmaPassLayer):
            n_pass += 1
        else:
            n_real += 1

    return {
        "active_params": total_params,
        "active_params_M": round(total_params / 1e6, 2),
        "n_real_layers": n_real,
        "n_pass_layers": n_pass,
        "n_total_layers": len(layers),
        "layer_sparsity_pct": round(n_pass / len(layers) * 100, 1) if len(layers) > 0 else 0.0,
    }

# ════════════════════════════════════════════════════════════════
# Dtype resolver
# ════════════════════════════════════════════════════════════════

def resolve_dtype(cfg):
    """Resolve dtype from config. No auto-detection — must be explicit."""
    dtype_str = cfg.get("eval", {}).get("dtype", None)
    if dtype_str is None:
        raise ValueError(
            "eval.dtype must be explicitly set in config (e.g. 'bfloat16' or 'float16'). "
            "Auto-detection is disabled for reproducibility."
        )
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from: {list(dtype_map.keys())}")
    return dtype_map[dtype_str], dtype_str

# ════════════════════════════════════════════════════════════════
# Gemma layer utilities (from gemma_prune_lora)
# ════════════════════════════════════════════════════════════════

def _get_layers(model):
    """Gemma: model.model.layers"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        return model.model.model.layers
    raise RuntimeError("Gemma layers not found (expected model.model.layers)")

def _set_layers(model, new_layers):
    layer_list = nn.ModuleList(list(new_layers))
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = layer_list
    elif hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        model.model.model.layers = layer_list
    else:
        raise RuntimeError("Cannot set Gemma layers")

def _detect_return_tuple(model):
    """Gemma의 현재 HF 구현은 tensor 반환이 기본 (return_tuple=False)"""
    try:
        core = model.model if hasattr(model, "model") else model
        src = inspect.getsource(core.forward)
        if "layer_outputs[0]" in src or "layer_outputs = decoder_layer" in src:
            return True
        if "hidden_states = decoder_layer" in src and "layer_outputs[0]" not in src:
            return False
    except Exception:
        pass
    return False  # Gemma default: tensor

class GemmaPassLayer(nn.Module):
    def __init__(self, hidden_size=0, return_tuple=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.return_tuple = return_tuple
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, cache_position=None,
                position_embeddings=None, output_attentions=False, **kw):
        if not self.return_tuple:
            return hidden_states
        outputs = (hidden_states,)
        if output_attentions: outputs += (None,)
        if use_cache: outputs += (past_key_values,)
        return outputs

def _ensure_original_layout(model, removed_indices, original_N):
    layers = _get_layers(model)
    cur_N = len(layers)
    removed = set(int(i) for i in removed_indices)
    kept = sorted(set(range(original_N)) - removed)
    ret_tuple = _detect_return_tuple(model)
    dev = next(model.parameters()).device
    hs = model.config.hidden_size

    if cur_N == original_N:
        for i in removed:
            layers[int(i)] = GemmaPassLayer(hs, ret_tuple).to(dev)
        return model, kept

    if cur_N != len(kept):
        raise ValueError(f"layer mismatch: {cur_N} vs compact={len(kept)} or sparse={original_N}")

    old = [layers[i] for i in range(cur_N)]
    new = [None] * original_N
    for pi, oi in enumerate(kept): new[oi] = old[pi]
    for i in removed: new[int(i)] = GemmaPassLayer(hs, ret_tuple).to(dev)
    assert all(l is not None for l in new)
    _set_layers(model, new)
    model.config.num_hidden_layers = original_N
    return model, kept

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
        with open(meta) as f:
            indices = json.load(f).get("indices", []) or json.load(open(meta)).get("layer_indices", [])
            if indices: return sorted(int(i) for i in indices)
    if not os.path.isdir(bdir): return []
    return sorted(int(re.match(r"layer_(\d+)", fn).group(1))
                  for fn in os.listdir(bdir) if re.match(r"layer_\d+\.safetensors", fn))

def _rehydrate(model, bdir, indices):
    if GemmaDecoderLayer is None:
        raise RuntimeError("GemmaDecoderLayer import failed")
    layers = _get_layers(model)
    dtype, dev = next(model.parameters()).dtype, next(model.parameters()).device
    for i in indices:
        try: nl = GemmaDecoderLayer(model.config, int(i))
        except TypeError: nl = GemmaDecoderLayer(model.config)
        nl = nl.to(device=dev, dtype=dtype)
        raw = load_file(_pick_file(bdir, int(i)))
        sd = {k: v.to(device=dev, dtype=dtype) for k, v in _extract_sd(raw, int(i)).items()}
        try: nl.load_state_dict(sd, strict=True)
        except RuntimeError: nl.load_state_dict(sd, strict=False)
        layers[int(i)] = nl

# ════════════════════════════════════════════════════════════════
# GPU utils
# ════════════════════════════════════════════════════════════════

def gpu_mem_mb():
    return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    gc.collect()

# ════════════════════════════════════════════════════════════════
# Model loaders
# ════════════════════════════════════════════════════════════════

def _require_hf_repo_id(path, key="original_model_dir"):
    repo_id = str(path).strip()
    if not repo_id:
        raise ValueError(f"{key} must be a non-empty Hugging Face repo ID")
    if repo_id.startswith(("./", "../", "/")) or os.path.exists(repo_id):
        raise ValueError(f"{key} must be a Hugging Face repo ID for Dense baseline, not a local path: {repo_id}")
    return repo_id

def _load_base(path, device, dtype, local_files_only=None):
    attempts = [
        {"torch_dtype": dtype, "attn_implementation": "eager", "trust_remote_code": True},
        {"torch_dtype": dtype, "trust_remote_code": True},
    ]
    for kw in attempts:
        if local_files_only is not None:
            kw["local_files_only"] = local_files_only
        try:
            model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True, **kw)
            model.to(device); model.config.use_cache = True; model.eval()
            return model
        except TypeError:
            continue
    raise RuntimeError(f"Failed to load model from {path}")

def load_stage1(cfg, device, dtype):
    model = _load_base(cfg["A_merged_dir"], device, dtype)
    removed = sorted(set(cfg["B_indices"] + cfg["C_indices"]))
    model, _ = _ensure_original_layout(model, removed, cfg["original_N"])
    return model

def load_stage2(cfg, device, dtype):
    model = _load_base(cfg["A_merged_dir"], device, dtype)
    removed = sorted(set(cfg["B_indices"] + cfg["C_indices"]))
    model, _ = _ensure_original_layout(model, removed, cfg["original_N"])
    _rehydrate(model, cfg["B_merged_dir"], cfg["B_indices"])
    return model

def load_stage3(cfg, device, dtype):
    if cfg.get("final_merged_dir"):
        return _load_base(cfg["final_merged_dir"], device, dtype)
    model = _load_base(cfg["A_merged_dir"], device, dtype)
    removed = sorted(set(cfg["B_indices"] + cfg["C_indices"]))
    model, _ = _ensure_original_layout(model, removed, cfg["original_N"])
    _rehydrate(model, cfg["B_merged_dir"], cfg["B_indices"])
    if cfg.get("C_merged_dir"):
        _rehydrate(model, cfg["C_merged_dir"], cfg["C_indices"])
    else:
        _rehydrate(model, cfg["C_bundle_dir"], cfg["C_indices"])
    return model

def load_dense(cfg, device, dtype):
    repo_id = _require_hf_repo_id(cfg["original_model_dir"])
    return _load_base(repo_id, device, dtype, local_files_only=False)

MAIN_LOADERS = {
    "Stage1": load_stage1, "Stage2": load_stage2,
    "Stage3": load_stage3, "Dense": load_dense,
}

# ════════════════════════════════════════════════════════════════
# 1. Perplexity (WikiText-2)
# ════════════════════════════════════════════════════════════════

def _strip_leading_bos(input_ids, bos_token_id):
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    ids = ids.long().cpu()
    if bos_token_id is None or ids.numel() == 0:
        return ids
    if int(ids[0]) == int(bos_token_id):
        return ids[1:]
    return ids

def _load_wikitext_test_ids(tokenizer):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    return tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0].long()

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, max_seq_len=2048, stride=512, seed=42):
    del stride  # Gemma uses BOS-prepended chunks, not sliding window
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if max_seq_len < 2:
        raise ValueError("Gemma perplexity evaluation requires max_seq_len >= 2")

    device = next(model.parameters()).device
    input_ids = _load_wikitext_test_ids(tokenizer)
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        raise RuntimeError("Gemma perplexity evaluation requires tokenizer.bos_token_id")

    ids = _strip_leading_bos(input_ids, bos_token_id)
    if ids.numel() == 0:
        return {"perplexity": float("nan"), "n_tokens": 0}

    step = max(1, int(max_seq_len) - 1)
    sum_nll = 0.0
    n_tokens = 0
    orig = model.config.use_cache
    model.config.use_cache = False
    bos = torch.tensor([[int(bos_token_id)]], dtype=torch.long, device=device)

    try:
        for start in range(0, ids.numel(), step):
            content = ids[start : start + step]
            if content.numel() == 0:
                break

            chunk = torch.cat([bos, content.unsqueeze(0).to(device)], dim=1)
            logits = model(
                input_ids=chunk,
                attention_mask=torch.ones_like(chunk, dtype=torch.long),
                use_cache=False,
            ).logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()
            vocab_size = shift_logits.size(-1)
            loss_tok = F.cross_entropy(
                shift_logits.float().view(-1, vocab_size),
                shift_labels.view(-1),
                reduction="none",
            ).view_as(shift_labels).float()

            sum_nll += float(loss_tok.sum().item())
            n_tokens += int(shift_labels.numel())
    finally:
        model.config.use_cache = orig

    if n_tokens == 0:
        return {"perplexity": float("nan"), "n_tokens": 0}

    mean_nll = sum_nll / n_tokens
    return {
        "perplexity": round(float(math.exp(mean_nll)), 2),
        "n_tokens": n_tokens,
        "max_seq_len": max_seq_len,
        "dataset": "wikitext-2-raw-v1",
        "seed": seed,
    }

# ════════════════════════════════════════════════════════════════
# 2. Zero-shot (lm-evaluation-harness, 0-shot)
# ════════════════════════════════════════════════════════════════

def evaluate_zero_shot(model, tokenizer, tasks=None, batch_size="auto",
                       num_fewshot=0, seed=42):
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("[WARN] lm-eval not installed")
        return {}, {}
    if tasks is None:
        tasks = ["arc_easy","arc_challenge","hellaswag","winogrande","piqa","boolq","openbookqa"]

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=num_fewshot,
        batch_size=batch_size, log_samples=False,
        random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed,
    )

    # ── Parsed summary (acc + stderr) ──
    parsed = {}
    for t, r in results.get("results", {}).items():
        acc = r.get("acc_norm,none", r.get("acc,none"))
        stderr = r.get("acc_norm_stderr,none", r.get("acc_stderr,none"))
        if acc is not None:
            entry = {"acc": round(float(acc) * 100, 2)}
            if stderr is not None:
                entry["stderr"] = round(float(stderr) * 100, 2)
            parsed[t] = entry

    if parsed:
        accs = [v["acc"] for v in parsed.values()]
        parsed["avg"] = {"acc": round(sum(accs) / len(accs), 2)}

    # ── Raw results for full reproducibility ──
    raw_meta = {
        "num_fewshot": num_fewshot,
        "seed": seed,
        "tasks": tasks,
        "lm_eval_version": getattr(lm_eval, "__version__", "unknown"),
    }
    if "configs" in results:
        raw_meta["task_configs"] = results["configs"]
    if "versions" in results:
        raw_meta["task_versions"] = results["versions"]
    if "n-shot" in results:
        raw_meta["n_shot"] = results["n-shot"]

    return parsed, raw_meta

# ════════════════════════════════════════════════════════════════
# 3. Results aggregation
# ════════════════════════════════════════════════════════════════

def aggregate_results(all_results, output_dir, env_info):
    os.makedirs(output_dir, exist_ok=True)

    # ── Full JSON with env info ──
    output = {
        "env": env_info,
        "results": all_results,
    }
    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {json_path}")

    # ── Raw lm-eval metadata per config ──
    for cfg_name, r in all_results.items():
        if "zero_shot_raw" in r:
            raw_path = os.path.join(output_dir, f"lm_eval_raw_{cfg_name}.json")
            with open(raw_path, "w") as f:
                json.dump(r["zero_shot_raw"], f, indent=2, ensure_ascii=False)
            print(f"[saved] {raw_path}")

    configs = list(all_results.keys())

    # ── Accuracy table (PPL + zero-shot with stderr) ──
    acc_path = os.path.join(output_dir, "accuracy_table.md")
    all_tasks = set()
    for c in configs:
        all_tasks.update(k for k in all_results[c].get("zero_shot", {}).keys() if k != "avg")
    task_list = sorted(all_tasks) + (
        ["avg"] if any("avg" in all_results[c].get("zero_shot", {}) for c in configs) else []
    )

    with open(acc_path, "w") as f:
        header = "| Config | PPL(↓) |" + "".join(f" {t} |" for t in task_list)
        sep = "|--------|--------|" + "".join("--------|" for _ in task_list)
        f.write(header + "\n" + sep + "\n")
        for c in configs:
            r = all_results[c]
            ppl = r.get("perplexity", {}).get("perplexity", "-")
            zs = r.get("zero_shot", {})
            cells = []
            for t in task_list:
                entry = zs.get(t, None)
                if entry is None:
                    cells.append("-")
                elif isinstance(entry, dict):
                    acc_str = str(entry["acc"])
                    if "stderr" in entry:
                        acc_str += f"±{entry['stderr']}"
                    cells.append(acc_str)
                else:
                    cells.append(str(entry))
            row = f"| {c} | {ppl} |" + "".join(f" {cell} |" for cell in cells)
            f.write(row + "\n")
    print(f"[saved] {acc_path}")

    # ── Parameter / sparsity table ──
    param_path = os.path.join(output_dir, "param_table.md")
    with open(param_path, "w") as f:
        f.write("| Config | Active Params (M) | Real Layers | Pass Layers | Layer Sparsity (%) |\n")
        f.write("|--------|-------------------|-------------|-------------|--------------------|\n")
        for c in configs:
            p = all_results[c].get("params", {})
            if not p:
                f.write(f"| {c} | - | - | - | - |\n")
                continue
            f.write(
                f"| {c} "
                f"| {p.get('active_params_M', '-')} "
                f"| {p.get('n_real_layers', '-')} "
                f"| {p.get('n_pass_layers', '-')} "
                f"| {p.get('layer_sparsity_pct', '-')} |\n"
            )
    print(f"[saved] {param_path}")

# ════════════════════════════════════════════════════════════════
# 4. Config generator
# ════════════════════════════════════════════════════════════════

def generate_sample_config(path):
    sample = {
        "original_model_dir": "google/gemma-7b", "original_N": 28,
        "A_merged_dir": "./merged_gemma/A_merged",
        "B_bundle_dir": "./gemma_results/pruning/bundles/B",
        "B_merged_dir": "./merged_gemma/B_merged",
        "C_bundle_dir": "./gemma_results/pruning/bundles/C",
        "C_merged_dir": "./merged_gemma/C_merged",
        "final_merged_dir": None, "B_indices": [], "C_indices": [],
        "eval": {
            "dtype": "bfloat16",
            "seed": 42,
            "seq_len": 2048,
            "ppl_stride": 512,
            "num_fewshot": 0,
            "zero_shot_tasks": [
                "arc_easy", "arc_challenge", "hellaswag",
                "winogrande", "piqa", "boolq", "openbookqa"
            ],
            "batch_size": "auto",
        }
    }
    with open(path, "w") as f: json.dump(sample, f, indent=2)

# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Gemma Progressive LoRA Evaluator")
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", default="./eval_results_gemma")
    p.add_argument("--only", default=None)
    p.add_argument("--skip_accuracy", action="store_true")
    p.add_argument("--gen_config", default=None)
    args = p.parse_args()
    if args.gen_config: generate_sample_config(args.gen_config); return

    with open(args.config) as f: cfg = json.load(f)
    if not cfg.get("B_indices") and cfg.get("B_bundle_dir"): cfg["B_indices"] = _load_bundle_indices(cfg["B_bundle_dir"])
    if not cfg.get("C_indices") and cfg.get("C_bundle_dir"): cfg["C_indices"] = _load_bundle_indices(cfg["C_bundle_dir"])

    config_names = [c.strip() for c in args.only.split(",")] if args.only else list(MAIN_LOADERS.keys())

    # ── Dtype: explicit from config ──
    dtype, dtype_str = resolve_dtype(cfg)

    # ── Seed: explicit from config ──
    eval_cfg = cfg.get("eval", {})
    seed = eval_cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # ── Environment info ──
    env_info = collect_env_info(dtype_str)
    env_info["seed"] = seed
    env_info["config_file"] = os.path.abspath(args.config)
    print(f"\n[ENV] {json.dumps(env_info, indent=2)}\n")

    device = torch.device(os.environ.get("DEVICE","cuda:0") if torch.cuda.is_available() else "cpu")

    # Tokenizer: always from original model for cross-stage consistency.
    tok_dir = cfg.get("original_model_dir")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, trust_remote_code=True, local_files_only=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    tokenizer.padding_side = "left"

    if dtype == torch.float16:
        print("[WARN] Gemma PPL can be unstable in fp16; bf16 recommended.")

    num_fewshot = eval_cfg.get("num_fewshot", 0)
    all_results = {}

    print(f"\n{'='*70}\n  Gemma Pipeline Evaluation\n  Configs: {config_names}\n  dtype: {dtype_str}  seed: {seed}  num_fewshot: {num_fewshot}\n{'='*70}\n")

    for cfg_name in config_names:
        loader = MAIN_LOADERS.get(cfg_name)
        if not loader: continue
        if cfg_name == "Dense" and not cfg.get("original_model_dir"): continue

        print(f"\n{'─'*60}\n  {cfg_name}\n{'─'*60}")
        reset_gpu_stats()
        model = loader(cfg, device, dtype)
        result = {}

        # ── Sync pad/eos token ids to model config ──
        if tokenizer.pad_token_id is not None and getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None and getattr(model.generation_config, "pad_token_id", None) is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None and getattr(model.generation_config, "eos_token_id", None) is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id

        # ── Parameter count ──
        print("  [Params] counting active parameters ...")
        result["params"] = count_active_params(model)
        print(f"    → {result['params']['active_params_M']}M params, "
              f"{result['params']['n_real_layers']} real / "
              f"{result['params']['n_pass_layers']} pass layers "
              f"({result['params']['layer_sparsity_pct']}% sparsity)")

        # ── Peak memory after load ──
        result["peak_memory_mb"] = round(gpu_mem_mb(), 1)

        if not args.skip_accuracy:
            print("  [PPL] ...")
            result["perplexity"] = evaluate_perplexity(
                model, tokenizer,
                eval_cfg.get("seq_len", 2048),
                eval_cfg.get("ppl_stride", 512),
                seed=seed,
            )
            print(f"    → {result['perplexity']['perplexity']}")

            print("  [Zero-shot] ...")
            parsed, raw_meta = evaluate_zero_shot(
                model, tokenizer,
                eval_cfg.get("zero_shot_tasks"),
                eval_cfg.get("batch_size", "auto"),
                num_fewshot=num_fewshot,
                seed=seed,
            )
            result["zero_shot"] = parsed
            result["zero_shot_raw"] = raw_meta
            for t, v in parsed.items():
                if isinstance(v, dict):
                    acc_str = f"{v['acc']}"
                    if "stderr" in v:
                        acc_str += f" ± {v['stderr']}"
                    print(f"    {t}: {acc_str}")
                else:
                    print(f"    {t}: {v}")

        all_results[cfg_name] = result
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    aggregate_results(all_results, args.output_dir, env_info)
    print(f"\n{'='*70}\n  Done! → {args.output_dir}/\n{'='*70}")

if __name__ == "__main__":
    main()