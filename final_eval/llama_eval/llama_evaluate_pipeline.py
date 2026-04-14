#!/usr/bin/env python3
"""
LLaMA Progressive LoRA – Pipeline Evaluator

메인 평가 구성:
  Stage1: A_merged + B,C=PassLayer
  Stage2: A_merged + B_merged + C=PassLayer
  Stage3: A_merged + B_merged + C_merged
  Dense:  original full model (baseline)

Usage:
# llama2-7b
  CUDA_VISIBLE_DEVICES=1 python -m final_eval.llama_eval.llama_evaluate_pipeline \
    --config ./final_eval/llama_eval/llama7b_eval_config.json --output_dir ./eval_results/llama2_7b_2

# llama2-13b
CUDA_VISIBLE_DEVICES=5 python -m final_eval.llama_eval.llama_evaluate_pipeline \
    --config ./final_eval/llama_eval/llama13b_eval_config.json --output_dir ./eval_results/llama2_13b

"""


import os, sys, json, re, inspect, argparse, gc, platform
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from peft import PeftModel
from datasets import load_dataset

try:
    from transformers.utils.hub import cached_file
except Exception:
    cached_file = None

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
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    try:
        import peft
        info["peft_version"] = peft.__version__
    except ImportError:
        pass
    try:
        import lm_eval
        info["lm_eval_version"] = getattr(lm_eval, "__version__", "unknown")
    except ImportError:
        pass
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
    """Count active (non-PassLayer) parameters and compute sparsity."""
    total_params = 0
    active_params = 0
    pass_layer_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        # PassLayer has no real parameters, but check by module type
        active_params += n

    # Count PassLayer positions separately by inspecting layers
    layers = _layers(model)
    n_pass = 0
    n_real = 0
    for i, layer in enumerate(layers):
        if isinstance(layer, PassLayer):
            n_pass += 1
        else:
            n_real += 1

    # For param count, total_params from named_parameters already excludes
    # PassLayer weights (PassLayer has zero parameters).
    # But we need to report what the Dense model's param count would be.
    return {
        "active_params": total_params,
        "active_params_M": round(total_params / 1e6, 2),
        "n_real_layers": n_real,
        "n_pass_layers": n_pass,
        "n_total_layers": len(layers),
        "layer_sparsity_pct": round(n_pass / len(layers) * 100, 1) if len(layers) > 0 else 0.0,
    }


# ════════════════════════════════════════════════════════════════
# Layer utilities (from llama_prune_lora.optimized_lora)
# ════════════════════════════════════════════════════════════════

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

def _ensure_original_layout(model, removed_indices, original_N):
    layers = _layers(model)
    cur_N = len(layers)
    removed = set(int(i) for i in removed_indices)
    kept = sorted(set(range(original_N)) - removed)
    use_t = _returns_tuple()
    dev = next(model.parameters()).device

    if cur_N == original_N:
        for i in removed_indices:
            layers[int(i)] = PassLayer(use_t).to(dev)
        return model, kept

    if cur_N != len(kept):
        raise ValueError(f"layer mismatch: {cur_N} vs compact={len(kept)} or sparse={original_N}")

    old = [layers[i] for i in range(cur_N)]
    new = [None] * original_N
    for pi, oi in enumerate(kept): new[oi] = old[pi]
    for i in removed_indices: new[int(i)] = PassLayer(use_t).to(dev)
    assert all(l is not None for l in new)

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model.model.layers = nn.ModuleList(new)
    elif hasattr(model, 'model') and hasattr(model.model, 'model'):
        model.model.model.layers = nn.ModuleList(new)
    else:
        raise RuntimeError("cannot find layers path")
    model.config.num_hidden_layers = original_N
    _invalidate(model)
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
        with open(meta) as f: return sorted(json.load(f).get("indices", []))
    return sorted(int(re.match(r"layer_(\d+)", fn).group(1))
                  for fn in os.listdir(bdir) if re.match(r"layer_\d+\.safetensors", fn))

def _rehydrate(model, bdir, indices, load_trace=None, component="rehydrate"):
    layers = _layers(model)
    dtype, dev = next(model.parameters()).dtype, next(model.parameters()).device
    loaded_files = []
    for i in indices:
        try: nl = LlamaDecoderLayer(model.config, layer_idx=int(i))
        except TypeError: nl = LlamaDecoderLayer(model.config)
        nl = nl.to(device=dev, dtype=dtype)
        layer_path = _pick_file(bdir, int(i))
        loaded_files.append({
            "kind": "layer_safetensors",
            "file": os.path.basename(layer_path),
            "path": os.path.abspath(layer_path),
            "layer_index": int(i),
        })
        raw = load_file(layer_path)
        sd = {k: v.to(device=dev, dtype=dtype) for k, v in _extract_sd(raw, int(i)).items()}
        try: nl.load_state_dict(sd, strict=True)
        except RuntimeError: nl.load_state_dict(sd, strict=False)
        layers[int(i)] = nl
    _emit_load_trace(load_trace, {
        "component": component,
        "loader": "safetensors.torch.load_file",
        "source": bdir,
        "resolved_source": _normalize_source_path(bdir),
        "files": loaded_files,
        "indices": [int(i) for i in indices],
    })

# ════════════════════════════════════════════════════════════════
# GPU utils
# ════════════════════════════════════════════════════════════════

def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0

def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()

# ════════════════════════════════════════════════════════════════
# Load trace helpers
# ════════════════════════════════════════════════════════════════

def _normalize_source_path(path):
    if isinstance(path, str) and os.path.exists(path):
        return os.path.abspath(path)
    return path

def _resolve_model_file(source, filename, local_files_only=None):
    if not source:
        return None
    if os.path.isdir(source):
        candidate = os.path.join(source, filename)
        return os.path.abspath(candidate) if os.path.isfile(candidate) else None
    if os.path.isfile(source):
        return os.path.abspath(source) if os.path.basename(source) == filename else None
    if cached_file is None:
        return None

    variants = [
        {
            "local_files_only": False if local_files_only is None else local_files_only,
            "_raise_exceptions_for_gated_repo": False,
            "_raise_exceptions_for_missing_entries": False,
            "_raise_exceptions_for_connection_errors": False,
        },
        {"local_files_only": False if local_files_only is None else local_files_only},
        {},
    ]
    for kwargs in variants:
        try:
            resolved = cached_file(source, filename, **kwargs)
            if resolved:
                return resolved
        except TypeError:
            continue
        except Exception:
            return None
    return None

def _resolve_weight_shards(source, index_path, local_files_only=None):
    try:
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
    except Exception:
        return []

    files = []
    for shard_name in dict.fromkeys(weight_map.values()):
        shard_path = _resolve_model_file(source, shard_name, local_files_only)
        if shard_path is None and index_path:
            shard_path = os.path.abspath(os.path.join(os.path.dirname(index_path), shard_name))
        files.append({
            "kind": "weight_shard",
            "file": shard_name,
            "path": shard_path,
        })
    return files

def _describe_from_pretrained_artifacts(source, local_files_only=None):
    artifacts = []
    for kind, filename in [
        ("config", "config.json"),
        ("generation_config", "generation_config.json"),
    ]:
        resolved = _resolve_model_file(source, filename, local_files_only)
        if resolved:
            artifacts.append({
                "kind": kind,
                "file": filename,
                "path": resolved,
            })

    for index_kind, index_name, weight_name in [
        ("weights_index", "model.safetensors.index.json", "model.safetensors"),
        ("weights_index", "pytorch_model.bin.index.json", "pytorch_model.bin"),
    ]:
        index_path = _resolve_model_file(source, index_name, local_files_only)
        if index_path:
            artifacts.append({
                "kind": index_kind,
                "file": index_name,
                "path": index_path,
            })
            artifacts.extend(_resolve_weight_shards(source, index_path, local_files_only))
            return artifacts

        weight_path = _resolve_model_file(source, weight_name, local_files_only)
        if weight_path:
            artifacts.append({
                "kind": "weights",
                "file": weight_name,
                "path": weight_path,
            })
            return artifacts

    return artifacts

def _emit_load_trace(load_trace, entry, max_console_files=8):
    if load_trace is None:
        return

    load_trace.append(entry)
    source = entry.get("resolved_source") or entry.get("source")
    print(f"  [LoadTrace] {entry.get('component', 'model')}: {source}")

    files = entry.get("files", [])
    for item in files[:max_console_files]:
        path = item.get("path") or item.get("file")
        if not path:
            continue
        suffix = f" (layer {item['layer_index']})" if "layer_index" in item else ""
        print(f"    - {item.get('kind', 'file')}: {path}{suffix}")
    if len(files) > max_console_files:
        print(f"    - ... {len(files) - max_console_files} more files recorded")

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

def _load_base(path, device, dtype, local_files_only=True, load_trace=None, component="base_model"):
    _emit_load_trace(load_trace, {
        "component": component,
        "loader": "AutoModelForCausalLM.from_pretrained",
        "source": path,
        "resolved_source": _normalize_source_path(path),
        "local_files_only": local_files_only,
        "files": _describe_from_pretrained_artifacts(path, local_files_only),
    })
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, device_map=None, local_files_only=local_files_only)
    model.to(device)
    model.config.use_cache = True
    model.eval()
    return model

def load_stage1(cfg, device, dtype, load_trace=None):
    """Stage 1: A_merged + B,C = PassLayer"""
    model = _load_base(
        cfg["A_merged_dir"], device, dtype,
        load_trace=load_trace, component="A_merged_base",
    )
    removed = sorted(set(cfg["B_indices"] + cfg["C_indices"]))
    model, _ = _ensure_original_layout(model, removed, cfg["original_N"])
    return model

def load_stage2(cfg, device, dtype, load_trace=None):
    """Stage 2: A_merged + B_merged + C = PassLayer"""
    model = _load_base(
        cfg["A_merged_dir"], device, dtype,
        load_trace=load_trace, component="A_merged_base",
    )
    removed = sorted(set(cfg["B_indices"] + cfg["C_indices"]))
    model, _ = _ensure_original_layout(model, removed, cfg["original_N"])
    _rehydrate(model, cfg["B_merged_dir"], cfg["B_indices"], load_trace=load_trace, component="B_rehydrate")
    return model

def load_stage3(cfg, device, dtype, load_trace=None):
    """Stage 3: A_merged + B_merged + C_merged"""
    if cfg.get("final_merged_dir"):
        return _load_base(
            cfg["final_merged_dir"], device, dtype,
            load_trace=load_trace, component="final_merged_base",
        )
    model = _load_base(
        cfg["A_merged_dir"], device, dtype,
        load_trace=load_trace, component="A_merged_base",
    )
    removed = sorted(set(cfg["B_indices"] + cfg["C_indices"]))
    model, _ = _ensure_original_layout(model, removed, cfg["original_N"])
    _rehydrate(model, cfg["B_merged_dir"], cfg["B_indices"], load_trace=load_trace, component="B_rehydrate")
    if cfg.get("C_merged_dir"):
        _rehydrate(model, cfg["C_merged_dir"], cfg["C_indices"], load_trace=load_trace, component="C_rehydrate")
    else:
        _rehydrate(
            model, cfg["C_bundle_dir"], cfg["C_indices"],
            load_trace=load_trace, component="C_bundle_rehydrate",
        )
    return model

def load_dense(cfg, device, dtype, load_trace=None):
    """Dense baseline"""
    repo_id = _require_hf_repo_id(cfg["original_model_dir"])
    return _load_base(
        repo_id, device, dtype,
        local_files_only=False, load_trace=load_trace, component="dense_base",
    )

MAIN_LOADERS = {
    "Stage1": load_stage1,
    "Stage2": load_stage2,
    "Stage3": load_stage3,
    "Dense":  load_dense,
}

# ════════════════════════════════════════════════════════════════
# 1. Perplexity (WikiText-2)
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, max_seq_len=2048, stride=512, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    nlls, n_tokens = [], 0
    orig_use_cache = model.config.use_cache
    model.config.use_cache = False

    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + max_seq_len, input_ids.size(1))
        chunk = input_ids[:, begin:end]
        trg = chunk.clone()
        if begin > 0:
            trg[:, :max_seq_len - stride] = -100
        outputs = model(input_ids=chunk, labels=trg)
        n_valid = (trg != -100).sum().item()
        nlls.append(outputs.loss.float().item() * n_valid)
        n_tokens += n_valid
        if end == input_ids.size(1): break

    model.config.use_cache = orig_use_cache
    ppl = float(np.exp(sum(nlls) / n_tokens))
    return {
        "perplexity": round(ppl, 2),
        "n_tokens": n_tokens,
        "max_seq_len": max_seq_len,
        "stride": stride,
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
    # Include task-level configs from lm-eval if available
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

    # ── Include env info at top level ──
    output = {
        "env": env_info,
        "results": all_results,
    }

    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {json_path}")

    # ── Also save raw lm-eval metadata per config ──
    for cfg_name, r in all_results.items():
        if "load_trace" in r:
            trace_path = os.path.join(output_dir, f"load_trace_{cfg_name}.json")
            with open(trace_path, "w") as f:
                json.dump(r["load_trace"], f, indent=2, ensure_ascii=False)
            print(f"[saved] {trace_path}")
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
        "original_model_dir": "meta-llama/Llama-2-7b-chat-hf",
        "original_N": 32,
        "A_dir": "./7b_results/pruning/A",
        "A_merged_dir": "./merged_models/A_merged",
        "B_bundle_dir": "./7b_results/pruning/bundles/B",
        "B_merged_dir": "./merged_models/B_merged",
        "C_bundle_dir": "./7b_results/pruning/bundles/C",
        "C_merged_dir": "./merged_models/C_merged",
        "final_merged_dir": None,
        "B_indices": [], "C_indices": [],
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
    with open(path, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"[generated] {path}")

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
# Main
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="LLaMA Progressive LoRA Evaluator")
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", default="./eval_results_llama")
    p.add_argument("--only", default=None)
    p.add_argument("--skip_accuracy", action="store_true")
    p.add_argument("--gen_config", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    if args.gen_config:
        generate_sample_config(args.gen_config); return

    with open(args.config) as f:
        cfg = json.load(f)

    if not cfg.get("B_indices") and cfg.get("B_bundle_dir"):
        cfg["B_indices"] = _load_bundle_indices(cfg["B_bundle_dir"])
    if not cfg.get("C_indices") and cfg.get("C_bundle_dir"):
        cfg["C_indices"] = _load_bundle_indices(cfg["C_bundle_dir"])

    config_names = [c.strip() for c in args.only.split(",")] if args.only else list(MAIN_LOADERS.keys())

    # ── Dtype: explicit from config, no auto-detection ──
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

    device = torch.device(os.environ.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")

    tok_dir = cfg.get("original_model_dir")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, local_files_only=False)
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left" # gemma와 동일하게 수정

    num_fewshot = eval_cfg.get("num_fewshot", 0)
    all_results = {}

    print(f"\n{'='*70}\n  LLaMA Pipeline Evaluation\n  Configs: {config_names}\n  dtype: {dtype_str}  seed: {seed}  num_fewshot: {num_fewshot}\n{'='*70}\n")

    for cfg_name in config_names:
        loader = MAIN_LOADERS.get(cfg_name)
        if not loader: print(f"[SKIP] {cfg_name}"); continue
        if cfg_name == "Dense" and not cfg.get("original_model_dir"): print("[SKIP] Dense"); continue

        print(f"\n{'─'*60}\n  {cfg_name}\n{'─'*60}")
        reset_gpu_stats()
        load_trace = []
        model = loader(cfg, device, dtype, load_trace=load_trace)
        result = {"load_trace": load_trace}

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
