#!/usr/bin/env python3
"""
Gemma Progressive LoRA – Pipeline Evaluator

메인 평가 구성:
  Stage1: A_merged + B,C=GemmaPassLayer
  Stage2: A_merged + B_merged + C=GemmaPassLayer
  Stage3: A_merged + B_merged + C_merged
  Dense:  original full model (baseline)

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m gemma_prune_lora.evaluate_pipeline \
    --config eval_config.json --output_dir ./eval_results_gemma
"""

import os, sys, json, re, time, inspect, argparse, gc
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import torch
import torch.nn as nn
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

def _load_base(path, device, dtype):
    attempts = [
        {"torch_dtype": dtype, "attn_implementation": "eager", "trust_remote_code": True},
        {"torch_dtype": dtype, "trust_remote_code": True},
    ]
    for kw in attempts:
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
    return _load_base(cfg["original_model_dir"], device, dtype)

MAIN_LOADERS = {
    "Stage1": load_stage1, "Stage2": load_stage2,
    "Stage3": load_stage3, "Dense": load_dense,
}

# ════════════════════════════════════════════════════════════════
# Evaluators
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, max_seq_len=2048, stride=512):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    nlls, n_tokens = [], 0
    orig = model.config.use_cache; model.config.use_cache = False
    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + max_seq_len, input_ids.size(1))
        chunk = input_ids[:, begin:end]
        trg = chunk.clone()
        if begin > 0: trg[:, :max_seq_len - stride] = -100
        out = model(input_ids=chunk, labels=trg)
        n_valid = (trg != -100).sum().item()
        nlls.append(out.loss.float().item() * n_valid); n_tokens += n_valid
        if end == input_ids.size(1): break
    model.config.use_cache = orig
    return {"perplexity": round(float(np.exp(sum(nlls) / n_tokens)), 2), "n_tokens": n_tokens}

def evaluate_zero_shot(model, tokenizer, tasks=None, batch_size="auto"):
    try:
        import lm_eval; from lm_eval.models.huggingface import HFLM
    except ImportError: return {}
    if tasks is None:
        tasks = ["arc_easy","arc_challenge","hellaswag","winogrande","piqa","boolq","openbookqa"]
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0,
                                       batch_size=batch_size, log_samples=False)
    parsed = {}
    for t, r in results.get("results", {}).items():
        acc = r.get("acc_norm,none", r.get("acc,none"))
        if acc is not None: parsed[t] = round(float(acc) * 100, 2)
    if parsed: parsed["avg"] = round(sum(parsed.values()) / len(parsed), 2)
    return parsed

@torch.no_grad()
def benchmark_system(model, tokenizer, prompt_lengths=[128,512,1024],
                     gen_tokens=128, n_warmup=3, n_repeat=10):
    device = next(model.parameters()).device
    results = {}
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    try:
        layers = _get_layers(model)
        pass_p = sum(p.numel() for l in layers if isinstance(l, GemmaPassLayer) for p in l.parameters())
        active_params = (sum(p.numel() for p in model.parameters()) - pass_p) / 1e6
    except: active_params = total_params

    for pl in prompt_lengths:
        ids = torch.randint(100, 30000, (1, pl), device=device)
        attn = torch.ones_like(ids)
        for _ in range(n_warmup):
            model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=1, do_sample=False)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        ttft_times = []
        for _ in range(n_repeat):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=1, do_sample=False)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            ttft_times.append((time.perf_counter() - t0) * 1000)
        ttft = float(np.mean(ttft_times))
        for _ in range(n_warmup):
            model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=gen_tokens, do_sample=False)
        gen_times, mems = [], []
        for _ in range(n_repeat):
            reset_gpu_stats()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=gen_tokens, do_sample=False)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            gen_times.append((time.perf_counter() - t0) * 1000); mems.append(gpu_mem_mb())
        ag = out.shape[1] - pl; avg_gen = float(np.mean(gen_times))
        results[f"prompt_{pl}"] = {
            "ttft_ms": round(ttft, 2), "decode_ms_per_token": round((avg_gen - ttft) / max(ag-1,1), 2),
            "throughput_tok_s": round(ag / (avg_gen / 1000), 1),
            "gen_total_ms": round(avg_gen, 2), "peak_mem_mb": round(max(mems), 1),
        }
    results["model_info"] = {"total_params_M": round(total_params,1), "active_params_M": round(active_params,1),
                              "pruning_ratio_pct": round((1 - active_params/total_params)*100,1) if total_params>0 else 0}
    return results

# ════════════════════════════════════════════════════════════════
# Results & Config
# ════════════════════════════════════════════════════════════════

def aggregate_results(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    configs = list(all_results.keys())
    all_tasks = set()
    for c in configs: all_tasks.update(all_results[c].get("zero_shot", {}).keys())
    all_tasks.discard("avg")
    task_list = sorted(all_tasks) + (["avg"] if any("avg" in all_results[c].get("zero_shot",{}) for c in configs) else [])
    with open(os.path.join(output_dir, "accuracy_table.md"), "w") as f:
        f.write("| Config | PPL(↓) |" + "".join(f" {t} |" for t in task_list) + "\n")
        f.write("|--------|--------|" + "".join("--------|" for _ in task_list) + "\n")
        for c in configs:
            r = all_results[c]; zs = r.get("zero_shot", {})
            f.write(f"| {c} | {r.get('perplexity',{}).get('perplexity','-')} |" + "".join(f" {zs.get(t,'-')} |" for t in task_list) + "\n")
    print(f"[saved] {output_dir}/")

def generate_sample_config(path):
    sample = {
        "original_model_dir": "google/gemma-7b", "original_N": 28,
        "A_merged_dir": "./merged_gemma/A_merged",
        "B_bundle_dir": "./gemma_results/pruning/bundles/B",
        "B_merged_dir": "./merged_gemma/B_merged",
        "C_bundle_dir": "./gemma_results/pruning/bundles/C",
        "C_merged_dir": "./merged_gemma/C_merged",
        "final_merged_dir": None, "B_indices": [], "C_indices": [],
        "eval": {"seq_len": 2048, "ppl_stride": 512,
                 "zero_shot_tasks": ["arc_easy","arc_challenge","hellaswag","winogrande","piqa","boolq","openbookqa"],
                 "system_prompt_lengths": [128,512,1024], "system_gen_tokens": 128,
                 "system_n_warmup": 3, "system_n_repeat": 10, "batch_size": "auto"}
    }
    with open(path, "w") as f: json.dump(sample, f, indent=2)

# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Gemma Progressive LoRA Evaluator")
    p.add_argument("--config", required=True); p.add_argument("--output_dir", default="./eval_results_gemma")
    p.add_argument("--only", default=None); p.add_argument("--skip_accuracy", action="store_true")
    p.add_argument("--skip_system", action="store_true"); p.add_argument("--gen_config", default=None)
    args = p.parse_args()
    if args.gen_config: generate_sample_config(args.gen_config); return

    with open(args.config) as f: cfg = json.load(f)
    if not cfg.get("B_indices") and cfg.get("B_bundle_dir"): cfg["B_indices"] = _load_bundle_indices(cfg["B_bundle_dir"])
    if not cfg.get("C_indices") and cfg.get("C_bundle_dir"): cfg["C_indices"] = _load_bundle_indices(cfg["C_bundle_dir"])

    config_names = [c.strip() for c in args.only.split(",")] if args.only else list(MAIN_LOADERS.keys())
    device = torch.device(os.environ.get("DEVICE","cuda:0") if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    tok_dir = cfg.get("A_merged_dir") or cfg.get("original_model_dir")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token

    if dtype == torch.float16:
        print("[WARN] Gemma PPL can be unstable in fp16; bf16 recommended.")

    eval_cfg = cfg.get("eval", {}); all_results = {}
    print(f"\n{'='*70}\n  Gemma Pipeline Evaluation\n  Configs: {config_names}\n{'='*70}\n")

    for cfg_name in config_names:
        loader = MAIN_LOADERS.get(cfg_name)
        if not loader: continue
        if cfg_name == "Dense" and not cfg.get("original_model_dir"): continue
        print(f"\n{'─'*60}\n  {cfg_name}\n{'─'*60}")
        reset_gpu_stats(); model = loader(cfg, device, dtype); result = {}
        if not args.skip_accuracy:
            result["perplexity"] = evaluate_perplexity(model, tokenizer, eval_cfg.get("seq_len",2048), eval_cfg.get("ppl_stride",512))
            print(f"    PPL = {result['perplexity']['perplexity']}")
            result["zero_shot"] = evaluate_zero_shot(model, tokenizer, eval_cfg.get("zero_shot_tasks"), eval_cfg.get("batch_size","auto"))
            for t,v in result["zero_shot"].items(): print(f"    {t}: {v}")
        if not args.skip_system:
            result["system"] = benchmark_system(model, tokenizer, eval_cfg.get("system_prompt_lengths",[128,512,1024]),
                eval_cfg.get("system_gen_tokens",128), eval_cfg.get("system_n_warmup",3), eval_cfg.get("system_n_repeat",10))
        all_results[cfg_name] = result
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    aggregate_results(all_results, args.output_dir)
    print(f"\n  Done! → {args.output_dir}/")

if __name__ == "__main__":
    main()