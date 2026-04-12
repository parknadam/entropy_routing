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
CUDA_VISIBLE_DEVICES=1 python -m final_eval.llama_eval.llama_evaluate_pipeline \
    --config ./final_eval/llama_eval/llama7b_eval_config.json --output_dir ./eval_results/llama2_7b

"""

import os, sys, json, re, time, inspect, argparse, gc
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

KST = timezone(timedelta(hours=9))
CANON_PATH = "model.layers"

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

def _rehydrate(model, bdir, indices):
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
# Model loaders
# ════════════════════════════════════════════════════════════════

def _load_base(path, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, device_map=None, local_files_only=True)
    model.to(device)
    model.config.use_cache = True
    model.eval()
    return model

def load_stage1(cfg, device, dtype):
    """Stage 1: A_merged + B,C = PassLayer"""
    model = _load_base(cfg["A_merged_dir"], device, dtype)
    removed = sorted(set(cfg["B_indices"] + cfg["C_indices"]))
    model, _ = _ensure_original_layout(model, removed, cfg["original_N"])
    return model

def load_stage2(cfg, device, dtype):
    """Stage 2: A_merged + B_merged + C = PassLayer"""
    model = _load_base(cfg["A_merged_dir"], device, dtype)
    removed = sorted(set(cfg["B_indices"] + cfg["C_indices"]))
    model, _ = _ensure_original_layout(model, removed, cfg["original_N"])
    _rehydrate(model, cfg["B_merged_dir"], cfg["B_indices"])
    return model

def load_stage3(cfg, device, dtype):
    """Stage 3: A_merged + B_merged + C_merged"""
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
    """Dense baseline"""
    return _load_base(cfg["original_model_dir"], device, dtype)

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
def evaluate_perplexity(model, tokenizer, max_seq_len=2048, stride=512):
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
    return {"perplexity": round(ppl, 2), "n_tokens": n_tokens}

# ════════════════════════════════════════════════════════════════
# 2. Zero-shot (lm-evaluation-harness, 0-shot)
# ════════════════════════════════════════════════════════════════

def evaluate_zero_shot(model, tokenizer, tasks=None, batch_size="auto"):
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("[WARN] lm-eval not installed")
        return {}
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

# ════════════════════════════════════════════════════════════════
# 2-b. MMLU (5-shot, lm-evaluation-harness)
# ════════════════════════════════════════════════════════════════

def evaluate_mmlu(model, tokenizer, batch_size="auto"):
    """
    MMLU 5-shot evaluation via lm-evaluation-harness.
    Deterministic (log-likelihood scoring) → 단일 실행 결과를 논문에 바로 사용 가능.
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("[WARN] lm-eval not installed – skipping MMLU")
        return {}
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=lm, tasks=["mmlu"], num_fewshot=5,
        batch_size=batch_size, log_samples=False)
    parsed = {}
    for t, r in results.get("results", {}).items():
        acc = r.get("acc,none", r.get("acc_norm,none"))
        if acc is not None:
            parsed[t] = round(float(acc) * 100, 2)
    # subcategory 평균 → mmlu_avg
    subs = {k: v for k, v in parsed.items() if k.startswith("mmlu_") and k != "mmlu_avg"}
    if subs:
        parsed["mmlu_avg"] = round(sum(subs.values()) / len(subs), 2)
    elif "mmlu" in parsed:
        parsed["mmlu_avg"] = parsed["mmlu"]
    return parsed

# ════════════════════════════════════════════════════════════════
# 3. System benchmark
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_ttft(model, input_ids, attention_mask, n_warmup=3, n_repeat=10):
    for _ in range(n_warmup):
        model.generate(input_ids=input_ids, attention_mask=attention_mask,
                       max_new_tokens=1, do_sample=False)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(input_ids=input_ids, attention_mask=attention_mask,
                       max_new_tokens=1, do_sample=False)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))

@torch.no_grad()
def benchmark_system(model, tokenizer, prompt_lengths=[128, 512, 1024],
                     gen_tokens=128, n_warmup=3, n_repeat=10):
    device = next(model.parameters()).device
    results = {}

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    try:
        layers = _layers(model)
        pass_params = sum(p.numel() for l in layers if isinstance(l, PassLayer) for p in l.parameters())
        active_params = (sum(p.numel() for p in model.parameters()) - pass_params) / 1e6
    except:
        active_params = total_params

    for prompt_len in prompt_lengths:
        dummy_ids = torch.randint(100, 30000, (1, prompt_len), device=device)
        attn = torch.ones_like(dummy_ids)

        ttft_mean, ttft_std = measure_ttft(model, dummy_ids, attn, n_warmup, n_repeat)

        for _ in range(n_warmup):
            model.generate(input_ids=dummy_ids, attention_mask=attn,
                           max_new_tokens=gen_tokens, do_sample=False)

        gen_times, peak_mems = [], []
        for _ in range(n_repeat):
            reset_gpu_stats()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(input_ids=dummy_ids, attention_mask=attn,
                                 max_new_tokens=gen_tokens, do_sample=False)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            gen_times.append((time.perf_counter() - t0) * 1000)
            peak_mems.append(gpu_mem_mb())

        actual_gen = out.shape[1] - prompt_len
        avg_gen = float(np.mean(gen_times))
        decode_total = avg_gen - ttft_mean
        decode_per_tok = decode_total / max(actual_gen - 1, 1)
        throughput = actual_gen / (avg_gen / 1000)

        results[f"prompt_{prompt_len}"] = {
            "ttft_ms": round(ttft_mean, 2), "ttft_std_ms": round(ttft_std, 2),
            "decode_ms_per_token": round(decode_per_tok, 2),
            "throughput_tok_s": round(throughput, 1),
            "gen_total_ms": round(avg_gen, 2),
            "peak_mem_mb": round(max(peak_mems), 1),
            "actual_gen_tokens": actual_gen,
        }

    results["model_info"] = {
        "total_params_M": round(total_params, 1),
        "active_params_M": round(active_params, 1),
        "pruning_ratio_pct": round((1 - active_params / total_params) * 100, 1) if total_params > 0 else 0,
    }
    return results

# ════════════════════════════════════════════════════════════════
# 4. Results aggregation
# ════════════════════════════════════════════════════════════════

def aggregate_results(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {json_path}")

    configs = list(all_results.keys())

    # ── Accuracy table (zero-shot + MMLU) ──
    acc_path = os.path.join(output_dir, "accuracy_table.md")
    all_tasks = set()
    for c in configs:
        all_tasks.update(all_results[c].get("zero_shot", {}).keys())
    all_tasks.discard("avg")
    task_list = sorted(all_tasks) + (["avg"] if any("avg" in all_results[c].get("zero_shot", {}) for c in configs) else [])

    has_mmlu = any("mmlu" in all_results[c] for c in configs)

    with open(acc_path, "w") as f:
        mmlu_col = " MMLU(5) |" if has_mmlu else ""
        header = "| Config | PPL(↓) |" + "".join(f" {t} |" for t in task_list) + mmlu_col
        sep = "|--------|--------|" + "".join("--------|" for _ in task_list) + ("--------|" if has_mmlu else "")
        f.write(header + "\n" + sep + "\n")
        for c in configs:
            r = all_results[c]
            ppl = r.get("perplexity", {}).get("perplexity", "-")
            zs = r.get("zero_shot", {})
            mmlu_val = r.get("mmlu", {}).get("mmlu_avg", "-") if has_mmlu else ""
            row = f"| {c} | {ppl} |" + "".join(f" {zs.get(t, '-')} |" for t in task_list)
            if has_mmlu: row += f" {mmlu_val} |"
            f.write(row + "\n")
    print(f"[saved] {acc_path}")

    # ── System table ──
    sys_path = os.path.join(output_dir, "system_table.md")
    prompt_keys = sorted(set(k for c in configs for k in all_results[c].get("system", {}) if k.startswith("prompt_")))
    with open(sys_path, "w") as f:
        header = "| Config | Active(M) | Prune% |" + "".join(f" TTFT@{pk.split('_')[1]}(ms) | Dec(ms/tok) | Tok/s | Mem(MB) |" for pk in prompt_keys)
        f.write(header + "\n")
        for c in configs:
            sr = all_results[c].get("system", {})
            info = sr.get("model_info", {})
            row = f"| {c} | {info.get('active_params_M', '-')} | {info.get('pruning_ratio_pct', '-')} |"
            for pk in prompt_keys:
                pr = sr.get(pk, {})
                row += f" {pr.get('ttft_ms', '-')} | {pr.get('decode_ms_per_token', '-')} | {pr.get('throughput_tok_s', '-')} | {pr.get('peak_mem_mb', '-')} |"
            f.write(row + "\n")
    print(f"[saved] {sys_path}")

# ════════════════════════════════════════════════════════════════
# 5. Config generator
# ════════════════════════════════════════════════════════════════

def generate_sample_config(path):
    sample = {
        "original_model_dir": "./original_llama2-7b",
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
            "seq_len": 2048, "ppl_stride": 512,
            "zero_shot_tasks": ["arc_easy","arc_challenge","hellaswag","winogrande","piqa","boolq","openbookqa"],
            "system_prompt_lengths": [128, 512, 1024],
            "system_gen_tokens": 128, "system_n_warmup": 3, "system_n_repeat": 10,
            "batch_size": "auto",
        }
    }
    with open(path, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"[generated] {path}")

# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="LLaMA Progressive LoRA Evaluator")
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", default="./eval_results_llama")
    p.add_argument("--only", default=None)
    p.add_argument("--skip_accuracy", action="store_true")
    p.add_argument("--skip_system", action="store_true")
    p.add_argument("--skip_mmlu", action="store_true")
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

    device = torch.device(os.environ.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    tok_dir = cfg.get("A_merged_dir") or cfg.get("A_dir") or cfg.get("original_model_dir")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, local_files_only=True)
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token

    eval_cfg = cfg.get("eval", {})
    all_results = {}

    print(f"\n{'='*70}\n  LLaMA Pipeline Evaluation\n  Configs: {config_names}\n{'='*70}\n")

    for cfg_name in config_names:
        loader = MAIN_LOADERS.get(cfg_name)
        if not loader: print(f"[SKIP] {cfg_name}"); continue
        if cfg_name == "Dense" and not cfg.get("original_model_dir"): print("[SKIP] Dense"); continue

        print(f"\n{'─'*60}\n  {cfg_name}\n{'─'*60}")
        reset_gpu_stats()
        model = loader(cfg, device, dtype)
        result = {}

        if not args.skip_accuracy:
            print("  [PPL] ...")
            result["perplexity"] = evaluate_perplexity(model, tokenizer,
                eval_cfg.get("seq_len", 2048), eval_cfg.get("ppl_stride", 512))
            print(f"    → {result['perplexity']['perplexity']}")

            print("  [Zero-shot] ...")
            result["zero_shot"] = evaluate_zero_shot(model, tokenizer,
                eval_cfg.get("zero_shot_tasks"), eval_cfg.get("batch_size", "auto"))
            for t, v in result["zero_shot"].items(): print(f"    {t}: {v}")

        if not args.skip_mmlu:
            print("  [MMLU 5-shot] ...")
            result["mmlu"] = evaluate_mmlu(model, tokenizer,
                eval_cfg.get("batch_size", "auto"))
            mmlu_avg = result["mmlu"].get("mmlu_avg", "-")
            print(f"    → MMLU avg: {mmlu_avg}")

        if not args.skip_system:
            print("  [System] ...")
            result["system"] = benchmark_system(model, tokenizer,
                eval_cfg.get("system_prompt_lengths", [128, 512, 1024]),
                eval_cfg.get("system_gen_tokens", 128),
                eval_cfg.get("system_n_warmup", 3), eval_cfg.get("system_n_repeat", 10))
            info = result["system"].get("model_info", {})
            print(f"    Active: {info.get('active_params_M')}M ({info.get('pruning_ratio_pct')}%)")

        all_results[cfg_name] = result
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    aggregate_results(all_results, args.output_dir)
    print(f"\n{'='*70}\n  Done! → {args.output_dir}/\n{'='*70}")

if __name__ == "__main__":
    main()