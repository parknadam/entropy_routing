#python eval_a/evaluate_stage_a.py
"""
Stage-A Pruning Ratio Comparison Evaluator

Original baseline vs 프루닝 비율별 A_merged 모델 비교.
"A 단계만으로 쓸 수 있는 모델인가?"를 검증.

Usage:
  # 전체 평가
  CUDA_VISIBLE_DEVICES=0 python eval_a/evaluate_stage_a.py \
    --config eval_a/stage_a_config.json --output_dir ./eval_stage_a

  # 시스템만 빠르게
  CUDA_VISIBLE_DEVICES=0 python eval_a/evaluate_stage_a.py \
    --config ./stage_a_config.json --output_dir ./eval_stage_a \
    --skip_accuracy

  # 특정 variant만
  CUDA_VISIBLE_DEVICES=0 python eval_a/evaluate_stage_a.py \
    --config ./stage_a_config.json --output_dir ./eval_stage_a \
    --only "original,7layers"

  # 샘플 config 생성
  python eval_a/evaluate_stage_a.py --gen_config ./stage_a_config.json
"""

import os, sys, json, re, time, inspect, argparse, gc
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from peft import PeftModel
from datasets import load_dataset

KST = timezone(timedelta(hours=9))

# ════════════════════════════════════════════════════════════════
# 유틸리티 (모델 레이아웃/PassLayer 관련)
# ════════════════════════════════════════════════════════════════

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
        raise ValueError(f"layer mismatch: {cur_N} vs expected compact={len(kept)} or sparse={original_N}")

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


# ════════════════════════════════════════════════════════════════
# GPU 유틸
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
# 모델 로더
# ════════════════════════════════════════════════════════════════

def load_original(cfg, device, dtype):
    """프루닝 전 원본 모델"""
    model = AutoModelForCausalLM.from_pretrained(
        cfg["original_model_dir"], torch_dtype=dtype, device_map=None, local_files_only=True)
    model.to(device)
    model.config.use_cache = True
    model.eval()
    return model

def load_a_merged(variant_cfg, device, dtype):
    """A_merged 모델 로드 (adapter 이미 병합된 상태)"""
    model = AutoModelForCausalLM.from_pretrained(
        variant_cfg["a_merged_dir"], torch_dtype=dtype, device_map=None, local_files_only=True)
    model.to(device)
    model.config.use_cache = True
    model.eval()

    # A_merged에서 제거된 레이어(B+C)는 PassLayer로 채움
    removed = sorted(set(variant_cfg.get("removed_layers", [])))
    original_N = variant_cfg["original_N"]
    if removed:
        model, _ = _ensure_original_layout(model, removed, original_N)

    return model

def load_a_with_adapter(variant_cfg, device, dtype):
    """A(프루닝 직후) + adapter (LoRA 미병합) 로드"""
    model = AutoModelForCausalLM.from_pretrained(
        variant_cfg["a_dir"], torch_dtype=dtype, device_map=None, local_files_only=True)
    model.to(device)
    model.config.use_cache = True
    model.eval()

    # 제거된 레이어를 PassLayer로 채움
    removed = sorted(set(variant_cfg.get("removed_layers", [])))
    original_N = variant_cfg["original_N"]
    if removed:
        model, _ = _ensure_original_layout(model, removed, original_N)

    # LoRA adapter 적용 (병합하지 않음)
    model = PeftModel.from_pretrained(model, variant_cfg["a_adapter_dir"])
    model.eval()
    return model

def _detect_mode(variant_cfg):
    """variant config에서 로드 모드 자동 감지
    Returns: 'a_merged' | 'a_adapter'
    """
    has_merged = bool(variant_cfg.get("a_merged_dir"))
    has_adapter = bool(variant_cfg.get("a_dir")) and bool(variant_cfg.get("a_adapter_dir"))

    if has_adapter:
        return "a_adapter"
    if has_merged:
        return "a_merged"
    raise ValueError(
        f"variant '{variant_cfg.get('name', '?')}': "
        f"a_dir + a_adapter_dir 또는 a_merged_dir 중 하나는 필수"
    )


# ════════════════════════════════════════════════════════════════
# Perplexity
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, max_seq_len=2048, stride=512):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    nlls = []
    n_tokens = 0
    seq_len = input_ids.size(1)

    for begin in range(0, seq_len, stride):
        end = min(begin + max_seq_len, seq_len)
        chunk = input_ids[:, begin:end]
        trg = chunk.clone()
        if begin > 0:
            overlap = max_seq_len - stride
            trg[:, :overlap] = -100

        outputs = model(input_ids=chunk, labels=trg)
        n_valid = (trg != -100).sum().item()
        nlls.append(outputs.loss.float().item() * n_valid)
        n_tokens += n_valid

        if end == seq_len:
            break

    ppl = float(np.exp(sum(nlls) / n_tokens))
    return {"perplexity": round(ppl, 2), "n_tokens": n_tokens}


# ════════════════════════════════════════════════════════════════
# Zero-shot (lm-evaluation-harness)
# ════════════════════════════════════════════════════════════════

def evaluate_zero_shot(model, tokenizer, tasks=None, num_fewshot=0, batch_size="auto"):
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("[WARN] lm-eval not installed. pip install lm-eval>=0.4.0")
        return {}

    if tasks is None:
        tasks = ["arc_easy", "arc_challenge", "hellaswag",
                 "winogrande", "piqa", "boolq", "openbookqa"]

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=lm, tasks=tasks, num_fewshot=num_fewshot,
        batch_size=batch_size, log_samples=False,
    )

    parsed = {}
    for task_name, task_res in results.get("results", {}).items():
        acc = task_res.get("acc_norm,none", task_res.get("acc,none", None))
        if acc is not None:
            parsed[task_name] = round(float(acc) * 100, 2)

    if parsed:
        parsed["avg"] = round(sum(parsed.values()) / len(parsed), 2)
    return parsed


# ════════════════════════════════════════════════════════════════
# 시스템 벤치마크
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def benchmark_system(model, tokenizer, prompt_lengths=[128, 512, 1024],
                     gen_tokens=128, n_warmup=3, n_repeat=10):
    device = next(model.parameters()).device
    results = {}

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    try:
        layers = _layers(model)
        pass_params = sum(
            p.numel() for i, layer in enumerate(layers)
            if isinstance(layer, PassLayer)
            for p in layer.parameters()
        )
        active_params = (sum(p.numel() for p in model.parameters()) - pass_params) / 1e6
    except:
        active_params = total_params

    # PassLayer 수 / 실제 레이어 수
    try:
        layers = _layers(model)
        n_pass = sum(1 for l in layers if isinstance(l, PassLayer))
        n_real = len(layers) - n_pass
    except:
        n_pass, n_real = 0, 0

    for prompt_len in prompt_lengths:
        dummy_ids = torch.randint(100, 30000, (1, prompt_len), device=device)
        attention_mask = torch.ones_like(dummy_ids)

        # Warmup
        for _ in range(n_warmup):
            _ = model.generate(
                input_ids=dummy_ids, attention_mask=attention_mask,
                max_new_tokens=16, do_sample=False)

        if torch.cuda.is_available(): torch.cuda.synchronize()

        # Prefill
        prefill_times = []
        for _ in range(n_repeat):
            reset_gpu_stats()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids=dummy_ids, attention_mask=attention_mask)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            prefill_times.append((time.perf_counter() - t0) * 1000)

        # Generation
        gen_times = []
        peak_mems = []
        for _ in range(n_repeat):
            reset_gpu_stats()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(
                input_ids=dummy_ids, attention_mask=attention_mask,
                max_new_tokens=gen_tokens, do_sample=False)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            gen_times.append((time.perf_counter() - t0) * 1000)
            peak_mems.append(gpu_mem_mb())

        actual_gen = out.shape[1] - prompt_len
        avg_gen_ms = np.mean(gen_times)
        avg_prefill_ms = np.mean(prefill_times)
        decode_total_ms = avg_gen_ms - avg_prefill_ms
        decode_per_token = decode_total_ms / max(actual_gen, 1)
        throughput = actual_gen / (avg_gen_ms / 1000)

        results[f"prompt_{prompt_len}"] = {
            "prefill_ms": round(avg_prefill_ms, 2),
            "decode_ms_per_token": round(decode_per_token, 2),
            "throughput_tok_s": round(throughput, 1),
            "gen_total_ms": round(avg_gen_ms, 2),
            "peak_mem_mb": round(max(peak_mems), 1),
        }

    results["model_info"] = {
        "total_params_M": round(total_params, 1),
        "active_params_M": round(active_params, 1),
        "real_layers": n_real,
        "pass_layers": n_pass,
        "pruning_ratio_pct": round((1 - active_params / total_params) * 100, 1)
            if total_params > 0 else 0,
    }
    return results


# ════════════════════════════════════════════════════════════════
# 결과 저장
# ════════════════════════════════════════════════════════════════

def save_results(all_results, output_dir, eval_cfg):
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, "stage_a_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {json_path}")

    variants = list(all_results.keys())

    # ── 정확도 테이블 ──
    acc_path = os.path.join(output_dir, "stage_a_accuracy.md")
    all_tasks = set()
    for v in variants:
        zs = all_results[v].get("zero_shot", {})
        all_tasks.update(zs.keys())
    all_tasks.discard("avg")
    task_list = sorted(all_tasks)
    if any("avg" in all_results[v].get("zero_shot", {}) for v in variants):
        task_list.append("avg")

    with open(acc_path, "w") as f:
        f.write("# Stage A — Pruning ratio comparison (accuracy)\n\n")
        header = "| Variant | Mode | Pruned layers | Pruning % | PPL(↓) |"
        sep =    "|---------|------|---------------|-----------|--------|"
        for t in task_list:
            header += f" {t} |"
            sep += "--------|"
        f.write(header + "\n" + sep + "\n")

        for v in variants:
            r = all_results[v]
            meta = r.get("meta", {})
            n_pruned = meta.get("pruned_layers", "-")
            prune_pct = meta.get("pruning_pct", "-")
            mode = meta.get("mode", "-")
            ppl = r.get("perplexity", {}).get("perplexity", "-")
            row = f"| {v} | {mode} | {n_pruned} | {prune_pct} | {ppl} |"
            zs = r.get("zero_shot", {})
            for t in task_list:
                row += f" {zs.get(t, '-')} |"
            f.write(row + "\n")

    print(f"[saved] {acc_path}")

    # ── 시스템 테이블 ──
    sys_path = os.path.join(output_dir, "stage_a_system.md")
    prompt_keys = set()
    for v in variants:
        sys_r = all_results[v].get("system", {})
        prompt_keys.update(k for k in sys_r if k.startswith("prompt_"))
    prompt_keys = sorted(prompt_keys)

    with open(sys_path, "w") as f:
        f.write("# Stage A — Pruning ratio comparison (system)\n\n")
        header = "| Variant | Active(M) | Real layers | Pruning % |"
        sep =    "|---------|-----------|-------------|-----------|"
        for pk in prompt_keys:
            plen = pk.split("_")[1]
            header += f" TTFT@{plen} | ms/tok | Tok/s | Mem(MB) |"
            sep += "--------|--------|-------|---------|"
        f.write(header + "\n" + sep + "\n")

        for v in variants:
            sys_r = all_results[v].get("system", {})
            info = sys_r.get("model_info", {})
            row = (f"| {v} | {info.get('active_params_M', '-')} | "
                   f"{info.get('real_layers', '-')} | "
                   f"{info.get('pruning_ratio_pct', '-')} |")
            for pk in prompt_keys:
                pr = sys_r.get(pk, {})
                row += (f" {pr.get('prefill_ms', '-')} |"
                        f" {pr.get('decode_ms_per_token', '-')} |"
                        f" {pr.get('throughput_tok_s', '-')} |"
                        f" {pr.get('peak_mem_mb', '-')} |")
            f.write(row + "\n")

    print(f"[saved] {sys_path}")

    # ── 요약 (논문용 한줄 비교) ──
    summary_path = os.path.join(output_dir, "stage_a_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Stage A — Quick summary\n\n")
        f.write("| Variant | Pruned | PPL | Avg acc | Speedup | Mem saving |\n")
        f.write("|---------|--------|-----|---------|---------|------------|\n")

        # baseline 기준 speedup/mem 계산
        base_sys = all_results.get("original", {}).get("system", {})
        base_prefill = None
        base_mem = None
        for pk in prompt_keys:
            if "512" in pk:  # prompt_512 기준
                base_prefill = base_sys.get(pk, {}).get("prefill_ms")
                base_mem = base_sys.get(pk, {}).get("peak_mem_mb")
                break

        for v in variants:
            r = all_results[v]
            meta = r.get("meta", {})
            ppl = r.get("perplexity", {}).get("perplexity", "-")
            avg_acc = r.get("zero_shot", {}).get("avg", "-")
            n_pruned = meta.get("pruned_layers", "-")

            sys_r = r.get("system", {})
            speedup = "-"
            mem_save = "-"
            for pk in prompt_keys:
                if "512" in pk:
                    cur_prefill = sys_r.get(pk, {}).get("prefill_ms")
                    cur_mem = sys_r.get(pk, {}).get("peak_mem_mb")
                    if base_prefill and cur_prefill and cur_prefill > 0:
                        speedup = f"{base_prefill / cur_prefill:.2f}x"
                    if base_mem and cur_mem:
                        mem_save = f"{(1 - cur_mem / base_mem) * 100:.1f}%"
                    break

            f.write(f"| {v} | {n_pruned} | {ppl} | {avg_acc} | {speedup} | {mem_save} |\n")

    print(f"[saved] {summary_path}")
    return json_path


# ════════════════════════════════════════════════════════════════
# Config 생성
# ════════════════════════════════════════════════════════════════

def generate_sample_config(path):
    sample = {
        "_comment": "프루닝 비율별 A 단계 비교 설정",
        "_modes": "variant마다 a_dir + a_adapter_dir (미병합) 또는 a_merged_dir (병합) 중 택1",

        "original_model_dir": "./original_llama2-7b",
        "original_N": 32,

        "variants": [
            {
                "name": "7layers_adapter",
                "a_dir": "./pruning_7layers/A",
                "a_adapter_dir": "./lora_results_7layers/adapters/A_lora/stageA",
                "removed_layers": [5, 8, 11, 14, 17, 20, 23],
                "original_N": 32,
                "_note": "7개 제거 (21.9%) — A + adapter 미병합"
            },
            {
                "name": "10layers_adapter",
                "a_dir": "./pruning_10layers/A",
                "a_adapter_dir": "./lora_results_10layers/adapters/A_lora/stageA",
                "removed_layers": [3, 5, 8, 10, 13, 15, 18, 20, 23, 25],
                "original_N": 32,
                "_note": "10개 제거 (31.3%) — A + adapter 미병합"
            },
            {
                "name": "13layers_adapter",
                "a_dir": "./pruning_13layers/A",
                "a_adapter_dir": "./lora_results_13layers/adapters/A_lora/stageA",
                "removed_layers": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26],
                "original_N": 32,
                "_note": "13개 제거 (40.6%) — A + adapter 미병합"
            },
            {
                "_comment_merged_example": "a_merged_dir를 쓰면 병합 모델로 평가 (혼용 가능)",
                "name": "7layers_merged",
                "a_merged_dir": "./merged_7layers/A_merged",
                "removed_layers": [5, 8, 11, 14, 17, 20, 23],
                "original_N": 32,
                "_note": "7개 제거 — A_merged 병합 완료"
            }
        ],

        "eval": {
            "seq_len": 2048,
            "ppl_stride": 512,
            "zero_shot_tasks": [
                "arc_easy", "arc_challenge", "hellaswag",
                "winogrande", "piqa", "boolq", "openbookqa"
            ],
            "system_prompt_lengths": [128, 512, 1024],
            "system_gen_tokens": 128,
            "system_n_warmup": 3,
            "system_n_repeat": 10,
            "batch_size": "auto"
        }
    }
    with open(path, "w") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"[generated] sample config → {path}")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Stage A Pruning Ratio Evaluator")
    p.add_argument("--config", help="config JSON 경로")
    p.add_argument("--output_dir", default="./eval_stage_a")
    p.add_argument("--only", default=None,
                   help="평가할 variant 이름 (쉼표 구분). 예: original,7layers,10layers")
    p.add_argument("--skip_accuracy", action="store_true")
    p.add_argument("--skip_system", action="store_true")
    p.add_argument("--skip_original", action="store_true",
                   help="원본 baseline 평가 스킵")
    p.add_argument("--gen_config", default=None,
                   help="샘플 config 생성 (이것만 실행하고 종료)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.gen_config:
        generate_sample_config(args.gen_config)
        return

    if not args.config:
        print("ERROR: --config 필요. 샘플 생성: --gen_config stage_a_config.json")
        sys.exit(1)

    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device(os.environ.get("DEVICE", "cuda:0")
                          if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16
             if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else torch.float16)

    tok_dir = cfg.get("original_model_dir") or cfg["variants"][0]["a_merged_dir"]
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, local_files_only=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    eval_cfg = cfg.get("eval", {})

    # 평가 대상 구성
    only_set = set(args.only.split(",")) if args.only else None

    run_list = []
    if not args.skip_original:
        if only_set is None or "original" in only_set:
            run_list.append(("original", None))

    for v in cfg["variants"]:
        name = v["name"]
        if only_set is None or name in only_set:
            run_list.append((name, v))

    print(f"\n{'='*60}")
    print(f"  Stage A — Pruning Ratio Comparison")
    print(f"  Variants: {[r[0] for r in run_list]}")
    print(f"  Accuracy: {'OFF' if args.skip_accuracy else 'ON'}")
    print(f"  System:   {'OFF' if args.skip_system else 'ON'}")
    print(f"{'='*60}\n")

    all_results = {}

    for name, variant_cfg in run_list:
        print(f"\n{'─'*50}")
        print(f"  [{name}]")
        print(f"{'─'*50}")

        reset_gpu_stats()
        t0 = time.time()

        if name == "original":
            model = load_original(cfg, device, dtype)
            meta = {"pruned_layers": 0, "pruning_pct": "0%",
                    "real_layers": cfg["original_N"], "mode": "original"}
        else:
            mode = _detect_mode(variant_cfg)
            if mode == "a_adapter":
                model = load_a_with_adapter(variant_cfg, device, dtype)
                print(f"  Mode: A + adapter (LoRA 미병합)")
            else:
                model = load_a_merged(variant_cfg, device, dtype)
                print(f"  Mode: A_merged (병합 완료)")
            removed = variant_cfg.get("removed_layers", [])
            orig_N = variant_cfg.get("original_N", cfg.get("original_N", 32))
            meta = {
                "pruned_layers": len(removed),
                "pruning_pct": f"{len(removed)/orig_N*100:.1f}%",
                "removed_indices": removed,
                "real_layers": orig_N - len(removed),
                "mode": mode,
            }

        load_s = time.time() - t0
        print(f"  Loaded in {load_s:.1f}s | {gpu_mem_mb():.0f} MB")

        result = {"meta": meta, "load_time_s": round(load_s, 1)}

        # 정확도
        if not args.skip_accuracy:
            print("  [PPL] WikiText-2 ...")
            ppl = evaluate_perplexity(
                model, tokenizer,
                max_seq_len=eval_cfg.get("seq_len", 2048),
                stride=eval_cfg.get("ppl_stride", 512))
            result["perplexity"] = ppl
            print(f"    → PPL = {ppl['perplexity']}")

            print("  [Zero-shot] 7 tasks ...")
            zs = evaluate_zero_shot(
                model, tokenizer,
                tasks=eval_cfg.get("zero_shot_tasks"),
                batch_size=eval_cfg.get("batch_size", "auto"))
            result["zero_shot"] = zs
            for t, v in zs.items():
                print(f"    {t}: {v}")

        # 시스템
        if not args.skip_system:
            print("  [System] Latency / throughput / memory ...")
            sys_r = benchmark_system(
                model, tokenizer,
                prompt_lengths=eval_cfg.get("system_prompt_lengths", [128, 512, 1024]),
                gen_tokens=eval_cfg.get("system_gen_tokens", 128),
                n_warmup=eval_cfg.get("system_n_warmup", 3),
                n_repeat=eval_cfg.get("system_n_repeat", 10))
            result["system"] = sys_r
            info = sys_r.get("model_info", {})
            print(f"    Active: {info.get('active_params_M')}M | "
                  f"Real layers: {info.get('real_layers')} | "
                  f"Pruning: {info.get('pruning_ratio_pct')}%")

        all_results[name] = result

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 저장
    save_results(all_results, args.output_dir, eval_cfg)

    print(f"\n{'='*60}")
    print(f"  Done! Results → {args.output_dir}/")
    print(f"    stage_a_results.json   (raw)")
    print(f"    stage_a_accuracy.md    (정확도 테이블)")
    print(f"    stage_a_system.md      (시스템 테이블)")
    print(f"    stage_a_summary.md     (논문용 요약)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()