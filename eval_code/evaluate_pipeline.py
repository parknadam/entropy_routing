#!/usr/bin/env python3
"""
Progressive LoRA Pruning – Pipeline Evaluator

6개 구성을 자동 평가:
  1) A                       — 프루닝 직후 (B,C=PassLayer)
  2) A + A_adapter           — Stage1 LoRA 적용 (미병합)
  3) A_merged + B            — A 병합 후 B 복원
  4) A_merged + B + B_adapter — Stage2 LoRA 적용 (미병합)
  5) A_merged + B_merged + C — B 병합 후 C 복원
  6) A_merged + B_merged + C_merged — 전체 병합 완료

평가 항목:
  [정확도]  WikiText-2 perplexity + zero-shot 7-task (lm-evaluation-harness)
  [시스템]  Prefill latency, decode latency, throughput, peak GPU memory, param count

Usage:
  # 전체 평가 (6개 구성 모두)
  CUDA_VISIBLE_DEVICES=0 python evaluate_pipeline.py \
    --config eval_config.json --output_dir ./eval_results

  # 특정 구성만 평가
  CUDA_VISIBLE_DEVICES=0 python evaluate_pipeline.py \
    --config eval_config.json --output_dir ./eval_results \
    --only A,A+adapter

  # 정확도만 (시스템 벤치마크 스킵)
  CUDA_VISIBLE_DEVICES=0 python evaluate_pipeline.py \
    --config eval_config.json --output_dir ./eval_results \
    --skip_system

  # 시스템만 (정확도 스킵)
  CUDA_VISIBLE_DEVICES=0 python evaluate_pipeline.py \
    --config eval_config.json --output_dir ./eval_results \
    --skip_accuracy

필요 패키지:
  pip install lm-eval>=0.4.0 transformers peft safetensors datasets torch
"""

import os, sys, json, re, time, inspect, argparse, gc
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
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

# ════════════════════════════════════════════════════════════════
# 유틸리티 (training 코드에서 가져옴 — 자체 포함)
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
# GPU 메모리 유틸
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

@contextmanager
def track_memory():
    """Peak GPU memory 측정 context manager"""
    reset_gpu_stats()
    yield
    peak = gpu_mem_mb()
    return peak

# ════════════════════════════════════════════════════════════════
# 모델 로더: 6개 구성
# ════════════════════════════════════════════════════════════════

def _load_base(path, device, dtype):
    """base model 로드 + 캐시 비활성화"""
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, device_map=None, local_files_only=True)
    model.to(device)
    model.config.use_cache = True  # 추론 시에는 KV cache 활용
    model.eval()
    return model

def load_config_A(cfg, device, dtype):
    """구성 1: A만 (B,C = PassLayer)"""
    model = _load_base(cfg["A_dir"], device, dtype)
    removed = sorted(set(cfg["B_indices"] + cfg["C_indices"]))
    model, _ = _ensure_original_layout(model, removed, cfg["original_N"])
    return model

def load_config_A_adapter(cfg, device, dtype):
    """구성 2: A + A_adapter (LoRA 미병합)"""
    model = load_config_A(cfg, device, dtype)
    model = PeftModel.from_pretrained(model, cfg["A_adapter_dir"])
    model.eval()
    return model

def load_config_A_merged_B(cfg, device, dtype):
    """구성 3: A_merged + B 복원 (C = PassLayer)"""
    model = _load_base(cfg["A_merged_dir"], device, dtype)
    removed = sorted(set(cfg["B_indices"] + cfg["C_indices"]))
    model, _ = _ensure_original_layout(model, removed, cfg["original_N"])
    _rehydrate(model, cfg["B_bundle_dir"], cfg["B_indices"])
    return model

def load_config_A_merged_B_adapter(cfg, device, dtype):
    """구성 4: A_merged + B + B_adapter (LoRA 미병합)"""
    model = load_config_A_merged_B(cfg, device, dtype)
    model = PeftModel.from_pretrained(model, cfg["B_adapter_dir"])
    model.eval()
    return model

def load_config_A_merged_B_merged_C(cfg, device, dtype):
    """구성 5: A_merged + B_merged + C 복원"""
    model = _load_base(cfg["A_merged_dir"], device, dtype)
    removed = sorted(set(cfg["B_indices"] + cfg["C_indices"]))
    model, _ = _ensure_original_layout(model, removed, cfg["original_N"])
    _rehydrate(model, cfg["B_merged_dir"], cfg["B_indices"])
    _rehydrate(model, cfg["C_bundle_dir"], cfg["C_indices"])
    return model

def load_config_A_merged_B_merged_C_merged(cfg, device, dtype):
    """구성 6: 전체 병합 완료 (= 최종 복원 모델)"""
    # C_merged가 별도 디렉토리면 그걸 로드, 아니면 구성5 + C_adapter 병합
    if cfg.get("final_merged_dir"):
        model = _load_base(cfg["final_merged_dir"], device, dtype)
        return model
    # fallback: 구성5 + C_adapter
    model = load_config_A_merged_B_merged_C(cfg, device, dtype)
    if cfg.get("C_adapter_dir"):
        model = PeftModel.from_pretrained(model, cfg["C_adapter_dir"])
        model = model.merge_and_unload()
    model.eval()
    return model

# 구성 이름 → 로더 매핑
CONFIG_LOADERS = {
    "A":                        load_config_A,
    "A+adapter":                load_config_A_adapter,
    "A_merged+B":               load_config_A_merged_B,
    "A_merged+B+adapter":       load_config_A_merged_B_adapter,
    "A_merged+B_merged+C":      load_config_A_merged_B_merged_C,
    "A_merged+B_merged+C_merged": load_config_A_merged_B_merged_C_merged,
}

# ════════════════════════════════════════════════════════════════
# 1. Perplexity (WikiText-2)
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, dataset_name="wikitext",
                        dataset_config="wikitext-2-raw-v1",
                        split="test", max_seq_len=2048, stride=512):
    """
    Sliding-window perplexity on WikiText-2.
    stride < max_seq_len → overlapping windows로 정확한 PPL 측정.
    """
    ds = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    nlls = []
    n_tokens = 0
    seq_len = input_ids.size(1)

    for begin in range(0, seq_len, stride):
        end = min(begin + max_seq_len, seq_len)
        chunk = input_ids[:, begin:end]
        target_len = end - begin

        # 이전 window와 겹치는 부분은 loss에서 제외
        trg = chunk.clone()
        if begin > 0:
            overlap = max_seq_len - stride
            trg[:, :overlap] = -100

        outputs = model(input_ids=chunk, labels=trg)
        # loss는 non-(-100) 토큰에 대한 평균 → 총 NLL로 변환
        n_valid = (trg != -100).sum().item()
        nlls.append(outputs.loss.float().item() * n_valid)
        n_tokens += n_valid

        if end == seq_len:
            break

    ppl = float(np.exp(sum(nlls) / n_tokens))
    return {"perplexity": round(ppl, 2), "n_tokens": n_tokens}


# ════════════════════════════════════════════════════════════════
# 2. Zero-shot 벤치마크 (lm-evaluation-harness)
# ════════════════════════════════════════════════════════════════

def evaluate_zero_shot(model, tokenizer, tasks=None, num_fewshot=0, batch_size="auto"):
    """
    lm-evaluation-harness v0.4+ 사용.

    기본 태스크 (ASPLOS 프루닝 논문 표준):
      - arc_easy, arc_challenge  (추론)
      - hellaswag                (상식)
      - winogrande               (대명사 해소)
      - piqa                     (물리 직관)
      - boolq                    (예/아니오 QA)
      - mmlu                     (지식 종합, 5-shot 권장)
      - openbookqa               (과학 QA)
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("[WARN] lm-eval not installed. pip install lm-eval>=0.4.0")
        return {}

    if tasks is None:
        tasks = [
            "arc_easy", "arc_challenge",
            "hellaswag",
            "winogrande",
            "piqa",
            "boolq",
            "openbookqa",
        ]

    # lm-eval에 이미 로드된 모델 전달
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        log_samples=False,
    )

    # 결과 정리
    parsed = {}
    for task_name, task_res in results.get("results", {}).items():
        # acc 또는 acc_norm 추출
        acc = task_res.get("acc_norm,none", task_res.get("acc,none", None))
        if acc is not None:
            parsed[task_name] = round(float(acc) * 100, 2)
    
    # 평균
    if parsed:
        parsed["avg"] = round(sum(parsed.values()) / len(parsed), 2)

    return parsed


def evaluate_mmlu(model, tokenizer, batch_size="auto"):
    """MMLU 5-shot (별도 — 다른 태스크는 0-shot이므로 분리)"""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        return {}

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=lm, tasks=["mmlu"], num_fewshot=5,
        batch_size=batch_size, log_samples=False,
    )
    acc = results.get("results", {}).get("mmlu", {}).get("acc,none")
    if acc is not None:
        return {"mmlu_5shot": round(float(acc) * 100, 2)}
    return {}


# ════════════════════════════════════════════════════════════════
# 3. 시스템 벤치마크 (latency / throughput / memory)
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def benchmark_latency(model, tokenizer, prompt_lengths=[128, 512, 1024],
                      gen_tokens=128, n_warmup=3, n_repeat=10):
    """
    Prefill latency (TTFT) + decode latency + throughput 측정.

    Returns:
      dict[prompt_len] → {
        prefill_ms, decode_ms_per_token, throughput_tok_s,
        peak_mem_mb, total_params_M, active_params_M
      }
    """
    device = next(model.parameters()).device
    results = {}

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    # PassLayer 제외 active params
    active_params = sum(
        p.numel() for n, p in model.named_parameters()
        if not any(isinstance(m, PassLayer) for m in [])  # fallback
    ) / 1e6

    # active params: PassLayer가 아닌 레이어의 파라미터만 카운트
    try:
        layers = _layers(model)
        pass_param_count = sum(
            p.numel() for i, layer in enumerate(layers)
            if isinstance(layer, PassLayer)
            for p in layer.parameters()
        )
        active_params = (sum(p.numel() for p in model.parameters()) - pass_param_count) / 1e6
    except:
        active_params = total_params

    for prompt_len in prompt_lengths:
        # 더미 입력 생성
        dummy_ids = torch.randint(100, 30000, (1, prompt_len), device=device)
        attention_mask = torch.ones_like(dummy_ids)

        # Warmup
        for _ in range(n_warmup):
            _ = model.generate(
                input_ids=dummy_ids, attention_mask=attention_mask,
                max_new_tokens=16, do_sample=False,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # ── Prefill latency (TTFT) ──
        prefill_times = []
        for _ in range(n_repeat):
            reset_gpu_stats()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids=dummy_ids, attention_mask=attention_mask)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            prefill_times.append((time.perf_counter() - t0) * 1000)

        # ── End-to-end generation (prefill + decode) ──
        gen_times = []
        peak_mems = []
        for _ in range(n_repeat):
            reset_gpu_stats()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(
                input_ids=dummy_ids, attention_mask=attention_mask,
                max_new_tokens=gen_tokens, do_sample=False,
            )
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
            "actual_gen_tokens": actual_gen,
        }

    results["model_info"] = {
        "total_params_M": round(total_params, 1),
        "active_params_M": round(active_params, 1),
        "pruning_ratio_pct": round((1 - active_params / total_params) * 100, 1)
            if total_params > 0 else 0,
    }

    return results


# ════════════════════════════════════════════════════════════════
# 4. 결과 집계 및 저장
# ════════════════════════════════════════════════════════════════

def aggregate_results(all_results, output_dir):
    """모든 결과를 JSON + 마크다운 테이블로 저장"""
    os.makedirs(output_dir, exist_ok=True)

    # JSON 전체 저장
    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {json_path}")

    # ── 정확도 테이블 (Markdown) ──
    acc_path = os.path.join(output_dir, "accuracy_table.md")
    configs = list(all_results.keys())

    # zero-shot 태스크 이름 수집
    all_tasks = set()
    for cfg_name in configs:
        zs = all_results[cfg_name].get("zero_shot", {})
        all_tasks.update(zs.keys())
    all_tasks.discard("avg")
    task_list = sorted(all_tasks) + (["avg"] if any("avg" in all_results[c].get("zero_shot", {}) for c in configs) else [])

    with open(acc_path, "w") as f:
        # 헤더
        header = "| Config | PPL(↓) |"
        sep = "|--------|--------|"
        for t in task_list:
            header += f" {t} |"
            sep += "--------|"
        if any("mmlu_5shot" in all_results[c].get("mmlu", {}) for c in configs):
            header += " MMLU(5) |"
            sep += "---------|"
        f.write(header + "\n" + sep + "\n")

        # 행
        for cfg_name in configs:
            r = all_results[cfg_name]
            ppl = r.get("perplexity", {}).get("perplexity", "-")
            row = f"| {cfg_name} | {ppl} |"
            zs = r.get("zero_shot", {})
            for t in task_list:
                row += f" {zs.get(t, '-')} |"
            mmlu_val = r.get("mmlu", {}).get("mmlu_5shot", "-")
            if any("mmlu_5shot" in all_results[c].get("mmlu", {}) for c in configs):
                row += f" {mmlu_val} |"
            f.write(row + "\n")

    print(f"[saved] {acc_path}")

    # ── 시스템 테이블 (Markdown) ──
    sys_path = os.path.join(output_dir, "system_table.md")
    with open(sys_path, "w") as f:
        f.write("| Config | Active Params(M) | Pruning% |")
        # prompt 길이 수집
        prompt_keys = set()
        for cfg_name in configs:
            sys_r = all_results[cfg_name].get("system", {})
            prompt_keys.update(k for k in sys_r if k.startswith("prompt_"))
        prompt_keys = sorted(prompt_keys)
        for pk in prompt_keys:
            plen = pk.split("_")[1]
            f.write(f" TTFT@{plen}(ms) | Decode(ms/tok) | Tok/s | Mem(MB) |")
        f.write("\n")

        f.write("|--------|-----------------|----------|")
        for _ in prompt_keys:
            f.write("-------------|----------------|-------|---------|")
        f.write("\n")

        for cfg_name in configs:
            sys_r = all_results[cfg_name].get("system", {})
            info = sys_r.get("model_info", {})
            row = f"| {cfg_name} | {info.get('active_params_M', '-')} | {info.get('pruning_ratio_pct', '-')} |"
            for pk in prompt_keys:
                pr = sys_r.get(pk, {})
                row += (f" {pr.get('prefill_ms', '-')} |"
                        f" {pr.get('decode_ms_per_token', '-')} |"
                        f" {pr.get('throughput_tok_s', '-')} |"
                        f" {pr.get('peak_mem_mb', '-')} |")
            f.write(row + "\n")

    print(f"[saved] {sys_path}")
    return json_path


# ════════════════════════════════════════════════════════════════
# 5. 설정 파일 생성 도우미
# ════════════════════════════════════════════════════════════════

def generate_sample_config(path):
    """샘플 eval_config.json 생성"""
    sample = {
        "_comment": "경로를 실제 환경에 맞게 수정하세요",

        "original_model_dir": "./original_llama2-7b",
        "original_N": 32,

        "A_dir":          "./7b_results/pruning/A",
        "A_adapter_dir":  "./lora_results/adapters/A_lora/stageA",
        "A_merged_dir":   "./new_merged_models_llama_7b_lora/A_merged",

        "B_bundle_dir":   "./7b_results/pruning/bundles/B",
        "B_adapter_dir":  "./lora_results/adapters/B_lora/stageB",
        "B_merged_dir":   "./new_merged_models_llama_7b_lora/B_merged",

        "C_bundle_dir":   "./7b_results/pruning/bundles/C",
        "C_adapter_dir":  "./lora_results/adapters/C_lora/stageC",

        "final_merged_dir": None,

        "B_indices": [],
        "C_indices": [],

        "_indices_note": "B_indices, C_indices가 비어있으면 bundle_meta.json에서 자동 탐지",

        "eval": {
            "seq_len": 2048,
            "ppl_stride": 512,
            "zero_shot_tasks": [
                "arc_easy", "arc_challenge", "hellaswag",
                "winogrande", "piqa", "boolq", "openbookqa"
            ],
            "run_mmlu": True,
            "system_prompt_lengths": [128, 512, 1024],
            "system_gen_tokens": 128,
            "system_n_warmup": 3,
            "system_n_repeat": 10,
            "batch_size": "auto",
        }
    }
    with open(path, "w") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"[generated] sample config → {path}")
    return path


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Progressive LoRA Pruning Evaluator")
    p.add_argument("--config", required=True, help="eval_config.json 경로")
    p.add_argument("--output_dir", default="./eval_results")
    p.add_argument("--only", default=None,
                   help="평가할 구성 (쉼표 구분). 예: A,A+adapter,A_merged+B")
    p.add_argument("--skip_accuracy", action="store_true", help="정확도 평가 스킵")
    p.add_argument("--skip_system", action="store_true", help="시스템 벤치마크 스킵")
    p.add_argument("--skip_mmlu", action="store_true", help="MMLU 5-shot 스킵 (오래 걸림)")
    p.add_argument("--include_original", action="store_true",
                   help="프루닝 전 원본 모델도 평가 (baseline)")
    p.add_argument("--gen_config", default=None,
                   help="샘플 config 생성 경로 (이것만 실행하고 종료)")
    return p.parse_args()


def main():
    args = parse_args()

    # 샘플 config 생성 모드
    if args.gen_config:
        generate_sample_config(args.gen_config)
        return

    # Config 로드
    with open(args.config) as f:
        cfg = json.load(f)

    # 인덱스 자동 탐지
    if not cfg.get("B_indices") and cfg.get("B_bundle_dir"):
        cfg["B_indices"] = _load_bundle_indices(cfg["B_bundle_dir"])
        print(f"[auto] B_indices: {cfg['B_indices']}")
    if not cfg.get("C_indices") and cfg.get("C_bundle_dir"):
        cfg["C_indices"] = _load_bundle_indices(cfg["C_bundle_dir"])
        print(f"[auto] C_indices: {cfg['C_indices']}")

    # 평가할 구성 결정
    if args.only:
        config_names = [c.strip() for c in args.only.split(",")]
    else:
        config_names = list(CONFIG_LOADERS.keys())

    if args.include_original:
        config_names = ["original"] + config_names

    device = torch.device(os.environ.get("DEVICE", "cuda:0")
                          if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16
             if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else torch.float16)

    # Tokenizer (아무 모델 디렉토리에서 로드 — 동일 모델 계열)
    tok_dir = cfg.get("A_merged_dir") or cfg.get("A_dir") or cfg.get("original_model_dir")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, local_files_only=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    eval_cfg = cfg.get("eval", {})
    all_results = {}

    print(f"\n{'='*70}")
    print(f"  Progressive LoRA Pruning – Pipeline Evaluation")
    print(f"  Configs: {config_names}")
    print(f"  Accuracy: {'OFF' if args.skip_accuracy else 'ON'}")
    print(f"  System:   {'OFF' if args.skip_system else 'ON'}")
    print(f"  MMLU:     {'OFF' if args.skip_mmlu else 'ON'}")
    print(f"{'='*70}\n")

    for cfg_name in config_names:
        print(f"\n{'─'*60}")
        print(f"  Evaluating: {cfg_name}")
        print(f"{'─'*60}")

        # ── 모델 로드 ──
        reset_gpu_stats()
        t_load = time.time()

        if cfg_name == "original":
            if not cfg.get("original_model_dir"):
                print("[SKIP] original_model_dir not set")
                continue
            model = _load_base(cfg["original_model_dir"], device, dtype)
        else:
            loader = CONFIG_LOADERS.get(cfg_name)
            if loader is None:
                print(f"[SKIP] Unknown config: {cfg_name}")
                continue
            model = loader(cfg, device, dtype)

        load_time = time.time() - t_load
        print(f"  Loaded in {load_time:.1f}s, GPU mem: {gpu_mem_mb():.0f} MB")

        result = {"load_time_s": round(load_time, 1)}

        # ── 정확도 평가 ──
        if not args.skip_accuracy:
            # Perplexity
            print("  [PPL] WikiText-2 ...")
            ppl_result = evaluate_perplexity(
                model, tokenizer,
                max_seq_len=eval_cfg.get("seq_len", 2048),
                stride=eval_cfg.get("ppl_stride", 512),
            )
            result["perplexity"] = ppl_result
            print(f"    → PPL = {ppl_result['perplexity']}")

            # Zero-shot
            print("  [Zero-shot] 7 tasks ...")
            zs_result = evaluate_zero_shot(
                model, tokenizer,
                tasks=eval_cfg.get("zero_shot_tasks"),
                batch_size=eval_cfg.get("batch_size", "auto"),
            )
            result["zero_shot"] = zs_result
            for t, v in zs_result.items():
                print(f"    {t}: {v}")

            # MMLU (optional)
            if not args.skip_mmlu and eval_cfg.get("run_mmlu", True):
                print("  [MMLU] 5-shot ...")
                mmlu_result = evaluate_mmlu(
                    model, tokenizer,
                    batch_size=eval_cfg.get("batch_size", "auto"),
                )
                result["mmlu"] = mmlu_result
                for t, v in mmlu_result.items():
                    print(f"    {t}: {v}")

        # ── 시스템 벤치마크 ──
        if not args.skip_system:
            print("  [System] Latency / throughput / memory ...")
            sys_result = benchmark_latency(
                model, tokenizer,
                prompt_lengths=eval_cfg.get("system_prompt_lengths", [128, 512, 1024]),
                gen_tokens=eval_cfg.get("system_gen_tokens", 128),
                n_warmup=eval_cfg.get("system_n_warmup", 3),
                n_repeat=eval_cfg.get("system_n_repeat", 10),
            )
            result["system"] = sys_result
            info = sys_result.get("model_info", {})
            print(f"    Active params: {info.get('active_params_M', '?')}M "
                  f"(pruning {info.get('pruning_ratio_pct', '?')}%)")
            for k, v in sys_result.items():
                if k.startswith("prompt_"):
                    print(f"    {k}: TTFT={v['prefill_ms']}ms, "
                          f"decode={v['decode_ms_per_token']}ms/tok, "
                          f"{v['throughput_tok_s']} tok/s, "
                          f"mem={v['peak_mem_mb']}MB")

        all_results[cfg_name] = result

        # 모델 해제
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 결과 저장 ──
    json_path = aggregate_results(all_results, args.output_dir)

    print(f"\n{'='*70}")
    print(f"  Evaluation complete!")
    print(f"  Results: {args.output_dir}/")
    print(f"    - eval_results.json    (전체 결과)")
    print(f"    - accuracy_table.md    (정확도 테이블)")
    print(f"    - system_table.md      (시스템 벤치마크)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()