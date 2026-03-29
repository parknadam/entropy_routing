#!/usr/bin/env python3
"""
머지된 모델의 벤치마크 평가 (MMLU 5-shot, HellaSwag 10-shot, GSM8K 5-shot CoT).

★ 순차적 로드: A → +B → +C 누적 복원 ★

  Step 1: A_merged 로드 → PassLayer 설치 (B,C 위치) → 평가
  Step 2: B_merged 복원 (B 위치 PassLayer → 실제 레이어, C는 그대로 PassLayer) → 평가
  Step 3: C_merged 복원 (C 위치 PassLayer → 실제 레이어) → 평가

  모델 1회 로드, bundle 순차 누적 — GPU 재로드 없음.

사용법:
  # 기본 (A → AB → ABC 순차)
  CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0 \
  python eval_benchmarks_mergedmodel.py \
    --model_path ./new_merged_models_llama_7b_lora/A_merged \
    --b_bundle ./new_merged_models_llama_7b_lora/B_merged \
    --c_bundle ./new_merged_models_llama_7b_lora/C_merged \
    --device cuda:0

  # 특정 벤치마크만
  CUDA_VISIBLE_DEVICES=2 DEVICE=cuda:0 \
  python eval_benchmarks_mergedmodel.py \
    --model_path ./new_merged_models_llama_7b_lora/A_merged \
    --b_bundle ./new_merged_models_llama_7b_lora/B_merged \
    --c_bundle ./new_merged_models_llama_7b_lora/C_merged \
    --benchmarks mmlu

  # 원본 모델 비교
  python eval_benchmarks_mergedmodel.py \
    --model_path mistralai/Mistral-7B-v0.1 ./merged_models/A_merged \
    --b_bundle ./merged_models/B_merged \
    --c_bundle ./merged_models/C_merged \
    --tokenizer_path mistralai/Mistral-7B-v0.1

  # 구버전 bundles_dir
  python eval_benchmarks_mergedmodel.py \
    --model_path ./merged_models/A_merged \
    --bundles_dir ./pruning/bundles

  # OOM 시 배치 줄이기
  CUDA_VISIBLE_DEVICES=2 DEVICE=cuda:0 \
  python eval_benchmarks_mergedmodel.py \
    --model_path ./merged_models_llama_7b/A_merged \
    --b_bundle ./merged_models_llama_7b/B_merged \
    --c_bundle ./merged_models_llama_7b/C_merged \
    --batch_size 4

  # 결과 저장 폴더 변경 (기본: ./eval_results/)
  python eval_benchmarks_mergedmodel.py \
    --model_path ./merged_models/A_merged \
    --b_bundle ./merged_models/B_merged \
    --c_bundle ./merged_models/C_merged \
    --output_dir ./my_results
"""

from __future__ import annotations

import argparse, gc, inspect, json, math, os, re, sys, time, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None

import lm_eval
from lm_eval.models.huggingface import HFLM


# ============================================================
# 벤치마크 설정
# ============================================================
BENCHMARKS = {
    "mmlu": dict(
        tasks=["mmlu"], num_fewshot=5,
        display="MMLU (5-shot)", metric_key="acc,none",
    ),
    "hellaswag": dict(
        tasks=["hellaswag"], num_fewshot=10,
        display="HellaSwag (10-shot)", metric_key="acc_norm,none",
    ),
    "gsm8k": dict(
        tasks=["gsm8k"], num_fewshot=5,
        display="GSM8K (5-shot CoT)", metric_key="exact_match,strict-match",
    ),
}


# ============================================================
# Seed 고정
# ============================================================
def _fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# PassLayer (파라미터 0, hidden_states 통과)
# ============================================================
class PassLayer(nn.Module):
    def __init__(self, return_tuple: bool = True):
        super().__init__()
        self._rt = return_tuple

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, **kw):
        if not self._rt:
            return hidden_states
        outs = (hidden_states,)
        if output_attentions:
            outs += (None,)
        if use_cache:
            outs += (past_key_value,)
        return outs


# ============================================================
# 레이어 컨테이너 탐색 (Llama / Mistral / Falcon)
# ============================================================
def _get_layers(model) -> nn.ModuleList:
    for chain in [
        ("model", "layers"),
        ("transformer", "h"),
        ("model", "model", "layers"),
    ]:
        obj = model
        for a in chain:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if obj is not None and isinstance(obj, nn.ModuleList):
            return obj
    raise RuntimeError("레이어 컨테이너를 찾을 수 없습니다.")


def _get_decoder_layer_class(model):
    mt = getattr(getattr(model, "config", None), "model_type", "").lower()
    for key, mod_path, cls_name in [
        ("mistral", "transformers.models.mistral.modeling_mistral", "MistralDecoderLayer"),
        ("llama",   "transformers.models.llama.modeling_llama",     "LlamaDecoderLayer"),
        ("falcon",  "transformers.models.falcon.modeling_falcon",   "FalconDecoderLayer"),
    ]:
        if key in mt:
            try:
                mod = __import__(mod_path, fromlist=[cls_name])
                return getattr(mod, cls_name)
            except Exception:
                pass
    for layer in _get_layers(model):
        if not isinstance(layer, PassLayer):
            return type(layer)
    return None


# ============================================================
# manifest.json 읽기
# ============================================================
def _read_dropped_layers(model_path: str) -> List[int]:
    fp = os.path.join(model_path, "manifest.json")
    if not os.path.isfile(fp):
        return []
    try:
        m = json.load(open(fp))
    except Exception:
        return []
    stages = m.get("stages", {})
    dropped = stages.get("A", {}).get("dropped_layers", [])
    if not dropped:
        B = stages.get("B", {}).get("removed_layers", [])
        C = stages.get("C", {}).get("removed_layers", [])
        dropped = sorted(set(B + C))
    if not dropped:
        dropped = m.get("simdrop", {}).get("removed_layers", [])
    return sorted(set(int(i) for i in dropped))


def _read_bc_indices(model_path: str) -> Tuple[List[int], List[int]]:
    fp = os.path.join(model_path, "manifest.json")
    if not os.path.isfile(fp):
        return [], []
    try:
        m = json.load(open(fp))
    except Exception:
        return [], []
    stages = m.get("stages", {})
    b = sorted(set(int(i) for i in stages.get("B", {}).get("removed_layers", [])))
    c = sorted(set(int(i) for i in stages.get("C", {}).get("removed_layers", [])))
    return b, c


# ============================================================
# return_tuple 감지
# ============================================================
def _detect_return_tuple(model) -> bool:
    try:
        core = model.model if hasattr(model, "model") else model
        if hasattr(core, "transformer"):
            core = core.transformer
        src = inspect.getsource(core.forward)
        if "layer_outputs[0]" in src:
            return True
        if "hidden_states = " in src and "layer_outputs" not in src:
            return False
    except Exception:
        pass
    return True


# ============================================================
# PassLayer 설치
# ============================================================
def _install_passlayers(model, dropped: List[int], return_tuple: bool = True):
    if not dropped:
        return model
    layers = _get_layers(model)
    for idx in dropped:
        if 0 <= idx < len(layers):
            old = layers[idx]
            dev = next(old.parameters()).device if sum(1 for _ in old.parameters()) > 0 else torch.device("cpu")
            layers[idx] = PassLayer(return_tuple=return_tuple).to(dev)
            del old
    print(f"    PassLayer 설치: {dropped} ({len(dropped)}개)")
    return model


# ============================================================
# Bundle 유틸리티
# ============================================================
_LAYER_FILE_RE = re.compile(r"^layer_(\d+)\.safetensors$")


def _build_layer_map(bundle_dir: Optional[Path]) -> Dict[int, Path]:
    if bundle_dir is None or not bundle_dir.is_dir():
        return {}
    out = {}
    for f in bundle_dir.iterdir():
        m = _LAYER_FILE_RE.match(f.name)
        if m:
            out[int(m.group(1))] = f
    return out


def _maybe_shift_indices(layer_map: Dict[int, Path], num_layers: int) -> Dict[int, Path]:
    if not layer_map:
        return layer_map
    indices = sorted(layer_map.keys())
    if all(0 <= i < num_layers for i in indices):
        return layer_map
    if all(1 <= i <= num_layers for i in indices):
        return {i - 1: p for i, p in layer_map.items()}
    raise ValueError(f"Bundle index mismatch (num_layers={num_layers}, indices={indices})")


def _strip_layer_prefix(sd: dict, layer_idx: int) -> dict:
    needles = [
        f"model.layers.{layer_idx}.", f"model.model.layers.{layer_idx}.",
        f"transformer.h.{layer_idx}.", f"layers.{layer_idx}.",
    ]
    out = {}
    for k, v in sd.items():
        nk = k
        for nd in needles:
            if nd in nk:
                nk = nk.split(nd, 1)[1]
                break
        out[nk] = v
    return out


def _is_bundle_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "config.json")):
        return False
    return any(_LAYER_FILE_RE.match(f) for f in os.listdir(path))


# ============================================================
# Bundle 복원: PassLayer → 실제 DecoderLayer (in-place)
# ============================================================
def _restore_bundle(model, bundle_dir: Path, device: str, dtype: torch.dtype) -> List[int]:
    layers = _get_layers(model)
    DecoderLayer = _get_decoder_layer_class(model)
    num_layers = len(layers)

    layer_map = _build_layer_map(bundle_dir)
    layer_map = _maybe_shift_indices(layer_map, num_layers)

    if not layer_map:
        print(f"    [warn] {bundle_dir}에 복원할 레이어 없음")
        return []

    restored = []
    for idx in sorted(layer_map.keys()):
        if idx < 0 or idx >= num_layers:
            print(f"    [warn] index {idx} 범위 초과")
            continue
        try:
            new_layer = DecoderLayer(model.config, idx)
        except TypeError:
            new_layer = DecoderLayer(model.config)

        new_layer = new_layer.to(device, dtype=dtype)
        sd = load_file(str(layer_map[idx]), device="cpu")
        sd = _strip_layer_prefix(sd, idx)
        new_layer.load_state_dict(sd, strict=False)

        old = layers[idx]
        layers[idx] = new_layer
        del old
        restored.append(idx)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"    복원 완료: layer {restored} ({len(restored)}개)")
    return restored


# ============================================================
# 레이어 상태 집계
# ============================================================
def _layer_status(model) -> Tuple[int, int]:
    layers = _get_layers(model)
    n_pass = sum(1 for l in layers if isinstance(l, PassLayer))
    return len(layers) - n_pass, n_pass


# ============================================================
# 모델 로드
# ============================================================
def _load_model(model_path: str, dtype: torch.dtype, device: str):
    print(f"    모델 로드: {model_path}")
    resolved = os.path.abspath(model_path) if os.path.exists(model_path) else model_path
    load_kw = dict(torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, attn_implementation="eager", **load_kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(resolved, **load_kw)

    dropped = _read_dropped_layers(model_path)
    if dropped:
        rt = _detect_return_tuple(model)
        model = _install_passlayers(model, dropped, return_tuple=rt)

    model = model.to(device)
    model.eval()
    return model, dropped


def _model_info(model, dropped: List[int]) -> str:
    n_layers = model.config.num_hidden_layers
    n_active = n_layers - len(dropped)
    n_params = sum(p.numel() for p in model.parameters())
    mt = model.config.model_type
    tag = f"{n_active} active + {len(dropped)} PassLayer" if dropped else f"{n_layers} layers"
    return f"{mt} | {tag} | {n_params / 1e6:.1f}M params"


# ============================================================
# 토크나이저
# ============================================================
def _load_tokenizer(path: str, fallbacks=None):
    for p in [path] + (fallbacks or []):
        try:
            return AutoTokenizer.from_pretrained(p, trust_remote_code=True)
        except Exception:
            continue
    raise RuntimeError(f"토크나이저 로드 실패: {path}")


def _find_tokenizer_fallbacks(model_path: str):
    fbs = []
    for name in ["adapter_config.json", "config.json"]:
        fp = os.path.join(model_path, name)
        if os.path.isfile(fp):
            try:
                cfg = json.load(open(fp))
                base = cfg.get("base_model_name_or_path") or cfg.get("_name_or_path")
                if base:
                    fbs.append(base)
            except Exception:
                pass
    return fbs


# ============================================================
# 벤치마크 실행
# ============================================================
def _extract_score(results: dict, bench_key: str) -> float:
    cfg = BENCHMARKS[bench_key]
    mk = cfg["metric_key"]
    res = results.get("results", {})

    for tn in cfg["tasks"]:
        if tn in res and mk in res[tn]:
            return res[tn][mk]

    # subtask 평균 (MMLU)
    sub = []
    for tn in cfg["tasks"]:
        for k, v in res.items():
            if k.startswith(tn) and isinstance(v, dict) and mk in v:
                sub.append(v[mk])
    if sub:
        return sum(sub) / len(sub)

    for v in res.values():
        if isinstance(v, dict):
            for mk2, mv in v.items():
                if "acc" in mk2 and isinstance(mv, (int, float)):
                    return mv
    return float("nan")


def run_benchmarks(model, tokenizer, bench_keys: List[str],
                   batch_size: int, device: str) -> Dict[str, float]:
    lm = HFLM(pretrained=model, tokenizer=tokenizer,
              batch_size=batch_size, device=device)

    scores = {}
    for bk in bench_keys:
        cfg = BENCHMARKS[bk]
        print(f"      ▶ {cfg['display']} ...", end=" ", flush=True)
        t0 = time.time()
        res = lm_eval.simple_evaluate(
            model=lm, tasks=cfg["tasks"],
            num_fewshot=cfg["num_fewshot"],
            batch_size=batch_size, device=device,
            log_samples=False,
        )
        sc = _extract_score(res, bk)
        scores[bk] = sc
        dt = time.time() - t0
        print(f"{sc * 100:.2f}%  ({dt:.1f}s)" if not math.isnan(sc) else f"FAILED ({dt:.1f}s)")

    return scores


# ============================================================
# 출력
# ============================================================
def _print_box(label: str, scores: Dict[str, float]):
    w = 52
    print(f"\n    ┌{'─' * w}┐")
    print(f"    │ {label:<{w - 1}}│")
    print(f"    ├{'─' * w}┤")
    for bk, sc in scores.items():
        val = f"{sc * 100:.2f}%" if not math.isnan(sc) else "N/A"
        line = f"{BENCHMARKS[bk]['display']}: {val}"
        print(f"    │ {line:<{w - 1}}│")
    print(f"    └{'─' * w}┘")


def _print_summary(all_results: List[Tuple[str, Dict[str, float]]],
                   bench_keys: List[str]):
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    headers = [BENCHMARKS[bk]["display"] for bk in bench_keys]
    hdr = f"  {'Label':<28s}"
    for h in headers:
        hdr += f"  {h:>18s}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for label, scores in all_results:
        row = f"  {label:<28s}"
        for bk in bench_keys:
            sc = scores.get(bk, float("nan"))
            val = f"{sc * 100:.2f}%" if not math.isnan(sc) else "N/A"
            row += f"  {val:>18s}"
        print(row)
    print(f"{'=' * 80}")


# ============================================================
# 메인
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="머지된 모델 벤치마크 평가 (순차 로드: A → +B → +C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model_path", type=str, nargs="+", required=True,
                    help="전체 모델 경로 (A_merged). bundle dir 자동 분류.")
    ap.add_argument("--b_bundle", type=str, default=None,
                    help="머지된 B 번들 디렉터리")
    ap.add_argument("--c_bundle", type=str, default=None,
                    help="머지된 C 번들 디렉터리")
    ap.add_argument("--bundles_dir", type=str, default=None,
                    help="구버전 호환: bundles_dir/B, bundles_dir/C")
    ap.add_argument("--stages", type=str, default=None,
                    help="평가 stage (기본: bundle에 따라 자동)")
    ap.add_argument("--benchmarks", type=str, default="mmlu,hellaswag,gsm8k",
                    help="벤치마크 (기본: mmlu,hellaswag,gsm8k)")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--output_dir", type=str, default="./eval_results",
                    help="결과 JSON 저장 폴더 (기본: ./eval_results)")
    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    _fix_seed(args.seed)

    bench_keys = [b.strip().lower() for b in args.benchmarks.split(",") if b.strip()]
    for bk in bench_keys:
        if bk not in BENCHMARKS:
            sys.exit(f"[error] 알 수 없는 벤치마크: {bk}")

    # ── bundle 경로 ──
    b_dir = Path(args.b_bundle) if args.b_bundle else None
    c_dir = Path(args.c_bundle) if args.c_bundle else None
    bundles_dir = Path(args.bundles_dir) if args.bundles_dir else None
    if bundles_dir and b_dir is None and (bundles_dir / "B").is_dir():
        b_dir = bundles_dir / "B"
    if bundles_dir and c_dir is None and (bundles_dir / "C").is_dir():
        c_dir = bundles_dir / "C"

    # ── model_path 자동 분류 ──
    full_model_paths = []
    for mpath in args.model_path:
        if _is_bundle_dir(mpath):
            name = os.path.basename(mpath).lower()
            if ("b_merged" in name or "/b" in mpath.lower()) and b_dir is None:
                b_dir = Path(mpath)
                print(f"  ★ Bundle B 자동 감지: {mpath}")
            elif ("c_merged" in name or "/c" in mpath.lower()) and c_dir is None:
                c_dir = Path(mpath)
                print(f"  ★ Bundle C 자동 감지: {mpath}")
            else:
                print(f"  ⚠ {mpath} → B/C 판별 불가, 스킵")
        else:
            full_model_paths.append(mpath)

    has_b = b_dir is not None
    has_c = c_dir is not None

    if not full_model_paths:
        sys.exit("[error] 전체 모델 경로가 최소 1개 필요합니다.")

    # ── stage 결정: 순차 누적이므로 A → AB → FULL 순서 고정 ──
    if args.stages is not None:
        stage_list = [s.strip().upper() for s in args.stages.split(",") if s.strip()]
    else:
        stage_list = ["A"]
        if has_b: stage_list.append("AB")
        if has_c: stage_list.append("FULL")
        print(f"  (stages 자동: {stage_list})")

    if not has_b and "AB" in stage_list:
        print("[warn] AB는 --b_bundle 필요. 제거.")
        stage_list = [s for s in stage_list if s != "AB"]
    if not has_c and "FULL" in stage_list:
        print("[warn] FULL은 --c_bundle 필요. 제거.")
        stage_list = [s for s in stage_list if s != "FULL"]

    # ── 헤더 ──
    print(f"\n{'=' * 80}")
    print("Merged Model Benchmark Evaluation  (sequential load: A → +B → +C)")
    print(f"{'=' * 80}")
    print(f"  Models:     {full_model_paths}")
    if has_b: print(f"  B bundle:   {b_dir}")
    if has_c: print(f"  C bundle:   {c_dir}")
    print(f"  Stages:     {stage_list}")
    print(f"  Benchmarks: {[BENCHMARKS[bk]['display'] for bk in bench_keys]}")
    print(f"  Device:     {args.device} ({args.dtype})")
    print(f"  Batch:      {args.batch_size}")
    print(f"  Seed:       {args.seed}")
    try:
        print(f"  lm-eval:    {lm_eval.__version__}")
    except AttributeError:
        pass
    print(f"  Output:     {args.output_dir}/")
    print(f"{'=' * 80}")

    all_results: List[Tuple[str, Dict[str, float]]] = []

    # ── 모델별 루프 ──
    for mi, mpath in enumerate(full_model_paths):
        print(f"\n{'─' * 65}")
        print(f"[{mi + 1}/{len(full_model_paths)}] {mpath}")
        print(f"{'─' * 65}")

        # 토크나이저
        tok_path = args.tokenizer_path or mpath
        fallbacks = None if args.tokenizer_path else _find_tokenizer_fallbacks(mpath)
        tok = _load_tokenizer(tok_path, fallbacks)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # manifest 확인
        b_idx, c_idx = _read_bc_indices(mpath)
        if b_idx or c_idx:
            print(f"  manifest → B: {b_idx}, C: {c_idx}")

        model = None
        dropped = []

        for stage in stage_list:
            stage_tag = {"A": "A", "AB": "AB", "FULL": "ABC"}.get(stage, stage)
            label = f"{os.path.basename(mpath)}[{stage_tag}]"

            print(f"\n  ━━━ Stage {stage} ━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # ──────────────────────────────────────────────
            # A: 모델 로드 + PassLayer 설치
            # ──────────────────────────────────────────────
            if stage == "A":
                _fix_seed(args.seed)
                model, dropped = _load_model(mpath, dtype, args.device)
                print(f"    ✓ {_model_info(model, dropped)}")

            # ──────────────────────────────────────────────
            # AB: 기존 모델에 B bundle 누적 복원
            # ──────────────────────────────────────────────
            elif stage == "AB":
                assert model is not None, "A stage를 먼저 실행해야 합니다."
                print(f"    B 번들 복원: {b_dir}")
                _restore_bundle(model, b_dir, args.device, dtype)

            # ──────────────────────────────────────────────
            # FULL: 기존 모델에 C bundle 누적 복원
            # ──────────────────────────────────────────────
            elif stage == "FULL":
                assert model is not None, "A stage를 먼저 실행해야 합니다."
                print(f"    C 번들 복원: {c_dir}")
                _restore_bundle(model, c_dir, args.device, dtype)

            # 현재 레이어 상태
            n_act, n_pass = _layer_status(model)
            print(f"    → {n_act} active + {n_pass} PassLayer")

            # 벤치마크 실행
            print(f"    벤치마크:")
            scores = run_benchmarks(
                model, tok, bench_keys,
                batch_size=args.batch_size, device=args.device,
            )
            _print_box(label, scores)
            all_results.append((label, scores))

        # 모델 해제
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 요약 ──
    _print_summary(all_results, bench_keys)

    # ── eval_results/ 에 JSON 자동 저장 ──
    out_dir = Path(args.output_dir)
    _save_results(out_dir, all_results, bench_keys, args, full_model_paths,
                  b_dir, c_dir, stage_list)


# ============================================================
# JSON 저장
# ============================================================
def _save_results(out_dir: Path,
                  all_results: List[Tuple[str, Dict[str, float]]],
                  bench_keys: List[str],
                  args, model_paths: List[str],
                  b_dir: Optional[Path], c_dir: Optional[Path],
                  stage_list: List[str]):
    """eval_results/ 에 per-stage JSON + summary JSON 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 타임스탬프
    ts = time.strftime("%Y%m%d_%H%M%S")

    # 공통 메타
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "model_paths": model_paths,
        "b_bundle": str(b_dir) if b_dir else None,
        "c_bundle": str(c_dir) if c_dir else None,
        "stages": stage_list,
        "benchmarks": bench_keys,
    }
    try:
        meta["lm_eval_version"] = lm_eval.__version__
    except AttributeError:
        pass
    try:
        import transformers
        meta["transformers_version"] = transformers.__version__
    except Exception:
        pass
    meta["torch_version"] = torch.__version__

    # ── per-stage JSON ──
    for label, scores in all_results:
        # label: "A_merged[A]", "A_merged[AB]", "A_merged[ABC]"
        safe_name = label.replace("/", "_").replace("[", "_").replace("]", "")
        stage_data = {
            "label": label,
            "meta": meta,
            "results": {},
        }
        for bk in bench_keys:
            sc = scores.get(bk, float("nan"))
            stage_data["results"][bk] = {
                "display_name": BENCHMARKS[bk]["display"],
                "score": round(sc * 100, 2) if not math.isnan(sc) else None,
                "score_raw": sc if not math.isnan(sc) else None,
                "num_fewshot": BENCHMARKS[bk]["num_fewshot"],
            }

        fp = out_dir / f"{safe_name}_{ts}.json"
        with open(fp, "w") as f:
            json.dump(stage_data, f, indent=2, ensure_ascii=False)
        print(f"  저장: {fp}")

    # ── summary JSON (전 stage 비교표) ──
    summary = {
        "meta": meta,
        "summary_table": {},
    }
    for label, scores in all_results:
        summary["summary_table"][label] = {}
        for bk in bench_keys:
            sc = scores.get(bk, float("nan"))
            summary["summary_table"][label][BENCHMARKS[bk]["display"]] = (
                round(sc * 100, 2) if not math.isnan(sc) else None
            )

    # 모델 이름 추출 (파일명용)
    model_name = os.path.basename(model_paths[0]).replace("/", "_") if model_paths else "model"
    sfp = out_dir / f"summary_{model_name}_{ts}.json"
    with open(sfp, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  저장: {sfp}")

    print(f"\n전체 결과: {out_dir}/")


if __name__ == "__main__":
    main()