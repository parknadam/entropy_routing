#!/usr/bin/env python3
"""
머지된 Falcon 모델의 SQuAD EM/F1 평가 스크립트.

훈련 시 사용한 프롬프트 포맷과 동일하게 맞춰서 생성 → EM/F1 계산.

사용법:
  # 머지 모델 평가
  CUDA_VISIBLE_DEVICES=2 DEVICE=cuda:0 \
  python falcon_prune_lora/falcon_eval_squad.py \
    --model_path ./merged_models_falcon/A_merged \
    --device cuda:0

  # 원본 모델과 비교
  CUDA_VISIBLE_DEVICES=2 DEVICE=cuda:0 \
  python falcon_eval_squad.py \
    --model_path tiiuae/falcon-7b-instruct ./merged_models_falcon/A_merged \
    --tokenizer_path tiiuae/falcon-7b-instruct \
    --device cuda:0

  # stage별 비교
  python falcon_eval_squad.py \
    --model_path ./merged_models_falcon/A_merged \
    --b_bundle ./merged_models_falcon/B_merged \
    --c_bundle ./merged_models_falcon/C_merged \
    --stages A,AB,FULL \
    --device cuda:0

  # answer-only PPL도 함께 측정
  python falcon_eval_squad.py \
    --model_path ./merged_models_falcon/A_merged \
    --eval_ppl \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import string
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None

FalconDecoderLayer = None
try:
    from transformers.models.falcon.modeling_falcon import FalconDecoderLayer as _FDL
    FalconDecoderLayer = _FDL
except ImportError:
    pass


# ============================================================
# EM / F1 metrics (SQuAD official)
# ============================================================
def _normalize_answer(s: str) -> str:
    """SQuAD 공식 정규화: 소문자, 관사/구두점/공백 제거."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    return max(metric_fn(prediction, gt) for gt in ground_truths)


# ============================================================
# PassLayer (eval_ppl_mergedmodel과 동일)
# ============================================================
class FalconPassLayer(nn.Module):
    def __init__(self, return_tuple: bool = True):
        super().__init__()
        self.return_tuple = return_tuple

    def forward(self, hidden_states, **kwargs):
        if not self.return_tuple:
            return hidden_states
        if kwargs.get("use_cache", False):
            return (hidden_states, kwargs.get("layer_past", None))
        return (hidden_states,)


# ============================================================
# 모델 로드 유틸 (eval_ppl_mergedmodel에서 가져옴)
# ============================================================
def _get_falcon_layers(model) -> nn.ModuleList:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model"):
        if hasattr(model.model, "transformer") and hasattr(model.model.transformer, "h"):
            return model.model.transformer.h
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model") and hasattr(base.model, "transformer"):
            return base.model.transformer.h
        if hasattr(base, "transformer"):
            return base.transformer.h
    raise RuntimeError("Cannot find Falcon layers (expected transformer.h)")


def _read_dropped_layers(model_path: str) -> List[int]:
    manifest_path = os.path.join(model_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        return []
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception:
        return []
    stages = manifest.get("stages", {})
    dropped = stages.get("A", {}).get("dropped_layers", [])
    if not dropped:
        B = stages.get("B", {}).get("removed_layers", [])
        C = stages.get("C", {}).get("removed_layers", [])
        dropped = sorted(set(B + C))
    if not dropped:
        dropped = manifest.get("simdrop", {}).get("removed_layers", [])
    return sorted(set(int(i) for i in dropped))


def _install_passlayers(model, dropped_indices, return_tuple=True):
    if not dropped_indices:
        return model
    layers = _get_falcon_layers(model)
    for idx in dropped_indices:
        if 0 <= idx < len(layers):
            dev = next(layers[idx].parameters()).device if sum(1 for _ in layers[idx].parameters()) > 0 else torch.device("cpu")
            layers[idx] = FalconPassLayer(return_tuple=return_tuple).to(dev)
    print(f"  ✓ PassLayer 설치: {dropped_indices}")
    return model


def _detect_layer_return_tuple(model) -> bool:
    try:
        import inspect
        core = model.transformer if hasattr(model, "transformer") else model
        src = inspect.getsource(core.forward)
        if "layer_outputs[0]" in src:
            return True
    except Exception:
        pass
    return True


def _load_model(model_path: str, dtype, device: str):
    print(f"  Loading model from: {model_path}")
    resolved = os.path.abspath(model_path) if os.path.exists(model_path) else model_path
    try:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, torch_dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation="eager", trust_remote_code=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
        )
    dropped = _read_dropped_layers(model_path)
    if dropped:
        rt = _detect_layer_return_tuple(model)
        model = _install_passlayers(model, dropped, return_tuple=rt)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ {n_params/1e6:.1f}M params | {device}")
    return model


def _load_tokenizer(model_path: str, tokenizer_path: str = None):
    path = tokenizer_path or model_path
    resolved = os.path.abspath(path) if os.path.exists(path) else path
    try:
        tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    except Exception:
        # fallback: manifest에서 base_model 읽기
        manifest_path = os.path.join(model_path, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                base = json.load(f).get("base_model")
            if base:
                tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
            else:
                raise
        else:
            raise
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ============================================================
# DynamicStageManager (simplified from eval_ppl_mergedmodel)
# ============================================================
_LAYER_RE = re.compile(r"layer_(\d+)\.safetensors$")


def _build_layer_map(dir_path: Path) -> Dict[int, Path]:
    m = {}
    if not dir_path.exists():
        return m
    for p in dir_path.glob("layer_*.safetensors"):
        mm = _LAYER_RE.search(p.name)
        if mm:
            m[int(mm.group(1))] = p
    return m


class DynamicStageManager:
    def __init__(self, model, device, dtype, passlayer_rt,
                 b_dir=None, c_dir=None, bundles_dir=None):
        if load_file is None:
            raise RuntimeError("safetensors required")
        self.model = model
        self.layers = _get_falcon_layers(model)
        self.device = device
        self.dtype = dtype
        self.passlayer_rt = passlayer_rt
        self.num_layers = len(self.layers)

        actual_b = b_dir if b_dir else (bundles_dir / "B" if bundles_dir else None)
        actual_c = c_dir if c_dir else (bundles_dir / "C" if bundles_dir else None)
        self.B_map = _build_layer_map(actual_b) if actual_b else {}
        self.C_map = _build_layer_map(actual_c) if actual_c else {}
        self.B_idx = sorted(self.B_map.keys())
        self.C_idx = sorted(self.C_map.keys())
        self.removed = sorted(set(self.B_idx) | set(self.C_idx))

    def _restore(self, i):
        p = self.B_map.get(i) or self.C_map.get(i)
        if not p:
            raise FileNotFoundError(f"layer_{i}.safetensors not found")
        try:
            new = FalconDecoderLayer(self.model.config, i)
        except TypeError:
            new = FalconDecoderLayer(self.model.config)
        new = new.to(self.device, dtype=self.dtype)
        sd = load_file(str(p), device="cpu")
        # strip prefix
        prefixes = [f"transformer.h.{i}.", f"h.{i}.", f"model.transformer.h.{i}."]
        out = {}
        for k, v in sd.items():
            for pf in prefixes:
                if k.startswith(pf):
                    out[k[len(pf):]] = v
                    break
            else:
                out[k] = v
        new.load_state_dict(out, strict=False)
        self.layers[i] = new

    def set_stage(self, stage: str):
        stage = stage.upper()
        pass_set = set(self.removed) if stage == "A" else (set(self.C_idx) if stage == "AB" else set())
        for i in self.removed:
            is_pass = isinstance(self.layers[i], FalconPassLayer)
            if i in pass_set and not is_pass:
                self.layers[i] = FalconPassLayer(self.passlayer_rt).to(self.device)
            elif i not in pass_set and is_pass:
                self._restore(i)
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


# ============================================================
# 프롬프트 구성 (훈련 코드와 동일)
# ============================================================
def build_prompt(tokenizer, context: str, question: str) -> str:
    """훈련 시와 동일한 프롬프트 포맷 생성."""
    sys_msg = "You are a helpful QA assistant. Answer the question based on the given context."
    user_msg = f"Context: {context}\n\nQuestion: {question}"
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = f"### System:\n{sys_msg}\n### User:\n{user_msg}\n### Assistant:\n"
    return prompt


# ============================================================
# 생성 + 평가
# ============================================================
@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, device: str,
                    max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1920)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,           # greedy
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # 생성된 부분만 디코드
    gen_ids = outputs[0, input_ids.shape[1]:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    # 첫 줄만 (multi-line 방지)
    answer = answer.split("\n")[0].strip()
    return answer


@torch.no_grad()
def eval_answer_ppl(model, tokenizer, context, question, answer, device, seq_len=1024):
    """Answer 부분에만 loss를 걸어 PPL 측정 (훈련과 동일한 방식)."""
    prompt = build_prompt(tokenizer, context, question)
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    ans_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id:
        ans_ids = ans_ids + [tokenizer.eos_token_id]

    full = prompt_ids + ans_ids
    prompt_len = len(prompt_ids)

    if len(full) > seq_len:
        cut = len(full) - seq_len
        full = full[cut:]
        prompt_len = max(0, prompt_len - cut)

    input_ids = torch.tensor([full], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits  # [1, L, V]

    # answer 영역만 loss
    # prompt_len 위치부터가 answer의 첫 토큰 예측
    if prompt_len >= len(full) - 1:
        return 0.0, 0

    shift_logits = logits[:, prompt_len - 1:-1, :]  # answer 예측 logits
    shift_labels = input_ids[:, prompt_len:]          # answer 실제 토큰

    V = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.float().reshape(-1, V),
        shift_labels.reshape(-1),
        reduction="sum",
    )
    n_tokens = shift_labels.numel()
    return float(loss.item()), n_tokens


def evaluate_squad(model, tokenizer, dataset, device: str,
                   max_samples: int = None, max_new_tokens: int = 64,
                   eval_ppl: bool = False, seq_len: int = 1024):
    """SQuAD validation set에서 EM/F1 (+ optional answer-only PPL) 측정."""
    total_em = 0.0
    total_f1 = 0.0
    total_nll = 0.0
    total_tok = 0
    count = 0

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for i in tqdm(range(n), desc="Evaluating"):
        example = dataset[i]
        context = example["context"]
        question = example["question"]
        gold_answers = example["answers"]["text"]

        if not gold_answers:
            continue

        prompt = build_prompt(tokenizer, context, question)
        if i < 2:
            print(f"  [DEBUG PROMPT]\n{repr(prompt[:300])}")
        prediction = generate_answer(model, tokenizer, prompt, device, max_new_tokens)

        em = metric_max_over_ground_truths(compute_exact, prediction, gold_answers)
        f1 = metric_max_over_ground_truths(compute_f1, prediction, gold_answers)
        total_em += em
        total_f1 += f1
        count += 1

        if eval_ppl:
            nll, ntok = eval_answer_ppl(
                model, tokenizer, context, question, gold_answers[0], device, seq_len
            )
            total_nll += nll
            total_tok += ntok

        # 처음 5개 예시 출력
        if i < 5:
            print(f"\n  --- Example {i+1} ---")
            print(f"  Q: {question[:100]}...")
            print(f"  Gold: {gold_answers[0][:80]}")
            print(f"  Pred: {prediction[:80]}")
            print(f"  EM={em:.0f}  F1={f1:.4f}")

    results = {
        "exact_match": total_em / count * 100 if count else 0,
        "f1": total_f1 / count * 100 if count else 0,
        "count": count,
    }
    if eval_ppl and total_tok > 0:
        mean_nll = total_nll / total_tok
        results["answer_ppl"] = math.exp(min(mean_nll, 100.0))
        results["answer_mean_nll"] = mean_nll
        results["answer_tokens"] = total_tok

    return results


def _print_result_box(label: str, r: dict):
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │ {label:<44s}│")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │ EM         = {r['exact_match']:<30.2f} │")
    print(f"  │ F1         = {r['f1']:<30.2f} │")
    print(f"  │ Samples    = {r['count']:<30d} │")
    if "answer_ppl" in r:
        print(f"  │ Answer PPL = {r['answer_ppl']:<30.4f} │")
        print(f"  │ Answer NLL = {r['answer_mean_nll']:<30.6f} │")
    print(f"  └─────────────────────────────────────────────┘")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="머지된 Falcon 모델의 SQuAD EM/F1 평가")
    ap.add_argument("--model_path", type=str, nargs="+", required=True)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    # 데이터
    ap.add_argument("--dataset", default="squad", choices=["squad", "squad_v2"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max_samples", type=int, default=1000,
                    help="평가할 샘플 수 (전체: None)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)

    # PPL
    ap.add_argument("--eval_ppl", action="store_true",
                    help="Answer-only PPL도 함께 측정")
    ap.add_argument("--seqlen", type=int, default=1024)

    # Stage 관련
    ap.add_argument("--b_bundle", default=None)
    ap.add_argument("--c_bundle", default=None)
    ap.add_argument("--bundles_dir", default=None)
    ap.add_argument("--stages", default=None)

    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Bundle paths
    b_dir = Path(args.b_bundle) if args.b_bundle else None
    c_dir = Path(args.c_bundle) if args.c_bundle else None
    bundles_dir = Path(args.bundles_dir) if args.bundles_dir else None
    has_bundles = b_dir or c_dir or bundles_dir

    if args.stages:
        stage_list = [s.strip().upper() for s in args.stages.split(",")]
    elif has_bundles:
        stage_list = ["A", "AB", "FULL"]
    else:
        stage_list = ["A"]

    # 데이터셋 로드
    print(f"\nLoading {args.dataset} ({args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"  ✓ {len(ds)} examples (evaluating {args.max_samples or 'all'})")

    results = []

    print(f"\n{'='*60}")
    print(f"SQuAD EM/F1 Evaluation")
    print(f"{'='*60}")

    for mpath in args.model_path:
        print(f"\n{'─'*60}")
        print(f"Model: {mpath}")
        print(f"{'─'*60}")

        tok = _load_tokenizer(mpath, args.tokenizer_path)
        model = _load_model(mpath, dtype, args.device)

        mgr = None
        if has_bundles and any(s in ("AB", "FULL") for s in stage_list):
            rt = _detect_layer_return_tuple(model)
            mgr = DynamicStageManager(model, args.device, dtype, rt,
                                      b_dir=b_dir, c_dir=c_dir, bundles_dir=bundles_dir)
            print(f"  Stage meta: B={mgr.B_idx}, C={mgr.C_idx}")

        for stage in stage_list:
            if stage in ("AB", "FULL") and mgr:
                mgr.set_stage(stage)
            elif stage == "A" and mgr:
                mgr.set_stage("A")

            label = f"{Path(mpath).name}[{stage}]"
            print(f"\n  [{stage}] Evaluating EM/F1...")

            r = evaluate_squad(
                model, tok, ds, args.device,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                eval_ppl=args.eval_ppl,
                seq_len=args.seqlen,
            )
            _print_result_box(label, r)
            results.append({"model": mpath, "stage": stage, "label": label, **r})

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"{'Label':<40s} {'EM':>8s} {'F1':>8s} {'N':>6s}")
        print(f"{'─'*40} {'─'*8} {'─'*8} {'─'*6}")
        for r in results:
            print(f"{r['label']:<40s} {r['exact_match']:>8.2f} {r['f1']:>8.2f} {r['count']:>6d}")

    # JSON 저장
    out_path = "squad_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()