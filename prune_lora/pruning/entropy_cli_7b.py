""" # prune_lora/pruning/entropy_cli.py
# REPL: prompt 입력 → (A/AB/FULL) 단계별 prompt entropy/PPL 측정 → stage 결정 → 해당 stage로 생성
#
# 사용:
   """ 
   python -m prune_lora.pruning.entropy_cli \
     --base_model meta-llama/Llama-2-7b-hf \
     --bundles_dir ./results/pruning/bundles \
     --max_new_tokens 128
 """
# bundles 구조:
#   results/pruning/bundles/B/layer_021.safetensors ...
#   results/pruning/bundles/C/layer_025.safetensors ...
#
# NOTE:
# - PassLayer는 Tensor만 반환하도록 구현(튜플 반환하면 transformers loop에서 터짐)
# - bundles 파일명 0-padding(layer_021) 대응: 디렉토리 스캔으로 idx->path map 구성
# - KV cache는 안정성 위해 CLI에서는 기본 use_cache=False로 평가/생성

""" from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# PassLayer: 삭제 레이어 자리 유지
# -----------------------------
class LlamaPassLayer(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        # ✅ transformers llama loop는 보통 Tensor hidden_states를 기대
        return hidden_states


def _get_llama_layers(model) -> nn.ModuleList:
    # HF 버전 차이 대응
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        return model.model.model.layers
    raise RuntimeError("LLaMA layers 경로를 찾지 못했어요. (예: model.model.layers)")


# -----------------------------
# bundles 스캔 유틸
# -----------------------------
_LAYER_RE = re.compile(r"layer_(\d+)\.safetensors$")


def _build_layer_map(dir_path: Path) -> Dict[int, Path]:
    """
    #bundles/B 또는 bundles/C 스캔해서 {idx:int -> path:Path} 맵 구성.
    #layer_021.safetensors 같은 0-padding도 idx=21로 파싱됨.
    """
    m: Dict[int, Path] = {}
    if not dir_path.exists():
        return m
    for p in dir_path.glob("layer_*.safetensors"):
        mm = _LAYER_RE.search(p.name)
        if mm:
            idx = int(mm.group(1))
            m[idx] = p
    return m


def _strip_layer_prefix(sd: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, torch.Tensor]:
    """
    #bundles의 key가 다양한 prefix를 가질 수 있어 레이어 내부 키로 정규화:
    #  - model.layers.{i}.xxx
    #  - model.model.layers.{i}.xxx
    #  - layers.{i}.xxx
    #  - 이미 xxx
    """
    out = {}
    needles = [
        f"model.layers.{layer_idx}.",
        f"model.model.layers.{layer_idx}.",
        f"layers.{layer_idx}.",
    ]
    for k, v in sd.items():
        nk = k
        for nd in needles:
            if nd in nk:
                nk = nk.split(nd, 1)[1]
                break
        out[nk] = v
    return out


# -----------------------------
# transformers 버전에 따라 LlamaDecoderLayer import 경로가 다름
# -----------------------------
try:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
except Exception:
    LlamaDecoderLayer = None


# -----------------------------
# Stage 스왑 매니저: A / AB / FULL
# -----------------------------
class DynamicStageManager:
    def __init__(
        self,
        model,
        bundles_dir: Path,
        device: str,
        dtype: torch.dtype,
    ):
        if LlamaDecoderLayer is None:
            raise RuntimeError(
                "transformers에서 LlamaDecoderLayer import에 실패했어요. "
                "transformers 버전이 너무 다르거나 llama 모델이 아닐 수 있어요."
            )

        self.model = model
        self.layers = _get_llama_layers(model)
        self.device = device
        self.dtype = dtype

        # ✅ 먼저 디렉토리 경로 정의
        self.B_dir = bundles_dir / "B"
        self.C_dir = bundles_dir / "C"

        # ✅ idx->path 맵 (0-padding 대응)
        self.B_map = _build_layer_map(self.B_dir)
        self.C_map = _build_layer_map(self.C_dir)

        self.B_idx = sorted(self.B_map.keys())
        self.C_idx = sorted(self.C_map.keys())
        self.removed = sorted(set(self.B_idx) | set(self.C_idx))

        # 시작은 A (removed 모두 PassLayer)
        self.set_stage("A")

    def stage_meta(self) -> Dict:
        return {"B": self.B_idx, "C": self.C_idx, "removed": self.removed}

    def _bundle_path(self, layer_i: int) -> Optional[Path]:
        if layer_i in self.B_map:
            return self.B_map[layer_i]
        if layer_i in self.C_map:
            return self.C_map[layer_i]
        return None

    def _restore_one_layer(self, layer_i: int):
        p = self._bundle_path(layer_i)
        if p is None:
            raise FileNotFoundError(f"layer_{layer_i}.*.safetensors를 B/C에서 찾지 못했어요.")

        # 새 디코더 레이어 생성: 버전에 따라 인자 다를 수 있음
        try:
            new_layer = LlamaDecoderLayer(self.model.config, layer_i)
        except TypeError:
            new_layer = LlamaDecoderLayer(self.model.config)

        new_layer = new_layer.to(self.device, dtype=self.dtype)

        sd = load_file(str(p), device="cpu")
        sd = _strip_layer_prefix(sd, layer_i)
        new_layer.load_state_dict(sd, strict=False)

        old = self.layers[layer_i]
        self.layers[layer_i] = new_layer
        del old

    def _pass_one_layer(self, layer_i: int):
        old = self.layers[layer_i]
        self.layers[layer_i] = LlamaPassLayer().to(self.device)
        del old

    def set_stage(self, stage: str):
        stage = stage.upper()
        if stage not in ("A", "AB", "FULL"):
            raise ValueError("stage는 A / AB / FULL 중 하나여야 해요.")

        if stage == "A":
            restore_set = set()
            pass_set = set(self.removed)
        elif stage == "AB":
            restore_set = set(self.B_idx)
            pass_set = set(self.C_idx)
        else:  # FULL
            restore_set = set(self.removed)
            pass_set = set()

        # removed 레이어에 대해서만 필요한 변경 적용
        for i in self.removed:
            cur = self.layers[i]
            is_pass = isinstance(cur, LlamaPassLayer)

            if i in pass_set:
                if not is_pass:
                    self._pass_one_layer(i)
            else:  # restore
                if is_pass:
                    self._restore_one_layer(i)

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


# -----------------------------
# Metrics: prompt entropy / ppl
# -----------------------------
@torch.no_grad()
def prompt_entropy_and_ppl(model, tokenizer, prompt: str, max_length: int, device: str) -> Dict[str, float]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)      # [1, L]
    attn = enc["attention_mask"].to(device)      # [1, L]

    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits                           # [1, L, V]

    # shift
    logits_use = logits[:, :-1, :]               # [1, L-1, V]
    target = input_ids[:, 1:]                    # [1, L-1]
    mask = attn[:, 1:].to(logits_use.dtype)      # [1, L-1]
    denom = mask.sum().clamp_min(1.0)

    # PPL: teacher-forcing NLL
    log_probs = F.log_softmax(logits_use, dim=-1)
    token_logp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
    token_nll = -token_logp * mask
    nll_mean = (token_nll.sum() / denom).item()
    ppl = float(math.exp(nll_mean))

    # predictive entropy mean
    probs = F.softmax(logits_use, dim=-1)
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)            # [1, L-1]
    ent_mean = ((ent * mask).sum() / denom).item()

    # last entropy (next token distribution at prompt end)
    last_logits = logits[:, -1, :]
    last_probs = F.softmax(last_logits, dim=-1)
    last_ent = float((-(last_probs * last_probs.clamp_min(1e-12).log()).sum()).item())

    return {
        "prompt_entropy_mean": ent_mean,
        "prompt_last_entropy": last_ent,
        "prompt_ppl": ppl,
        "prompt_nll": nll_mean,
        "prompt_tokens_scored": int(denom.item()),
        "prompt_len_tokens": int(attn.sum().item()),
    }


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,  # ✅ 안정성 우선
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(gen[0], skip_special_tokens=True)


# -----------------------------
# Router: entropy + ppl + min_tokens로 승급
# -----------------------------
@dataclass
class RouterConfig:
    # 짧은 프롬프트는 last_entropy가 흔들리기 쉬움 → 보수적으로 승급
    min_tokens: int = 50

    # A → AB
    thr_A_ent_mean: float = 2.6
    thr_A_last_ent: float = 2.5
    thr_A_ppl: float = 40.0

    # AB → FULL
    thr_AB_ent_mean: float = 2.7
    thr_AB_last_ent: float = 2.6
    thr_AB_ppl: float = 35.0


def decide_stage(mA: Dict[str, float], mAB: Optional[Dict[str, float]], cfg: RouterConfig) -> str:
    # 0) 너무 짧으면 A 확신 금지
    if mA["prompt_tokens_scored"] < cfg.min_tokens:
        return "AB"

    need_up_A = (
        (mA["prompt_entropy_mean"] >= cfg.thr_A_ent_mean) or
        (mA["prompt_last_entropy"] >= cfg.thr_A_last_ent) or
        (mA["prompt_ppl"] >= cfg.thr_A_ppl)
    )
    if not need_up_A:
        return "A"

    if mAB is None:
        return "AB"

    need_up_AB = (
        (mAB["prompt_entropy_mean"] >= cfg.thr_AB_ent_mean) or
        (mAB["prompt_last_entropy"] >= cfg.thr_AB_last_ent) or
        (mAB["prompt_ppl"] >= cfg.thr_AB_ppl)
    )
    return "FULL" if need_up_AB else "AB"


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="원본 모델 경로/HF id (예: meta-llama/Llama-2-7b-hf)")
    ap.add_argument("--bundles_dir", required=True, help="layeronly_drop.py의 save_removed_dir (B/, C/ 포함)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=128)

    # 기존 CLI 인자 호환(너가 이미 쓰고 있어서)
    ap.add_argument("--thr_A_to_AB", type=float, default=2.5, help="(호환용) A->AB last_entropy 기준. 실제 결정은 RouterConfig 사용")
    ap.add_argument("--thr_AB_to_FULL", type=float, default=2.9, help="(호환용) AB->FULL last_entropy 기준. 실제 결정은 RouterConfig 사용")

    # 라우터 파라미터(필요시 CLI로 튜닝 가능하게)
    ap.add_argument("--min_tokens", type=int, default=50)
    ap.add_argument("--thr_A_ent_mean", type=float, default=2.6)
    ap.add_argument("--thr_A_last_ent", type=float, default=None)
    ap.add_argument("--thr_A_ppl", type=float, default=40.0)

    ap.add_argument("--thr_AB_ent_mean", type=float, default=2.7)
    ap.add_argument("--thr_AB_last_ent", type=float, default=None)
    ap.add_argument("--thr_AB_ppl", type=float, default=35.0)

    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    mgr = DynamicStageManager(
        model=model,
        bundles_dir=Path(args.bundles_dir),
        device=args.device,
        dtype=dtype,
    )

    # 기존 인자(thr_A_to_AB, thr_AB_to_FULL)를 last_entropy 기본값으로 반영
    cfg = RouterConfig(
        min_tokens=args.min_tokens,
        thr_A_ent_mean=args.thr_A_ent_mean,
        thr_A_last_ent=(args.thr_A_last_ent if args.thr_A_last_ent is not None else args.thr_A_to_AB),
        thr_A_ppl=args.thr_A_ppl,
        thr_AB_ent_mean=args.thr_AB_ent_mean,
        thr_AB_last_ent=(args.thr_AB_last_ent if args.thr_AB_last_ent is not None else args.thr_AB_to_FULL),
        thr_AB_ppl=args.thr_AB_ppl,
    )

    print("\n=== GROUP META (from bundles) ===")
    print(mgr.stage_meta())
    print("\n[REPL] 프롬프트 입력. 종료: Ctrl+C\n")

    while True:
        prompt = input("PROMPT> ").strip()
        if not prompt:
            continue

        # 1) A에서 측정
        mgr.set_stage("A")
        mA = prompt_entropy_and_ppl(model, tok, prompt, args.max_length, args.device)

        # 2) AB에서 측정(항상 측정: 디버그/안정 우선)
        mAB = None
        if len(mgr.B_idx) > 0:
            mgr.set_stage("AB")
            mAB = prompt_entropy_and_ppl(model, tok, prompt, args.max_length, args.device)

        # 3) stage 결정
        stage = decide_stage(mA, mAB, cfg)

        # 4) stage로 세팅 후 생성
        mgr.set_stage(stage)
        out = generate_text(model, tok, prompt, args.max_new_tokens, args.device)

        print("\n--- decision ---")
        print(f"stage = {stage}")
        if stage == "A":
            print("loaded groups: A only (B,C are PassLayer)")
        elif stage == "AB":
            print(f"loaded groups: A + B (restored indices: {mgr.B_idx}) / C are PassLayer")
        else:
            print(f"loaded groups: A + B + C (restored indices: {mgr.removed})")

        print("\n--- prompt metrics (A) ---")
        for k, v in mA.items():
            print(f"{k}: {v}")

        if mAB is not None:
            print("\n--- prompt metrics (AB) ---")
            for k, v in mAB.items():
                print(f"{k}: {v}")

        print("\n--- output ---")
        print(out)
        print()

if __name__ == "__main__":
    main()


