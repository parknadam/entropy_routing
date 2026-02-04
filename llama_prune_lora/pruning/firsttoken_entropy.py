# prune_lora/pruning/firsttoken_entropy.py
# REPL: prompt 입력 → (A/AB/FULL) 단계별 prompt entropy/PPL 측정 → stage 결정 → 해당 stage로 생성
#
# 사용 예:
""" 
python -m prune_lora.pruning.firsttoken_entropy \
  --base_model ./results/pruning/A \
  --bundles_dir ./results/pruning/bundles \
  --max_new_tokens 128

"""
#
# 또는 (HF gated 접근 권한 있을 때):
#   python -m prune_lora.pruning.firsttoken_entropy \
#     --base_model meta-llama/Llama-2-13b-chat-hf \
#     --bundles_dir ./results/pruning/bundles \
#     --chat auto \
#     --max_new_tokens 128
#
# NOTE:
# - bundles 파일명 0-padding(layer_021) 대응: 디렉토리 스캔으로 idx->path map 구성
# - KV cache는 안정성 위해 CLI에서는 기본 use_cache=False로 평가/생성

from __future__ import annotations

import argparse
import inspect
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# transformers 버전에 따라 LlamaDecoderLayer import 경로가 다름
# -----------------------------
try:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
except Exception:
    LlamaDecoderLayer = None


def _get_llama_layers(model) -> nn.ModuleList:
    # HF 버전 차이 대응
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        return model.model.model.layers
    raise RuntimeError("LLaMA layers 경로를 찾지 못했어요. (예: model.model.layers)")


# -----------------------------
# PassLayer: 삭제 레이어 자리 유지
# -----------------------------
class LlamaPassLayer(nn.Module):
    """
    transformers 내부 loop가 'Tensor를 기대'하는지 'tuple을 기대'하는지 버전에 따라 달라질 수 있어
    return_tuple 플래그로 맞춘다.
    """
    def __init__(self, return_tuple: bool = True):
        super().__init__()
        self.return_tuple = return_tuple

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
        if not self.return_tuple:
            return hidden_states

        # LlamaDecoderLayer가 보통 반환하는 형태를 최대한 모사
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (None,)
        if use_cache:
            outputs = outputs + (past_key_value,)
        return outputs


def _detect_layer_return_tuple(model) -> bool:
    """
    모델 forward 구현을 소스에서 대충 파싱해서
    decoder_layer 출력이 tuple로 처리되는지 추정.
    실패하면 안전하게 tuple로 간주.
    """
    try:
        # 보통 model.model이 LlamaModel
        core = model.model if hasattr(model, "model") else model
        src = inspect.getsource(core.forward)
        # tuple일 때 흔히 layer_outputs[0] 접근이 존재
        if "layer_outputs[0]" in src or "layer_outputs = decoder_layer" in src:
            return True
        # tensor일 때 hidden_states = decoder_layer(...) 같은 패턴이 있을 수 있음
        if "hidden_states = decoder_layer" in src and "layer_outputs[0]" not in src:
            return False
    except Exception:
        pass
    return True


# -----------------------------
# bundles 스캔 유틸
# -----------------------------
_LAYER_RE = re.compile(r"layer_(\d+)\.safetensors$")


def _build_layer_map(dir_path: Path) -> Dict[int, Path]:
    """
    bundles/B 또는 bundles/C 스캔해서 {idx:int -> path:Path} 맵 구성.
    layer_021.safetensors 같은 0-padding도 idx=21로 파싱됨.
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
    bundles의 key가 다양한 prefix를 가질 수 있어 레이어 내부 키로 정규화:
      - model.layers.{i}.xxx
      - model.model.layers.{i}.xxx
      - layers.{i}.xxx
      - 이미 xxx
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


def _maybe_shift_indices_to_zero_based(
    B_map: Dict[int, Path],
    C_map: Dict[int, Path],
    num_layers: int,
) -> Tuple[Dict[int, Path], Dict[int, Path], int]:
    """
    bundles가 1-based로 저장된 경우(예: layer_1 ... layer_32) 자동으로 -1 shift.
    - shift 적용 시 (키를 -1한 새 map) 반환, shift=-1
    - 아니면 원본 반환, shift=0
    """
    all_idx = sorted(set(B_map.keys()) | set(C_map.keys()))
    if not all_idx:
        return B_map, C_map, 0

    out_of_range = any((i < 0 or i >= num_layers) for i in all_idx)
    if not out_of_range:
        return B_map, C_map, 0

    # 1-based로 가정해볼 조건: 모든 idx가 [1..num_layers] 범위에 들어가면 shift 가능
    one_based_ok = all((1 <= i <= num_layers) for i in all_idx)
    if one_based_ok:
        B2 = {i - 1: p for i, p in B_map.items()}
        C2 = {i - 1: p for i, p in C_map.items()}
        return B2, C2, -1

    # 혼종/완전 mismatch
    raise ValueError(
        f"bundles layer index가 모델 레이어 수(num_layers={num_layers})와 맞지 않아요. "
        f"예: max_index={max(all_idx)}. "
        "A(베이스) 모델과 bundles(B/C)가 같은 기반 모델(13b-chat 등)에서 생성됐는지, "
        "그리고 layer 인덱스가 0-based인지(0~L-1) 확인해 주세요."
    )


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
        passlayer_return_tuple: bool,
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
        self.passlayer_return_tuple = passlayer_return_tuple

        self.num_layers = len(self.layers)

        # 디렉토리
        self.B_dir = bundles_dir / "B"
        self.C_dir = bundles_dir / "C"

        # idx->path 맵
        B_map_raw = _build_layer_map(self.B_dir)
        C_map_raw = _build_layer_map(self.C_dir)

        # 0-based/1-based 자동 보정
        self.B_map, self.C_map, self.index_shift = _maybe_shift_indices_to_zero_based(
            B_map=B_map_raw,
            C_map=C_map_raw,
            num_layers=self.num_layers,
        )

        self.B_idx = sorted(self.B_map.keys())
        self.C_idx = sorted(self.C_map.keys())
        self.removed = sorted(set(self.B_idx) | set(self.C_idx))

        # 최종 검증 (여기서 한번 더 안전하게)
        for i in self.removed:
            if i < 0 or i >= self.num_layers:
                raise ValueError(
                    f"Invalid layer index in bundles: {i} (num_layers={self.num_layers}). "
                    "bundles와 base_model이 같은 모델인지 확인해 주세요."
                )

        # 시작은 A (removed 모두 PassLayer)
        self.set_stage("A")

    def stage_meta(self) -> Dict:
        meta = {
            "num_layers": self.num_layers,
            "index_shift_applied": self.index_shift,  # -1이면 bundles가 1-based였음
            "B": self.B_idx,
            "C": self.C_idx,
            "removed": self.removed,
        }
        return meta

    def _bundle_path(self, layer_i: int) -> Optional[Path]:
        if layer_i in self.B_map:
            return self.B_map[layer_i]
        if layer_i in self.C_map:
            return self.C_map[layer_i]
        return None

    def _restore_one_layer(self, layer_i: int):
        p = self._bundle_path(layer_i)
        if p is None:
            raise FileNotFoundError(f"layer_{layer_i}.safetensors를 B/C에서 찾지 못했어요.")

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
        self.layers[layer_i] = LlamaPassLayer(return_tuple=self.passlayer_return_tuple).to(self.device)
        del old

    def set_stage(self, stage: str):
        stage = stage.upper()
        if stage not in ("A", "AB", "FULL"):
            raise ValueError("stage는 A / AB / FULL 중 하나여야 해요.")

        if stage == "A":
            pass_set = set(self.removed)
        elif stage == "AB":
            pass_set = set(self.C_idx)
        else:  # FULL
            pass_set = set()

        for i in self.removed:
            cur = self.layers[i]
            is_pass = isinstance(cur, LlamaPassLayer)
            if i in pass_set:
                if not is_pass:
                    self._pass_one_layer(i)
            else:
                if is_pass:
                    self._restore_one_layer(i)

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


# -----------------------------
# Chat prompt formatting
# -----------------------------
def _is_chat_model_auto(tokenizer, base_model_id: str, chat_arg: str) -> bool:
    if chat_arg == "on":
        return True
    if chat_arg == "off":
        return False

    # auto
    if getattr(tokenizer, "chat_template", None):
        return True
    lowered = (base_model_id or "").lower()
    if "chat" in lowered or "instruct" in lowered:
        return True
    return False


def _encode_prompt(tokenizer, prompt: str, max_length: int, device: str, chat: bool, system_prompt: str):
    if chat and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        attn = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attn}

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}


# -----------------------------
# Metrics: prompt entropy / ppl
# -----------------------------
@torch.no_grad()
def get_next_token_metrics(model, tokenizer, prompt: str, device: str, chat: bool, system_prompt: str) -> Dict[str, float]:
    """
    Stage A 모델이 프롬프트 다음에 올 첫 번째 토큰을 얼마나 확신하는지 측정합니다.
    """
    enc = _encode_prompt(tokenizer, prompt, max_length=2048, device=device, chat=chat, system_prompt=system_prompt)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    # Forward pass (Stage A)
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    
    # 마지막 토큰의 Logits만 추출 [Batch, Seq, Vocab] -> [Vocab]
    last_logits = out.logits[0, -1, :] 
    probs = F.softmax(last_logits, dim=-1)
    log_probs = F.log_softmax(last_logits, dim=-1)

    # 1. 확신도 (Max Probability): 가장 높은 확률값이 얼마인가?
    conf, _ = torch.max(probs, dim=-1)
    
    # 2. 예측 엔트로피: 확률 분포가 얼마나 퍼져 있는가?
    # H = -sum(p * log(p))
    entropy = -(probs * log_probs.clamp_min(1e-12)).sum(dim=-1)

    return {
        "next_token_confidence": float(conf.item()), # 0~1 사이, 높을수록 확신
        "next_token_entropy": float(entropy.item()), # 낮을수록 확신
    }

# -----------------------------
# Router: 첫 토큰 확신도 기반 결정
# -----------------------------
def decide_stage_by_confidence(metrics: Dict[str, float], cfg: RouterConfig) -> str:
    conf = metrics["next_token_confidence"]
    ent = metrics["next_token_entropy"]

    # 예시 임계값: 확신도가 낮거나 엔트로피가 높으면 승급
    # 연구를 통해 이 Threshold 값을 최적화하는 것이 목표가 됩니다.
    
    # Stage A -> AB 판단
    if conf < 0.6 or ent > 2.0:
        # Stage AB -> FULL 판단 (추가 지표가 없다면 순차적으로 AB 시도)
        if conf < 0.3 or ent > 3.5:
            return "FULL"
        return "AB"
    
    return "A"

@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, device: str, chat: bool, system_prompt: str) -> str:
    if chat and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        gen = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        # prompt를 제외하고 새로 생성된 부분만 디코드
        new_tokens = gen[0, input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # non-chat
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_len = enc["input_ids"].shape[1]
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = gen[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# -----------------------------
# Router: entropy + ppl + min_tokens로 승급
# -----------------------------
@dataclass
class RouterConfig:
    # 최소 토큰 수 (프롬프트가 너무 짧으면 무조건 로드하는 등의 로직용)
    min_tokens: int = 50

    # Stage A 관련 임계값
    thr_A_ent_mean: float = 2.6
    thr_A_last_ent: float = 2.5
    thr_A_ppl: float = 40.0
    
    # (추가한 Confidence 로직용 필드 - 필요시 사용)
    thr_A_conf: float = 0.6
    thr_A_ent: float = 2.0

    # Stage AB 관련 임계값
    thr_AB_ent_mean: float = 2.7
    thr_AB_last_ent: float = 2.6
    thr_AB_ppl: float = 35.0
    
    # (추가한 Confidence 로직용 필드 - 필요시 사용)
    thr_AB_conf: float = 0.3
    thr_AB_ent: float = 3.5


def decide_stage_by_confidence(metrics: Dict[str, float], cfg: RouterConfig) -> str:
    conf = metrics["next_token_confidence"]
    ent = metrics["next_token_entropy"]

    # Stage A -> AB/FULL 판단
    if conf < cfg.thr_A_conf or ent > cfg.thr_A_ent:
        # 더 심각한 불확실성일 경우 바로 FULL로 점프하거나 AB 거치기
        if conf < cfg.thr_AB_conf or ent > cfg.thr_AB_ent:
            return "FULL"
        return "AB"
    
    return "A"


# -----------------------------
# main
# -----------------------------
def _load_model(base_model: str, dtype: torch.dtype, device: str):
    # transformers 버전에 따라 torch_dtype vs dtype 파라미터가 다를 수 있어 try/except
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="A(프루닝 베이스) 모델 경로/HF id")
    ap.add_argument("--bundles_dir", required=True, help="bundles dir (B/, C/ 포함)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=128)

    # chat formatting
    ap.add_argument("--chat", default="auto", choices=["auto", "on", "off"], help="chat template 사용 여부")
    ap.add_argument("--system_prompt", default="", help="chat일 때 system prompt (비우면 없음)")

    # 기존 CLI 인자 호환(너가 이미 쓰고 있어서)
    ap.add_argument("--thr_A_to_AB", type=float, default=2.5, help="(호환용) A->AB last_entropy 기준")
    ap.add_argument("--thr_AB_to_FULL", type=float, default=2.9, help="(호환용) AB->FULL last_entropy 기준")

    # 라우터 파라미터
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

    chat_mode = _is_chat_model_auto(tok, args.base_model, args.chat)

    model = _load_model(args.base_model, dtype=dtype, device=args.device)
    model.eval()

    # PassLayer return 형태 자동 추정
    passlayer_return_tuple = _detect_layer_return_tuple(model)

    mgr = DynamicStageManager(
        model=model,
        bundles_dir=Path(args.bundles_dir),
        device=args.device,
        dtype=dtype,
        passlayer_return_tuple=passlayer_return_tuple,
    )

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
    print(f"\nchat_mode={chat_mode} | passlayer_return_tuple={passlayer_return_tuple}")
    if chat_mode:
        print("※ Chat 모델은 [INST]/<<SYS>> 포맷(또는 chat template)을 맞춰야 성능이 안정적이에요.")
    print("\n[REPL] 프롬프트 입력. 종료: Ctrl+C\n")

    while True:
        prompt = input("PROMPT> ").strip()
        if not prompt: continue

        # 1) Stage A Probing
        mgr.set_stage("A")
        metrics = get_next_token_metrics(model, tok, prompt, args.device, chat_mode, args.system_prompt)

        # 2) Decision
        stage = decide_stage_by_confidence(metrics, cfg)

        # 3) 결정된 Stage로 실제 생성
        if stage != "A":
            mgr.set_stage(stage)
        
        # 변수명 out으로 통일
        out = generate_text(model, tok, prompt, args.max_new_tokens, args.device, chat_mode, args.system_prompt)

        # 4) 결과 출력
        print(f"\n[Decision] Stage: {stage}")
        print(f" - Confidence: {metrics['next_token_confidence']:.4f} (Threshold A: {cfg.thr_A_conf})")
        print(f" - Entropy: {metrics['next_token_entropy']:.4f} (Threshold A: {cfg.thr_A_ent})")
        
        print(f"\n--- Output ({stage}) ---")
        print(out)
        print("-" * 30 + "\n")


if __name__ == "__main__":
    main()
