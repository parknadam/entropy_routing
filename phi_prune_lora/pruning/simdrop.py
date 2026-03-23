# simdrop.py  (Phi-3 전용)
# Angular-distance 기반 연속 n개 레이어 드랍
# ─────────────────────────────────────────────────────────────
# 변경 사유:
#   1. is_opt 파라미터 완전 제거
#      → Phi-3는 항상 model.model.layers 경로만 사용하므로
#        OPT의 model.model.decoder.layers 분기가 불필요
#   2. LlamaPassLayer → Phi3PassLayer로 교체
#   3. _compute_layer_inputs에서 레이어 접근을 model.model.layers로 고정
#   4. drop_consecutive_layers도 동일하게 단순화
#   5. 나머지 알고리즘 로직(angular distance 계산, 후보 선택)은 동일
#      → 이 부분은 모델 아키텍처에 무관한 수학적 연산이므로
# ─────────────────────────────────────────────────────────────

import math
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from .identity import Phi3PassLayer


def _to_inputs(
    batch: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    device: torch.device,
) -> dict:
    """
    dataloader가 dict/tuple/tensor 무엇을 주더라도 HF CausalLM이 기대하는 형태로 정규화.
    (원본과 동일 — 데이터 전처리는 모델 아키텍처에 무관)
    """
    if isinstance(batch, dict):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return {
                "input_ids": batch[0].to(device, non_blocking=True),
                "attention_mask": batch[1].to(device, non_blocking=True),
            }
        else:
            return {"input_ids": batch[0].to(device, non_blocking=True)}
    else:
        return {"input_ids": batch.to(device, non_blocking=True)}


@torch.no_grad()
def _compute_layer_inputs(
    model,
    dataloader,
    device: torch.device,
    max_batches: int = 64,
) -> Tuple[Dict[int, List[torch.Tensor]], int]:
    """
    각 레이어 입력의 마지막 토큰 hidden state를 캡처.

    변경점: is_opt 분기 제거 → model.model.layers 직접 사용
    """
    layers = model.model.layers  # Phi-3는 항상 이 경로
    L = len(layers)

    captured: Dict[int, List[torch.Tensor]] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(ell: int):
        def hook(mod: nn.Module, inputs):
            x = inputs[0]  # hidden_states: [B, T, H]
            x_last = x[:, -1, :].detach().float().cpu()
            captured.setdefault(ell, []).append(x_last)
        return hook

    for ell, block in enumerate(layers):
        h = block.register_forward_pre_hook(make_hook(ell))
        handles.append(h)

    seen = 0
    for batch in dataloader:
        if seen >= max_batches:
            break
        try:
            inputs = _to_inputs(batch, device)
            model(**inputs)
        except Exception:
            pass  # 데이터/모델 미스매치 시 다음 배치로 넘어감
        seen += 1

    for h in handles:
        h.remove()

    if not captured:
        raise RuntimeError("No activations were captured. Check dataloader/model wiring.")

    min_len = min(len(v) for v in captured.values())
    if min_len == 0:
        raise RuntimeError("Captured activations are empty for at least one layer.")
    for k in list(captured.keys()):
        captured[k] = captured[k][:min_len]

    return captured, L


def _mean_cos_over_batches(A_list: List[torch.Tensor], B_list: List[torch.Tensor]) -> float:
    """
    배치 단위로 per-sample cosine을 계산해 평균 반환.
    (원본과 동일 — 순수 수학 연산)
    """
    cos_vals = []
    eps = 1e-8
    for A, B in zip(A_list, B_list):
        dot = (A * B).sum(dim=1)
        denom = A.norm(dim=1) * B.norm(dim=1) + eps
        cos = dot / denom
        cos_vals.append(cos.mean().item())
    return float(sum(cos_vals) / len(cos_vals))


@torch.no_grad()
def choose_block_to_drop(
    model,
    dataloader,
    device: torch.device,
    n: int,
    keep_last_layer: bool = True,
    max_batches: int = 64,
) -> Tuple[int, float, int]:
    """
    d(l) = (1/pi) * arccos( mean_cos( x(l), x(l+n) ) ) 를 최소화하는 시작 인덱스 l 선택.

    변경점: is_opt 파라미터 제거 (Phi-3 전용이므로)
    반환: (best_l, best_d, 총 레이어 수 L)
    """
    model.eval()
    captured, L = _compute_layer_inputs(model, dataloader, device, max_batches=max_batches)

    last_idx_exclusive = L - n
    if keep_last_layer:
        last_idx_exclusive -= 1
    if last_idx_exclusive <= 0:
        raise ValueError(f"n={n} too large for L={L} (keep_last_layer={keep_last_layer})")

    eligible = [i for i in range(last_idx_exclusive) if i in captured and (i + n) in captured]
    if not eligible:
        raise RuntimeError("No eligible start index for block drop (capture failed?).")

    best_ell, best_d = None, float("inf")
    for ell in eligible:
        cos_val = _mean_cos_over_batches(captured[ell], captured[ell + n])
        cos_val = max(min(cos_val, 1.0), -1.0)
        d = math.acos(cos_val) / math.pi
        if d < best_d:
            best_d, best_ell = d, ell

    return best_ell, best_d, L


def drop_consecutive_layers(model, ell: int, n: int):
    """
    지정된 연속 레이어를 Phi3PassLayer로 교체.

    변경점:
      - is_opt 파라미터 제거
      - LlamaPassLayer → Phi3PassLayer
      - model.model.layers 직접 접근
    """
    layers = model.model.layers
    hidden_size = model.config.hidden_size

    removed_indices = list(range(ell, ell + n))
    for i in removed_indices:
        layers[i] = Phi3PassLayer(hidden_size)

    return model, removed_indices