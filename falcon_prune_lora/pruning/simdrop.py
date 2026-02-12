# simdrop.py
# Angular-distance 기반 연속 n개 레이어 드랍 (Falcon 구조 대응)

import math
from typing import Dict, List, Tuple, Union
from .identity import FalconPassLayer

import torch
import torch.nn as nn


def _get_falcon_layers(model):
    """Falcon 모델의 디코더 레이어 리스트를 반환"""
    return model.transformer.h


def _to_inputs(
    batch: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    device: torch.device,
) -> dict:
    """
    dataloader가 dict/tuple/tensor 무엇을 주더라도 HF CausalLM이 기대하는 형태로 정규화.

    주의: get_loaders()는 (input_ids, targets) 튜플을 반환함.
          targets(두 번째 요소)는 attention_mask가 아니므로 input_ids만 사용.
    """
    if isinstance(batch, dict):
        # dict인 경우 input_ids / attention_mask 키가 있을 수 있음
        out = {}
        for k in ("input_ids", "attention_mask", "position_ids"):
            if k in batch:
                out[k] = batch[k].to(device, non_blocking=True)
        if not out:
            # fallback: 모든 키 전달
            out = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        return out
    elif isinstance(batch, (list, tuple)):
        # get_loaders 반환값: (input_ids, targets) — targets는 사용하지 않음
        input_ids = batch[0].to(device, non_blocking=True)
        return {"input_ids": input_ids}
    else:
        return {"input_ids": batch.to(device, non_blocking=True)}


@torch.no_grad()
def _compute_layer_inputs(
    model,
    dataloader,
    device: torch.device,
    max_batches: int = 64,
) -> Tuple[Dict[int, List[torch.Tensor]], int]:

    layers = _get_falcon_layers(model)
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
    first_error_logged = False
    for batch in dataloader:
        if seen >= max_batches:
            break
        try:
            inputs = _to_inputs(batch, device)
            # Falcon의 max_position_embeddings 초과 시 잘라냄
            max_len = getattr(model.config, "max_position_embeddings", None)
            if max_len and inputs["input_ids"].shape[-1] > max_len:
                inputs["input_ids"] = inputs["input_ids"][:, :max_len]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][:, :max_len]
            model(**inputs)
        except Exception as e:
            if not first_error_logged:
                print(f"[_compute_layer_inputs] first batch error: {type(e).__name__}: {e}")
                first_error_logged = True
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
    """배치 단위로 per-sample cosine을 계산해 정확한 평균을 반환"""
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
    d(l) = (1/pi) * arccos( mean_cos( x(l), x(l+n) ) ) 를 최소화하는 시작 인덱스 l 선택
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
    """Falcon 모델에서 연속 n개 레이어를 PassLayer로 치환"""
    layers = _get_falcon_layers(model)
    hidden_size = model.config.hidden_size

    removed_indices = list(range(ell, ell + n))
    for i in removed_indices:
        layers[i] = FalconPassLayer(hidden_size)

    return model, removed_indices