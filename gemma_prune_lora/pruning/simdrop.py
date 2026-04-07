# simdrop.py
# Angular-distance 기반 연속 n개 레이어 드랍 (Gemma / LLaMA / OPT 공용)

import math
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from .identity import PassLayer
from .model_utils import get_layers


def _to_inputs(
    batch: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    device: torch.device,
) -> dict:
    """
    dataloader 출력을 HF CausalLM forward에 맞게 정규화.
    - dict: 그대로 전달 (input_ids, attention_mask 등 이미 포함)
    - tuple/list: data.py의 (input_ids, targets) 형태 → input_ids만 사용
      (targets는 -100 값이므로 attention_mask로 쓰면 안 됨)
    - tensor: input_ids로 간주
    """
    if isinstance(batch, dict):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        # data.py는 (input_ids, targets) 튜플을 반환
        # targets는 loss 계산용이지 attention_mask가 아님 — input_ids만 전달
        return {"input_ids": batch[0].to(device, non_blocking=True)}
    else:
        return {"input_ids": batch.to(device, non_blocking=True)}


@torch.no_grad()
def _compute_layer_inputs(
    model,
    dataloader,
    device: torch.device,
    arch: str,
    max_batches: int = 64,
) -> Tuple[Dict[int, List[torch.Tensor]], int]:

    layers = get_layers(model, arch)
    L = len(layers)

    captured: Dict[int, List[torch.Tensor]] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(ell: int):
        def hook(mod: nn.Module, inputs):
            x = inputs[0]
            x_last = x[:, -1, :].detach().float().cpu()
            captured.setdefault(ell, []).append(x_last)
        return hook

    for ell, block in enumerate(layers):
        h = block.register_forward_pre_hook(make_hook(ell))
        handles.append(h)

    seen = 0
    errors = 0
    first_error = None
    for batch in dataloader:
        if seen >= max_batches:
            break
        try:
            inputs = _to_inputs(batch, device)
            model(**inputs)
        except Exception as e:
            errors += 1
            if first_error is None:
                first_error = e
        seen += 1

    for h in handles:
        h.remove()

    if errors > 0:
        print(f"  [warn] Forward pass failed on {errors}/{seen} batches. First error: {first_error}")

    if not captured:
        raise RuntimeError(
            f"No activations were captured ({errors}/{seen} batches failed). "
            f"First error: {first_error}"
        )

    min_len = min(len(v) for v in captured.values())
    if min_len == 0:
        raise RuntimeError("Captured activations are empty for at least one layer.")
    for k in list(captured.keys()):
        captured[k] = captured[k][:min_len]

    print(f"  [ok] Captured {min_len} batches across {len(captured)} layers")
    return captured, L


def _mean_cos_over_batches(A_list: List[torch.Tensor], B_list: List[torch.Tensor]) -> float:
    cos_vals = []
    eps = 1e-8
    for A, B in zip(A_list, B_list):
        finite_mask = torch.isfinite(A).all(dim=1) & torch.isfinite(B).all(dim=1)
        if not torch.any(finite_mask):
            continue
        A = A[finite_mask]
        B = B[finite_mask]
        dot = (A * B).sum(dim=1)
        denom = A.norm(dim=1) * B.norm(dim=1) + eps
        cos = dot / denom
        cos = cos[torch.isfinite(cos)]
        if cos.numel() == 0:
            continue
        cos_vals.append(cos.mean().item())
    if not cos_vals:
        return float("nan")
    return float(sum(cos_vals) / len(cos_vals))


@torch.no_grad()
def choose_block_to_drop(
    model,
    dataloader,
    device: torch.device,
    n: int,
    arch: str = "gemma",
    keep_last_layer: bool = True,
    max_batches: int = 64,
) -> Tuple[int, float, int]:
    """
    d(l) = (1/pi) * arccos( mean_cos( x(l), x(l+n) ) ) 를 최소화하는 시작 인덱스 l 선택
    반환: (best_l, best_d, 총 레이어 수 L)
    """
    model.eval()
    captured, L = _compute_layer_inputs(model, dataloader, device, arch, max_batches=max_batches)

    last_idx_exclusive = L - n
    if keep_last_layer:
        last_idx_exclusive -= 1
    if last_idx_exclusive <= 0:
        raise ValueError(f"n={n} too large for L={L} (keep_last_layer={keep_last_layer})")

    eligible = [i for i in range(last_idx_exclusive) if i in captured and (i + n) in captured]
    if not eligible:
        captured_keys = sorted(captured.keys())
        raise RuntimeError(
            f"No eligible start index for block drop. "
            f"L={L}, n={n}, last_idx_exclusive={last_idx_exclusive}, "
            f"captured layers={captured_keys}"
        )

    best_ell, best_d = None, float("inf")
    invalid_count = 0
    for ell in eligible:
        cos_val = _mean_cos_over_batches(captured[ell], captured[ell + n])
        if not math.isfinite(cos_val):
            invalid_count += 1
            continue
        cos_val = max(min(cos_val, 1.0), -1.0)
        d = math.acos(cos_val) / math.pi
        if not math.isfinite(d):
            invalid_count += 1
            continue
        if d < best_d:
            best_d, best_ell = d, ell

    if best_ell is None:
        best_ell = eligible[len(eligible) // 2]
        best_d = 0.5
        print(
            "  [warn] All eligible candidates produced non-finite distances "
            f"(invalid={invalid_count}/{len(eligible)}). Falling back to start={best_ell}."
        )

    return best_ell, best_d, L


def drop_consecutive_layers(model, ell: int, n: int, arch: str = "gemma"):
    layers = get_layers(model, arch)
    hidden_size = model.config.hidden_size

    removed_indices = list(range(ell, ell + n))
    for i in removed_indices:
        layers[i] = PassLayer(hidden_size)

    return model, removed_indices
