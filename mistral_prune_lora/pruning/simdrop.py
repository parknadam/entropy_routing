# simdrop.py
# Angular-distance 기반 연속 n개 레이어 드랍 (Deeper Layers)
# Mistral/LLaMA 모두 지원

import math
from typing import Dict, List, Tuple, Union
from .identity import PassLayer

import torch
import torch.nn as nn


def _looks_like_attention_mask(mask: torch.Tensor, input_ids: torch.Tensor) -> bool:
    """
    attention_mask 형태인지 보수적으로 판별.
    - input_ids와 같은 shape
    - 값이 0/1 또는 bool
    """
    if not torch.is_tensor(mask):
        return False
    if mask.shape != input_ids.shape:
        return False
    if mask.dtype == torch.bool:
        return True
    return bool(((mask == 0) | (mask == 1)).all().item())


def _detect_model_type(model) -> str:
    """
    모델 타입 자동 감지
    Returns: 'mistral', 'llama', 'opt', 'unknown'
    """
    model_type = getattr(model.config, "model_type", "").lower()
    
    if "mistral" in model_type:
        return "mistral"
    elif "llama" in model_type:
        return "llama"
    elif "opt" in model_type:
        return "opt"
    
    # fallback: 구조로 판단
    if hasattr(model, "model"):
        if hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
            return "opt"
        elif hasattr(model.model, "layers"):
            # Mistral과 LLaMA 모두 model.model.layers 사용
            # config로 구분
            arch_type = getattr(model.config, "architectures", [])
            if any("Mistral" in a for a in arch_type):
                return "mistral"
            return "llama"
    
    return "unknown"


def _get_layers(model, model_type: str = None):
    """모델 타입에 따라 레이어 접근"""
    if model_type is None:
        model_type = _detect_model_type(model)
    
    if model_type == "opt":
        return model.model.decoder.layers
    elif model_type in ("mistral", "llama"):
        return model.model.layers
    else:
        # fallback
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        elif hasattr(model, "model") and hasattr(model.model, "decoder"):
            return model.model.decoder.layers
        raise RuntimeError(f"Cannot find layers for model type: {model_type}")


def _to_inputs(
    batch: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    device: torch.device,
) -> dict:
    """
    dataloader가 dict/tuple/tensor 무엇을 주더라도 HF CausalLM이 기대하는 형태로 정규화
    - 우선순위: {'input_ids','attention_mask'} -> ('input_ids','attention_mask') -> input_ids only
    """
    if isinstance(batch, dict):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        input_ids = batch[0].to(device, non_blocking=True)
        if len(batch) >= 2:
            second = batch[1].to(device, non_blocking=True)
            # (input_ids, labels) 형태를 attention_mask로 오해하지 않도록 필터링
            if _looks_like_attention_mask(second, input_ids):
                return {
                    "input_ids": input_ids,
                    "attention_mask": second,
                }
            return {"input_ids": input_ids}
        else:
            return {"input_ids": input_ids}
    else:
        return {"input_ids": batch.to(device, non_blocking=True)}


@torch.no_grad()
def _compute_layer_inputs(
    model,
    dataloader,
    device: torch.device,
    model_type: str = None,
    max_batches: int = 64,
) -> Tuple[Dict[int, List[torch.Tensor]], int]:
    """
    각 레이어의 입력 활성화를 캡처
    
    Args:
        model: Mistral/LLaMA/OPT 모델
        dataloader: 캘리브레이션 데이터
        device: 디바이스
        model_type: 'mistral', 'llama', 'opt' (None이면 자동 감지)
        max_batches: 최대 배치 수
        
    Returns:
        (captured: Dict[layer_idx -> List[Tensor]], total_layers: int)
    """
    if model_type is None:
        model_type = _detect_model_type(model)
    
    layers = _get_layers(model, model_type)
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
    successful = 0
    failed = 0
    last_error = None
    for batch in dataloader:
        if seen >= max_batches:
            break
        try:
            inputs = _to_inputs(batch, device)
            model(**inputs)
            successful += 1
        except Exception as exc:
            failed += 1
            last_error = f"{type(exc).__name__}: {exc}"
        seen += 1

    for h in handles:
        h.remove()

    if successful == 0:
        raise RuntimeError(
            "No successful forward passes while capturing activations. "
            f"attempted={seen}, failed={failed}, last_error={last_error}"
        )
    if failed > 0:
        print(f"Warning: {failed}/{seen} calibration batches failed during capture.")

    if not captured:
        raise RuntimeError("No activations were captured. Check dataloader/model wiring.")

    missing = [i for i in range(L) if i not in captured]
    if missing:
        raise RuntimeError(
            "Missing activation captures for some layers. "
            f"missing={missing[:8]}{'...' if len(missing) > 8 else ''}"
        )

    # 모든 레이어가 동일한 배치 수가 되도록 잘라 맞춤
    min_len = min(len(v) for v in captured.values())
    if min_len == 0:
        raise RuntimeError("Captured activations are empty for at least one layer.")
    for k in list(captured.keys()):
        captured[k] = captured[k][:min_len]

    return captured, L


def _mean_cos_over_batches(A_list: List[torch.Tensor], B_list: List[torch.Tensor]) -> float:
    """
    배치 단위로 per-sample cosine을 계산해 '정확한 평균'을 반환
    """
    if not A_list or not B_list:
        raise ValueError("Cannot compute cosine mean with empty activation lists.")

    cos_vals = []
    eps = 1e-8
    for A, B in zip(A_list, B_list):
        # A,B: [B, H] (float32, cpu)
        dot = (A * B).sum(dim=1)
        denom = A.norm(dim=1) * B.norm(dim=1) + eps
        cos = dot / denom
        cos_vals.append(cos.mean().item())

    if not cos_vals:
        raise RuntimeError("No cosine values were computed from captured activations.")

    return float(sum(cos_vals) / len(cos_vals))


@torch.no_grad()
def choose_block_to_drop(
    model,
    dataloader,
    device: torch.device,
    n: int,
    model_type: str = None,
    keep_last_layer: bool = True,
    max_batches: int = 64,
) -> Tuple[int, float, int]:
    """
    d(l) = (1/pi) * arccos( mean_cos( x(l), x(l+n) ) ) 를 최소화하는 시작 인덱스 l 선택
    
    Args:
        model: Mistral/LLaMA/OPT 모델
        dataloader: 캘리브레이션 데이터
        device: 디바이스
        n: 제거할 연속 레이어 개수
        model_type: 'mistral', 'llama', 'opt' (None이면 자동 감지)
        keep_last_layer: 마지막 레이어 유지 여부
        max_batches: 최대 배치 수
        
    Returns:
        (best_l: 최적 시작 인덱스, best_d: 최소 angular distance, L: 총 레이어 수)
    """
    model.eval()
    
    if model_type is None:
        model_type = _detect_model_type(model)
    
    print(f"Detected model type: {model_type}")
    
    captured, L = _compute_layer_inputs(model, dataloader, device, model_type, max_batches=max_batches)
    
    last_idx_exclusive = L - n
    if keep_last_layer:
        last_idx_exclusive -= 1
    if last_idx_exclusive <= 0:
        raise ValueError(f"n={n} too large for L={L} (keep_last_layer={keep_last_layer})")

    # l과 l+n 모두 캡처된 경우만 후보
    eligible = [i for i in range(last_idx_exclusive) if i in captured and (i+n) in captured]
    if not eligible:
        raise RuntimeError("No eligible start index for block drop (capture failed?).")

    best_ell, best_d = None, float("inf")
    for ell in eligible:
        cos_val = _mean_cos_over_batches(captured[ell], captured[ell + n])
        # 수치 안정화
        cos_val = max(min(cos_val, 1.0), -1.0)
        d = math.acos(cos_val) / math.pi
        if d < best_d:
            best_d, best_ell = d, ell

    return best_ell, best_d, L


def drop_consecutive_layers(model, ell: int, n: int, model_type: str = None):
    """
    연속된 n개 레이어를 PassLayer로 교체
    
    Args:
        model: Mistral/LLaMA/OPT 모델
        ell: 시작 레이어 인덱스
        n: 제거할 레이어 개수
        model_type: 'mistral', 'llama', 'opt' (None이면 자동 감지)
        
    Returns:
        (model, removed_indices)
    """
    if model_type is None:
        model_type = _detect_model_type(model)
    
    layers = _get_layers(model, model_type)
    hidden_size = model.config.hidden_size

    removed_indices = list(range(ell, ell + n))
    for i in removed_indices:
        layers[i] = PassLayer(hidden_size)

    return model, removed_indices


# Backward compatibility
def choose_block_to_drop_legacy(model, dataloader, device, n: int, is_opt: bool = False, 
                               keep_last_layer: bool = True, max_batches: int = 64):
    """Legacy function for backward compatibility"""
    model_type = "opt" if is_opt else _detect_model_type(model)
    return choose_block_to_drop(model, dataloader, device, n, model_type, keep_last_layer, max_batches)


def drop_consecutive_layers_legacy(model, ell: int, n: int, is_opt: bool = False):
    """Legacy function for backward compatibility"""
    model_type = "opt" if is_opt else _detect_model_type(model)
    return drop_consecutive_layers(model, ell, n, model_type)
