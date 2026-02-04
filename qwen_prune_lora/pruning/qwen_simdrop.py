# qwen_prune_lora/pruning/qwen_simdrop.py
# Forward hook 사용 버전 (pre_hook 대신)

import math
from typing import Dict, List, Tuple, Union
from .qwen_identity import QwenPassLayer
import torch
import torch.nn as nn

def _to_inputs(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return {"input_ids": batch[0].to(device), "attention_mask": batch[1].to(device)}
        return {"input_ids": batch[0].to(device)}
    return {"input_ids": batch.to(device)}

def _get_layers(model):
    # Qwen2/Qwen2.5 / LLaMA 스타일
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # 구 Qwen 스타일
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find transformer blocks: model.model.layers or model.transformer.h")

@torch.no_grad()
def _compute_layer_inputs(model, dataloader, device, max_batches=64):
    """
    Forward hook 사용: 각 레이어의 OUTPUT을 캡처
    (pre_hook은 Qwen에서 제대로 작동하지 않음)
    """
    layers = _get_layers(model)
    L = len(layers)
    captured = {}
    handles = []
    
    def make_hook(ell):
        def hook(mod, input, output):
            x = output

            if isinstance(x, (tuple, list)):
                x = x[0] if len(x) > 0 else None

            if isinstance(x, dict):
                x = x.get("hidden_states", None)

            # 일부 ModelOutput류 대응
            if x is None and hasattr(output, "hidden_states"):
                x = output.hidden_states

            if torch.is_tensor(x) and x.dim() == 3 and x.size(1) > 0:
                x_last = x[:, -1, :].detach().float().cpu()
                # 일단 디버깅 단계에선 필터링 완화(아래 참고)
                captured.setdefault(ell, []).append(x_last)
        return hook

    
    # Forward hook 등록 (pre_hook 아님!)
    for ell, block in enumerate(layers):
        handles.append(block.register_forward_hook(make_hook(ell)))
    
    seen, successful = 0, 0
    for batch in dataloader:
        if seen >= max_batches:
            break
        try:
            model(**_to_inputs(batch, device))
            successful += 1
        except Exception as e:
            print(f"  Batch {seen} failed: {str(e)[:80]}")
        seen += 1
    
    for h in handles:
        h.remove()
    
    if not captured:
        raise RuntimeError(
            f"No activations captured. "
            f"Batches: {successful}/{seen} succeeded. "
            f"This usually means model forward failed completely."
        )
    
    # 레이어 개수 확인
    if len(captured) < L:
        print(f"  ⚠ Warning: Only {len(captured)}/{L} layers captured")
        print(f"    Captured layers: {sorted(captured.keys())}")
    
    min_len = min(len(v) for v in captured.values())
    if min_len == 0:
        raise RuntimeError("All captured activations are empty")
    
    for k in list(captured.keys()):
        captured[k] = captured[k][:min_len]
    
    print(f"  ✓ Captured {min_len} samples across {len(captured)} layers")
    return captured, L

def _mean_cos_over_batches(A_list, B_list):
    cos_vals = []
    eps = 1e-8
    
    for A, B in zip(A_list, B_list):
        if torch.isnan(A).any() or torch.isinf(A).any():
            continue
        if torch.isnan(B).any() or torch.isinf(B).any():
            continue
        
        dot = (A * B).sum(dim=1)
        norm_A = A.norm(dim=1)
        norm_B = B.norm(dim=1)
        
        valid = (norm_A > eps) & (norm_B > eps)
        if valid.sum() == 0:
            continue
        
        cos = (dot / (norm_A * norm_B + eps))[valid]
        if len(cos) > 0:
            cos_vals.append(cos.mean().item())
    
    return float(sum(cos_vals) / len(cos_vals)) if cos_vals else 0.5

@torch.no_grad()
def choose_block_to_drop(model, dataloader, device, n, keep_last_layer=True, max_batches=64):
    model.eval()
    print(f"Computing layer activations (max {max_batches} batches)...")
    captured, L = _compute_layer_inputs(model, dataloader, device, max_batches)
    
    # 캡처 실패 시 에러 메시지 개선
    if len(captured) < n + 1:
        raise RuntimeError(
            f"Not enough layers captured: {len(captured)}/{L} captured, need at least {n+1}. "
            f"Captured layers: {sorted(captured.keys())}. "
            f"This usually means hooks didn't fire properly."
        )
    
    last_idx = L - n - (1 if keep_last_layer else 0)
    if last_idx <= 0:
        raise ValueError(f"Cannot drop {n} from {L} layers (keep_last_layer={keep_last_layer})")
    
    eligible = [i for i in range(last_idx) if i in captured and (i+n) in captured]
    if not eligible:
        # 상세 디버그 정보
        available_starts = [i for i in captured.keys() if i < last_idx]
        available_ends = [i for i in captured.keys() if i >= n]
        raise RuntimeError(
            f"No eligible positions for n={n} block drop.\n"
            f"  Captured layers: {sorted(captured.keys())}\n"
            f"  Available start positions: {available_starts}\n"
            f"  Available end positions (i+{n}): {available_ends}\n"
            f"  Need both i and i+{n} to be captured."
        )
    
    print(f"  Evaluating {len(eligible)} positions...")
    
    best_ell, best_d = None, float("inf")
    
    for ell in eligible:
        cos_val = max(min(_mean_cos_over_batches(captured[ell], captured[ell+n]), 1.0), -1.0)
        try:
            d = math.acos(cos_val) / math.pi
            if math.isfinite(d) and d < best_d:
                best_d, best_ell = d, ell
        except:
            pass
    
    if best_ell is None:
        # Fallback: 중간 위치
        best_ell = eligible[len(eligible) // 2]
        best_d = 0.5
        print(f"  ⚠ Using fallback position: {best_ell}")
    
    print(f"  ✓ Selected layer {best_ell} (distance: {best_d:.4f})")
    return best_ell, best_d, L

def drop_consecutive_layers(model, ell, n, return_tuple=True):
    layers = _get_layers(model)
    hidden_size = model.config.hidden_size
    removed = list(range(ell, ell + n))
    
    print(f"  Replacing layers {removed[0]}-{removed[-1]} with PassLayer...")
    for i in removed:
        layers[i] = QwenPassLayer(hidden_size, return_tuple)
    
    return model, removed