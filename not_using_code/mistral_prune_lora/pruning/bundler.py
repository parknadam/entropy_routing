# bundler.py
# 점진적 프루닝을 위해 제거한 레이어를 B/C 두 번들로 나누어 저장하는 코드
# Mistral/LLaMA/OPT 모두 지원

import os
import json
from typing import List, Tuple, Dict, Set
from safetensors.torch import save_file
from transformers import PretrainedConfig


def _detect_model_type_from_config(config: PretrainedConfig) -> str:
    """Config에서 모델 타입 감지"""
    model_type = getattr(config, "model_type", "").lower()
    if "mistral" in model_type:
        return "mistral"
    elif "llama" in model_type:
        return "llama"
    elif "opt" in model_type:
        return "opt"
    
    arch_type = getattr(config, "architectures", [])
    if any("Mistral" in a for a in arch_type):
        return "mistral"
    elif any("Llama" in a or "LLaMA" in a for a in arch_type):
        return "llama"
    
    return "unknown"


# ---------------------------
# Split utilities
# ---------------------------
def split_indices(indices: List[int], policy: str = "half", ratio: float = 0.5) -> Tuple[List[int], List[int]]:
    """
    제거할 레이어 인덱스 리스트를 B/C 두 그룹으로 나눕니다.
    - policy="half": 앞/뒤 반반
    - policy="ratio": 앞쪽 비율로 절단(0.0~1.0)
    반환: (B_idx, C_idx)
    """
    indices = list(indices)
    n = len(indices)
    if n == 0:
        return [], []
    if policy == "half":
        k = n // 2
        return indices[:k], indices[k:]
    elif policy == "ratio":
        k = int(round(n * ratio))
        return indices[:k], indices[k:]
    else:
        raise ValueError(f"Unknown split policy: {policy}")


def _ensure_disjoint_and_cover(removed: List[int], B_idx: List[int], C_idx: List[int]) -> None:
    """B와 C가 서로소이며 removed를 완전히 커버하는지 검증"""
    set_removed, setB, setC = set(removed), set(B_idx), set(C_idx)
    if not setB.isdisjoint(setC):
        both = sorted(setB.intersection(setC))
        raise AssertionError(f"[bundlers] B/C overlap on layers: {both}")
    if setB.union(setC) != set_removed:
        missing = sorted(set_removed - (setB.union(setC)))
        extra = sorted((setB.union(setC)) - set_removed)
        raise AssertionError(f"[bundlers] Split does not cover removed. missing={missing}, extra={extra}")


# ---------------------------
# Layer helpers
# ---------------------------
def _get_layers(model, model_type: str):
    """모델 타입에 따라 레이어 접근"""
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


def _bundle_meta(config: PretrainedConfig, indices: List[int], model_type: str = None) -> Dict:
    """번들 메타데이터 생성"""
    if model_type is None:
        model_type = _detect_model_type_from_config(config)
    
    return {
        "base_model": getattr(config, "_name_or_path", ""),
        "arch": model_type,
        "model_type": model_type,
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "indices": list(indices),
        "format": "safetensors",
        "granularity": "layer",
    }


# ---------------------------
# Single-bundle export (레이어 단위 파일)
# ---------------------------
def export_layer_bundle(
    model,
    indices: List[int],
    out_dir: str,
    model_type: str,
    config: PretrainedConfig,
) -> None:
    """
    지정된 레이어 인덱스들의 파라미터를 레이어별 safetensors 파일로 저장합니다.
    - out_dir/bundle_meta.json: 메타 정보
    - out_dir/layer_{idx:03d}.safetensors: 각 레이어의 state_dict
    
    Args:
        model: Mistral/LLaMA/OPT 모델
        indices: 저장할 레이어 인덱스 리스트
        out_dir: 출력 디렉토리
        model_type: 'mistral', 'llama', 'opt'
        config: 모델 config
    """
    os.makedirs(out_dir, exist_ok=True)
    layers = _get_layers(model, model_type)

    # 메타
    meta = _bundle_meta(config, indices, model_type)
    with open(os.path.join(out_dir, "bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 레이어 단위 원자성 보장: 한 레이어는 한 파일에 전부 저장
    for i in indices:
        sd = layers[i].state_dict()
        sd_cpu = {k: v.detach().cpu() for k, v in sd.items()}
        save_file(sd_cpu, os.path.join(out_dir, f"layer_{i:03d}.safetensors"))


# ---------------------------
# Two-bundle export (B/C 동시 저장)
# ---------------------------
def export_two_bundles(
    model,
    removed_indices: List[int],
    out_root: str,
    model_type: str,
    config: PretrainedConfig,
    split_policy: str = "half",
    split_ratio: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """
    removed_indices를 split_policy에 따라 B/C로 분할 후, 각각 out_root/B, out_root/C에 저장.
    
    Args:
        model: Mistral/LLaMA/OPT 모델
        removed_indices: 제거된 레이어 인덱스 리스트
        out_root: 번들 루트 디렉토리
        model_type: 'mistral', 'llama', 'opt'
        config: 모델 config
        split_policy: 'half' or 'ratio'
        split_ratio: ratio policy일 때 비율
        
    Returns:
        (B_idx, C_idx)
    """
    B_idx, C_idx = split_indices(removed_indices, policy=split_policy, ratio=split_ratio)

    # 무결성 검사: 중복/누락 방지
    _ensure_disjoint_and_cover(removed_indices, B_idx, C_idx)

    B_dir = os.path.join(out_root, "B")
    C_dir = os.path.join(out_root, "C")

    export_layer_bundle(model, B_idx, B_dir, model_type, config)
    export_layer_bundle(model, C_idx, C_dir, model_type, config)

    # 저장 후 빠른 검증: 파일 이름 기반으로 레이어 충돌 확인
    _verify_bundle_atomicity_files(B_dir, C_dir)

    return B_idx, C_idx


# ---------------------------
# Post-save verification
# ---------------------------
def _list_layer_files(dir_path: str) -> Set[int]:
    """디렉토리에서 layer_*.safetensors 파일의 인덱스 추출"""
    out = set()
    if not os.path.isdir(dir_path):
        return out
    
    for name in os.listdir(dir_path):
        if name.startswith("layer_") and name.endswith(".safetensors"):
            try:
                idx = int(name[len("layer_"):len("layer_")+3])
            except ValueError:
                # layer_XX?.safetensors가 아닐 가능성도 처리
                try:
                    idx = int(name.split("_")[1].split(".")[0])
                except Exception:
                    continue
            out.add(idx)
    return out


def _verify_bundle_atomicity_files(B_dir: str, C_dir: str) -> None:
    """B와 C 번들이 서로 겹치지 않는지 파일 레벨에서 검증"""
    if not (os.path.isdir(B_dir) and os.path.isdir(C_dir)):
        return
    b_layers = _list_layer_files(B_dir)
    c_layers = _list_layer_files(C_dir)
    inter = b_layers.intersection(c_layers)
    if inter:
        raise AssertionError(f"[bundlers] Same layers appear in both B and C: {sorted(inter)}")


# ---------------------------
# Legacy compatibility
# ---------------------------
def export_layer_bundle_legacy(model, indices, out_dir, is_opt, config):
    """Legacy function for backward compatibility"""
    model_type = "opt" if is_opt else _detect_model_type_from_config(config)
    return export_layer_bundle(model, indices, out_dir, model_type, config)


def export_two_bundles_legacy(model, removed_indices, out_root, is_opt, config, 
                              split_policy="half", split_ratio=0.5):
    """Legacy function for backward compatibility"""
    model_type = "opt" if is_opt else _detect_model_type_from_config(config)
    return export_two_bundles(model, removed_indices, out_root, model_type, config,
                             split_policy, split_ratio)