# qwen_prune_lora/pruning/qwen_bundler.py
# Qwen 모델용 레이어 번들 저장 코드

import os
import json
from typing import List, Tuple, Dict, Set
from safetensors.torch import save_file
from transformers import PretrainedConfig


def split_indices(indices: List[int], policy: str = "half", ratio: float = 0.5) -> Tuple[List[int], List[int]]:
    """레이어 인덱스를 B/C 두 그룹으로 분할"""
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
    """B와 C가 겹치지 않고 removed를 완전히 커버하는지 검증"""
    set_removed, setB, setC = set(removed), set(B_idx), set(C_idx)
    if not setB.isdisjoint(setC):
        both = sorted(setB.intersection(setC))
        raise AssertionError(f"[bundlers] B/C overlap on layers: {both}")
    if setB.union(setC) != set_removed:
        missing = sorted(set_removed - (setB.union(setC)))
        extra = sorted((setB.union(setC)) - set_removed)
        raise AssertionError(f"[bundlers] Split mismatch. missing={missing}, extra={extra}")


def _get_layers(model):
    """Qwen 모델의 레이어 접근"""
    return model.transformer.h


def _bundle_meta(config: PretrainedConfig, indices: List[int]) -> Dict:
    """번들 메타데이터 생성"""
    return {
        "base_model": getattr(config, "_name_or_path", ""),
        "arch": "qwen",
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "indices": list(indices),
        "format": "safetensors",
        "granularity": "layer",
    }


def export_layer_bundle(
    model,
    indices: List[int],
    out_dir: str,
    config: PretrainedConfig,
) -> None:
    """
    지정된 레이어들을 개별 safetensors 파일로 저장 (Qwen용)
    - out_dir/bundle_meta.json: 메타 정보
    - out_dir/layer_{idx:03d}.safetensors: 각 레이어의 state_dict
    """
    os.makedirs(out_dir, exist_ok=True)
    layers = _get_layers(model)

    # 메타데이터 저장
    meta = _bundle_meta(config, indices)
    with open(os.path.join(out_dir, "bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 레이어별 저장
    for i in indices:
        sd = layers[i].state_dict()
        sd_cpu = {k: v.detach().cpu() for k, v in sd.items()}
        save_file(sd_cpu, os.path.join(out_dir, f"layer_{i:03d}.safetensors"))


def export_two_bundles(
    model,
    removed_indices: List[int],
    out_root: str,
    config: PretrainedConfig,
    split_policy: str = "half",
    split_ratio: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """
    removed_indices를 B/C로 분할 후 저장 (Qwen용)
    반환: (B_idx, C_idx)
    """
    B_idx, C_idx = split_indices(removed_indices, policy=split_policy, ratio=split_ratio)

    # 무결성 검사
    _ensure_disjoint_and_cover(removed_indices, B_idx, C_idx)

    B_dir = os.path.join(out_root, "B")
    C_dir = os.path.join(out_root, "C")

    export_layer_bundle(model, B_idx, B_dir, config)
    export_layer_bundle(model, C_idx, C_dir, config)

    # 파일 기반 검증
    _verify_bundle_atomicity_files(B_dir, C_dir)

    return B_idx, C_idx


def _list_layer_files(dir_path: str) -> Set[int]:
    """디렉토리에서 layer_*.safetensors 파일들의 인덱스 추출"""
    out = set()
    if not os.path.isdir(dir_path):
        return out
    for name in os.listdir(dir_path):
        if name.startswith("layer_") and name.endswith(".safetensors"):
            try:
                idx = int(name[len("layer_"):len("layer_")+3])
            except ValueError:
                try:
                    idx = int(name.split("_")[1].split(".")[0])
                except Exception:
                    continue
            out.add(idx)
    return out


def _verify_bundle_atomicity_files(B_dir: str, C_dir: str) -> None:
    """B와 C에 동일한 레이어가 없는지 파일 기반으로 검증"""
    if not (os.path.isdir(B_dir) and os.path.isdir(C_dir)):
        return
    b_layers = _list_layer_files(B_dir)
    c_layers = _list_layer_files(C_dir)
    inter = b_layers.intersection(c_layers)
    if inter:
        raise AssertionError(f"[bundlers] Same layers in both B and C: {sorted(inter)}")