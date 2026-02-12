# bundler.py
# 점진적 프루닝을 위해 제거한 레이어를 B/C 두 번들로 나누어 저장하는 코드 (Falcon 구조 대응)

import os, json
from typing import List, Tuple, Dict, Set
from safetensors.torch import save_file
from transformers import PretrainedConfig


# ---------------------------
# Split utilities
# ---------------------------
def split_indices(indices: List[int], policy: str = "half", ratio: float = 0.5) -> Tuple[List[int], List[int]]:
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
    set_removed, setB, setC = set(removed), set(B_idx), set(C_idx)
    if not setB.isdisjoint(setC):
        both = sorted(setB.intersection(setC))
        raise AssertionError(f"[bundler] B/C overlap on layers: {both}")
    if setB.union(setC) != set_removed:
        missing = sorted(set_removed - (setB.union(setC)))
        extra = sorted((setB.union(setC)) - set_removed)
        raise AssertionError(f"[bundler] Split does not cover removed. missing={missing}, extra={extra}")


# ---------------------------
# Layer helpers (Falcon)
# ---------------------------
def _get_layers(model):
    """Falcon 모델의 디코더 레이어 리스트 반환"""
    return model.transformer.h


def _bundle_meta(config: PretrainedConfig, indices: List[int]) -> Dict:
    return {
        "base_model": getattr(config, "_name_or_path", ""),
        "arch": getattr(config, "model_type", "falcon"),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "indices": list(indices),
        "format": "safetensors",
        "granularity": "layer",
    }


# ---------------------------
# Single-bundle export
# ---------------------------
def export_layer_bundle(
    model,
    indices: List[int],
    out_dir: str,
    config: PretrainedConfig,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    layers = _get_layers(model)

    meta = _bundle_meta(config, indices)
    with open(os.path.join(out_dir, "bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

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
    config: PretrainedConfig,
    split_policy: str = "half",
    split_ratio: float = 0.5,
) -> Tuple[List[int], List[int]]:
    B_idx, C_idx = split_indices(removed_indices, policy=split_policy, ratio=split_ratio)
    _ensure_disjoint_and_cover(removed_indices, B_idx, C_idx)

    B_dir = os.path.join(out_root, "B")
    C_dir = os.path.join(out_root, "C")

    export_layer_bundle(model, B_idx, B_dir, config)
    export_layer_bundle(model, C_idx, C_dir, config)

    _verify_bundle_atomicity_files(B_dir, C_dir)

    return B_idx, C_idx


# ---------------------------
# Post-save verification
# ---------------------------
def _list_layer_files(dir_path: str) -> Set[int]:
    out = set()
    for name in os.listdir(dir_path):
        if name.startswith("layer_") and name.endswith(".safetensors"):
            try:
                idx = int(name[len("layer_"):len("layer_") + 3])
            except ValueError:
                try:
                    idx = int(name.split("_")[1].split(".")[0])
                except Exception:
                    continue
            out.add(idx)
    return out


def _verify_bundle_atomicity_files(B_dir: str, C_dir: str) -> None:
    if not (os.path.isdir(B_dir) and os.path.isdir(C_dir)):
        return
    b_layers = _list_layer_files(B_dir)
    c_layers = _list_layer_files(C_dir)
    inter = b_layers.intersection(c_layers)
    if inter:
        raise AssertionError(f"[bundler] Same layers appear in both B and C: {sorted(inter)}")