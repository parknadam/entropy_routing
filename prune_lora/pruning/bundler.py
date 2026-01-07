# bundlers.py
# 점진적 프루닝을 위해 제거한 레이어를 B/C 두 번들로 나누어 저장하는 코드
import os, json
from typing import List, Tuple, Dict, Set
from safetensors.torch import save_file
from transformers import PretrainedConfig

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
    indices = list(indices)  # 사본
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
        raise AssertionError(f"[bundlers] B/C overlap on layers: {both}")
    if setB.union(setC) != set_removed:
        missing = sorted(set_removed - (setB.union(setC)))
        extra = sorted((setB.union(setC)) - set_removed)
        raise AssertionError(f"[bundlers] Split does not cover removed. missing={missing}, extra={extra}")

# ---------------------------
# Layer helpers
# ---------------------------
def _get_layers(model, is_opt: bool):
    return model.model.decoder.layers if is_opt else model.model.layers

def _bundle_meta(config: PretrainedConfig, indices: List[int]) -> Dict:
    return {
        "base_model": getattr(config, "_name_or_path", ""),
        "arch": getattr(config, "model_type", ""),
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
    is_opt: bool,
    config: PretrainedConfig,
) -> None:
    """
    지정된 레이어 인덱스들의 파라미터를 레이어별 safetensors 파일로 저장합니다.
    - out_dir/bundle_meta.json: 메타 정보
    - out_dir/layer_{idx:03d}.safetensors: 각 레이어의 state_dict
    """
    os.makedirs(out_dir, exist_ok=True)
    layers = _get_layers(model, is_opt)

    # 메타
    meta = _bundle_meta(config, indices)
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
    is_opt: bool,
    config: PretrainedConfig,
    split_policy: str = "half",
    split_ratio: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """
    removed_indices를 split_policy에 따라 B/C로 분할 후, 각각 out_root/B, out_root/C에 저장.
    반환: (B_idx, C_idx)
    """
    B_idx, C_idx = split_indices(removed_indices, policy=split_policy, ratio=split_ratio)

    # 무결성 검사: 중복/누락 방지
    _ensure_disjoint_and_cover(removed_indices, B_idx, C_idx)

    B_dir = os.path.join(out_root, "B")
    C_dir = os.path.join(out_root, "C")

    export_layer_bundle(model, B_idx, B_dir, is_opt, config)
    export_layer_bundle(model, C_idx, C_dir, is_opt, config)

    # 저장 후 빠른 검증: 파일 이름 기반으로 레이어 충돌 확인
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
    if not (os.path.isdir(B_dir) and os.path.isdir(C_dir)):
        return
    b_layers = _list_layer_files(B_dir)
    c_layers = _list_layer_files(C_dir)
    inter = b_layers.intersection(c_layers)
    if inter:
        raise AssertionError(f"[bundlers] Same layers appear in both B and C: {sorted(inter)}")