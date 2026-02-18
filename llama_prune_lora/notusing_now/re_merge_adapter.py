#!/usr/bin/env python3
"""
A(완전한 HF 모델)과 B/C(레이어 번들)를 "분리"해서 LoRA를 merge하는 유틸.

- A: HF 모델 폴더(config.json 있음) -> PeftModel merge -> HF 모델 저장
- B/C: 번들 폴더(config.json 없음) -> skeleton 모델(A_merged 등)에 번들 가중치 주입 -> LoRA merge
       -> 번들 형태로 다시 저장(입력 번들 레이아웃을 유지)

지원 번들 레이아웃:
1) per-layer: layer_{idx}.safetensors 다수 (+ bundle_meta.json optional)
2) single-file: model.safetensors 1개 (키 subset로 저장돼 있는 경우)

Usage:
  # A merge
  python -m prune_lora.pruning.merge_split_progressive \
    --a_model ./7b_results/pruning/A \
    --a_adapter ./kd_lora_results/adapters/A_lora/stageA/stageA \
    --out_a ./merged_models/A_merged \
    --device cuda:0 --dtype fp16

  # Bundle merge (B or C)
  python -m prune_lora.pruning.merge_split_progressive \
    --skeleton_model ./merged_models/A_merged \
    --bundle_dir ./7b_results/pruning/bundles/B \
    --bundle_adapter ./kd_lora_results/adapters/B_lora/stageB \
    --out_bundle ./merged_models/bundles/B_merged \
    --device cuda:0 --dtype fp16

# 1) A merge
python -m prune_lora.pruning.merge_split_progressive \
  --a_model ./7b_results/pruning/A \
  --a_adapter ./kd_lora_results/adapters/A_lora/stageA/stageA \
  --out_a ./merged_models/A_merged \
  --device cuda:0 --dtype fp16

# 2) B 번들 + stageB merge → B_merged_bundle 생성
python -m prune_lora.pruning.re_merge_adapter \
  --skeleton_model ./merged_models/A_merged \
  --bundle_dir ./7b_results/pruning/bundles/B \
  --bundle_adapter ./kd_lora_results/adapters/B_lora/stageB \
  --out_bundle ./merged_models/B_merged \
  --device cuda:0 --dtype fp16

# 3) C 번들 + stageC merge → C_merged_bundle 생성
python -m prune_lora.pruning.re_merge_adapter \
  --skeleton_model ./merged_models/A_merged \
  --bundle_dir ./7b_results/pruning/bundles/C \
  --bundle_adapter ./kd_lora_results/adapters/C_lora/stageC \
  --out_bundle ./merged_models/C_merged \
  --device cuda:0 --dtype fp16

"""

import os
import re
import glob
import json
import argparse
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from safetensors.torch import load_file, save_file


# -----------------------------
# Helpers
# -----------------------------
def _parse_dtype(dtype_str: str):
    s = dtype_str.lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str} (use fp16/bf16/fp32)")

def _device_map(device: str):
    # accelerate가 설치된 환경이면 {"": "cuda:0"} 같은 형태도 잘 동작하는 편
    if device == "auto":
        return "auto"
    return {"": device}

def _is_hf_model_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _list_bundle_layout(bundle_dir: str) -> Tuple[str, List[str]]:
    """
    returns: ("per_layer", [layer_files]) or ("single", [model_file])
    """
    layer_files = sorted(glob.glob(os.path.join(bundle_dir, "layer_*.safetensors")))
    if layer_files:
        return "per_layer", layer_files
    model_file = os.path.join(bundle_dir, "model.safetensors")
    if os.path.isfile(model_file):
        return "single", [model_file]
    # 혹시 이름이 다른 경우도 고려: *.safetensors 하나뿐이라면 single로 처리
    all_st = sorted(glob.glob(os.path.join(bundle_dir, "*.safetensors")))
    if len(all_st) == 1:
        return "single", all_st
    raise FileNotFoundError(
        f"Bundle dir has no layer_*.safetensors or model.safetensors: {bundle_dir}"
    )

def _extract_layer_idx_from_filename(path: str) -> Optional[int]:
    m = re.search(r"layer_(\d+)\.safetensors$", os.path.basename(path))
    return int(m.group(1)) if m else None

def _infer_layer_indices_from_single_safetensors(keys: List[str]) -> List[int]:
    """
    model.safetensors 내부 키들에서 layer index를 추정.
    예: 'model.layers.12.self_attn.q_proj.weight' -> 12
    """
    idxs = set()
    for k in keys:
        m = re.search(r"\blayers\.(\d+)\b", k)
        if m:
            idxs.add(int(m.group(1)))
    return sorted(idxs)

def _load_model(model_dir: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=_device_map(device),
        trust_remote_code=True
    )
    return model

def merge_hf_adapter(base_model_dir: str, adapter_dir: str, out_dir: str, device: str, dtype: torch.dtype):
    print("=" * 60)
    print("Merge HF model + LoRA adapter")
    print("=" * 60)
    print(f"Base:   {base_model_dir}")
    print(f"Adapter:{adapter_dir}")
    print(f"Out:    {out_dir}")

    if not _is_hf_model_dir(base_model_dir):
        raise ValueError(f"Base model must be HF model dir with config.json: {base_model_dir}")

    model = _load_model(base_model_dir, device=device, dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged = model.merge_and_unload()

    _ensure_dir(out_dir)
    merged.save_pretrained(out_dir)

    # tokenizer 저장(가능하면)
    try:
        tok = AutoTokenizer.from_pretrained(base_model_dir)
        tok.save_pretrained(out_dir)
        print("Tokenizer saved.")
    except Exception as e:
        print(f"Tokenizer save skipped: {e}")

    print(f"✓ Saved HF merged model: {out_dir}")
    return out_dir

# -----------------------------
# Bundle merge: inject bundle weights -> merge adapter -> export bundle again
# -----------------------------
def _inject_per_layer_bundle(model, layer_files: List[str]):
    """
    layer_{idx}.safetensors를 해당 model.model.layers[idx]에 주입.
    키가 prefix 포함/미포함 둘 다 처리.
    """
    for f in layer_files:
        idx = _extract_layer_idx_from_filename(f)
        if idx is None:
            raise ValueError(f"Cannot parse layer idx from filename: {f}")

        sd = load_file(f)
        # 1) 키가 submodule 기준(self_attn.q_proj.weight 등)이라면 layer module에 바로 load
        # 2) 키가 model.layers.{idx}. ... 처럼 prefix가 있으면 전체 모델 strict=False로 load
        try:
            missing, unexpected = model.model.layers[idx].load_state_dict(sd, strict=False)
            # 너무 많이 unexpected가 나오면 prefix 케이스일 가능성이 있으니 fallback
            if len(sd) > 0 and (len(unexpected) / max(len(sd), 1)) > 0.5:
                raise RuntimeError("Too many unexpected keys; try full-model load fallback")
        except Exception:
            # full-model fallback
            missing, unexpected = model.load_state_dict(sd, strict=False)

def _inject_single_bundle(model, model_file: str):
    """
    model.safetensors(키 subset) 형태 번들을 전체 모델 strict=False로 주입
    """
    sd = load_file(model_file)
    model.load_state_dict(sd, strict=False)
    return list(sd.keys())

def _export_per_layer_bundle(model, layer_indices: List[int], out_dir: str, meta_src: Optional[str] = None):
    _ensure_dir(out_dir)
    for idx in layer_indices:
        layer_sd = model.model.layers[idx].state_dict()
        out_f = os.path.join(out_dir, f"layer_{idx}.safetensors")
        save_file(layer_sd, out_f)
    # meta copy/create
    if meta_src and os.path.isfile(meta_src):
        with open(meta_src, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}
    meta["merged"] = True
    meta["layer_indices"] = layer_indices
    with open(os.path.join(out_dir, "bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _export_single_bundle(model, keys_to_save: List[str], out_dir: str, out_name: str = "model.safetensors", meta_src: Optional[str] = None):
    _ensure_dir(out_dir)
    full_sd = model.state_dict()
    subset = {k: full_sd[k].detach().cpu() for k in keys_to_save if k in full_sd}
    out_f = os.path.join(out_dir, out_name)
    save_file(subset, out_f)

    # meta
    if meta_src and os.path.isfile(meta_src):
        with open(meta_src, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}
    meta["merged"] = True
    meta["saved_keys"] = len(subset)
    with open(os.path.join(out_dir, "bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def merge_bundle_adapter(
    skeleton_model_dir: str,
    bundle_dir: str,
    bundle_adapter_dir: str,
    out_bundle_dir: str,
    device: str,
    dtype: torch.dtype,
):
    print("=" * 60)
    print("Merge Bundle(B/C) + LoRA adapter (split)")
    print("=" * 60)
    print(f"Skeleton: {skeleton_model_dir}")
    print(f"Bundle:   {bundle_dir}")
    print(f"Adapter:  {bundle_adapter_dir}")
    print(f"Out:      {out_bundle_dir}")

    if not _is_hf_model_dir(skeleton_model_dir):
        raise ValueError(f"Skeleton model must be HF model dir with config.json: {skeleton_model_dir}")

    layout, files = _list_bundle_layout(bundle_dir)
    meta_src = os.path.join(bundle_dir, "bundle_meta.json")
    print(f"Detected bundle layout: {layout}")

    # 1) skeleton 로드
    model = _load_model(skeleton_model_dir, device=device, dtype=dtype)

    # 2) 번들 가중치 주입
    if layout == "per_layer":
        layer_files = files
        layer_indices = [_extract_layer_idx_from_filename(f) for f in layer_files]
        layer_indices = [i for i in layer_indices if i is not None]
        _inject_per_layer_bundle(model, layer_files)
        keys_to_save = None
    else:
        model_file = files[0]
        keys_to_save = _inject_single_bundle(model, model_file)
        layer_indices = _infer_layer_indices_from_single_safetensors(keys_to_save)

    # 3) 어댑터 merge
    model = PeftModel.from_pretrained(model, bundle_adapter_dir)
    merged = model.merge_and_unload()

    # 4) 번들 형태로 재저장
    if layout == "per_layer":
        if not layer_indices:
            raise ValueError("Could not infer layer indices for per_layer bundle.")
        _export_per_layer_bundle(merged, layer_indices, out_bundle_dir, meta_src=meta_src)
    else:
        if not keys_to_save:
            raise ValueError("Could not collect keys for single bundle.")
        _export_single_bundle(merged, keys_to_save, out_bundle_dir, out_name=os.path.basename(files[0]), meta_src=meta_src)

    print(f"✓ Saved merged bundle: {out_bundle_dir}")
    return out_bundle_dir


def main():
    ap = argparse.ArgumentParser()
    # A merge mode
    ap.add_argument("--a_model", type=str, default=None, help="HF base model dir for A (has config.json)")
    ap.add_argument("--a_adapter", type=str, default=None, help="Adapter dir for A (has adapter_model.safetensors+adapter_config.json)")
    ap.add_argument("--out_a", type=str, default=None, help="Output dir for merged A model")

    # Bundle merge mode (B or C)
    ap.add_argument("--skeleton_model", type=str, default=None, help="HF model dir used as skeleton (e.g., merged_models/A_merged)")
    ap.add_argument("--bundle_dir", type=str, default=None, help="Bundle dir (B or C) containing layer_*.safetensors or model.safetensors")
    ap.add_argument("--bundle_adapter", type=str, default=None, help="Adapter dir for that bundle stage")
    ap.add_argument("--out_bundle", type=str, default=None, help="Output dir for merged bundle")

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    args = ap.parse_args()
    dtype = _parse_dtype(args.dtype)

    did_something = False

    # A merge
    if args.a_model and args.a_adapter and args.out_a:
        merge_hf_adapter(args.a_model, args.a_adapter, args.out_a, args.device, dtype)
        did_something = True

    # Bundle merge
    if args.skeleton_model and args.bundle_dir and args.bundle_adapter and args.out_bundle:
        merge_bundle_adapter(args.skeleton_model, args.bundle_dir, args.bundle_adapter, args.out_bundle, args.device, dtype)
        did_something = True

    if not did_something:
        raise ValueError(
            "No action. Provide either:\n"
            "- --a_model --a_adapter --out_a\n"
            "or\n"
            "- --skeleton_model --bundle_dir --bundle_adapter --out_bundle\n"
        )

if __name__ == "__main__":
    main()
