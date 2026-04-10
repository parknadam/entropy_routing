#!/usr/bin/env python3
"""
Utility to merge LoRA adapters into Gemma/LLaMA base models or bundles.

Usage:
# A merge
python -m gemma_prune_lora.pruning.merge_adapter \
  --base_model ./gemma_7b_results/pruning/A \
  --adapter_path ./lora_results/adapters/A_lora/stageA \
  --output_dir ./merged_models/A_merged

# B merge (bundle-only auto detection)
python -m gemma_prune_lora.pruning.merge_adapter \
  --base_model ./gemma_7b_results/pruning/bundles/B \
  --adapter_path ./lora_results/adapters/B_lora/stageB \
  --output_dir ./merged_models/B_merged --device cuda:0

# C merge
python -m gemma_prune_lora.pruning.merge_adapter \
  --base_model ./gemma_7b_results/pruning/bundles/C \
  --adapter_path ./lora_results/adapters/C_lora/stageC \
  --output_dir ./merged_models/C_merged --device cuda:0
"""

import argparse
import json
import os
import re
import shutil

import torch
import torch.nn as nn
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
except Exception:
    GemmaDecoderLayer = None

from .identity import PassLayer
from .model_utils import detect_layer_return_tuple


# ============================================================
# regex helpers
# ============================================================
_BUNDLE_LAYER_FILE_RE = re.compile(r"^layer_(\d+)\.safetensors$")
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _unique_sorted_ints(indices):
    return sorted(set(int(i) for i in indices)) if indices else []


# ============================================================
# manifest / adapter config helpers
# ============================================================
def _read_stage_layers_from_manifest(base_model_path: str):
    manifest_path = os.path.join(base_model_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        return {
            "L_full": None,
            "A_kept": [],
            "A_dropped": [],
            "B_removed": [],
            "C_removed": [],
            "simdrop_removed": [],
        }
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return {
            "L_full": None,
            "A_kept": [],
            "A_dropped": [],
            "B_removed": [],
            "C_removed": [],
            "simdrop_removed": [],
        }

    stages = manifest.get("stages", {}) or {}
    counts = manifest.get("counts", {}) or {}
    return {
        "L_full": counts.get("L_full"),
        "A_kept": _unique_sorted_ints(stages.get("A", {}).get("kept_layers", [])),
        "A_dropped": _unique_sorted_ints(stages.get("A", {}).get("dropped_layers", [])),
        "B_removed": _unique_sorted_ints(stages.get("B", {}).get("removed_layers", [])),
        "C_removed": _unique_sorted_ints(stages.get("C", {}).get("removed_layers", [])),
        "simdrop_removed": _unique_sorted_ints(
            (manifest.get("simdrop", {}) or {}).get("removed_layers", [])
        ),
    }


def _read_manifest(base_model_path: str):
    manifest_path = os.path.join(base_model_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest if isinstance(manifest, dict) else {}
    except Exception:
        return {}


def _resolve_bundle_dirs_from_manifest(base_model_path: str):
    manifest = _read_manifest(base_model_path)
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    out = {}
    for stage in ("B", "C"):
        entry = artifacts.get(stage, {}) if isinstance(artifacts, dict) else {}
        dir_path = entry.get("dir") if isinstance(entry, dict) else None
        if isinstance(dir_path, str) and dir_path:
            out[stage] = os.path.abspath(dir_path) if os.path.exists(dir_path) else dir_path
    return out


def _infer_adapter_stage(adapter_path: str):
    if not adapter_path:
        return None
    p = adapter_path.replace("\\", "/").lower()
    if "stageb" in p or "/b_lora/" in p:
        return "B"
    if "stagec" in p or "/c_lora/" in p:
        return "C"
    if "stagea" in p or "/a_lora/" in p:
        return "A"
    return None


def _read_adapter_config(adapter_path: str):
    if not adapter_path:
        return {}
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_adapter_layers_to_transform(adapter_path: str):
    return _unique_sorted_ints(_read_adapter_config(adapter_path).get("layers_to_transform", []))


def _read_adapter_base_model_path(adapter_path: str):
    return _read_adapter_config(adapter_path).get("base_model_name_or_path")


def _is_hf_model_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def _list_bundle_layer_files(bundle_dir: str):
    if not os.path.isdir(bundle_dir):
        return []
    return [
        os.path.join(bundle_dir, f)
        for f in sorted(os.listdir(bundle_dir))
        if _BUNDLE_LAYER_FILE_RE.match(f)
    ]


def _load_bundle_indices(bundle_dir: str):
    meta_path = os.path.join(bundle_dir, "bundle_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            indices = meta.get("indices", []) or meta.get("layer_indices", [])
            if indices:
                return _unique_sorted_ints(indices)
        except Exception:
            pass
    idxs = []
    for layer_file in _list_bundle_layer_files(bundle_dir):
        m = _BUNDLE_LAYER_FILE_RE.match(os.path.basename(layer_file))
        if m:
            idxs.append(int(m.group(1)))
    return _unique_sorted_ints(idxs)


def _pick_bundle_layer_file(bundle_dir: str, idx: int):
    for fname in [f"layer_{idx:03d}.safetensors", f"layer_{idx}.safetensors"]:
        p = os.path.join(bundle_dir, fname)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Layer file not found for idx={idx} in bundle: {bundle_dir}")


def _extract_layer_sd(raw_sd: dict, idx: int):
    prefixes = [
        f"model.layers.{idx}.",
        f"layers.{idx}.",
        f"base_model.model.model.layers.{idx}.",
        f"model.model.layers.{idx}.",
    ]
    out = {}
    for k, v in raw_sd.items():
        for p in prefixes:
            if k.startswith(p):
                out[k[len(p):]] = v
                break
    return out if out else raw_sd


# ============================================================
# model layer access
# ============================================================
def _get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        return model.model.model.layers
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model") and hasattr(base.model, "model") and hasattr(base.model.model, "layers"):
            return base.model.model.layers
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            return base.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise RuntimeError("Cannot find decoder layers")


def _set_model_layers(model, new_layers):
    layer_list = nn.ModuleList(list(new_layers))
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = layer_list
        return
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        model.model.model.layers = layer_list
        return
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model") and hasattr(base.model, "model") and hasattr(base.model.model, "layers"):
            base.model.model.layers = layer_list
            return
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            base.model.layers = layer_list
            return
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        model.model.decoder.layers = layer_list
        return
    raise RuntimeError("Cannot set decoder layers")


# ============================================================
# PassLayer restore
# ============================================================
def _restore_passlayers(model, dropped_indices: list):
    if not dropped_indices:
        return model
    layers = _get_model_layers(model)
    hidden_size = model.config.hidden_size
    return_tuple = detect_layer_return_tuple(model)
    restored = []
    for idx in dropped_indices:
        if 0 <= idx < len(layers):
            old = layers[idx]
            layers[idx] = PassLayer(hidden_size, return_tuple=return_tuple)
            del old
            restored.append(idx)
    if restored:
        print(f"  [ok] PassLayer restored: {len(restored)} layers {restored}")
    return model


def _ensure_original_layout(model, stage_info: dict):
    stage_info = stage_info or {}
    layers = _get_model_layers(model)
    current_num_layers = len(layers)

    removed = _unique_sorted_ints(stage_info.get("A_dropped", []))
    kept = _unique_sorted_ints(stage_info.get("A_kept", []))
    original_num_layers = int(
        stage_info.get("L_full")
        or (max(removed + kept) + 1 if (removed or kept) else current_num_layers)
    )

    if not removed and current_num_layers == original_num_layers:
        model.config.num_hidden_layers = original_num_layers
        return model

    if not kept and removed:
        removed_set = set(removed)
        kept = [idx for idx in range(original_num_layers) if idx not in removed_set]

    try:
        ref_param = next(model.parameters())
        device, dtype = ref_param.device, ref_param.dtype
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.float32

    return_tuple = detect_layer_return_tuple(model)
    hidden_size = int(model.config.hidden_size)

    if current_num_layers == original_num_layers:
        changed = []
        for idx in removed:
            if 0 <= idx < len(layers) and not isinstance(layers[idx], PassLayer):
                old = layers[idx]
                layers[idx] = PassLayer(hidden_size, return_tuple=return_tuple).to(device=device, dtype=dtype)
                del old
                changed.append(idx)
        model.config.num_hidden_layers = original_num_layers
        if changed:
            print(f"  Reapplied PassLayer layout: {changed}")
        return model

    if kept and current_num_layers == len(kept):
        print(f"  Expanding compact layout: {current_num_layers} -> {original_num_layers}")
        old_layers = [layers[i] for i in range(current_num_layers)]
        new_layers = [None] * original_num_layers
        for packed_idx, original_idx in enumerate(kept):
            new_layers[original_idx] = old_layers[packed_idx]
        for idx in removed:
            new_layers[idx] = PassLayer(hidden_size, return_tuple=return_tuple).to(device=device, dtype=dtype)
        if any(layer is None for layer in new_layers):
            raise RuntimeError("Expanded layout contains empty layer slots.")
        _set_model_layers(model, new_layers)
        model.config.num_hidden_layers = original_num_layers
        return model

    raise RuntimeError(
        f"Cannot reconstruct original layout: loaded={current_num_layers}, "
        f"kept={len(kept)}, original={original_num_layers}"
    )


def _rehydrate_layer_if_needed(model, idx: int):
    layers = _get_model_layers(model)
    if idx < 0 or idx >= len(layers):
        raise IndexError(f"Layer idx {idx} out of range [0, {len(layers) - 1}]")

    layer = layers[idx]
    if not isinstance(layer, PassLayer):
        return layer

    if GemmaDecoderLayer is None:
        raise RuntimeError("GemmaDecoderLayer import failed. Check transformers version.")

    try:
        dev = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dev, dtype = torch.device("cpu"), torch.float32

    try:
        new_layer = GemmaDecoderLayer(model.config, idx)
    except TypeError:
        new_layer = GemmaDecoderLayer(model.config)
    new_layer = new_layer.to(device=dev, dtype=dtype)

    old = layers[idx]
    layers[idx] = new_layer
    del old
    return new_layer


def _restore_pruned_a_layout(model, stage_info: dict):
    return _ensure_original_layout(model, stage_info)


# ============================================================
# bundle injection
# ============================================================
def _inject_bundle_layers_into_model(model, bundle_dir: str, indices: list):
    if not indices:
        return
    layers = _get_model_layers(model)
    print(f"  Injecting bundle layers from: {bundle_dir}")
    for idx in indices:
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"Bundle layer idx {idx} out of range [0, {len(layers) - 1}]")
        _rehydrate_layer_if_needed(model, idx)
        sf_path = _pick_bundle_layer_file(bundle_dir, idx)
        raw_sd = load_file(sf_path)
        layer_sd = _extract_layer_sd(raw_sd, idx)
        try:
            ref_param = next(layers[idx].parameters())
            dev, dtype = ref_param.device, ref_param.dtype
            layer_sd = {k: v.to(device=dev, dtype=dtype) for k, v in layer_sd.items()}
        except StopIteration:
            pass
        try:
            layers[idx].load_state_dict(layer_sd, strict=True)
        except RuntimeError:
            layers[idx].load_state_dict(layer_sd, strict=False)
        print(f"    [ok] layer {idx} <- {os.path.basename(sf_path)}")


# ============================================================
# save helpers
# ============================================================
def _clear_output_dir(output_dir: str):
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        p = os.path.join(output_dir, name)
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def _save_bundle_only(model, output_dir: str, layer_indices: list, source_bundle_dir: str = None):
    if not layer_indices:
        raise ValueError("Bundle save requires non-empty layer_indices")
    _clear_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    layers = _get_model_layers(model)
    saved = []
    layer_indices = sorted(set(int(i) for i in layer_indices))
    for idx in layer_indices:
        sd = {k: v.detach().to("cpu") for k, v in layers[idx].state_dict().items()}
        out_f = os.path.join(output_dir, f"layer_{idx:03d}.safetensors")
        save_file(sd, out_f)
        saved.append(os.path.basename(out_f))

    meta = {}
    meta_src = os.path.join(source_bundle_dir, "bundle_meta.json") if source_bundle_dir else None
    if meta_src and os.path.isfile(meta_src):
        try:
            with open(meta_src, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass
    meta["layer_indices"] = layer_indices
    meta["indices"] = layer_indices
    meta["merged"] = True
    with open(os.path.join(output_dir, "bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"  Saved bundle: {saved} + bundle_meta.json ({total_bytes / (1024**3):.2f} GB)")


def _save_complete_model(model, tokenizer, output_dir: str, base_model_path: str = None):
    os.makedirs(output_dir, exist_ok=True)
    print("  Saving model weights...")
    model.save_pretrained(output_dir, safe_serialization=True)
    print("  Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)
    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(output_dir)
    except Exception:
        if base_model_path:
            gen_cfg = os.path.join(base_model_path, "generation_config.json")
            if os.path.isfile(gen_cfg):
                shutil.copy2(gen_cfg, os.path.join(output_dir, "generation_config.json"))
    if base_model_path:
        manifest_src = os.path.join(base_model_path, "manifest.json")
        if os.path.isfile(manifest_src):
            shutil.copy2(manifest_src, os.path.join(output_dir, "manifest.json"))
    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"  Total size: {total_bytes / (1024**3):.2f} GB")


def _extract_layer_idx(name: str):
    m = _LAYER_INDEX_RE.search(name)
    return int(m.group(1)) if m else None


def _save_complete_model_excluding_layers(
    model, tokenizer, output_dir, exclude_layer_indices, base_model_path=None,
):
    exclude_set = set(int(i) for i in (exclude_layer_indices or []))
    if not exclude_set:
        return _save_complete_model(model, tokenizer, output_dir, base_model_path)
    os.makedirs(output_dir, exist_ok=True)
    state = model.state_dict()
    filtered = {}
    dropped = 0
    for k, v in state.items():
        idx = _extract_layer_idx(k)
        if idx is not None and idx in exclude_set:
            dropped += 1
            continue
        filtered[k] = v
    print(f"  Saving model (excluding {dropped} keys from layers {sorted(exclude_set)})...")
    model.save_pretrained(output_dir, safe_serialization=True, state_dict=filtered)
    tokenizer.save_pretrained(output_dir)
    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(output_dir)
    except Exception:
        if base_model_path:
            gen_cfg = os.path.join(base_model_path, "generation_config.json")
            if os.path.isfile(gen_cfg):
                shutil.copy2(gen_cfg, os.path.join(output_dir, "generation_config.json"))
    if base_model_path:
        manifest_src = os.path.join(base_model_path, "manifest.json")
        if os.path.isfile(manifest_src):
            shutil.copy2(manifest_src, os.path.join(output_dir, "manifest.json"))
    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"  Total size: {total_bytes / (1024**3):.2f} GB")


# ============================================================
# LoRA scope enforcement
# ============================================================
def _collect_lora_layers(model):
    layers = set()
    for name, _ in model.named_parameters():
        if "lora_" not in name.lower():
            continue
        idx = _extract_layer_idx(name)
        if idx is not None:
            layers.add(idx)
    return sorted(layers)


def _zero_lora_weights_outside_layers(model, target_layers: list):
    if not target_layers:
        return
    target_set = set(target_layers)
    zeroed_layers = set()
    zeroed_tensors = 0
    with torch.no_grad():
        for module_name, module in model.named_modules():
            layer_idx = _extract_layer_idx(module_name)
            if layer_idx is None or layer_idx in target_set:
                continue
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)
            if lora_a is not None and lora_b is not None and hasattr(lora_a, "keys"):
                for adapter_name in list(lora_a.keys()):
                    if adapter_name in lora_a and hasattr(lora_a[adapter_name], "weight"):
                        lora_a[adapter_name].weight.zero_()
                        zeroed_tensors += 1
                        zeroed_layers.add(layer_idx)
                    if adapter_name in lora_b and hasattr(lora_b[adapter_name], "weight"):
                        lora_b[adapter_name].weight.zero_()
                        zeroed_tensors += 1
                        zeroed_layers.add(layer_idx)
    if zeroed_tensors:
        print(f"  [ok] Zeroed {zeroed_tensors} LoRA tensors outside target layers {sorted(zeroed_layers)}")


def _enforce_adapter_scope(model, target_layers: list):
    lora_layers = _collect_lora_layers(model)
    if not lora_layers:
        print("  [warn] No LoRA parameters found in adapter.")
        return
    print(f"  LoRA layers found: {lora_layers}")
    if not target_layers:
        return
    target_set = set(target_layers)
    scoped = sorted(i for i in lora_layers if i in target_set)
    if not scoped:
        raise RuntimeError(
            f"Adapter has no LoRA params on target layers {target_layers}. Found: {lora_layers}"
        )
    unexpected = sorted(i for i in lora_layers if i not in target_set)
    if unexpected:
        print(f"  [warn] Non-target LoRA layers detected: {unexpected}")
        _zero_lora_weights_outside_layers(model, target_layers)
    print(f"  [ok] Effective merge target layers: {target_layers}")


# ============================================================
# target layer resolution
# ============================================================
def _resolve_target_layers(base_model_path, adapter_path, stage_info=None):
    if stage_info is None:
        stage_info = _read_stage_layers_from_manifest(base_model_path)
    stage = _infer_adapter_stage(adapter_path)
    adapter_layers = _read_adapter_layers_to_transform(adapter_path)
    target_layers = []
    source = None
    if stage == "B" and stage_info["B_removed"]:
        target_layers = stage_info["B_removed"]
        source = "manifest.stages.B"
    elif stage == "C" and stage_info["C_removed"]:
        target_layers = stage_info["C_removed"]
        source = "manifest.stages.C"
    elif adapter_layers:
        target_layers = adapter_layers
        source = "adapter_config.layers_to_transform"
    if target_layers:
        print(f"  Target layers ({source}): {target_layers}")
    return stage, target_layers


def _read_dropped_layers_from_manifest(base_model_path, merge_stage=None, stage_info=None):
    if stage_info is None:
        stage_info = _read_stage_layers_from_manifest(base_model_path)
    a_dropped = stage_info["A_dropped"]
    b_removed = stage_info["B_removed"]
    c_removed = stage_info["C_removed"]
    simdrop_removed = stage_info["simdrop_removed"]
    if merge_stage == "B":
        if c_removed:
            return c_removed
        if a_dropped and b_removed:
            return _unique_sorted_ints(set(a_dropped) - set(b_removed))
        return []
    if merge_stage == "C":
        return []
    if a_dropped:
        return a_dropped
    if b_removed or c_removed:
        return _unique_sorted_ints(b_removed + c_removed)
    if simdrop_removed:
        return simdrop_removed
    return []


# ============================================================
# tokenizer helpers
# ============================================================
def _resolve_tokenizer(*candidate_paths, trust_remote_code=True):
    errors = []
    for path in candidate_paths:
        if path is None:
            continue
        try:
            resolved = os.path.abspath(path) if os.path.exists(path) else path
            tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=trust_remote_code)
            print(f"  [ok] Tokenizer loaded from: {path}")
            return tok
        except Exception as e:
            errors.append((path, str(e)))
    msg = ["Tokenizer load failed:"] + [f"  - {p}: {e}" for p, e in errors]
    raise RuntimeError("\n".join(msg))


def _find_tokenizer_candidates(base_model_path, adapter_path=None, tokenizer_path=None):
    candidates = []
    if tokenizer_path:
        candidates.append(tokenizer_path)
    candidates.append(base_model_path)
    orig_cfg_dir = os.path.join(base_model_path, "original_config")
    if os.path.isdir(orig_cfg_dir):
        candidates.append(orig_cfg_dir)
    if adapter_path:
        base_name = _read_adapter_base_model_path(adapter_path)
        if base_name:
            candidates.append(base_name)
    manifest_path = os.path.join(base_model_path, "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                original_model = json.load(f).get("base_model")
            if original_model:
                candidates.append(original_model)
        except Exception:
            pass
    return candidates


def _device_map_from_arg(device: str):
    if device in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return device
    return {"": device}


def _merge_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ============================================================
# verification
# ============================================================
def _verify_merged_model(output_dir: str):
    print(f"\n[Verify] {output_dir}")
    files = os.listdir(output_dir) if os.path.isdir(output_dir) else []
    has_config = "config.json" in files
    has_tok = any(f in files for f in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"])
    has_weights = any("model.safetensors" in f or "model-00001-of-" in f for f in files)
    ok = has_config and has_tok and has_weights
    print(f"  config={has_config} tokenizer={has_tok} weights={has_weights} -> {'PASS' if ok else 'FAIL'}")
    return ok


def _verify_bundle_output(output_dir: str):
    print(f"\n[Verify] {output_dir}")
    if not os.path.isdir(output_dir):
        return False
    files = os.listdir(output_dir)
    layer_files = [f for f in files if _BUNDLE_LAYER_FILE_RE.match(f)]
    has_meta = "bundle_meta.json" in files
    ok = bool(layer_files) and has_meta
    print(f"  layers={len(layer_files)} meta={has_meta} -> {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================
# single adapter merge
# ============================================================
def merge_single_adapter(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    device: str = "cuda:0",
    tokenizer_path: str = None,
    verify: bool = True,
    save_bundle_only: bool = False,
    force_full_model: bool = False,
):
    print(f"\n{'='*60}\nMerge Adapter (Gemma/LLaMA)\n{'='*60}")
    print(f"Base: {base_model_path}\nAdapter: {adapter_path}\nOutput: {output_dir}")

    bundle_mode = False
    bundle_dir = None
    bundle_indices = []
    effective_base = base_model_path

    if not _is_hf_model_dir(base_model_path):
        bundle_indices = _load_bundle_indices(base_model_path)
        if bundle_indices:
            bundle_mode = True
            bundle_dir = base_model_path
            if not save_bundle_only and not force_full_model:
                save_bundle_only = True
                print("  [mode] Bundle dir -> bundle-only save")
            adapter_base = _read_adapter_base_model_path(adapter_path)
            if not adapter_base:
                raise RuntimeError("Bundle dir but adapter has no base_model_name_or_path")
            effective_base = os.path.abspath(adapter_base) if os.path.exists(adapter_base) else adapter_base
            if not _is_hf_model_dir(effective_base):
                raise RuntimeError(f"Skeleton HF dir not found: {effective_base}")
            print(f"  Bundle layers: {bundle_indices}\n  Skeleton: {effective_base}")
        else:
            raise RuntimeError(f"No config.json or bundle files: {base_model_path}")

    # [1] routing
    print("\n[1/6] Layer routing...")
    stage_info = _read_stage_layers_from_manifest(effective_base)
    adapter_stage, target_layers = _resolve_target_layers(effective_base, adapter_path, stage_info)
    if adapter_stage in {"B", "C"} and not save_bundle_only and not force_full_model:
        save_bundle_only = True
    if bundle_mode and not target_layers:
        target_layers = list(bundle_indices)
    dropped_layers = _read_dropped_layers_from_manifest(effective_base, adapter_stage, stage_info)
    bundle_dirs = _resolve_bundle_dirs_from_manifest(effective_base)

    # [2] tokenizer
    if save_bundle_only:
        print("\n[2/6] Skipping tokenizer (bundle-only)")
        tokenizer = None
    else:
        print("\n[2/6] Loading tokenizer...")
        tokenizer = _resolve_tokenizer(*_find_tokenizer_candidates(effective_base, adapter_path, tokenizer_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # [3] base model
    print("\n[3/6] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        effective_base,
        torch_dtype=_merge_dtype(),
        device_map=_device_map_from_arg(device), trust_remote_code=True,
    )
    n_layers = len(_get_model_layers(base_model))
    print(f"  {n_layers} layers, {sum(p.numel() for p in base_model.parameters())/1e6:.1f}M params")
    _restore_pruned_a_layout(base_model, stage_info)

    if not bundle_mode and adapter_stage in {"B", "C"} and target_layers:
        bundle_dir = bundle_dirs.get(adapter_stage)
        if bundle_dir and os.path.isdir(bundle_dir):
            _inject_bundle_layers_into_model(base_model, bundle_dir, target_layers)
        else:
            print(f"  [warn] Missing manifest bundle dir for stage {adapter_stage}; proceeding without bundle injection.")
    if bundle_mode:
        _inject_bundle_layers_into_model(base_model, bundle_dir, bundle_indices)

    # [4] merge
    print("\n[4/6] Merging adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    _enforce_adapter_scope(model, target_layers)
    merged_model = model.merge_and_unload()
    print("  [ok] LoRA fused")

    # [5] passlayers
    if save_bundle_only:
        print("\n[5/6] Skip PassLayer (bundle-only)")
    else:
        print("\n[5/6] PassLayer layout already applied on load")

    # [6] save
    if save_bundle_only:
        indices = bundle_indices or target_layers
        print(f"\n[6/6] Saving bundle (layers {indices})...")
        _save_bundle_only(merged_model, output_dir, indices, bundle_dir)
    else:
        print(f"\n[6/6] Saving model...")
        if adapter_stage == "A" and dropped_layers:
            _save_complete_model_excluding_layers(
                merged_model, tokenizer, output_dir, dropped_layers, effective_base
            )
        else:
            _save_complete_model(merged_model, tokenizer, output_dir, effective_base)

    del merged_model, model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verify:
        (_verify_bundle_output if save_bundle_only else _verify_merged_model)(output_dir)

    print(f"\n[ok] Merge completed: {output_dir}")
    return output_dir


# ============================================================
# sequential multi-adapter merge
# ============================================================
def merge_multiple_adapters(
    base_model_path: str,
    adapter_paths: list,
    output_dir: str,
    device: str = "cuda:0",
    tokenizer_path: str = None,
    verify: bool = True,
):
    if not _is_hf_model_dir(base_model_path):
        raise ValueError("merge_multiple_adapters requires HF base model dir.")

    print(f"\n{'='*60}\nSequential Merge ({len(adapter_paths)} adapters)\n{'='*60}")
    stage_info = _read_stage_layers_from_manifest(base_model_path)

    tokenizer = _resolve_tokenizer(
        *_find_tokenizer_candidates(base_model_path, adapter_paths[0] if adapter_paths else None, tokenizer_path)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bundle_dirs = _resolve_bundle_dirs_from_manifest(base_model_path)
    current_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=_merge_dtype(),
        device_map=_device_map_from_arg(device), trust_remote_code=True,
    )
    _restore_pruned_a_layout(current_model, stage_info)

    for i, adapter_path in enumerate(adapter_paths, 1):
        print(f"\n--- Stage {i}/{len(adapter_paths)}: {os.path.basename(adapter_path)} ---")

        adapter_stage, target_layers = _resolve_target_layers(base_model_path, adapter_path, stage_info)
        if adapter_stage in {"B", "C"} and target_layers:
            bundle_dir = bundle_dirs.get(adapter_stage)
            if bundle_dir and os.path.isdir(bundle_dir):
                _inject_bundle_layers_into_model(current_model, bundle_dir, target_layers)
            else:
                print(f"  [warn] Missing manifest bundle dir for stage {adapter_stage}; proceeding without bundle injection.")

        model = PeftModel.from_pretrained(current_model, adapter_path)
        _enforce_adapter_scope(model, target_layers)
        current_model = model.merge_and_unload()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _save_complete_model(current_model, tokenizer, output_dir, base_model_path)

    del current_model

    if verify:
        _verify_merged_model(output_dir)
    print(f"\n[ok] All {len(adapter_paths)} adapters merged: {output_dir}")
    return output_dir


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter(s) into Gemma/LLaMA model or bundle")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--adapter_paths", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--no_verify", action="store_true")
    parser.add_argument("--save_bundle_only", action="store_true")
    parser.add_argument("--save_full_model", action="store_true")
    args = parser.parse_args()

    if args.adapter_path and args.adapter_paths:
        raise ValueError("--adapter_path and --adapter_paths cannot be used together.")
    if not args.adapter_path and not args.adapter_paths:
        raise ValueError("Specify either --adapter_path or --adapter_paths.")

    if args.adapter_path:
        merge_single_adapter(
            base_model_path=args.base_model,
            adapter_path=args.adapter_path,
            output_dir=args.output_dir,
            device=args.device,
            tokenizer_path=args.tokenizer_path,
            verify=not args.no_verify,
            save_bundle_only=args.save_bundle_only,
            force_full_model=args.save_full_model,
        )
    else:
        merge_multiple_adapters(
            base_model_path=args.base_model,
            adapter_paths=args.adapter_paths,
            output_dir=args.output_dir,
            device=args.device,
            tokenizer_path=args.tokenizer_path,
            verify=not args.no_verify,
        )


if __name__ == "__main__":
    main()
