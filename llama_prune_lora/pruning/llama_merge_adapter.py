#!/usr/bin/env python3
"""
Utility to merge LoRA adapters into LLaMA base models or bundles.

Core behavior:
  A merge:  pruned A model + A LoRA -> save full HF model (PassLayer for B/C positions, ~10GB)
  B/C merge: bundle(B/C) + LoRA -> save only merged bundle layers (~1.6GB)

Usage:
# A merge (full model, B/C positions become PassLayer)
CUDA_VISIBLE_DEVICES=4 DEVICE=cuda:0 \
python -m progressiveserve.llama_prune_lora.pruning.llama_merge_adapter \
  --base_model ./7b_results/pruning/A \
  --adapter_path ./lora_results/adapters/A_lora/stageA/stageA \
  --output_dir ./new_merged_models_llama_7b_lora/A_merged

# B merge (bundle-only auto detection)
CUDA_VISIBLE_DEVICES=4 DEVICE=cuda:0 \
python -m llama_prune_lora.pruning.llama_merge_adapter \
  --base_model ./7b_results/pruning/bundles/B \
  --adapter_path ./lora_results/adapters/B_lora/stageB/stageB \
  --output_dir ./new_merged_models_llama_7b_lora/B_merged \
  --device cuda:0

# C merge (bundle-only auto detection)
python -m llama_prune_lora.pruning.llama_merge_adapter \
  --base_model ./7b_results/pruning/bundles/C \
  --adapter_path ./lora_results/adapters/C_lora/stageC/stageC \
  --output_dir ./new_merged_models_llama_7b_lora/C_merged \
  --device cuda:0
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


# ============================================================
# LlamaPassLayer: lightweight placeholder for pruned layers
# ============================================================
class LlamaPassLayer(nn.Module):
    def __init__(self, hidden_size: int = 0):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs


# ============================================================
# manifest / adapter config helpers
# ============================================================
_BUNDLE_LAYER_FILE_RE = re.compile(r"^layer_(\d+)\.safetensors$")
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _unique_sorted_ints(indices):
    if not indices:
        return []
    return sorted(set(int(i) for i in indices))


def _read_stage_layers_from_manifest(base_model_path: str):
    manifest_path = os.path.join(base_model_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"  [warn] manifest.json not found in {base_model_path}")
        return {"A_kept": [], "A_dropped": [], "B_removed": [], "C_removed": [], "simdrop_removed": []}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"  [warn] manifest.json parse failed: {e}")
        return {"A_kept": [], "A_dropped": [], "B_removed": [], "C_removed": [], "simdrop_removed": []}

    stages = manifest.get("stages", {}) or {}
    return {
        "A_kept": _unique_sorted_ints(stages.get("A", {}).get("kept_layers", [])),
        "A_dropped": _unique_sorted_ints(stages.get("A", {}).get("dropped_layers", [])),
        "B_removed": _unique_sorted_ints(stages.get("B", {}).get("removed_layers", [])),
        "C_removed": _unique_sorted_ints(stages.get("C", {}).get("removed_layers", [])),
        "simdrop_removed": _unique_sorted_ints(
            (manifest.get("simdrop", {}) or {}).get("removed_layers", [])
        ),
    }


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
    cfg = _read_adapter_config(adapter_path)
    return _unique_sorted_ints(cfg.get("layers_to_transform", []))


def _read_adapter_base_model_path(adapter_path: str):
    cfg = _read_adapter_config(adapter_path)
    return cfg.get("base_model_name_or_path")


def _is_hf_model_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def _list_bundle_layer_files(bundle_dir: str):
    if not os.path.isdir(bundle_dir):
        return []
    files = []
    for fname in sorted(os.listdir(bundle_dir)):
        if _BUNDLE_LAYER_FILE_RE.match(fname):
            files.append(os.path.join(bundle_dir, fname))
    return files


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

    raise RuntimeError("Cannot find LLaMA layers (expected model.layers)")


# ============================================================
# PassLayer restore
# ============================================================
def _restore_passlayers(model, dropped_indices: list):
    """Replace dropped positions with LlamaPassLayer (no parameters)."""
    if not dropped_indices:
        return model
    layers = _get_model_layers(model)
    hidden_size = model.config.hidden_size
    restored = []
    for idx in dropped_indices:
        if 0 <= idx < len(layers):
            old = layers[idx]
            layers[idx] = LlamaPassLayer(hidden_size)
            del old
            restored.append(idx)
    if restored:
        print(f"  [ok] LlamaPassLayer restored: {len(restored)} layers {restored}")
    return model


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
# output directory / save
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
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"Bundle save layer idx {idx} out of range [0, {len(layers)-1}]")
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
            meta = {}

    meta["arch"] = meta.get("arch", "llama")
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
    print(f"  Saved bundle files: {saved} + bundle_meta.json")
    print(f"  Total bundle size: {total_bytes / (1024**3):.2f} GB")


def _save_complete_model(model, tokenizer, output_dir: str, base_model_path: str = None):
    os.makedirs(output_dir, exist_ok=True)

    print("  Saving model weights...")
    model.save_pretrained(output_dir, safe_serialization=True)
    print("  Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(output_dir)
            print("  Saved generation_config.")
    except Exception:
        if base_model_path:
            gen_cfg_file = os.path.join(base_model_path, "generation_config.json")
            if os.path.isfile(gen_cfg_file):
                shutil.copy2(gen_cfg_file, os.path.join(output_dir, "generation_config.json"))
                print("  Copied generation_config from base model.")

    if base_model_path:
        manifest_src = os.path.join(base_model_path, "manifest.json")
        if os.path.isfile(manifest_src):
            shutil.copy2(manifest_src, os.path.join(output_dir, "manifest.json"))
            print("  Copied manifest.json.")

    saved_files = sorted(os.listdir(output_dir))
    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in saved_files
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"  Saved files: {saved_files}")
    print(f"  Total size: {total_bytes / (1024**3):.2f} GB")


def _save_complete_model_excluding_layers(
    model,
    tokenizer,
    output_dir: str,
    exclude_layer_indices: list,
    base_model_path: str = None,
):
    """
    Save a full HF model while excluding specific decoder-layer weights from state_dict.
    Useful for stage A output where B/C pass regions should not be serialized.
    """
    exclude_set = set(int(i) for i in (exclude_layer_indices or []))
    if not exclude_set:
        return _save_complete_model(model, tokenizer, output_dir, base_model_path)

    os.makedirs(output_dir, exist_ok=True)
    state = model.state_dict()
    filtered_state = {}
    dropped_keys = 0
    for k, v in state.items():
        idx = _extract_layer_idx(k)
        if idx is not None and idx in exclude_set:
            dropped_keys += 1
            continue
        filtered_state[k] = v

    print("  Saving model weights (excluding dropped B/C layer keys)...")
    model.save_pretrained(output_dir, safe_serialization=True, state_dict=filtered_state)
    print(f"  Excluded layer keys: {dropped_keys} (layers={sorted(exclude_set)})")

    print("  Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(output_dir)
            print("  Saved generation_config.")
    except Exception:
        if base_model_path:
            gen_cfg_file = os.path.join(base_model_path, "generation_config.json")
            if os.path.isfile(gen_cfg_file):
                shutil.copy2(gen_cfg_file, os.path.join(output_dir, "generation_config.json"))
                print("  Copied generation_config from base model.")

    if base_model_path:
        manifest_src = os.path.join(base_model_path, "manifest.json")
        if os.path.isfile(manifest_src):
            shutil.copy2(manifest_src, os.path.join(output_dir, "manifest.json"))
            print("  Copied manifest.json.")

    saved_files = sorted(os.listdir(output_dir))
    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in saved_files
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"  Saved files: {saved_files}")
    print(f"  Total size: {total_bytes / (1024**3):.2f} GB")


# ============================================================
# target layer resolution / LoRA scope enforcement
# ============================================================
def _resolve_target_layers(base_model_path: str, adapter_path: str, stage_info: dict = None):
    if stage_info is None:
        stage_info = _read_stage_layers_from_manifest(base_model_path)

    stage = _infer_adapter_stage(adapter_path)
    adapter_layers = _read_adapter_layers_to_transform(adapter_path)

    target_layers = []
    source = None
    if stage == "B" and stage_info["B_removed"]:
        target_layers = stage_info["B_removed"]
        source = "manifest.stages.B.removed_layers"
    elif stage == "C" and stage_info["C_removed"]:
        target_layers = stage_info["C_removed"]
        source = "manifest.stages.C.removed_layers"
    elif adapter_layers:
        target_layers = adapter_layers
        source = "adapter_config.layers_to_transform"

    if target_layers:
        print(f"  Target layers ({source}): {target_layers}")

    return stage, target_layers


def _read_dropped_layers_from_manifest(base_model_path: str, merge_stage: str = None, stage_info: dict = None):
    """
    Determine which layers should be restored as PassLayer for each merge stage.
      A: B+C become PassLayer
      B: only C becomes PassLayer
      C: none
    """
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

    # Stage A / fallback
    if a_dropped:
        return a_dropped
    if b_removed or c_removed:
        return _unique_sorted_ints(b_removed + c_removed)
    if simdrop_removed:
        return simdrop_removed
    return []


def _extract_layer_idx(name: str):
    m = _LAYER_INDEX_RE.search(name)
    return int(m.group(1)) if m else None


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

            lora_emb_a = getattr(module, "lora_embedding_A", None)
            lora_emb_b = getattr(module, "lora_embedding_B", None)
            if lora_emb_a is not None and lora_emb_b is not None and hasattr(lora_emb_a, "keys"):
                for adapter_name in list(lora_emb_a.keys()):
                    if adapter_name in lora_emb_a:
                        lora_emb_a[adapter_name].zero_()
                        zeroed_tensors += 1
                        zeroed_layers.add(layer_idx)
                    if adapter_name in lora_emb_b:
                        lora_emb_b[adapter_name].zero_()
                        zeroed_tensors += 1
                        zeroed_layers.add(layer_idx)

    if zeroed_tensors:
        print(
            f"  [ok] LoRA scope enforced: zeroed {zeroed_tensors} tensors outside target layers "
            f"{sorted(zeroed_layers)}"
        )


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
            f"Adapter has no LoRA params on target layers {target_layers}. "
            f"Found layers: {lora_layers}"
        )

    unexpected = sorted(i for i in lora_layers if i not in target_set)
    if unexpected:
        print(f"  [warn] Non-target LoRA layers detected: {unexpected}")
        _zero_lora_weights_outside_layers(model, target_layers)

    print(f"  [ok] Effective merge target layers: {target_layers}")


# ============================================================
# tokenizer (multi fallback)
# ============================================================
def _resolve_tokenizer(*candidate_paths: str, trust_remote_code: bool = True):
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

    msg_lines = ["Tokenizer load failed. tried:"]
    for p, err in errors:
        msg_lines.append(f"  - {p}: {err}")
    msg_lines.append("Use --tokenizer_path to set explicit tokenizer path.")
    raise RuntimeError("\n".join(msg_lines))


def _find_tokenizer_candidates(base_model_path: str, adapter_path: str = None, tokenizer_path: str = None):
    candidates = []
    if tokenizer_path:
        candidates.append(tokenizer_path)
    candidates.append(base_model_path)

    orig_cfg_dir = os.path.join(base_model_path, "original_config")
    if os.path.isdir(orig_cfg_dir):
        candidates.append(orig_cfg_dir)

    if adapter_path:
        cfg = _read_adapter_config(adapter_path)
        base_name = cfg.get("base_model_name_or_path")
        if base_name:
            candidates.append(base_name)

    manifest_path = os.path.join(base_model_path, "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            original_model = manifest.get("base_model")
            if original_model:
                candidates.append(original_model)
        except Exception:
            pass

    return candidates


def _device_map_from_arg(device: str):
    if device in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return device
    return {"": device}


# ============================================================
# verification
# ============================================================
def _verify_merged_model(output_dir: str):
    print(f"\n[Verify] Checking merged model at {output_dir}...")
    abs_dir = os.path.abspath(output_dir)
    files = os.listdir(abs_dir) if os.path.isdir(abs_dir) else []

    tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
    weight_patterns = ["model.safetensors", "model-00001-of-"]

    has_config = "config.json" in files
    has_tokenizer = any(tf in files for tf in tokenizer_files)
    has_weights = any(any(wp in f for wp in weight_patterns) for f in files)

    print(f"  config.json: {'OK' if has_config else 'MISSING'}")
    print(f"  tokenizer:   {'OK' if has_tokenizer else 'MISSING'}")
    print(f"  weights:     {'OK' if has_weights else 'MISSING'}")

    ok = has_config and has_tokenizer and has_weights
    print(f"  {'verification passed' if ok else 'verification failed'}")
    return ok


def _verify_bundle_output(output_dir: str):
    print(f"\n[Verify] Checking bundle output at {output_dir}...")
    abs_dir = os.path.abspath(output_dir)
    if not os.path.isdir(abs_dir):
        print("  directory not found")
        return False

    files = os.listdir(abs_dir)
    layer_files = sorted(f for f in files if _BUNDLE_LAYER_FILE_RE.match(f))
    has_meta = "bundle_meta.json" in files

    print(f"  layer files: {len(layer_files)} -> {layer_files}")
    print(f"  bundle_meta: {'OK' if has_meta else 'MISSING'}")

    total_bytes = sum(
        os.path.getsize(os.path.join(abs_dir, f))
        for f in files
        if os.path.isfile(os.path.join(abs_dir, f))
    )
    print(f"  total size:  {total_bytes / (1024**3):.2f} GB")

    ok = bool(layer_files) and has_meta
    print(f"  {'verification passed' if ok else 'verification failed'}")
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
    """Merge a single LoRA adapter into base model/bundle."""
    print(f"\n{'=' * 60}")
    print("Merging Adapter into Base Model (LLaMA)")
    print(f"{'=' * 60}")
    print(f"Base Model:  {base_model_path}")
    print(f"Adapter:     {adapter_path}")
    print(f"Output:      {output_dir}")
    if tokenizer_path:
        print(f"Tokenizer:   {tokenizer_path}")

    # Detect whether base_model is HF model dir or bundle dir.
    bundle_mode = False
    bundle_dir = None
    bundle_indices = []
    effective_base_model_path = base_model_path

    if not _is_hf_model_dir(base_model_path):
        bundle_indices = _load_bundle_indices(base_model_path)
        if bundle_indices:
            bundle_mode = True
            bundle_dir = base_model_path

            if not save_bundle_only and not force_full_model:
                save_bundle_only = True
                print("  [mode] Bundle dir detected -> auto-enabled bundle-only save mode")
            elif force_full_model:
                print("  [mode] --save_full_model set -> forcing full-model save mode")

            adapter_base = _read_adapter_base_model_path(adapter_path)
            if not adapter_base:
                raise RuntimeError(
                    "base_model is bundle dir, but adapter_config.json has no base_model_name_or_path."
                )

            resolved_adapter_base = os.path.abspath(adapter_base) if os.path.exists(adapter_base) else adapter_base
            if not _is_hf_model_dir(resolved_adapter_base):
                raise RuntimeError(
                    "Bundle merge requires a skeleton HF model dir with config.json. "
                    f"Not found/invalid: {resolved_adapter_base}"
                )

            effective_base_model_path = resolved_adapter_base
            print(f"  Bundle base:     {bundle_dir}")
            print(f"  Skeleton model:  {effective_base_model_path}")
            print(f"  Bundle layers:   {bundle_indices}")
        else:
            raise RuntimeError(
                f"--base_model has neither config.json nor bundle layer files: {base_model_path}"
            )

    # [1/6] manifest + adapter routing
    print("\n[1/6] Reading manifest/adapter for layer routing...")
    stage_info = _read_stage_layers_from_manifest(effective_base_model_path)
    adapter_stage, target_layers = _resolve_target_layers(
        effective_base_model_path, adapter_path, stage_info
    )

    # For stage B/C adapters, default to bundle-only save unless explicitly forced.
    if adapter_stage in {"B", "C"} and not save_bundle_only and not force_full_model:
        save_bundle_only = True
        print(f"  [mode] Stage {adapter_stage} adapter detected -> auto-enabled bundle-only save mode")

    if bundle_mode and not target_layers:
        target_layers = list(bundle_indices)

    dropped_layers = _read_dropped_layers_from_manifest(
        effective_base_model_path,
        merge_stage=adapter_stage,
        stage_info=stage_info,
    )

    if adapter_stage:
        print(f"  Adapter stage: {adapter_stage}")

    if bundle_mode:
        print(f"  Save mode: bundle-only (layers {bundle_indices} only)")
        if target_layers and set(target_layers) != set(bundle_indices):
            print(
                f"  [warn] target layers from manifest/config differ from bundle indices. "
                f"target={target_layers}, bundle={bundle_indices}"
            )
    else:
        if dropped_layers:
            print(f"  PassLayer targets: {dropped_layers} ({len(dropped_layers)} layers)")
        else:
            print("  PassLayer targets: none")

    if adapter_stage == "A":
        if dropped_layers:
            print("  [mode] Stage A merge -> save A-layer weights only (B/C dropped layers excluded)")
        else:
            print("  [warn] Stage A merge but no dropped layers resolved from manifest")

    # [2/6] tokenizer
    if save_bundle_only:
        print("\n[2/6] Skipping tokenizer load (bundle-only save)")
        tokenizer = None
    else:
        print("\n[2/6] Loading tokenizer...")
        tok_candidates = _find_tokenizer_candidates(effective_base_model_path, adapter_path, tokenizer_path)
        tokenizer = _resolve_tokenizer(*tok_candidates)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"  Set pad_token = eos_token ({tokenizer.eos_token})")

    # [3/6] load base skeleton and inject bundle layers if needed.
    print("\n[3/6] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        effective_base_model_path,
        torch_dtype=torch.float16,
        device_map=_device_map_from_arg(device),
        trust_remote_code=True,
    )
    n_layers = len(_get_model_layers(base_model))
    print(f"  Loaded: {n_layers} layers, {sum(p.numel() for p in base_model.parameters())/1e6:.1f}M params")

    if bundle_mode:
        _inject_bundle_layers_into_model(base_model, bundle_dir, bundle_indices)

    # [4/6] merge adapter
    print("\n[4/6] Loading and merging adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    _enforce_adapter_scope(model, target_layers)
    merged_model = model.merge_and_unload()
    print("  [ok] LoRA weights fused into base model")

    # [5/6] restore passlayers when saving full model.
    if save_bundle_only:
        print("\n[5/6] Skipping PassLayer restore (bundle-only save)")
    elif adapter_stage == "A":
        print("\n[5/6] Skipping PassLayer restore for stage A (will exclude dropped B/C layer keys on save)")
    else:
        print("\n[5/6] Restoring LlamaPassLayers for dropped layers...")
        merged_model = _restore_passlayers(merged_model, dropped_layers)

    # [6/6] save
    if save_bundle_only:
        bundle_save_indices = bundle_indices if bundle_indices else target_layers
        if not bundle_save_indices:
            raise RuntimeError("save_bundle_only=True but no layer indices resolved.")
        print(f"\n[6/6] Saving bundle-only (layers {bundle_save_indices}) to {output_dir}...")
        _save_bundle_only(
            merged_model,
            output_dir=output_dir,
            layer_indices=bundle_save_indices,
            source_bundle_dir=bundle_dir,
        )
    else:
        print(f"\n[6/6] Saving complete merged model to {output_dir}...")
        if adapter_stage == "A" and dropped_layers:
            _save_complete_model_excluding_layers(
                merged_model,
                tokenizer,
                output_dir=output_dir,
                exclude_layer_indices=dropped_layers,
                base_model_path=effective_base_model_path,
            )
        else:
            _save_complete_model(merged_model, tokenizer, output_dir, effective_base_model_path)

    # cleanup & verify
    del merged_model, model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verify:
        if save_bundle_only:
            _verify_bundle_output(output_dir)
        else:
            _verify_merged_model(output_dir)

    print(f"\n{'=' * 60}")
    print(f"[ok] Merge completed: {output_dir}")
    if save_bundle_only:
        print(f"  (bundle-only: layers {bundle_indices if bundle_indices else target_layers})")
    print(f"{'=' * 60}")
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
    """Merge multiple LoRA adapters sequentially (full-model mode only)."""
    if not _is_hf_model_dir(base_model_path):
        raise ValueError("merge_multiple_adapters requires HF base model dir (config.json).")

    print(f"\n{'=' * 60}")
    print("Sequential Adapter Merging (LLaMA)")
    print(f"{'=' * 60}")
    print(f"Base Model:   {base_model_path}")
    print(f"Adapters:     {adapter_paths}")
    print(f"Final Output: {output_dir}")

    print("\n[0a] Reading manifest for stage layers...")
    stage_info = _read_stage_layers_from_manifest(base_model_path)
    if stage_info["A_dropped"]:
        print(f"  A dropped: {stage_info['A_dropped']}")
    if stage_info["B_removed"]:
        print(f"  B layers:  {stage_info['B_removed']}")
    if stage_info["C_removed"]:
        print(f"  C layers:  {stage_info['C_removed']}")

    print("\n[0b] Loading tokenizer...")
    first_adapter = adapter_paths[0] if adapter_paths else None
    tok_candidates = _find_tokenizer_candidates(base_model_path, first_adapter, tokenizer_path)
    tokenizer = _resolve_tokenizer(*tok_candidates)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    current_model_path = base_model_path
    temp_dirs = []

    for i, adapter_path in enumerate(adapter_paths, 1):
        is_last = i == len(adapter_paths)
        print(f"\n{'-' * 40}")
        print(f"Stage {i}/{len(adapter_paths)}: {os.path.basename(adapter_path)}")
        print(f"{'-' * 40}")

        adapter_stage, target_layers = _resolve_target_layers(base_model_path, adapter_path, stage_info)
        dropped_layers = _read_dropped_layers_from_manifest(
            base_model_path,
            merge_stage=adapter_stage,
            stage_info=stage_info,
        )
        if adapter_stage:
            print(f"  Adapter stage: {adapter_stage}")
        print(f"  PassLayer targets: {dropped_layers if dropped_layers else '[]'}")

        print(f"  Loading model from {current_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            current_model_path,
            torch_dtype=torch.float16,
            device_map=_device_map_from_arg(device),
            trust_remote_code=True,
        )

        print(f"  Loading adapter: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        _enforce_adapter_scope(model, target_layers)
        merged_model = model.merge_and_unload()
        print(f"  [ok] Merged stage {i}")

        print("  Restoring LlamaPassLayers...")
        merged_model = _restore_passlayers(merged_model, dropped_layers)

        if is_last:
            print(f"\n  Saving final model to {output_dir}...")
            _save_complete_model(merged_model, tokenizer, output_dir, base_model_path)
        else:
            temp_dir = f"{output_dir}_temp_stage{i}"
            print(f"  Saving intermediate to {temp_dir}...")
            _save_complete_model(merged_model, tokenizer, temp_dir, base_model_path)
            current_model_path = temp_dir
            temp_dirs.append(temp_dir)

        del model, merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for td in temp_dirs:
        try:
            shutil.rmtree(td)
            print(f"  Cleaned up: {td}")
        except Exception as e:
            print(f"  Warning: {td}: {e}")

    if verify:
        _verify_merged_model(output_dir)

    print(f"\n{'=' * 60}")
    print(f"[ok] All {len(adapter_paths)} adapters merged: {output_dir}")
    print(f"{'=' * 60}")
    return output_dir


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter(s) into LLaMA base model or bundle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # A merge (full model, B/C become PassLayer ~10GB)
  python -m llama_prune_lora.pruning.llama_merge_adapter \\
    --base_model ./7b_results/pruning/A \\
    --adapter_path ./kd_lora_results/adapters/A_lora/stageA \\
    --output_dir ./merged_models_llama_7b/A_merged

  # B merge (bundle-only ~1.6GB)
  python -m llama_prune_lora.pruning.llama_merge_adapter \\
    --base_model ./7b_results/pruning/bundles/B \\
    --adapter_path ./kd_lora_results/adapters/B_lora/stageB \\
    --output_dir ./merged_models_llama_7b/B_merged

  # C merge (bundle-only ~1.6GB)
  python -m llama_prune_lora.pruning.llama_merge_adapter \\
    --base_model ./7b_results/pruning/bundles/C \\
    --adapter_path ./kd_lora_results/adapters/C_lora/stageC \\
    --output_dir ./merged_models_llama_7b/C_merged
        """,
    )
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to HF base model (A) or bundle dir (B/C)")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to single adapter")
    parser.add_argument("--adapter_paths", type=str, nargs="+", default=None,
                        help="Paths to multiple adapters for sequential full-model merge")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Explicit tokenizer path (used for full-model save)")
    parser.add_argument("--no_verify", action="store_true", help="Skip post-merge verification")
    parser.add_argument("--save_bundle_only", action="store_true",
                        help="Force bundle-only save")
    parser.add_argument("--save_full_model", action="store_true",
                        help="Force full-model save even when base_model is bundle dir")

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
        if args.save_bundle_only or args.save_full_model:
            raise ValueError("--save_bundle_only/--save_full_model are only for single-adapter mode.")
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
