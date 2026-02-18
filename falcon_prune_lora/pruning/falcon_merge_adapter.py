#!/usr/bin/env python3
"""
Utility to merge LoRA adapters into Falcon base models or bundles.

Core behavior:
  A merge:  pruned A model + A LoRA -> save full HF model
  B/C merge: bundle(B/C) + LoRA -> save only merged bundle layers

Usage:
# A merge (full model)
python -m falcon_prune_lora.pruning.falcon_merge_adapter \
  --base_model ./falcon_results/pruning/A \
  --adapter_path ./falcon_kd_lora_results/adapters/A_lora/stageA/stageA \
  --output_dir ./merged_models_falcon/A_merged \
  --device cuda:0

# B merge (bundle-only auto detection)
python -m falcon_prune_lora.pruning.falcon_merge_adapter \
  --base_model ./falcon_results/pruning/bundles/B \
  --adapter_path ./falcon_kd_lora_results/adapters/B_lora/stageB \
  --output_dir ./merged_models_falcon/B_merged \
  --device cuda:0

# C merge (bundle-only auto detection)
python -m falcon_prune_lora.pruning.falcon_merge_adapter \
  --base_model ./falcon_results/pruning/bundles/C \
  --adapter_path ./falcon_kd_lora_results/adapters/C_lora/stageC \
  --output_dir ./merged_models_falcon/C_merged \
  --device cuda:0
"""

import argparse
import json
import os
import re
import shutil

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


_BUNDLE_LAYER_FILE_RE = re.compile(r"^layer_(\d+)\.safetensors$")
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)(?:h|layers)\.(\d+)(?:\.|$)")


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
        f"transformer.h.{idx}.",
        f"h.{idx}.",
        f"model.transformer.h.{idx}.",
        f"base_model.model.transformer.h.{idx}.",
    ]
    out = {}
    for k, v in raw_sd.items():
        for p in prefixes:
            if k.startswith(p):
                out[k[len(p):]] = v
                break
    return out if out else raw_sd


def _get_model_layers(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "transformer") and hasattr(model.model.transformer, "h"):
        return model.model.transformer.h
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model") and hasattr(base.model, "transformer") and hasattr(base.model.transformer, "h"):
            return base.model.transformer.h
        if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
            return base.transformer.h
    raise RuntimeError("Cannot find Falcon layers (expected transformer.h)")


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
    meta["arch"] = meta.get("arch", "falcon")
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

            lora_A = getattr(module, "lora_A", None)
            lora_B = getattr(module, "lora_B", None)
            if lora_A is not None and lora_B is not None and hasattr(lora_A, "keys"):
                for adapter_name in list(lora_A.keys()):
                    if adapter_name in lora_A and hasattr(lora_A[adapter_name], "weight"):
                        lora_A[adapter_name].weight.zero_()
                        zeroed_tensors += 1
                        zeroed_layers.add(layer_idx)
                    if adapter_name in lora_B and hasattr(lora_B[adapter_name], "weight"):
                        lora_B[adapter_name].weight.zero_()
                        zeroed_tensors += 1
                        zeroed_layers.add(layer_idx)

            lora_emb_A = getattr(module, "lora_embedding_A", None)
            lora_emb_B = getattr(module, "lora_embedding_B", None)
            if lora_emb_A is not None and lora_emb_B is not None and hasattr(lora_emb_A, "keys"):
                for adapter_name in list(lora_emb_A.keys()):
                    if adapter_name in lora_emb_A:
                        lora_emb_A[adapter_name].zero_()
                        zeroed_tensors += 1
                        zeroed_layers.add(layer_idx)
                    if adapter_name in lora_emb_B:
                        lora_emb_B[adapter_name].zero_()
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
    print(f"\n{'=' * 60}")
    print("Merging Adapter into Base Model (Falcon)")
    print(f"{'=' * 60}")
    print(f"Base Model:  {base_model_path}")
    print(f"Adapter:     {adapter_path}")
    print(f"Output:      {output_dir}")
    if tokenizer_path:
        print(f"Tokenizer:   {tokenizer_path}")

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

    print("\n[1/5] Reading manifest/adapter for layer routing...")
    stage_info = _read_stage_layers_from_manifest(effective_base_model_path)
    adapter_stage, target_layers = _resolve_target_layers(
        effective_base_model_path, adapter_path, stage_info
    )
    if bundle_mode and not target_layers:
        target_layers = list(bundle_indices)

    if adapter_stage:
        print(f"  Adapter stage: {adapter_stage}")
    if bundle_mode and target_layers and set(target_layers) != set(bundle_indices):
        print(
            "  [warn] target layers from manifest/config differ from bundle indices. "
            f"target={target_layers}, bundle={bundle_indices}"
        )

    tokenizer = None
    if save_bundle_only:
        print("\n[2/5] Skipping tokenizer load (bundle-only save)")
    else:
        print("\n[2/5] Loading tokenizer...")
        tok_candidates = _find_tokenizer_candidates(effective_base_model_path, adapter_path, tokenizer_path)
        tokenizer = _resolve_tokenizer(*tok_candidates)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"  Set pad_token = eos_token ({tokenizer.eos_token})")

    print("\n[3/5] Loading base model...")
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

    print("\n[4/5] Loading and merging adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    _enforce_adapter_scope(model, target_layers)
    merged_model = model.merge_and_unload()
    print("  [ok] LoRA weights fused into base model")

    print("\n[5/5] Saving merged output...")
    if save_bundle_only:
        bundle_save_indices = bundle_indices if bundle_indices else target_layers
        if not bundle_save_indices:
            raise RuntimeError("save_bundle_only=True but no layer indices resolved.")
        _save_bundle_only(
            merged_model,
            output_dir=output_dir,
            layer_indices=bundle_save_indices,
            source_bundle_dir=bundle_dir,
        )
    else:
        _save_complete_model(merged_model, tokenizer, output_dir, effective_base_model_path)

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


def merge_multiple_adapters(
    base_model_path: str,
    adapter_paths: list,
    output_dir: str,
    device: str = "cuda:0",
    tokenizer_path: str = None,
    verify: bool = True,
):
    if not _is_hf_model_dir(base_model_path):
        raise ValueError("merge_multiple_adapters requires HF base model dir (config.json).")

    print(f"\n{'=' * 60}")
    print("Sequential Adapter Merging (Falcon)")
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

        _, target_layers = _resolve_target_layers(base_model_path, adapter_path, stage_info)

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


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter(s) into Falcon base model or bundle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # A merge (full model)
  python -m falcon_prune_lora.pruning.falcon_merge_adapter \\
    --base_model ./falcon_results/pruning/A \\
    --adapter_path ./falcon_kd_lora_results/adapters/A_lora/stageA \\
    --output_dir ./merged_models_falcon/A_merged

  # B merge (bundle-only auto mode)
  python -m falcon_prune_lora.pruning.falcon_merge_adapter \\
    --base_model ./falcon_results/pruning/bundles/B \\
    --adapter_path ./falcon_kd_lora_results/adapters/B_lora/stageB \\
    --output_dir ./merged_models_falcon/B_merged

  # C merge (bundle-only auto mode)
  python -m falcon_prune_lora.pruning.falcon_merge_adapter \\
    --base_model ./falcon_results/pruning/bundles/C \\
    --adapter_path ./falcon_kd_lora_results/adapters/C_lora/stageC \\
    --output_dir ./merged_models_falcon/C_merged
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
