#!/usr/bin/env python3
"""
머지된 Gemma 모델(pruned + LoRA merged)의 PPL을 평가하는 스크립트.

평가 모드:
  1) A-only:     A_merged 모델을 그대로 평가 (B/C는 PassLayer)
  2) A/AB/FULL:  --b_bundle, --c_bundle로 머지된 번들을 주입하여 stage별 비교

사용법:
# A-only 평가
python -m gemma_prune_lora.eval_ppl_mergedmodel \
  --model_path ./merged_models_gemma_7b/A_merged \
  --device cuda:0

# A / AB / FULL stage 비교 (머지된 B/C 번들 사용)
CUDA_VISIBLE_DEVICES=2 DEVICE=cuda:0 \
python -m gemma_prune_lora.eval_ppl_mergedmodel \
  --model_path ./merged_models_gemma_7b/A_merged \
  --b_bundle ./merged_models_gemma_7b/B_merged \
  --c_bundle ./merged_models_gemma_7b/C_merged \
  --stages A,AB,FULL \
  --device cuda:0

# 원본 모델과 비교
python -m gemma_prune_lora.eval_ppl_mergedmodel \
  --model_path google/gemma-7b ./merged_models_gemma_7b/A_merged \
  --tokenizer_path google/gemma-7b \
  --device cuda:0

# 원본 bundles_dir 구조로도 사용 가능 (구버전 호환)
python -m gemma_prune_lora.eval_ppl_mergedmodel \
  --model_path ./merged_models_gemma_7b/A_merged \
  --bundles_dir ./gemma_7b_results/pruning/bundles \
  --stages A,AB,FULL \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None

GemmaDecoderLayer = None
try:
    from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer as _GDL

    GemmaDecoderLayer = _GDL
except ImportError:
    pass

GET_LOADERS = None
for _mod_name in ["gemma_prune_lora.pruning.data", "pruning.data"]:
    try:
        _mod = __import__(_mod_name, fromlist=["get_loaders"])
        if hasattr(_mod, "get_loaders"):
            GET_LOADERS = getattr(_mod, "get_loaders")
            break
    except Exception:
        pass

PeftModel = None
try:
    from peft import PeftModel as _PM

    PeftModel = _PM
except ImportError:
    pass


class GemmaPassLayer(nn.Module):
    def __init__(self, hidden_size: int, return_tuple: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.return_tuple = return_tuple

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        output_attentions=False,
        **kwargs,
    ):
        if not self.return_tuple:
            return hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_values,)
        return outputs


def _get_layers(model) -> nn.ModuleList:
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
    raise RuntimeError("Cannot find Gemma decoder layers")


def _detect_layer_return_tuple(model) -> bool:
    try:
        core = model.model if hasattr(model, "model") else model
        src = inspect.getsource(core.forward)
        if "layer_outputs[0]" in src or "layer_outputs = decoder_layer" in src:
            return True
        if "hidden_states = decoder_layer" in src and "layer_outputs[0]" not in src:
            return False
    except Exception:
        pass
    return True


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalize_indices(indices: List[int], num_layers: int) -> List[int]:
    if not indices:
        return []
    uniq = sorted(set(int(i) for i in indices))
    if all(0 <= i < num_layers for i in uniq):
        return uniq
    if all(1 <= i <= num_layers for i in uniq):
        return [i - 1 for i in uniq]
    raise ValueError(f"Layer indices out of range for num_layers={num_layers}: {uniq}")


def _read_stage_layout(model_dir: Path, num_layers: int) -> Dict[str, Any]:
    manifest = _load_json(model_dir / "manifest.json")
    stages = manifest.get("stages", {}) if isinstance(manifest, dict) else {}
    manifest_layers = int((manifest.get("counts", {}) or {}).get("L_full", num_layers)) if manifest else num_layers

    layout = {
        "num_layers": manifest_layers,
        "A_kept": [],
        "A_dropped": [],
        "B_removed": [],
        "C_removed": [],
    }

    try:
        layout["A_kept"] = _normalize_indices(stages.get("A", {}).get("kept_layers", []), manifest_layers)
        layout["A_dropped"] = _normalize_indices(stages.get("A", {}).get("dropped_layers", []), manifest_layers)
        layout["B_removed"] = _normalize_indices(stages.get("B", {}).get("removed_layers", []), manifest_layers)
        layout["C_removed"] = _normalize_indices(stages.get("C", {}).get("removed_layers", []), manifest_layers)
    except ValueError as exc:
        print(f"[WARN] manifest stage indices look invalid: {exc}")

    if not layout["A_dropped"]:
        layers_map = _load_json(model_dir / "layers_map.json")
        raw_layers = layers_map.get("layers", {}) if isinstance(layers_map, dict) else {}
        inferred = []
        for idx_str, param_names in raw_layers.items():
            try:
                idx = int(idx_str)
            except Exception:
                continue
            if isinstance(param_names, list) and len(param_names) == 0:
                inferred.append(idx)
        try:
            layout["A_dropped"] = _normalize_indices(inferred, num_layers)
        except ValueError as exc:
            print(f"[WARN] layers_map indices look invalid: {exc}")

    if not layout["A_dropped"]:
        layout["A_dropped"] = sorted(set(layout["B_removed"]) | set(layout["C_removed"]))
    if not layout["A_kept"] and layout["A_dropped"]:
        dropped_set = set(layout["A_dropped"])
        layout["A_kept"] = [idx for idx in range(layout["num_layers"]) if idx not in dropped_set]

    return layout


def _read_dropped_layers(model_path: str, num_layers: Optional[int] = None) -> List[int]:
    if num_layers is None:
        cfg = _load_json(Path(model_path) / "config.json")
        num_layers = int(cfg.get("num_hidden_layers", 0)) if cfg else 0
    layout = _read_stage_layout(Path(model_path), int(num_layers or 0))
    return sorted(set(int(i) for i in layout.get("A_dropped", [])))


def _read_bc_indices(model_path: str, num_layers: Optional[int] = None) -> Tuple[List[int], List[int]]:
    if num_layers is None:
        cfg = _load_json(Path(model_path) / "config.json")
        num_layers = int(cfg.get("num_hidden_layers", 0)) if cfg else 0
    layout = _read_stage_layout(Path(model_path), int(num_layers or 0))
    return layout.get("B_removed", []), layout.get("C_removed", [])


def _install_passlayers(model, dropped_indices: List[int], return_tuple: bool = True):
    if not dropped_indices:
        return model
    layers = _get_layers(model)
    hidden_size = getattr(model.config, "hidden_size", 0)
    restored = []
    for idx in dropped_indices:
        if 0 <= idx < len(layers):
            old = layers[idx]
            try:
                dev = next(old.parameters()).device
            except StopIteration:
                dev = torch.device("cpu")
            layers[idx] = GemmaPassLayer(hidden_size, return_tuple=return_tuple).to(dev)
            del old
            restored.append(idx)
    if restored:
        print(f"  [ok] PassLayer installed: {restored} ({len(restored)} layers)")
    return model


def _set_layers(model, new_layers: List[nn.Module]) -> None:
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
    raise RuntimeError("Cannot find Gemma decoder layers while setting layers.")


def _ensure_original_layout(model, layout: Dict[str, Any], return_tuple: bool) -> Any:
    layers = _get_layers(model)
    current_num_layers = len(layers)
    original_num_layers = int(layout.get("num_layers", current_num_layers) or current_num_layers)
    removed = sorted(set(int(i) for i in layout.get("A_dropped", [])))
    kept = sorted(set(int(i) for i in layout.get("A_kept", [])))

    if not kept and removed:
        removed_set = set(removed)
        kept = [idx for idx in range(original_num_layers) if idx not in removed_set]

    hidden_size = getattr(model.config, "hidden_size", 0)

    if current_num_layers == original_num_layers:
        return _install_passlayers(model, removed, return_tuple=return_tuple)

    if kept and current_num_layers == len(kept):
        old_layers = [layers[i] for i in range(current_num_layers)]
        new_layers: List[Optional[nn.Module]] = [None] * original_num_layers
        for packed_idx, original_idx in enumerate(kept):
            new_layers[original_idx] = old_layers[packed_idx]
        for idx in removed:
            new_layers[idx] = GemmaPassLayer(hidden_size, return_tuple=return_tuple)
        if any(layer is None for layer in new_layers):
            raise RuntimeError("Expanded merged-model layout contains empty layer slots.")
        _set_layers(model, new_layers)
        model.config.num_hidden_layers = original_num_layers
        return model

    raise RuntimeError(
        f"Cannot reconstruct merged-model layout: loaded={current_num_layers}, "
        f"kept={len(kept)}, original={original_num_layers}"
    )


def _load_model(model_path: str, dtype: torch.dtype, device: str):
    print(f"  Loading model from: {model_path}")
    resolved = os.path.abspath(model_path) if os.path.exists(model_path) else model_path
    attempts = [
        {"torch_dtype": dtype, "attn_implementation": "eager", "trust_remote_code": True},
        {"torch_dtype": dtype, "trust_remote_code": True},
        {"dtype": dtype, "attn_implementation": "eager", "trust_remote_code": True},
        {"dtype": dtype, "trust_remote_code": True},
    ]

    last_error = None
    model = None
    for kwargs in attempts:
        try:
            model = AutoModelForCausalLM.from_pretrained(resolved, low_cpu_mem_usage=True, **kwargs)
            break
        except TypeError as exc:
            last_error = exc

    if model is None:
        raise last_error

    layout = _read_stage_layout(Path(model_path), len(_get_layers(model)))
    model = _ensure_original_layout(model, layout, return_tuple=_detect_layer_return_tuple(model))
    num_layers = len(_get_layers(model))
    dropped = sorted(set(int(i) for i in layout.get("A_dropped", [])))

    model = model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    n_params = sum(p.numel() for p in model.parameters())
    n_active = num_layers - len(dropped)
    if dropped:
        print(
            f"  [ok] {model.config.model_type} | {num_layers} layers "
            f"({n_active} active + {len(dropped)} PassLayer) | {n_params/1e6:.1f}M params | {device}"
        )
    else:
        print(f"  [ok] {model.config.model_type} | {num_layers} layers | {n_params/1e6:.1f}M params | {device}")
    return model


def _load_tokenizer(model_path: str, fallback_paths: Optional[List[str]] = None) -> AutoTokenizer:
    candidates = [model_path]
    if fallback_paths:
        candidates.extend(fallback_paths)

    errors = []
    for path in candidates:
        if path is None:
            continue
        try:
            resolved = os.path.abspath(path) if os.path.exists(path) else path
            tok = AutoTokenizer.from_pretrained(resolved, use_fast=True, trust_remote_code=True)
            if path != model_path:
                print(f"  [INFO] Tokenizer fallback: {path}")
            return tok
        except Exception as exc:
            errors.append((path, str(exc)))

    for path in candidates:
        if path is None:
            continue
        try:
            resolved = os.path.abspath(path) if os.path.exists(path) else path
            tok = AutoTokenizer.from_pretrained(resolved, use_fast=False, trust_remote_code=True)
            print(f"  [INFO] Slow tokenizer: {path}")
            return tok
        except Exception:
            continue

    em = "\n".join(f"  - {p}: {e}" for p, e in errors)
    raise RuntimeError(f"Tokenizer load failed:\n{em}\nUse --tokenizer_path.")


def _find_tokenizer_fallbacks(model_path: str) -> List[str]:
    fallbacks = []
    orig_cfg = os.path.join(model_path, "original_config")
    if os.path.isdir(orig_cfg):
        fallbacks.append(orig_cfg)

    manifest_path = os.path.join(model_path, "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            base_model = manifest.get("base_model")
            if base_model:
                fallbacks.append(base_model)
        except Exception:
            pass
    return fallbacks


_LAYER_RE = re.compile(r"layer_(\d+)\.safetensors$")


def _build_layer_map(dir_path: Optional[Path]) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    if dir_path is None or not dir_path.exists():
        return out
    for path in dir_path.glob("layer_*.safetensors"):
        match = _LAYER_RE.search(path.name)
        if match:
            out[int(match.group(1))] = path
    return out


def _strip_layer_prefix(sd: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, torch.Tensor]:
    prefixes = [
        f"model.layers.{layer_idx}.",
        f"model.model.layers.{layer_idx}.",
        f"layers.{layer_idx}.",
        f"base_model.model.model.layers.{layer_idx}.",
    ]
    out = {}
    for key, value in sd.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        out[new_key] = value
    return out


def _maybe_shift_indices_to_zero_based(
    b_map: Dict[int, Path],
    c_map: Dict[int, Path],
    num_layers: int,
) -> Tuple[Dict[int, Path], Dict[int, Path], int]:
    all_idx = sorted(set(b_map) | set(c_map))
    if not all_idx:
        return b_map, c_map, 0

    if all(0 <= i < num_layers for i in all_idx):
        return b_map, c_map, 0

    if all(1 <= i <= num_layers for i in all_idx):
        return ({i - 1: p for i, p in b_map.items()}, {i - 1: p for i, p in c_map.items()}, -1)

    raise ValueError(
        f"Bundle layer index mismatch (num_layers={num_layers}, max={max(all_idx)}). "
        "base_model and bundles may not match."
    )


class DynamicStageManager:
    def __init__(
        self,
        model,
        base_model_dir: Path,
        device: str,
        dtype: torch.dtype,
        passlayer_return_tuple: bool,
        b_dir: Optional[Path] = None,
        c_dir: Optional[Path] = None,
        bundles_dir: Optional[Path] = None,
    ):
        if GemmaDecoderLayer is None:
            raise RuntimeError("GemmaDecoderLayer import failed. Check transformers version.")
        if load_file is None:
            raise RuntimeError("safetensors import failed. Install safetensors first.")

        self.model = model
        self.layers = _get_layers(model)
        self.device = device
        self.dtype = dtype
        self.hidden_size = getattr(model.config, "hidden_size", 0)
        self.return_tuple = passlayer_return_tuple

        self.num_layers = len(self.layers)
        self.layout = _read_stage_layout(base_model_dir, self.num_layers)

        actual_b_dir = b_dir if b_dir is not None else (bundles_dir / "B" if bundles_dir is not None else None)
        actual_c_dir = c_dir if c_dir is not None else (bundles_dir / "C" if bundles_dir is not None else None)

        b_raw = _build_layer_map(actual_b_dir)
        c_raw = _build_layer_map(actual_c_dir)
        self.b_map, self.c_map, self.index_shift = _maybe_shift_indices_to_zero_based(b_raw, c_raw, self.num_layers)
        self.b_idx = self.layout["B_removed"] or sorted(self.b_map)
        self.c_idx = self.layout["C_removed"] or sorted(self.c_map)
        self.removed = self.layout["A_dropped"] or sorted(set(self.b_idx) | set(self.c_idx))

        if self.layout["num_layers"] != self.num_layers:
            print(
                f"[WARN] manifest L_full={self.layout['num_layers']} but loaded model has {self.num_layers} layers."
            )

        bundle_union = sorted(set(self.b_map) | set(self.c_map))
        stage_union = sorted(set(self.b_idx) | set(self.c_idx))
        if bundle_union and stage_union and bundle_union != stage_union:
            print(
                f"[WARN] manifest bundle indices {stage_union} differ from files on disk {bundle_union}. "
                "Stage-A masking follows manifest; AB/FULL restore requires matching bundle files."
            )

    def stage_meta(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "index_shift_applied": self.index_shift,
            "B": self.b_idx,
            "C": self.c_idx,
            "removed": self.removed,
            "manifest_num_layers": self.layout["num_layers"],
        }

    def _pass_one_layer(self, layer_idx: int):
        old = self.layers[layer_idx]
        self.layers[layer_idx] = GemmaPassLayer(self.hidden_size, self.return_tuple).to(self.device)
        del old

    def _restore_one_layer(self, layer_idx: int):
        bundle_path = self.b_map.get(layer_idx) or self.c_map.get(layer_idx)
        if bundle_path is None:
            raise FileNotFoundError(f"layer_{layer_idx}.safetensors not found in B/C bundle dirs.")

        try:
            new_layer = GemmaDecoderLayer(self.model.config, layer_idx)
        except TypeError:
            new_layer = GemmaDecoderLayer(self.model.config)
        new_layer = new_layer.to(self.device, dtype=self.dtype)

        state_dict = _strip_layer_prefix(load_file(str(bundle_path), device="cpu"), layer_idx)
        missing, unexpected = new_layer.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[WARN] layer {layer_idx}: missing={len(missing)} unexpected={len(unexpected)}")

        old = self.layers[layer_idx]
        self.layers[layer_idx] = new_layer
        del old

    def set_stage(self, stage: str):
        stage = stage.upper()
        if stage not in ("A", "AB", "FULL"):
            raise ValueError("stage must be A / AB / FULL")

        pass_set = set(self.removed) if stage == "A" else set(self.c_idx) if stage == "AB" else set()

        for idx in self.removed:
            is_pass = isinstance(self.layers[idx], GemmaPassLayer)
            if idx in pass_set and not is_pass:
                self._pass_one_layer(idx)
            if idx not in pass_set and is_pass:
                self._restore_one_layer(idx)

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x


def _fit_to_len(x: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
    x = _ensure_2d(x)
    if x.size(1) > target_len:
        return x[:, :target_len]
    if x.size(1) < target_len:
        pad = torch.full((x.size(0), target_len - x.size(1)), pad_value, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=1)
    return x


def _is_attn_mask_like(x: torch.Tensor) -> bool:
    if x.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8, torch.bool):
        return False
    uniq = torch.unique(x.detach().cpu())
    return all(int(v) in (0, 1) for v in uniq.tolist()[:10])


def _call_get_loaders(dataset: str, tok, nsamples: int, seed: int, seqlen: int):
    if GET_LOADERS is None:
        raise RuntimeError("get_loaders import failed. Use --text_file.")
    return GET_LOADERS(dataset, nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tok)


def _pick_split(obj: Any, split: str) -> Any:
    if isinstance(obj, dict):
        for key in (split, "test", "validation", "val", "train"):
            if key in obj:
                return obj[key]
        return next(iter(obj.values()))
    if isinstance(obj, tuple) and len(obj) >= 2:
        return obj[1] if split in ("test", "validation", "val") else obj[0]
    return obj


def _tokenize_text_corpus(tok: AutoTokenizer, text_file: Path) -> Optional[torch.Tensor]:
    text = text_file.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return None
    return tok(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0].long()


def _iter_textfile_batches(
    tok: AutoTokenizer,
    text_file: Path,
    seqlen: int,
    batch_size: int,
    device: str,
    max_batches: Optional[int],
) -> Iterator[Dict[str, torch.Tensor]]:
    token_ids = _tokenize_text_corpus(tok, text_file)
    if token_ids is None:
        return

    made = 0
    batch_buf: List[torch.Tensor] = []

    for start in range(0, token_ids.numel() - seqlen + 1, seqlen):
        batch_buf.append(token_ids[start : start + seqlen].clone().long())
        if len(batch_buf) == batch_size:
            input_ids = torch.stack(batch_buf, dim=0).to(device)
            yield {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids, dtype=torch.long)}
            batch_buf = []
            made += 1
            if max_batches is not None and made >= max_batches:
                return


def _extract_corpus_ids(raw_loader: Any) -> Optional[torch.Tensor]:
    if hasattr(raw_loader, "input_ids") and torch.is_tensor(raw_loader.input_ids):
        raw_loader = raw_loader.input_ids

    if torch.is_tensor(raw_loader):
        ids = raw_loader[0] if raw_loader.dim() == 2 else raw_loader
        return ids.long()

    return None


def _strip_leading_bos(input_ids: torch.Tensor, bos_token_id: Optional[int]) -> torch.Tensor:
    ids = _ensure_2d(input_ids).cpu()[0]
    if bos_token_id is None or ids.numel() == 0:
        return ids
    if int(ids[0]) == int(bos_token_id):
        return ids[1:]
    return ids


def _normalize_loader_to_batches(
    raw_loader: Any,
    seqlen: int,
    batch_size: int,
    device: str,
    max_batches: Optional[int],
) -> Iterator[Dict[str, torch.Tensor]]:
    if hasattr(raw_loader, "input_ids") and torch.is_tensor(raw_loader.input_ids):
        raw_loader = raw_loader.input_ids

    if torch.is_tensor(raw_loader):
        ids = raw_loader[0] if raw_loader.dim() == 2 else raw_loader

        def _from_tensor():
            made = 0
            buf: List[torch.Tensor] = []
            for start in range(0, ids.numel() - seqlen + 1, seqlen):
                buf.append(ids[start : start + seqlen].clone().long())
                if len(buf) == batch_size:
                    input_ids = torch.stack(buf, dim=0).to(device)
                    yield {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids, dtype=torch.long)}
                    buf = []
                    made += 1
                    if max_batches is not None and made >= max_batches:
                        return

        return _from_tensor()

    iterator = iter(raw_loader)

    def _from_iter():
        made = 0
        for batch in iterator:
            if isinstance(batch, dict) and "input_ids" in batch:
                input_ids = _ensure_2d(batch["input_ids"])[:, :seqlen].to(device)
                attn = batch.get("attention_mask")
                labels = batch.get("labels")
                out = {"input_ids": input_ids}
                out["attention_mask"] = (
                    _fit_to_len(attn, input_ids.size(1), 1).to(device)
                    if torch.is_tensor(attn)
                    else torch.ones_like(input_ids, dtype=torch.long)
                )
                if torch.is_tensor(labels):
                    out["labels"] = _fit_to_len(labels, input_ids.size(1), -100).to(device)
                yield out
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]):
                input_ids = _ensure_2d(batch[0])[:, :seqlen].to(device)
                second = _fit_to_len(batch[1], input_ids.size(1), 0 if _is_attn_mask_like(batch[1]) else -100).to(device)
                if second.shape == input_ids.shape and _is_attn_mask_like(second):
                    yield {"input_ids": input_ids, "attention_mask": second}
                else:
                    yield {
                        "input_ids": input_ids,
                        "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
                        "labels": second,
                    }
            elif torch.is_tensor(batch):
                input_ids = _ensure_2d(batch)[:, :seqlen].to(device)
                yield {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids, dtype=torch.long)}
            else:
                continue

            made += 1
            if max_batches is not None and made >= max_batches:
                return

    return _from_iter()


@torch.no_grad()
def eval_ppl_stride(
    model,
    input_ids: torch.Tensor,
    seqlen: int,
    stride: int,
    device: str,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    input_ids = _ensure_2d(input_ids).cpu()
    seq_len = input_ids.size(1)
    if seq_len < 2:
        return {"mean_nll": float("nan"), "ppl": float("nan"), "tokens": 0}

    stride = max(1, min(int(stride), int(seqlen)))
    sum_nll = 0.0
    sum_tok = 0
    prev_end = 0
    windows = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + seqlen, seq_len)
        chunk = input_ids[:, begin:end].to(device)
        if chunk.size(1) < 2:
            break

        logits = model(
            input_ids=chunk,
            attention_mask=torch.ones_like(chunk, dtype=torch.long),
            use_cache=False,
        ).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        target_start = max(prev_end, begin + 1)
        if target_start >= end:
            prev_end = end
            if end == seq_len:
                break
            continue

        active_from = target_start - (begin + 1)
        shift_mask = torch.zeros_like(shift_labels, dtype=torch.float32)
        shift_mask[:, active_from:] = 1.0

        vocab_size = shift_logits.size(-1)
        loss_tok = F.cross_entropy(
            shift_logits.float().view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels).float()

        sum_nll += float((loss_tok * shift_mask).sum().item())
        sum_tok += int(shift_mask.sum().item())

        prev_end = end
        windows += 1
        if max_batches is not None and windows >= max_batches:
            break
        if end == seq_len:
            break

    if sum_tok == 0:
        return {"mean_nll": float("nan"), "ppl": float("nan"), "tokens": 0}

    mean_nll = sum_nll / sum_tok
    return {"mean_nll": mean_nll, "ppl": math.exp(mean_nll), "tokens": sum_tok}


@torch.no_grad()
def eval_ppl_bos_blocks(
    model,
    input_ids: torch.Tensor,
    seqlen: int,
    bos_token_id: Optional[int],
    device: str,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    if bos_token_id is None:
        raise ValueError("bos_block mode requires tokenizer.bos_token_id")

    ids = _strip_leading_bos(input_ids, bos_token_id)
    if ids.numel() == 0 or seqlen < 2:
        return {"mean_nll": float("nan"), "ppl": float("nan"), "tokens": 0}

    step = max(1, int(seqlen) - 1)
    bos = torch.tensor([[int(bos_token_id)]], dtype=torch.long)
    sum_nll = 0.0
    sum_tok = 0
    blocks = 0

    for start in range(0, ids.numel(), step):
        content = ids[start : start + step]
        if content.numel() == 0:
            break

        chunk = torch.cat([bos, content.unsqueeze(0)], dim=1).to(device)
        logits = model(
            input_ids=chunk,
            attention_mask=torch.ones_like(chunk, dtype=torch.long),
            use_cache=False,
        ).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        vocab_size = shift_logits.size(-1)

        loss_tok = F.cross_entropy(
            shift_logits.float().view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels).float()

        sum_nll += float(loss_tok.sum().item())
        sum_tok += int(shift_labels.numel())
        blocks += 1
        if max_batches is not None and blocks >= max_batches:
            break

    if sum_tok == 0:
        return {"mean_nll": float("nan"), "ppl": float("nan"), "tokens": 0}

    mean_nll = sum_nll / sum_tok
    return {"mean_nll": mean_nll, "ppl": math.exp(mean_nll), "tokens": sum_tok}


@torch.no_grad()
def eval_ppl(model, loader: Iterator[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    sum_nll = 0.0
    sum_tok = 0

    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids, dtype=torch.long))
        labels = batch.get("labels")

        if labels is not None:
            labels = labels.clone()
            labels[attention_mask == 0] = -100
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            vocab_size = logits.size(-1)
            loss_sum = F.cross_entropy(
                logits.float().view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            sum_nll += float(loss_sum.item())
            sum_tok += int((labels != -100).sum().item())
            continue

        if input_ids.size(1) < 2:
            continue

        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()
        vocab_size = shift_logits.size(-1)

        loss_tok = F.cross_entropy(
            shift_logits.float().view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels).float()

        sum_nll += float((loss_tok * shift_mask).sum().item())
        sum_tok += int(shift_mask.sum().item())

    if sum_tok == 0:
        return {"mean_nll": float("nan"), "ppl": float("nan"), "tokens": 0}

    mean_nll = sum_nll / sum_tok
    return {"mean_nll": mean_nll, "ppl": math.exp(mean_nll), "tokens": sum_tok}


def _apply_lora_no_merge(model, adapter_path: str):
    if PeftModel is None:
        raise RuntimeError("peft import failed. Install peft first.")
    try:
        model_lora = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    except TypeError:
        model_lora = PeftModel.from_pretrained(model, adapter_path)
    model_lora.eval()
    return model_lora


def _parse_lora_list(spec: Optional[str]) -> List[str]:
    if not spec:
        return []
    return [path.strip() for path in spec.split(",") if path.strip()]


def _normalize_stage_name(stage: str) -> str:
    stage = stage.strip().upper()
    aliases = {
        "ABC": "FULL",
    }
    return aliases.get(stage, stage)


def _is_bundle_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "config.json")):
        return False
    for name in os.listdir(path):
        if _LAYER_RE.match(name):
            return True
    return False


def _print_result_box(label: str, metrics: Dict[str, Any], mode: Optional[str] = None):
    print("\n  +-----------------------------------------+")
    print(f"  | {label:<40s}|")
    print("  +-----------------------------------------+")
    print(f"  | PPL      = {metrics['ppl']:<28.4f} |")
    print(f"  | Mean NLL = {metrics['mean_nll']:<28.6f} |")
    print(f"  | Tokens   = {metrics['tokens']:<28d} |")
    if mode is not None:
        print(f"  | Mode     = {mode:<28s} |")
    print("  +-----------------------------------------+")


def _choose_eval_mode(args, tok, has_corpus_ids: bool) -> str:
    bos_block_ok = has_corpus_ids and tok.bos_token_id is not None
    if args.ppl_mode == "auto":
        return "bos_block" if bos_block_ok else "stride" if has_corpus_ids else "block"
    if args.ppl_mode == "bos_block" and not bos_block_ok:
        print("[WARN] bos_block mode needs raw corpus ids and tokenizer.bos_token_id; falling back to block mode.")
        return "block"
    if args.ppl_mode == "stride" and not has_corpus_ids:
        print("[WARN] stride mode needs raw corpus ids; falling back to block mode.")
        return "block"
    return args.ppl_mode


def _make_raw_loader(args, tok):
    if args.text_file:
        return _iter_textfile_batches(
            tok=tok,
            text_file=Path(args.text_file),
            seqlen=args.seqlen,
            batch_size=args.batch_size,
            device=args.device,
            max_batches=args.max_batches,
        )
    raw = _call_get_loaders(args.dataset, tok, args.nsamples, args.seed, args.seqlen)
    return _pick_split(raw, args.split)


def _evaluate_model(model, tok, args, corpus_ids: Optional[torch.Tensor]):
    raw_loader = None
    stride_ids = corpus_ids
    loader_mode = _choose_eval_mode(args, tok, corpus_ids is not None)

    if loader_mode == "bos_block":
        metrics = eval_ppl_bos_blocks(
            model=model,
            input_ids=stride_ids,
            seqlen=args.seqlen,
            bos_token_id=tok.bos_token_id,
            device=args.device,
            max_batches=args.max_batches,
        )
        return metrics, loader_mode, stride_ids

    if loader_mode == "stride":
        metrics = eval_ppl_stride(
            model=model,
            input_ids=stride_ids,
            seqlen=args.seqlen,
            stride=args.stride,
            device=args.device,
            max_batches=args.max_batches,
        )
        return metrics, loader_mode, stride_ids

    raw_loader = _make_raw_loader(args, tok)
    if corpus_ids is None:
        corpus_ids_local = _extract_corpus_ids(raw_loader)
        local_mode = _choose_eval_mode(args, tok, corpus_ids_local is not None)
        if local_mode == "stride" and corpus_ids_local is not None:
            stride_ids = corpus_ids_local
            metrics = eval_ppl_stride(
                model=model,
                input_ids=stride_ids,
                seqlen=args.seqlen,
                stride=args.stride,
                device=args.device,
                max_batches=args.max_batches,
            )
            return metrics, "stride", stride_ids
        if local_mode == "bos_block" and corpus_ids_local is not None:
            stride_ids = corpus_ids_local
            metrics = eval_ppl_bos_blocks(
                model=model,
                input_ids=stride_ids,
                seqlen=args.seqlen,
                bos_token_id=tok.bos_token_id,
                device=args.device,
                max_batches=args.max_batches,
            )
            return metrics, "bos_block", stride_ids

    batches = list(
        _normalize_loader_to_batches(
            raw_loader=raw_loader,
            seqlen=args.seqlen,
            batch_size=args.batch_size,
            device=args.device,
            max_batches=args.max_batches,
        )
    )
    return eval_ppl(model, iter(batches)), "block", stride_ids


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate PPL for merged Gemma models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # A-only
  python -m gemma_prune_lora.eval_ppl_mergedmodel \
    --model_path ./merged_models_gemma_7b/A_merged

  # A/AB/FULL compare
  python -m gemma_prune_lora.eval_ppl_mergedmodel \
    --model_path ./merged_models_gemma_7b/A_merged \
    --b_bundle ./merged_models_gemma_7b/B_merged \
    --c_bundle ./merged_models_gemma_7b/C_merged \
    --stages A,AB,FULL
        """,
    )
    ap.add_argument(
        "--model_path",
        type=str,
        nargs="+",
        required=True,
        help="Full model path(s) (A_merged etc). Do not pass bundle dirs here.",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    ap.add_argument("--dataset", default="wikitext2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--ppl_mode", default="auto", choices=["auto", "bos_block", "stride", "block"])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--text_file", default=None)

    ap.add_argument("--b_bundle", default=None, help="Merged B bundle directory")
    ap.add_argument("--c_bundle", default=None, help="Merged C bundle directory")
    ap.add_argument("--bundles_dir", default=None, help="Legacy bundles root with B/C subdirs")

    ap.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stages: A,AB,FULL. If omitted: A,AB,FULL when bundles exist, else A",
    )

    ap.add_argument("--lora_A", default=None, help="Optional comma-separated no-merge LoRA adapters for stage A")
    ap.add_argument("--lora_AB", default=None, help="Optional comma-separated no-merge LoRA adapters for stage AB")
    ap.add_argument("--lora_FULL", default=None, help="Optional comma-separated no-merge LoRA adapters for stage FULL")
    ap.add_argument("--tokenizer_path", default=None)
    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dtype == "fp16":
        print("[WARN] Gemma perplexity can become unstable in fp16; bf16 is recommended.")

    b_dir = Path(args.b_bundle) if args.b_bundle else None
    c_dir = Path(args.c_bundle) if args.c_bundle else None
    bundles_dir = Path(args.bundles_dir) if args.bundles_dir else None
    has_bundles = (b_dir is not None) or (c_dir is not None) or (bundles_dir is not None)

    if args.stages is not None:
        stage_list = [_normalize_stage_name(s) for s in args.stages.split(",") if s.strip()]
    elif has_bundles:
        stage_list = ["A"]
        if bundles_dir is not None or b_dir is not None:
            stage_list.append("AB")
        if bundles_dir is not None or (b_dir is not None and c_dir is not None):
            stage_list.append("FULL")
        print(f"  (--stages omitted, bundles detected -> {','.join(stage_list)})")
    else:
        stage_list = ["A"]

    invalid_stages = [stage for stage in stage_list if stage not in ("A", "AB", "FULL")]
    if invalid_stages:
        raise ValueError(f"Invalid stages: {invalid_stages}. Use A, AB, FULL (ABC alias supported).")

    if not has_bundles and any(s in ("AB", "FULL") for s in stage_list):
        print("[warn] AB/FULL require bundles. Falling back to A only.")
        stage_list = ["A"]

    full_model_paths = []
    auto_b = None
    auto_c = None

    for mpath in args.model_path:
        if _is_bundle_dir(mpath):
            name_lower = os.path.basename(mpath).lower()
            path_lower = mpath.lower().replace("\\", "/")
            if "b_merged" in name_lower or "/b/" in path_lower or path_lower.endswith("/b"):
                if b_dir is None:
                    auto_b = mpath
                    print(f"  [auto] bundle B: {mpath}")
                else:
                    print(f"  [warn] {mpath} is bundle dir but --b_bundle already set. skip")
            elif "c_merged" in name_lower or "/c/" in path_lower or path_lower.endswith("/c"):
                if c_dir is None:
                    auto_c = mpath
                    print(f"  [auto] bundle C: {mpath}")
                else:
                    print(f"  [warn] {mpath} is bundle dir but --c_bundle already set. skip")
            else:
                print(f"  [warn] {mpath} is bundle dir but cannot infer B/C. skip")
        else:
            full_model_paths.append(mpath)

    if auto_b and b_dir is None:
        b_dir = Path(auto_b)
    if auto_c and c_dir is None:
        c_dir = Path(auto_c)

    has_bundles = (b_dir is not None) or (c_dir is not None) or (bundles_dir is not None)
    if has_bundles:
        if b_dir is None and bundles_dir is None and "AB" in stage_list:
            print("[warn] Stage AB requires B bundle. Removing AB from stage list.")
            stage_list = [stage for stage in stage_list if stage != "AB"]
        if (bundles_dir is None and (b_dir is None or c_dir is None)) and "FULL" in stage_list:
            print("[warn] Stage FULL requires both B and C bundles. Removing FULL from stage list.")
            stage_list = [stage for stage in stage_list if stage != "FULL"]

    if not full_model_paths:
        raise RuntimeError(
            "At least one full model path (with config.json) is required. "
            "Use --b_bundle/--c_bundle for bundle dirs."
        )

    results = []

    print("\n" + "=" * 70)
    print("Merged Model PPL Evaluation (Gemma)")
    print("=" * 70)
    print(f"Models:      {full_model_paths}")
    if b_dir:
        print(f"B bundle:    {b_dir}")
    if c_dir:
        print(f"C bundle:    {c_dir}")
    if bundles_dir:
        print(f"Bundles dir: {bundles_dir}")
    print(f"Stages:      {stage_list}")
    print(f"Dataset:     {args.text_file or args.dataset} (split={args.split})")
    print(f"Seq length:  {args.seqlen}")
    print(f"PPL mode:    {args.ppl_mode}")
    if args.ppl_mode in ("auto", "stride"):
        print(f"Stride:      {min(args.stride, args.seqlen)}")
    print(f"Device:      {args.device} ({args.dtype})")
    print("=" * 70)

    for model_idx, mpath in enumerate(full_model_paths):
        print(f"\n{'-' * 60}")
        print(f"[{model_idx + 1}/{len(full_model_paths)}] {mpath}")
        print(f"{'-' * 60}")

        tok_path = args.tokenizer_path or mpath
        fallbacks = None if args.tokenizer_path else _find_tokenizer_fallbacks(mpath)
        print(f"\n  Loading tokenizer from: {tok_path}")
        tok = _load_tokenizer(tok_path, fallback_paths=fallbacks)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        corpus_ids = _tokenize_text_corpus(tok, Path(args.text_file)) if args.text_file else None

        print("\n  Loading model...")
        model = _load_model(mpath, dtype=dtype, device=args.device)

        mgr = None
        if has_bundles:
            print("  Initializing DynamicStageManager...")
            passlayer_rt = _detect_layer_return_tuple(model)
            mgr = DynamicStageManager(
                model=model,
                base_model_dir=Path(mpath),
                device=args.device,
                dtype=dtype,
                passlayer_return_tuple=passlayer_rt,
                b_dir=b_dir,
                c_dir=c_dir,
                bundles_dir=bundles_dir,
            )
            if not mgr.removed:
                b_manifest, c_manifest = _read_bc_indices(mpath, num_layers=mgr.num_layers)
                print(
                    "  [warn] No bundle layers detected. "
                    f"manifest B={b_manifest} C={c_manifest}"
                )
            print(f"  Stage meta: B={mgr.b_idx}, C={mgr.c_idx}")

        for stage in stage_list:
            stage_name = {"A": "A", "AB": "AB", "FULL": "ABC"}.get(stage, stage)
            label = f"{Path(mpath).name}[{stage_name}]"

            if mgr is not None:
                print(f"\n  [{stage}] Switching stage...")
                mgr.set_stage(stage)
                n_active = mgr.num_layers - sum(
                    1 for i in mgr.removed if isinstance(mgr.layers[i], GemmaPassLayer)
                )
                print(f"  [ok] Stage {stage}: {n_active} active layers")

            print(f"  [{stage}] Evaluating PPL...")
            metrics, loader_mode, stride_ids = _evaluate_model(model, tok, args, corpus_ids)
            _print_result_box(label, metrics, mode=loader_mode)
            results.append(
                {
                    "model": mpath,
                    "stage": stage,
                    "label": label,
                    "mode": loader_mode,
                    **metrics,
                }
            )

            lora_spec = {"A": args.lora_A, "AB": args.lora_AB, "FULL": args.lora_FULL}.get(stage)
            for adapter_path in _parse_lora_list(lora_spec):
                print(f"\n  Applying LoRA (no-merge): {adapter_path}")
                model_lora = _apply_lora_no_merge(model, adapter_path)

                if loader_mode == "bos_block":
                    metrics_lora = eval_ppl_bos_blocks(
                        model=model_lora,
                        input_ids=stride_ids,
                        seqlen=args.seqlen,
                        bos_token_id=tok.bos_token_id,
                        device=args.device,
                        max_batches=args.max_batches,
                    )
                elif loader_mode == "stride":
                    metrics_lora = eval_ppl_stride(
                        model=model_lora,
                        input_ids=stride_ids,
                        seqlen=args.seqlen,
                        stride=args.stride,
                        device=args.device,
                        max_batches=args.max_batches,
                    )
                else:
                    raw_loader = _make_raw_loader(args, tok)
                    batches = list(
                        _normalize_loader_to_batches(
                            raw_loader=raw_loader,
                            seqlen=args.seqlen,
                            batch_size=args.batch_size,
                            device=args.device,
                            max_batches=args.max_batches,
                        )
                    )
                    metrics_lora = eval_ppl(model_lora, iter(batches))

                lp_name = Path(adapter_path).name
                lora_label = f"{label}+LoRA({lp_name})"
                _print_result_box(lora_label, metrics_lora, mode=loader_mode)
                results.append(
                    {
                        "model": mpath,
                        "stage": f"{stage}+LoRA",
                        "label": lora_label,
                        "mode": loader_mode,
                        **metrics_lora,
                    }
                )
                del model_lora

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if len(results) > 1:
        print(f"\n{'=' * 86}")
        print("Summary")
        print(f"{'=' * 86}")
        print(f"{'Label':<42s} {'Mode':<10s} {'PPL':>12s} {'Mean NLL':>12s} {'Tokens':>10s}")
        print(f"{'-' * 42} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 10}")
        for row in results:
            print(
                f"{row['label']:<42s} {row['mode']:<10s} "
                f"{row['ppl']:>12.4f} {row['mean_nll']:>12.6f} {row['tokens']:>10d}"
            )
        print(f"{'=' * 86}")

    print("\nDone.")


if __name__ == "__main__":
    main()
