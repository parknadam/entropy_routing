#!/usr/bin/env python3
"""
머지된 모델(pruned + LoRA merged)의 PPL을 평가하는 스크립트.

평가 모드:
  1) A-only:     A_merged 모델을 그대로 평가 (B/C는 PassLayer)
  2) A/AB/FULL:  --b_bundle, --c_bundle로 머지된 번들을 주입하여 stage별 비교

사용법:
# A-only 평가
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path ./merged_models_mistral_7b/A_merged \
  --device cuda:0

# A / AB / FULL stage 비교 (머지된 B/C 번들 사용)
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path ./merged_models_mistral_7b/A_merged \
  --b_bundle ./merged_models_mistral_7b/B_merged \
  --c_bundle ./merged_models_mistral_7b/C_merged \
  --stages A,AB,FULL \
  --device cuda:0

# 원본 모델과 비교
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path mistralai/Mistral-7B-v0.1 ./merged_models_mistral_7b/A_merged \
  --tokenizer_path mistralai/Mistral-7B-v0.1 \
  --device cuda:0

# 원본 bundles_dir 구조로도 사용 가능 (구버전 호환)
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path ./merged_models_mistral_7b/A_merged \
  --bundles_dir ./25_mistral_results/pruning/bundles \
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

MistralDecoderLayer = None
try:
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer as _MDL
    MistralDecoderLayer = _MDL
except ImportError:
    pass

GET_LOADERS = None
for _mod_name in ["mistral_prune_lora.pruning.data", "pruning.data"]:
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


# ============================================================
# PassLayer
# ============================================================
class MistralPassLayer(nn.Module):
    def __init__(self, return_tuple: bool = True):
        super().__init__()
        self.return_tuple = return_tuple

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, **kwargs):
        if not self.return_tuple:
            return hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs


# ============================================================
# manifest에서 dropped layers 읽기 + PassLayer 설치
# ============================================================
def _read_dropped_layers(model_path: str) -> List[int]:
    """manifest.json에서 프루닝으로 제거된 레이어 인덱스를 읽습니다."""
    manifest_path = os.path.join(model_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        return []
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except Exception:
        return []

    stages = manifest.get("stages", {})
    dropped = stages.get("A", {}).get("dropped_layers", [])
    if not dropped:
        B = stages.get("B", {}).get("removed_layers", [])
        C = stages.get("C", {}).get("removed_layers", [])
        dropped = sorted(set(B + C))
    if not dropped:
        dropped = manifest.get("simdrop", {}).get("removed_layers", [])
    return sorted(set(int(i) for i in dropped))


def _read_bc_indices(model_path: str) -> Tuple[List[int], List[int]]:
    """manifest.json에서 B/C 레이어 인덱스를 따로 읽습니다."""
    manifest_path = os.path.join(model_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        return [], []
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except Exception:
        return [], []
    stages = manifest.get("stages", {})
    B = sorted(set(int(i) for i in stages.get("B", {}).get("removed_layers", [])))
    C = sorted(set(int(i) for i in stages.get("C", {}).get("removed_layers", [])))
    return B, C


def _install_passlayers(model, dropped_indices: List[int], return_tuple: bool = True):
    """from_pretrained() 후 dropped 레이어 위치에 PassLayer를 설치합니다."""
    if not dropped_indices:
        return model
    layers = _get_mistral_layers(model)
    for idx in dropped_indices:
        if 0 <= idx < len(layers):
            old = layers[idx]
            dev = next(old.parameters()).device if sum(1 for _ in old.parameters()) > 0 else torch.device("cpu")
            layers[idx] = MistralPassLayer(return_tuple=return_tuple).to(dev)
            del old
    print(f"  ✓ PassLayer 설치: {dropped_indices} ({len(dropped_indices)}개 레이어)")
    return model


# ============================================================
# 모델 & 토크나이저 로드
# ============================================================
def _get_mistral_layers(model) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Cannot find model.model.layers")


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


def _load_model(model_path: str, dtype: torch.dtype, device: str):
    print(f"  Loading model from: {model_path}")
    resolved = os.path.abspath(model_path) if os.path.exists(model_path) else model_path

    try:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, torch_dtype=dtype, low_cpu_mem_usage=True,
            attn_implementation="eager", trust_remote_code=True,
        )
    except TypeError:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                resolved, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                resolved, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
            )

    # ★ manifest.json이 있으면 dropped 레이어에 PassLayer 설치
    dropped = _read_dropped_layers(model_path)
    if dropped:
        return_tuple = _detect_layer_return_tuple(model)
        model = _install_passlayers(model, dropped, return_tuple=return_tuple)

    model = model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_params = sum(p.numel() for p in model.parameters())
    n_active = n_layers - len(dropped)
    if dropped:
        print(f"  ✓ {model.config.model_type} | {n_layers} layers ({n_active} active + {len(dropped)} PassLayer) | {n_params/1e6:.1f}M params | {device}")
    else:
        print(f"  ✓ {model.config.model_type} | {n_layers} layers | {n_params/1e6:.1f}M params | {device}")
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
            tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
            if path != model_path:
                print(f"  [INFO] Tokenizer fallback: {path}")
            return tok
        except Exception as e:
            errors.append((path, str(e)))
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
    raise RuntimeError(f"Tokenizer 로드 실패:\n{em}\n--tokenizer_path 로 명시해 주세요.")


def _find_tokenizer_fallbacks(model_path: str) -> List[str]:
    fallbacks = []
    orig_cfg = os.path.join(model_path, "original_config")
    if os.path.isdir(orig_cfg):
        fallbacks.append(orig_cfg)
    manifest_path = os.path.join(model_path, "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            base_model = manifest.get("base_model")
            if base_model:
                fallbacks.append(base_model)
        except Exception:
            pass
    return fallbacks


# ============================================================
# DynamicStageManager: B/C 번들 → stage별 레이어 교체
# ============================================================
_LAYER_RE = re.compile(r"layer_(\d+)\.safetensors$")


def _build_layer_map(dir_path: Path) -> Dict[int, Path]:
    m: Dict[int, Path] = {}
    if not dir_path.exists():
        return m
    for p in dir_path.glob("layer_*.safetensors"):
        mm = _LAYER_RE.search(p.name)
        if mm:
            m[int(mm.group(1))] = p
    return m


def _strip_layer_prefix(sd: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, torch.Tensor]:
    out = {}
    needles = [f"model.layers.{layer_idx}.", f"model.model.layers.{layer_idx}.", f"layers.{layer_idx}."]
    for k, v in sd.items():
        nk = k
        for nd in needles:
            if nd in nk:
                nk = nk.split(nd, 1)[1]
                break
        out[nk] = v
    return out


def _maybe_shift_indices(B_map, C_map, num_layers):
    all_idx = sorted(set(B_map.keys()) | set(C_map.keys()))
    if not all_idx:
        return B_map, C_map, 0
    if all(0 <= i < num_layers for i in all_idx):
        return B_map, C_map, 0
    if all(1 <= i <= num_layers for i in all_idx):
        return {i-1: p for i, p in B_map.items()}, {i-1: p for i, p in C_map.items()}, -1
    raise ValueError(f"Bundle layer index mismatch (num_layers={num_layers})")


class DynamicStageManager:
    """
    A:    removed 모두 PassLayer
    AB:   B 복구, C만 PassLayer
    FULL: B+C 모두 복구

    B/C 번들 경로를 개별적으로 받습니다.
    """

    def __init__(self, model, device: str, dtype: torch.dtype,
                 passlayer_return_tuple: bool,
                 b_dir: Optional[Path] = None,
                 c_dir: Optional[Path] = None,
                 bundles_dir: Optional[Path] = None):
        """
        b_dir / c_dir: 머지된 B/C 번들 경로 (--b_bundle, --c_bundle)
        bundles_dir: 구버전 호환 (bundles_dir/B, bundles_dir/C)
        둘 다 지정 시 b_dir/c_dir 우선
        """
        if MistralDecoderLayer is None:
            raise RuntimeError("MistralDecoderLayer import 실패")
        if load_file is None:
            raise RuntimeError("safetensors import 실패 (pip install safetensors)")

        self.model = model
        self.layers = _get_mistral_layers(model)
        self.device = device
        self.dtype = dtype
        self.passlayer_return_tuple = passlayer_return_tuple
        self.num_layers = len(self.layers)

        # B/C 디렉터리 결정
        actual_b_dir = b_dir if b_dir else (bundles_dir / "B" if bundles_dir else None)
        actual_c_dir = c_dir if c_dir else (bundles_dir / "C" if bundles_dir else None)

        B_raw = _build_layer_map(actual_b_dir) if actual_b_dir else {}
        C_raw = _build_layer_map(actual_c_dir) if actual_c_dir else {}
        self.B_map, self.C_map, _ = _maybe_shift_indices(B_raw, C_raw, self.num_layers)
        self.B_idx = sorted(self.B_map.keys())
        self.C_idx = sorted(self.C_map.keys())
        self.removed = sorted(set(self.B_idx) | set(self.C_idx))

    def stage_meta(self):
        return {"num_layers": self.num_layers, "B": self.B_idx, "C": self.C_idx, "removed": self.removed}

    def _bundle_path(self, layer_i):
        return self.B_map.get(layer_i) or self.C_map.get(layer_i)

    def _restore_one_layer(self, layer_i):
        p = self._bundle_path(layer_i)
        if p is None:
            raise FileNotFoundError(f"layer_{layer_i}.safetensors not found in B/C.")
        try:
            new_layer = MistralDecoderLayer(self.model.config, layer_i)
        except TypeError:
            new_layer = MistralDecoderLayer(self.model.config)
        new_layer = new_layer.to(self.device, dtype=self.dtype)
        sd = load_file(str(p), device="cpu")
        sd = _strip_layer_prefix(sd, layer_i)
        new_layer.load_state_dict(sd, strict=False)
        old = self.layers[layer_i]
        self.layers[layer_i] = new_layer
        del old

    def _pass_one_layer(self, layer_i):
        old = self.layers[layer_i]
        self.layers[layer_i] = MistralPassLayer(return_tuple=self.passlayer_return_tuple).to(self.device)
        del old

    def set_stage(self, stage: str):
        stage = stage.upper()
        if stage not in ("A", "AB", "FULL"):
            raise ValueError("stage must be A / AB / FULL")
        pass_set = set(self.removed) if stage == "A" else (set(self.C_idx) if stage == "AB" else set())

        for i in self.removed:
            is_pass = isinstance(self.layers[i], MistralPassLayer)
            if i in pass_set:
                if not is_pass:
                    self._pass_one_layer(i)
            else:
                if is_pass:
                    self._restore_one_layer(i)

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


# ============================================================
# 데이터 로더
# ============================================================
def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x


def _call_get_loaders(dataset: str, tok, nsamples: int, seed: int, seqlen: int):
    if GET_LOADERS is None:
        raise RuntimeError("get_loaders import 실패. --text_file 사용 필요.")
    return GET_LOADERS(dataset, nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tok)


def _pick_split(obj: Any, split: str) -> Any:
    if isinstance(obj, dict):
        for k in (split, "test", "validation", "val", "train"):
            if k in obj:
                return obj[k]
        return next(iter(obj.values()))
    if isinstance(obj, tuple) and len(obj) >= 2:
        return obj[1] if split in ("test", "validation", "val") else obj[0]
    return obj


def _tensor_to_batches(ids: torch.Tensor, seqlen: int, batch_size: int,
                       device: str, max_batches: Optional[int]) -> List[Dict[str, torch.Tensor]]:
    if ids.dim() == 2:
        ids = ids[0]
    ids = ids.long()
    batches = []
    cur = []
    for start in range(0, ids.numel() - seqlen + 1, seqlen):
        cur.append(ids[start: start + seqlen])
        if len(cur) == batch_size:
            x = torch.stack(cur, dim=0).to(device)
            batches.append({"input_ids": x, "attention_mask": torch.ones_like(x, dtype=torch.long)})
            cur = []
            if max_batches is not None and len(batches) >= max_batches:
                return batches
    return batches


def _extract_batch(batch: Any, device: str, seqlen: int) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(batch, dict):
        ids = batch.get("input_ids")
        if ids is None or not torch.is_tensor(ids):
            return None
        ids = _ensure_2d(ids)[:, :seqlen].to(device)
        attn = batch.get("attention_mask", torch.ones_like(ids, dtype=torch.long))
        attn = _ensure_2d(attn)[:, :seqlen].to(device)
        out = {"input_ids": ids, "attention_mask": attn}
        labels = batch.get("labels")
        if labels is not None and torch.is_tensor(labels):
            out["labels"] = _ensure_2d(labels)[:, :seqlen].to(device)
        return out
    if isinstance(batch, (tuple, list)) and len(batch) >= 1 and torch.is_tensor(batch[0]):
        ids = _ensure_2d(batch[0])[:, :seqlen].to(device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.long)}
    if torch.is_tensor(batch):
        ids = _ensure_2d(batch)[:, :seqlen].to(device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.long)}
    return None


def _pack_text_lines_to_batches(lines, tok, seqlen, batch_size, device, max_batches):
    buf = []
    batches = []
    cur = []
    for s in lines:
        if not isinstance(s, str) or not s.strip():
            continue
        ids = tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        buf.extend(ids)
        while len(buf) >= seqlen:
            cur.append(torch.tensor(buf[:seqlen], dtype=torch.long))
            buf = buf[seqlen:]
            if len(cur) == batch_size:
                x = torch.stack(cur, dim=0).to(device)
                batches.append({"input_ids": x, "attention_mask": torch.ones_like(x, dtype=torch.long)})
                cur = []
                if max_batches is not None and len(batches) >= max_batches:
                    return batches
    return batches


def _extract_input_ids_tensor(obj) -> Optional[torch.Tensor]:
    """BatchEncoding / TokenizerWrapper 등에서 input_ids 텐서 추출."""
    if isinstance(obj, dict) and "input_ids" in obj:
        val = obj["input_ids"]
        if torch.is_tensor(val):
            return val
    if hasattr(obj, "input_ids"):
        val = getattr(obj, "input_ids", None)
        if torch.is_tensor(val):
            return val
    return None


def prepare_batches(raw_loader, tok, seqlen, batch_size, device, max_batches):
    """★ Tokens=0 버그 수정: BatchEncoding/TokenizerWrapper를 올바르게 처리."""

    # Case 0: .input_ids를 가진 객체 (BatchEncoding, TokenizerWrapper)
    ids_tensor = _extract_input_ids_tensor(raw_loader)
    if ids_tensor is not None:
        print(f"  [data] input_ids tensor shape: {ids_tensor.shape}")
        return _tensor_to_batches(ids_tensor, seqlen, batch_size, device, max_batches)

    if torch.is_tensor(raw_loader):
        print(f"  [data] tensor shape: {raw_loader.shape}")
        return _tensor_to_batches(raw_loader, seqlen, batch_size, device, max_batches)

    if isinstance(raw_loader, str):
        return _pack_text_lines_to_batches(iter([raw_loader]), tok, seqlen, batch_size, device, max_batches)

    if isinstance(raw_loader, list):
        if not raw_loader:
            return []
        if isinstance(raw_loader[0], str):
            return _pack_text_lines_to_batches(iter(raw_loader), tok, seqlen, batch_size, device, max_batches)
        batches = []
        for item in raw_loader:
            b = _extract_batch(item, device, seqlen)
            if b is not None:
                batches.append(b)
                if max_batches is not None and len(batches) >= max_batches:
                    break
        print(f"  [data] list of {len(raw_loader)} items → {len(batches)} batches")
        return batches

    batches = []
    try:
        for item in raw_loader:
            b = _extract_batch(item, device, seqlen)
            if b is not None:
                batches.append(b)
                if max_batches is not None and len(batches) >= max_batches:
                    break
    except Exception as e:
        print(f"  [WARN] iteration failed: {e}")

    if not batches:
        print(f"  [ERROR] type={type(raw_loader).__name__}, 배치 추출 실패!")
    return batches


# ============================================================
# PPL 평가
# ============================================================
@torch.no_grad()
def eval_ppl(model, batches: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    sum_nll = 0.0
    sum_tok = 0
    for batch in batches:
        input_ids = batch["input_ids"]
        attn = batch.get("attention_mask", torch.ones_like(input_ids, dtype=torch.long))
        labels = batch.get("labels", None)

        if input_ids.dim() != 2:
            input_ids = _ensure_2d(input_ids)
        if attn.dim() != 2:
            attn = _ensure_2d(attn)

        if labels is not None:
            if labels.dim() != 2:
                labels = _ensure_2d(labels)
            labels = labels.clone()
            labels[attn == 0] = -100
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            V = out.logits.size(-1)
            loss_sum = F.cross_entropy(
                out.logits.float().view(-1, V), labels.view(-1),
                ignore_index=-100, reduction="sum",
            )
            sum_nll += float(loss_sum.item())
            sum_tok += int((labels != -100).sum().item())
            continue

        if input_ids.shape[1] < 2:
            continue
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attn[:, 1:].contiguous().float()
        V = shift_logits.size(-1)
        loss_tok = F.cross_entropy(
            shift_logits.float().view(-1, V), shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels).float()
        sum_nll += float((loss_tok * shift_mask).sum().item())
        sum_tok += int(shift_mask.sum().item())

    if sum_tok == 0:
        return {"mean_nll": float("nan"), "ppl": float("nan"), "tokens": 0}
    mean_nll = sum_nll / sum_tok
    ppl = math.exp(min(mean_nll, 100.0))
    return {"mean_nll": mean_nll, "ppl": ppl, "tokens": sum_tok}


# ============================================================
# LoRA (미머지 비교용)
# ============================================================
def _apply_lora_no_merge(model, adapter_path: str):
    if PeftModel is None:
        raise RuntimeError("peft import 실패. pip install peft")
    model_lora = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model_lora.eval()
    return model_lora


# ============================================================
# 유틸: bundle dir인지 판별
# ============================================================
def _is_bundle_dir(path: str) -> bool:
    """config.json 없고 layer_*.safetensors가 있으면 bundle dir."""
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "config.json")):
        return False
    for f in os.listdir(path):
        if _LAYER_RE.match(f):
            return True
    return False


def _print_result_box(label: str, m: dict):
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │ {label:<40s}│")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │ PPL      = {m['ppl']:<28.4f} │")
    print(f"  │ Mean NLL = {m['mean_nll']:<28.6f} │")
    print(f"  │ Tokens   = {m['tokens']:<28d} │")
    print(f"  └─────────────────────────────────────────┘")


# ============================================================
# 메인
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="머지된 모델의 PPL을 평가합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # A-only 평가
  python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \\
    --model_path ./merged_models/A_merged

  # A/AB/FULL 비교 (머지된 B/C 번들)
  python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \\
    --model_path ./merged_models/A_merged \\
    --b_bundle ./merged_models/B_merged \\
    --c_bundle ./merged_models/C_merged \\
    --stages A,AB,FULL

  # 원본 모델과 비교
  python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \\
    --model_path mistralai/Mistral-7B-v0.1 ./merged_models/A_merged \\
    --tokenizer_path mistralai/Mistral-7B-v0.1

  # 구버전 bundles_dir 호환
  python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \\
    --model_path ./merged_models/A_merged \\
    --bundles_dir ./pruning/bundles \\
    --stages A,AB,FULL
        """,
    )
    ap.add_argument("--model_path", type=str, nargs="+", required=True,
                    help="전체 모델 경로 (A_merged 등). bundle dir은 넣지 마세요.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    # 데이터셋
    ap.add_argument("--dataset", default="wikitext2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--text_file", default=None)

    # ★ 머지된 B/C 번들 (새로운 인터페이스)
    ap.add_argument("--b_bundle", default=None,
                    help="머지된 B 번들 디렉터리 (layer_XXX.safetensors)")
    ap.add_argument("--c_bundle", default=None,
                    help="머지된 C 번들 디렉터리 (layer_XXX.safetensors)")

    # 구버전 호환: bundles_dir (bundles_dir/B, bundles_dir/C 구조)
    ap.add_argument("--bundles_dir", default=None,
                    help="구버전 호환: B/C 서브폴더가 있는 번들 디렉터리")

    # stage 선택
    ap.add_argument("--stages", default=None,
                    help="평가할 stage (comma-separated: A,AB,FULL). "
                         "미지정 시: bundle 있으면 A,AB,FULL / 없으면 A")

    # 기타
    ap.add_argument("--lora_paths", type=str, nargs="*", default=None)
    ap.add_argument("--tokenizer_path", default=None)

    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── B/C 번들 경로 결정 ──
    b_dir = Path(args.b_bundle) if args.b_bundle else None
    c_dir = Path(args.c_bundle) if args.c_bundle else None
    bundles_dir = Path(args.bundles_dir) if args.bundles_dir else None

    has_bundles = (b_dir is not None) or (c_dir is not None) or (bundles_dir is not None)

    # ── stage 결정 ──
    if args.stages is not None:
        stage_list = [s.strip().upper() for s in args.stages.split(",") if s.strip()]
    elif has_bundles:
        stage_list = ["A", "AB", "FULL"]
        print("  (--stages 미지정, bundle 감지 → A,AB,FULL 자동 설정)")
    else:
        stage_list = ["A"]

    # AB/FULL인데 번들이 없으면 경고
    if not has_bundles and any(s in ("AB", "FULL") for s in stage_list):
        print("⚠ WARNING: AB/FULL 평가에는 --b_bundle/--c_bundle 또는 --bundles_dir 필요. A만 평가합니다.")
        stage_list = ["A"]

    # ── model_path에서 bundle dir을 자동 분류 ──
    full_model_paths = []
    auto_b = None
    auto_c = None

    for mpath in args.model_path:
        if _is_bundle_dir(mpath):
            # bundle dir → 이름에서 B/C 추론
            name_lower = os.path.basename(mpath).lower()
            if "b_merged" in name_lower or "/b" in mpath.lower():
                if b_dir is None:
                    auto_b = mpath
                    print(f"  ★ Bundle dir 자동 감지 (B): {mpath}")
                else:
                    print(f"  ⚠ {mpath}는 bundle dir이지만 --b_bundle이 이미 지정됨. 스킵.")
            elif "c_merged" in name_lower or "/c" in mpath.lower():
                if c_dir is None:
                    auto_c = mpath
                    print(f"  ★ Bundle dir 자동 감지 (C): {mpath}")
                else:
                    print(f"  ⚠ {mpath}는 bundle dir이지만 --c_bundle이 이미 지정됨. 스킵.")
            else:
                print(f"  ⚠ {mpath}는 bundle dir이지만 B/C를 판별할 수 없습니다. 스킵.")
        else:
            full_model_paths.append(mpath)

    # 자동 감지된 번들 적용
    if auto_b and b_dir is None:
        b_dir = Path(auto_b)
    if auto_c and c_dir is None:
        c_dir = Path(auto_c)

    # 번들이 감지되었는데 stages가 A뿐이면 자동 확장
    has_bundles = (b_dir is not None) or (c_dir is not None) or (bundles_dir is not None)
    if has_bundles and stage_list == ["A"] and args.stages is None:
        stage_list = ["A"]
        if b_dir is not None:
            stage_list.append("AB")
        if c_dir is not None or bundles_dir is not None:
            stage_list.append("FULL")
        print(f"  (bundle 감지 → stages 자동 확장: {stage_list})")

    if not full_model_paths:
        raise RuntimeError(
            "전체 모델(config.json이 있는)이 최소 1개 필요합니다.\n"
            "B/C 번들은 --b_bundle / --c_bundle로 지정하세요."
        )

    results = []

    print("\n" + "=" * 70)
    print("Merged Model PPL Evaluation")
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
    print(f"Device:      {args.device} ({args.dtype})")
    print("=" * 70)

    for model_idx, mpath in enumerate(full_model_paths):
        print(f"\n{'─' * 60}")
        print(f"[{model_idx + 1}/{len(full_model_paths)}] {mpath}")
        print(f"{'─' * 60}")

        # ── 토크나이저 ──
        tok_path = args.tokenizer_path or mpath
        fallbacks = None if args.tokenizer_path else _find_tokenizer_fallbacks(mpath)
        print(f"\n  Loading tokenizer from: {tok_path}")
        tok = _load_tokenizer(tok_path, fallback_paths=fallbacks)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # ── 데이터 ──
        print(f"\n  Loading evaluation data...")
        if args.text_file is not None:
            batches = _pack_text_lines_to_batches(
                iter(open(args.text_file, "r", encoding="utf-8", errors="ignore")),
                tok, args.seqlen, args.batch_size, args.device, args.max_batches,
            )
        else:
            raw = _call_get_loaders(args.dataset, tok, args.nsamples, args.seed, args.seqlen)
            raw_loader = _pick_split(raw, args.split)
            print(f"  raw_loader type: {type(raw_loader).__name__}")
            batches = prepare_batches(
                raw_loader, tok, args.seqlen,
                args.batch_size, args.device, args.max_batches,
            )
        print(f"  ✓ {len(batches)} batches (seqlen={args.seqlen})")
        if len(batches) == 0:
            print(f"  ✗ ERROR: 배치가 0개!")
            continue

        # ── 모델 로드 (1회, stage 전환으로 재사용) ──
        print(f"\n  Loading model...")
        model = _load_model(mpath, dtype=dtype, device=args.device)

        # DynamicStageManager 초기화 (AB/FULL 필요한 경우만)
        mgr = None
        needs_mgr = has_bundles and any(s in ("AB", "FULL") for s in stage_list)
        if needs_mgr:
            print(f"  Initializing DynamicStageManager...")
            passlayer_rt = _detect_layer_return_tuple(model)
            mgr = DynamicStageManager(
                model, args.device, dtype, passlayer_rt,
                b_dir=b_dir, c_dir=c_dir, bundles_dir=bundles_dir,
            )
            print(f"  Stage meta: B={mgr.B_idx}, C={mgr.C_idx}")

        # ── Stage별 평가 ──
        for stage in stage_list:
            stage_name = {"A": "A", "AB": "AB", "FULL": "ABC"}.get(stage, stage)
            label = f"{Path(mpath).name}[{stage_name}]"

            if stage in ("AB", "FULL") and mgr:
                print(f"\n  [{stage}] Restoring bundle layers...")
                mgr.set_stage(stage)
                n_active = mgr.num_layers - sum(1 for i in mgr.removed if isinstance(mgr.layers[i], MistralPassLayer))
                print(f"  ✓ Stage {stage}: {n_active} active layers")
            elif stage == "A" and mgr:
                mgr.set_stage("A")

            print(f"  [{stage}] Evaluating PPL...")
            m = eval_ppl(model, batches)
            _print_result_box(label, m)
            results.append({"model": mpath, "stage": stage, "label": label, **m})

            # LoRA 미머지 비교 (A stage, 첫 모델만)
            if model_idx == 0 and stage == "A" and args.lora_paths:
                for lp in args.lora_paths:
                    print(f"\n  Applying LoRA (no-merge): {lp}")
                    model_lora = _apply_lora_no_merge(model, lp)
                    m_lora = eval_ppl(model_lora, batches)
                    lp_name = Path(lp).name
                    _print_result_box(f"+LoRA({lp_name})", m_lora)
                    results.append({"model": mpath, "stage": "A+LoRA", "label": f"+LoRA({lp_name})", **m_lora})
                    del model_lora

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ── 요약 ──
    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print("Summary")
        print(f"{'=' * 70}")
        print(f"{'Label':<45s} {'PPL':>12s} {'Mean NLL':>12s} {'Tokens':>10s}")
        print(f"{'─' * 45} {'─' * 12} {'─' * 12} {'─' * 10}")
        for r in results:
            print(f"{r['label']:<45s} {r['ppl']:>12.4f} {r['mean_nll']:>12.6f} {r['tokens']:>10d}")
        print(f"{'=' * 70}")

    print("\nDone.")


if __name__ == "__main__":
    main()