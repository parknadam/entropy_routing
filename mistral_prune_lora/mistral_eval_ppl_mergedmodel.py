#!/usr/bin/env python3
"""
머지된 모델(pruned + LoRA merged)의 PPL을 평가하는 스크립트.

평가 모드:
  1) A-only (기본):  머지된 모델을 그대로 평가 (드롭된 레이어는 PassLayer → identity)
  2) A/AB/FULL:      B/C 번들을 로드하여 stage별 비교 평가
     - AB:   A_merged + B_merged (C는 PassLayer)
     - FULL: A_merged + B_merged + C_merged

A_merged 모델의 PassLayer에 대해:
  - PassLayer는 hidden_states를 그대로 통과시키는 identity layer입니다.
  - 따라서 A_merged 모델은 "프루닝 + LoRA 머지" 된 모델로 정상 추론/PPL 측정 가능합니다.
  - AB/FULL 평가는 --bundles_dir 또는 --bundle_B_dir/--bundle_C_dir로 번들을 지정하세요.

사용법:
# A-only 평가 (단순)
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path ./merged_models_mistral_7b/A_merged \
  --device cuda:0
  --seqlen 1024

# A/AB/FULL stage 비교 (기존 bundles/B,C 사용)
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path ./merged_models_mistral_7b/A_merged \
  --bundles_dir ./25_mistral_results/pruning/bundles \
  --stages A,AB,FULL \
  --device cuda:0

# A+AB 평가 (A_merged + B_merged, C=PassLayer)
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path ./merged_models_mistral_7b/A_merged \
  --bundle_B_dir ./merged_models_mistral_7b/B_merged \
  --stages A,AB \
  --device cuda:0

# FULL 평가 (A_merged + B_merged + C_merged)
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path ./merged_models_mistral_7b/A_merged \
  --bundle_B_dir ./merged_models_mistral_7b/B_merged \
  --bundle_C_dir ./merged_models_mistral_7b/C_merged \
  --stages FULL \
  --device cuda:0

# 여러 머지 모델 비교
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path ./merged_models/A_merged ./merged_models/AB_merged \
  --device cuda:0

# 원본 모델과 비교
python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \
  --model_path mistralai/Mistral-7B-v0.1 ./merged_models/A_merged \
  --tokenizer_path mistralai/Mistral-7B-v0.1 \
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

# safetensors (for bundle loading)
try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None

# MistralDecoderLayer (for bundle restoration)
MistralDecoderLayer = None
try:
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer as _MDL
    MistralDecoderLayer = _MDL
except ImportError:
    pass

# get_loaders import (optional)
GET_LOADERS = None
for _mod_name in [
    "mistral_prune_lora.pruning.data",
    "pruning.data",
]:
    try:
        _mod = __import__(_mod_name, fromlist=["get_loaders"])
        if hasattr(_mod, "get_loaders"):
            GET_LOADERS = getattr(_mod, "get_loaders")
            break
    except Exception:
        pass

# peft (optional)
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
# manifest.json에서 dropped layers 읽기 + PassLayer 설치
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

    # stages.A.dropped_layers (layeronly_drop.py 생성)
    stages = manifest.get("stages", {})
    dropped = stages.get("A", {}).get("dropped_layers", [])

    # B + C removed_layers
    if not dropped:
        B_removed = stages.get("B", {}).get("removed_layers", [])
        C_removed = stages.get("C", {}).get("removed_layers", [])
        dropped = sorted(set(B_removed + C_removed))

    # simdrop.removed_layers (구버전)
    if not dropped:
        dropped = manifest.get("simdrop", {}).get("removed_layers", [])

    return sorted(set(int(i) for i in dropped))


def _install_passlayers(model, dropped_indices: List[int], return_tuple: bool = True):
    """
    ★ 핵심: from_pretrained() 후 dropped 레이어 위치에 PassLayer를 설치합니다.

    from_pretrained()는 config.num_hidden_layers=32 기준으로 32개 MistralDecoderLayer를
    생성하고, safetensors에 가중치가 없는 레이어는 랜덤 초기화됩니다.
    이 랜덤 레이어가 hidden_states를 오염시키므로 반드시 PassLayer로 교체해야 합니다.
    """
    if not dropped_indices:
        return model

    layers = _get_mistral_layers(model)
    hidden_size = model.config.hidden_size

    for idx in dropped_indices:
        if 0 <= idx < len(layers):
            old = layers[idx]
            # PassLayer를 같은 디바이스에 배치
            dev = next(old.parameters()).device if sum(1 for _ in old.parameters()) > 0 else torch.device("cpu")
            layers[idx] = MistralPassLayer(return_tuple=return_tuple).to(dev)
            del old

    print(f"  ✓ PassLayer 설치: {dropped_indices} ({len(dropped_indices)}개 레이어)")
    return model


# ============================================================
# 모델 & 토크나이저 로드
# ============================================================
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
    #   from_pretrained()가 랜덤 초기화한 레이어를 identity로 교체
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
                print(f"  [INFO] Tokenizer fallback 사용: {path}")
            return tok
        except Exception as e:
            errors.append((path, str(e)))

    # slow tokenizer 시도
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
# DynamicStageManager: bundles에서 B/C 레이어 복원
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


_LAYER_RE = re.compile(r"layer_(\d+)\.safetensors$")


def _build_layer_map(dir_path: Optional[Path]) -> Dict[int, Path]:
    m: Dict[int, Path] = {}
    if dir_path is None or not dir_path.exists():
        return m
    for p in dir_path.glob("layer_*.safetensors"):
        mm = _LAYER_RE.search(p.name)
        if mm:
            m[int(mm.group(1))] = p
    return m


def _has_bundle_layers(dir_path: Optional[Path]) -> bool:
    return len(_build_layer_map(dir_path)) > 0


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
    """A: removed→PassLayer / AB: C만PassLayer / FULL: 전부 복구"""

    def __init__(
        self,
        model,
        device: str,
        dtype: torch.dtype,
        passlayer_return_tuple: bool,
        bundles_dir: Optional[Path] = None,
        B_dir: Optional[Path] = None,
        C_dir: Optional[Path] = None,
    ):
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

        if B_dir is None and bundles_dir is not None:
            B_dir = bundles_dir / "B"
        if C_dir is None and bundles_dir is not None:
            C_dir = bundles_dir / "C"

        self.B_dir = B_dir
        self.C_dir = C_dir

        B_raw = _build_layer_map(B_dir) if B_dir is not None else {}
        C_raw = _build_layer_map(C_dir) if C_dir is not None else {}
        if not B_raw and not C_raw:
            raise FileNotFoundError(
                "No bundle layer files found. "
                f"B_dir={B_dir}, C_dir={C_dir}, bundles_dir={bundles_dir}"
            )

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
# ★ 데이터 로더 — Tokens=0 버그의 핵심 수정 영역
# ============================================================
def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x


def _call_get_loaders(dataset: str, tok, nsamples: int, seed: int, seqlen: int):
    """
    ★ get_loaders 시그니처 문제 해결.

    data.py의 실제 시그니처:
        get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None)

    이전 코드 문제:
        GET_LOADERS(dataset, tok, seqlen=..., nsamples=..., seed=...)
        → 두 번째 positional로 tok이 들어가 nsamples=tok이 됨 → TypeError

    수정: keyword-only로 호출하여 시그니처 충돌 방지
    """
    if GET_LOADERS is None:
        raise RuntimeError(
            "get_loaders import 실패. --text_file로 평가하거나 "
            "pruning/data.py 경로를 확인해 주세요."
        )

    # ★ 안전한 호출: 모두 keyword argument로
    return GET_LOADERS(
        dataset,
        nsamples=nsamples,
        seed=seed,
        seqlen=seqlen,
        tokenizer=tok,
    )


def _pick_split(obj: Any, split: str) -> Any:
    """
    get_loaders 반환값에서 적절한 split을 선택.

    wikitext2/ptb: (trainloader, testenc)
    c4: (trainloader, valenc)
    """
    if isinstance(obj, dict):
        for k in (split, "test", "validation", "val", "train"):
            if k in obj:
                return obj[k]
        return next(iter(obj.values()))

    if isinstance(obj, tuple) and len(obj) >= 2:
        # obj = (trainloader, testenc)
        # split이 test/val이면 obj[1], 아니면 obj[0]
        if split in ("test", "validation", "val"):
            return obj[1]
        return obj[0]

    return obj


def _tensor_to_batches(
    ids: torch.Tensor, seqlen: int, batch_size: int,
    device: str, max_batches: Optional[int],
) -> List[Dict[str, torch.Tensor]]:
    """
    1D 또는 2D 텐서를 seqlen 블록으로 잘라 배치 리스트로 변환합니다.
    """
    if ids.dim() == 2:
        ids = ids[0]  # [1, N] → [N]
    ids = ids.long()

    batches = []
    cur: List[torch.Tensor] = []
    total = ids.numel()

    for start in range(0, total - seqlen + 1, seqlen):
        cur.append(ids[start: start + seqlen])
        if len(cur) == batch_size:
            x = torch.stack(cur, dim=0).to(device)
            batches.append({"input_ids": x, "attention_mask": torch.ones_like(x, dtype=torch.long)})
            cur = []
            if max_batches is not None and len(batches) >= max_batches:
                return batches

    return batches


def _extract_batch(batch: Any, device: str, seqlen: int) -> Optional[Dict[str, torch.Tensor]]:
    """(input_ids, target) 튜플 등 개별 배치를 정규화."""
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
        attn = torch.ones_like(ids, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": attn}

    if torch.is_tensor(batch):
        ids = _ensure_2d(batch)[:, :seqlen].to(device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.long)}

    return None


def _pack_text_lines_to_batches(
    lines: Iterator[str], tok: AutoTokenizer, seqlen: int,
    batch_size: int, device: str, max_batches: Optional[int],
) -> List[Dict[str, torch.Tensor]]:
    buf: List[int] = []
    batches = []
    cur: List[torch.Tensor] = []

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


def prepare_batches(
    raw_loader: Any,
    tok: AutoTokenizer,
    seqlen: int,
    batch_size: int,
    device: str,
    max_batches: Optional[int],
) -> List[Dict[str, torch.Tensor]]:
    """
    ★★★ Tokens=0 버그 수정의 핵심 함수 ★★★

    get_loaders가 반환하는 raw_loader의 실제 타입:
      - wikitext2 test → BatchEncoding  (dict 상속, iter하면 "input_ids" 등 키 문자열!)
      - c4 val        → TokenizerWrapper (.input_ids = tensor)
      - trainloader   → list of (input_ids_tensor, target_tensor) tuples

    이전 코드 버그:
      BatchEncoding을 iter()하면 "input_ids" 문자열이 나오는데,
      이걸 _extract_batch()에 넣으면 None → tokens=0

    수정: raw_loader 타입을 먼저 판별하여 적절히 텐서를 추출
    """

    # ─── Case 1: .input_ids 속성을 가진 객체 (TokenizerWrapper, BatchEncoding) ───
    # BatchEncoding은 dict를 상속하므로 isinstance(raw_loader, dict)로도 잡힘
    # 하지만 .input_ids 체크가 더 안전
    input_ids_tensor = None

    # 1a) dict-like (BatchEncoding 포함) → ["input_ids"] 키로 접근
    if isinstance(raw_loader, dict) and "input_ids" in raw_loader:
        val = raw_loader["input_ids"]
        if torch.is_tensor(val):
            input_ids_tensor = val
            print(f"  [DEBUG] raw_loader is dict/BatchEncoding, input_ids shape: {val.shape}")

    # 1b) .input_ids 속성 (TokenizerWrapper)
    if input_ids_tensor is None and hasattr(raw_loader, "input_ids"):
        val = getattr(raw_loader, "input_ids", None)
        if torch.is_tensor(val):
            input_ids_tensor = val
            print(f"  [DEBUG] raw_loader has .input_ids attr, shape: {val.shape}")

    if input_ids_tensor is not None:
        return _tensor_to_batches(input_ids_tensor, seqlen, batch_size, device, max_batches)

    # ─── Case 2: 텐서 (whole corpus) ───
    if torch.is_tensor(raw_loader):
        print(f"  [DEBUG] raw_loader is tensor, shape: {raw_loader.shape}")
        return _tensor_to_batches(raw_loader, seqlen, batch_size, device, max_batches)

    # ─── Case 3: 문자열 ───
    if isinstance(raw_loader, str):
        return _pack_text_lines_to_batches(iter([raw_loader]), tok, seqlen, batch_size, device, max_batches)

    # ─── Case 4: 리스트 ───
    if isinstance(raw_loader, list):
        if not raw_loader:
            return []

        # 문자열 리스트
        if isinstance(raw_loader[0], str):
            return _pack_text_lines_to_batches(iter(raw_loader), tok, seqlen, batch_size, device, max_batches)

        # (input_ids, target) 튜플 리스트 — trainloader 형태
        batches = []
        for item in raw_loader:
            b = _extract_batch(item, device, seqlen)
            if b is not None:
                batches.append(b)
                if max_batches is not None and len(batches) >= max_batches:
                    break
        if batches:
            print(f"  [DEBUG] raw_loader is list of {len(raw_loader)} items → {len(batches)} batches")
        return batches

    # ─── Case 5: 이터러블 (DataLoader 등) ───
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

    if batches:
        print(f"  [DEBUG] iterable → {len(batches)} batches")
    else:
        print(f"  [ERROR] raw_loader type={type(raw_loader).__name__}, 배치를 추출하지 못했습니다!")
        print(f"  [ERROR] raw_loader repr: {repr(raw_loader)[:200]}")

    return batches


# ============================================================
# PPL 평가
# ============================================================
@torch.no_grad()
def eval_ppl(model, batches: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """Token-weighted NLL / PPL."""
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

        # case 1) labels provided
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

        # case 2) next-token shift
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
# LoRA 어댑터 (미머지 비교용)
# ============================================================
def _apply_lora_no_merge(model, adapter_path: str):
    if PeftModel is None:
        raise RuntimeError("peft import 실패. pip install peft")
    model_lora = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model_lora.eval()
    return model_lora


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

  # A/AB/FULL stage 비교 (bundles 필요)
  python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \\
    --model_path ./merged_models/A_merged \\
    --bundles_dir ./25_mistral_results/pruning/bundles \\
    --stages A,AB,FULL

  # A+AB 평가 (A_merged + B_merged, C=PassLayer)
  python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \\
    --model_path ./merged_models/A_merged \\
    --bundle_B_dir ./merged_models_mistral_7b/B_merged \\
    --stages A,AB

  # FULL 평가 (A_merged + B_merged + C_merged)
  python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \\
    --model_path ./merged_models/A_merged \\
    --bundle_B_dir ./merged_models_mistral_7b/B_merged \\
    --bundle_C_dir ./merged_models_mistral_7b/C_merged \\
    --stages FULL

  # 여러 모델 비교
  python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \\
    --model_path mistralai/Mistral-7B-v0.1 ./merged/A_merged ./merged/AB_merged

  # LoRA 미머지 비교
  python -m mistral_prune_lora.mistral_eval_ppl_mergedmodel \\
    --model_path ./pruning/A --lora_paths ./adapters/A_lora/stageA
        """,
    )
    ap.add_argument("--model_path", type=str, nargs="+", required=True,
                    help="머지된 모델 경로 (여러 개 → 순차 비교)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    # 데이터셋
    ap.add_argument("--dataset", default="wikitext2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--text_file", default=None)

    # Stage 평가 (bundle source 필요)
    ap.add_argument("--bundles_dir", default=None,
                    help="기존 B/C 번들 루트 디렉터리 (예: ./pruning/bundles)")
    ap.add_argument("--bundle_B_dir", default=None,
                    help="B 번들(또는 B_merged) 디렉터리 override")
    ap.add_argument("--bundle_C_dir", default=None,
                    help="C 번들(또는 C_merged) 디렉터리 override")
    ap.add_argument("--stages", default="A",
                    help="평가할 stage (comma-separated: A,AB,FULL). AB는 B bundle, FULL은 B+C bundle 필요")

    # LoRA 비교
    ap.add_argument("--lora_paths", type=str, nargs="*", default=None)
    ap.add_argument("--tokenizer_path", default=None)

    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    stage_list = [s.strip().upper() for s in args.stages.split(",") if s.strip()]
    for stage in stage_list:
        if stage not in ("A", "AB", "FULL"):
            raise ValueError(f"Invalid stage in --stages: {stage} (use A,AB,FULL)")

    bundle_B_dir = Path(args.bundle_B_dir) if args.bundle_B_dir else (
        Path(args.bundles_dir) / "B" if args.bundles_dir else None
    )
    bundle_C_dir = Path(args.bundle_C_dir) if args.bundle_C_dir else (
        Path(args.bundles_dir) / "C" if args.bundles_dir else None
    )

    has_B_bundle = _has_bundle_layers(bundle_B_dir)
    has_C_bundle = _has_bundle_layers(bundle_C_dir)

    if "AB" in stage_list and not has_B_bundle:
        print(
            "⚠ WARNING: AB stage 평가에는 B bundle이 필요합니다. "
            f"(checked: {bundle_B_dir}) -> AB를 제외합니다."
        )
        stage_list = [s for s in stage_list if s != "AB"]

    if "FULL" in stage_list and (not has_B_bundle or not has_C_bundle):
        print(
            "⚠ WARNING: FULL stage 평가에는 B + C bundle이 모두 필요합니다. "
            f"(B: {bundle_B_dir}, C: {bundle_C_dir}) -> FULL을 제외합니다."
        )
        stage_list = [s for s in stage_list if s != "FULL"]

    if not stage_list:
        print("⚠ WARNING: 평가 가능한 stage가 없어 A만 평가합니다.")
        stage_list = ["A"]

    results = []

    print("=" * 70)
    print("Merged Model PPL Evaluation")
    print("=" * 70)
    print(f"Models:      {args.model_path}")
    print(f"Dataset:     {args.text_file or args.dataset} (split={args.split})")
    print(f"Seq length:  {args.seqlen}")
    print(f"Stages:      {stage_list}")
    print(f"B bundle:    {bundle_B_dir} ({'found' if has_B_bundle else 'missing'})")
    print(f"C bundle:    {bundle_C_dir} ({'found' if has_C_bundle else 'missing'})")
    print(f"Device:      {args.device} ({args.dtype})")
    print("=" * 70)

    for model_idx, mpath in enumerate(args.model_path):
        print(f"\n{'─' * 60}")
        print(f"[{model_idx + 1}/{len(args.model_path)}] {mpath}")
        print(f"{'─' * 60}")

        # ── 토크나이저 로드 ──
        tok_path = args.tokenizer_path or mpath
        fallbacks = None if args.tokenizer_path else _find_tokenizer_fallbacks(mpath)
        print(f"\n  Loading tokenizer from: {tok_path}")
        tok = _load_tokenizer(tok_path, fallback_paths=fallbacks)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # ── 데이터 로더 + 배치 준비 ──
        print(f"\n  Loading evaluation data...")
        if args.text_file is not None:
            batches = _pack_text_lines_to_batches(
                iter(open(args.text_file, "r", encoding="utf-8", errors="ignore")),
                tok, args.seqlen, args.batch_size, args.device, args.max_batches,
            )
        else:
            # ★ 수정된 호출: keyword-only로 시그니처 충돌 방지
            raw = _call_get_loaders(args.dataset, tok, args.nsamples, args.seed, args.seqlen)
            raw_loader = _pick_split(raw, args.split)
            print(f"  raw_loader type: {type(raw_loader).__name__}")

            # ★ 수정된 배치 변환: BatchEncoding/TokenizerWrapper 올바르게 처리
            batches = prepare_batches(
                raw_loader, tok, args.seqlen,
                args.batch_size, args.device, args.max_batches,
            )

        print(f"  ✓ Prepared {len(batches)} batches (seqlen={args.seqlen})")

        if len(batches) == 0:
            print(f"  ✗ ERROR: 배치가 0개입니다! 데이터 로딩을 확인하세요.")
            continue

        # ── Stage별 평가 ──
        for stage in stage_list:
            stage_label_map = {"A": "A", "AB": "AB", "FULL": "ABC"}
            label = f"{Path(mpath).name}[{stage_label_map.get(stage, stage)}]"

            print(f"\n  [{stage}] Loading model...")
            model = _load_model(mpath, dtype=dtype, device=args.device)

            # bundle(B/C) 로딩 → DynamicStageManager
            if stage in ("AB", "FULL"):
                print(f"  [{stage}] Loading bundle layers and restoring...")
                passlayer_rt = _detect_layer_return_tuple(model)
                mgr = DynamicStageManager(
                    model=model,
                    device=args.device,
                    dtype=dtype,
                    passlayer_return_tuple=passlayer_rt,
                    bundles_dir=Path(args.bundles_dir) if args.bundles_dir else None,
                    B_dir=bundle_B_dir,
                    C_dir=bundle_C_dir,
                )
                print(f"  Stage meta: B={mgr.B_idx}, C={mgr.C_idx}")
                mgr.set_stage(stage)
                print(f"  ✓ Stage {stage} set")

            print(f"  [{stage}] Evaluating PPL...")
            m = eval_ppl(model, batches)
            print(f"\n  ┌─────────────────────────────────────────┐")
            print(f"  │ {label:<40s}│")
            print(f"  ├─────────────────────────────────────────┤")
            print(f"  │ PPL      = {m['ppl']:<28.4f} │")
            print(f"  │ Mean NLL = {m['mean_nll']:<28.6f} │")
            print(f"  │ Tokens   = {m['tokens']:<28d} │")
            print(f"  └─────────────────────────────────────────┘")
            results.append({"model": mpath, "stage": stage, "label": label, **m})

            # LoRA 미머지 비교
            if model_idx == 0 and stage == "A" and args.lora_paths:
                for lp in args.lora_paths:
                    print(f"\n  Applying LoRA (no-merge): {lp}")
                    model_lora = _apply_lora_no_merge(model, lp)
                    m_lora = eval_ppl(model_lora, batches)
                    lp_name = Path(lp).name
                    print(f"  LoRA({lp_name}): ppl={m_lora['ppl']:.4f} | nll={m_lora['mean_nll']:.6f}")
                    results.append({"model": mpath, "stage": "A+LoRA", "label": f"+LoRA({lp_name})", **m_lora})
                    del model_lora

            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    # ── 요약 테이블 ──
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
