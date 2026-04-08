# gemma_prune_lora/eval_ppl_debug.py
#
# Gemma 1 7B pruning stage(A / AB / FULL) corpus perplexity evaluator
# with optional LoRA adapters applied without merge.
# This debug copy keeps the original script intact while adding safer
# stage-by-stage checks for memory-constrained investigations.
#
# Example:
"""
CUDA_VISIBLE_DEVICES=4 DEVICE=cuda:0 \
python -m gemma_prune_lora.eval_ppl_debug \
    --base_model ./20_gemma_7b_results/pruning/A \
    --bundles_dir ./20_gemma_7b_results/pruning/bundles \
    --text_file ./data/wikitext2_test.txt \
    --seqlen 1024 --batch_size 1 --max_batches 64 \
    --device cuda:0 --dtype bf16 \
    --stage FULL --restore_mode strict \
    --loss_dtype model --report_cuda_mem

CUDA_VISIBLE_DEVICES=6 DEVICE=cuda:0 \
python -m gemma_prune_lora.eval_ppl_debug \
    --base_model ./gemma_7b_results/pruning/A \
    --bundles_dir ./gemma_7b_results/pruning/bundles \
    --text_file ./data/wikitext2_test.txt \
    --seqlen 1024 --batch_size 1 --max_batches 64 \
    --device cuda:0 --dtype bf16 \
    --lora_A ./gemma_lora_results/adapters/A_lora/stageA/stageA \
    --lora_AB ./gemma_lora_results/adapters/B_lora/stageB/stageB \
    --lora_FULL ./gemma_lora_results/adapters/C_lora/stageC/stageC
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

try:
    from gemma_prune_lora.pruning.data import get_loaders
except Exception:
    get_loaders = None

try:
    from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
except Exception:
    GemmaDecoderLayer = None


def _get_layers(model) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        return model.model.model.layers
    raise RuntimeError("Gemma layers path not found. Expected model.model.layers")


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


def _build_layer_map(dir_path: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    if not dir_path.exists():
        return out
    for path in dir_path.glob("layer_*.safetensors"):
        match = _LAYER_RE.search(path.name)
        if match:
            out[int(match.group(1))] = path
    return out


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


def _read_stage_layout(base_model_dir: Path, num_layers: int) -> Dict[str, Any]:
    manifest = _load_json(base_model_dir / "manifest.json")
    stages = manifest.get("stages", {}) if isinstance(manifest, dict) else {}

    layout = {
        "num_layers": int((manifest.get("counts", {}) or {}).get("L_full", num_layers)) if manifest else num_layers,
        "A_dropped": [],
        "B_removed": [],
        "C_removed": [],
    }

    try:
        layout["A_dropped"] = _normalize_indices(stages.get("A", {}).get("dropped_layers", []), num_layers)
        layout["B_removed"] = _normalize_indices(stages.get("B", {}).get("removed_layers", []), num_layers)
        layout["C_removed"] = _normalize_indices(stages.get("C", {}).get("removed_layers", []), num_layers)
    except ValueError as exc:
        print(f"[WARN] manifest stage indices look invalid: {exc}")

    if not layout["A_dropped"]:
        layers_map = _load_json(base_model_dir / "layers_map.json")
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

    return layout


def _strip_layer_prefix(sd: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, torch.Tensor]:
    prefixes = [
        f"model.layers.{layer_idx}.",
        f"model.model.layers.{layer_idx}.",
        f"layers.{layer_idx}.",
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
    B_map: Dict[int, Path],
    C_map: Dict[int, Path],
    num_layers: int,
) -> Tuple[Dict[int, Path], Dict[int, Path], int]:
    all_idx = sorted(set(B_map) | set(C_map))
    if not all_idx:
        return B_map, C_map, 0

    if all(0 <= i < num_layers for i in all_idx):
        return B_map, C_map, 0

    if all(1 <= i <= num_layers for i in all_idx):
        return ({i - 1: p for i, p in B_map.items()}, {i - 1: p for i, p in C_map.items()}, -1)

    raise ValueError(
        f"Bundle layer index mismatch (num_layers={num_layers}, max={max(all_idx)}). "
        "base_model and bundles may not match."
    )


class DynamicStageManager:
    def __init__(
        self,
        model,
        base_model_dir: Path,
        bundles_dir: Path,
        device: str,
        dtype: torch.dtype,
        passlayer_return_tuple: bool,
        restore_mode: str,
    ):
        if GemmaDecoderLayer is None:
            raise RuntimeError("GemmaDecoderLayer import failed. Check transformers version.")

        self.model = model
        self.layers = _get_layers(model)
        self.device = device
        self.dtype = dtype
        self.hidden_size = getattr(model.config, "hidden_size", 0)
        self.return_tuple = passlayer_return_tuple
        self.restore_mode = restore_mode

        self.num_layers = len(self.layers)
        self.layout = _read_stage_layout(base_model_dir, self.num_layers)
        B_raw = _build_layer_map(bundles_dir / "B")
        C_raw = _build_layer_map(bundles_dir / "C")
        self.B_map, self.C_map, self.index_shift = _maybe_shift_indices_to_zero_based(B_raw, C_raw, self.num_layers)
        self.B_idx = self.layout["B_removed"] or sorted(self.B_map)
        self.C_idx = self.layout["C_removed"] or sorted(self.C_map)
        self.removed = self.layout["A_dropped"] or sorted(set(self.B_idx) | set(self.C_idx))

        if self.layout["num_layers"] != self.num_layers:
            print(
                f"[WARN] manifest L_full={self.layout['num_layers']} but loaded model has {self.num_layers} layers."
            )

        bundle_union = sorted(set(self.B_map) | set(self.C_map))
        stage_union = sorted(set(self.B_idx) | set(self.C_idx))
        if bundle_union and stage_union and bundle_union != stage_union:
            print(
                f"[WARN] manifest bundle indices {stage_union} differ from files on disk {bundle_union}. "
                "Stage-A masking follows manifest; AB/FULL restore requires matching bundle files."
            )

        self.set_stage("A")

    def stage_meta(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "index_shift_applied": self.index_shift,
            "B": self.B_idx,
            "C": self.C_idx,
            "removed": self.removed,
            "manifest_num_layers": self.layout["num_layers"],
        }

    def _pass_one_layer(self, layer_idx: int):
        old = self.layers[layer_idx]
        self.layers[layer_idx] = GemmaPassLayer(self.hidden_size, self.return_tuple).to(self.device)
        del old

    def _restore_one_layer(self, layer_idx: int):
        bundle_path = self.B_map.get(layer_idx) or self.C_map.get(layer_idx)
        if bundle_path is None:
            raise FileNotFoundError(f"layer_{layer_idx}.safetensors not found in bundles.")

        try:
            new_layer = GemmaDecoderLayer(self.model.config, layer_idx)
        except TypeError:
            new_layer = GemmaDecoderLayer(self.model.config)
        new_layer = new_layer.to(self.device, dtype=self.dtype)

        state_dict = _strip_layer_prefix(load_file(str(bundle_path), device="cpu"), layer_idx)
        try:
            if self.restore_mode == "strict":
                new_layer.load_state_dict(state_dict, strict=True)
            else:
                missing, unexpected = new_layer.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    print(f"[WARN] layer {layer_idx}: missing={len(missing)} unexpected={len(unexpected)}")
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to restore layer {layer_idx} from {bundle_path.name} "
                f"(restore_mode={self.restore_mode}): {exc}"
            ) from exc

        old = self.layers[layer_idx]
        self.layers[layer_idx] = new_layer
        del old
        del state_dict

    def set_stage(self, stage: str):
        stage = stage.upper()
        if stage not in ("A", "AB", "FULL"):
            raise ValueError("stage must be A / AB / FULL")

        pass_set = set(self.removed) if stage == "A" else set(self.C_idx) if stage == "AB" else set()

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


def _iter_textfile_batches(
    tok: AutoTokenizer,
    text_file: Path,
    seqlen: int,
    batch_size: int,
    device: str,
    max_batches: Optional[int],
) -> Iterator[Dict[str, torch.Tensor]]:
    # Preserve the raw corpus layout so text_file evaluation is much closer to
    # the built-in wikitext2 loader than line-wise tokenization with stripped blanks.
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


def _tokenize_text_corpus(tok: AutoTokenizer, text_file: Path) -> Optional[torch.Tensor]:
    text = text_file.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return None
    return tok(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0].long()


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
                    yield {
                        "input_ids": input_ids,
                        "attention_mask": second,
                    }
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


def _cast_logits_for_loss(logits: torch.Tensor, loss_dtype: Optional[torch.dtype]) -> torch.Tensor:
    if loss_dtype is None or logits.dtype == loss_dtype:
        return logits
    return logits.to(loss_dtype)


def _reset_cuda_peak(device: str):
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return
    torch.cuda.reset_peak_memory_stats(torch.device(device))


def _maybe_report_cuda_mem(enabled: bool, label: str, device: str):
    if not enabled:
        return
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        print(f"[CUDA_MEM] {label}: unavailable")
        return

    dev = torch.device(device)
    gib = 1024 ** 3
    allocated = torch.cuda.memory_allocated(dev) / gib
    reserved = torch.cuda.memory_reserved(dev) / gib
    peak_allocated = torch.cuda.max_memory_allocated(dev) / gib
    peak_reserved = torch.cuda.max_memory_reserved(dev) / gib
    print(
        f"[CUDA_MEM] {label}: "
        f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB "
        f"peak_allocated={peak_allocated:.2f}GiB peak_reserved={peak_reserved:.2f}GiB"
    )


@torch.no_grad()
def eval_ppl_stride(
    model,
    input_ids: torch.Tensor,
    seqlen: int,
    stride: int,
    device: str,
    loss_dtype: Optional[torch.dtype] = torch.float32,
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
            _cast_logits_for_loss(shift_logits, loss_dtype).view(-1, vocab_size),
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


def eval_ppl_bos_blocks(
    model,
    input_ids: torch.Tensor,
    seqlen: int,
    bos_token_id: Optional[int],
    device: str,
    loss_dtype: Optional[torch.dtype] = torch.float32,
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
            _cast_logits_for_loss(shift_logits, loss_dtype).view(-1, vocab_size),
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
def eval_ppl(
    model,
    loader: Iterator[Dict[str, torch.Tensor]],
    loss_dtype: Optional[torch.dtype] = torch.float32,
) -> Dict[str, float]:
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
                _cast_logits_for_loss(logits, loss_dtype).view(-1, vocab_size),
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
            _cast_logits_for_loss(shift_logits, loss_dtype).view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels).float()

        sum_nll += float((loss_tok * shift_mask).sum().item())
        sum_tok += int(shift_mask.sum().item())

    if sum_tok == 0:
        return {"mean_nll": float("nan"), "ppl": float("nan"), "tokens": 0}

    mean_nll = sum_nll / sum_tok
    return {"mean_nll": mean_nll, "ppl": math.exp(mean_nll), "tokens": sum_tok}


def _load_model(base_model: str, dtype: torch.dtype, device: str):
    attempts = [
        {"torch_dtype": dtype, "attn_implementation": "eager", "trust_remote_code": True},
        {"torch_dtype": dtype, "trust_remote_code": True},
        {"dtype": dtype, "attn_implementation": "eager", "trust_remote_code": True},
        {"dtype": dtype, "trust_remote_code": True},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            model = AutoModelForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True, **kwargs)
            model = model.to(device)
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            return model
        except TypeError as exc:
            last_error = exc
    raise last_error


def _pick_split(obj: Any, split: str) -> Any:
    if isinstance(obj, dict):
        if split in obj:
            return obj[split]
        for key in (split, "test", "validation", "val", "train"):
            if key in obj:
                return obj[key]
        return next(iter(obj.values()))
    if isinstance(obj, tuple) and len(obj) >= 2:
        return obj[1] if split in ("test", "validation", "val") else obj[0]
    return obj


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--bundles_dir", required=True)
    ap.add_argument("--baseline_model", default=None)
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
    ap.add_argument("--lora_A", default=None)
    ap.add_argument("--lora_AB", default=None)
    ap.add_argument("--lora_FULL", default=None)
    ap.add_argument("--stage", default="ALL", choices=["ALL", "A", "AB", "FULL"])
    ap.add_argument("--restore_mode", default="strict", choices=["strict", "warn"])
    ap.add_argument("--loss_dtype", default="fp32", choices=["model", "bf16", "fp16", "fp32"])
    ap.add_argument("--report_cuda_mem", action="store_true")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    loss_dtype = {
        "model": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.loss_dtype]

    try:
        tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    except Exception:
        try:
            tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        except Exception:
            try:
                tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
            except Exception:
                tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if args.dtype == "fp16":
        print("[WARN] Gemma perplexity can become unstable in fp16; bf16 is recommended.")

    def choose_eval_mode(has_corpus_ids: bool) -> str:
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

    def make_raw_loader():
        if args.text_file:
            return _iter_textfile_batches(
                tok=tok,
                text_file=Path(args.text_file),
                seqlen=args.seqlen,
                batch_size=args.batch_size,
                device=args.device,
                max_batches=args.max_batches,
            )
        if get_loaders is None:
            raise RuntimeError("gemma_prune_lora.pruning.data.get_loaders import failed. Use --text_file instead.")
        raw = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, tokenizer=tok)
        return _pick_split(raw, args.split)

    corpus_ids = _tokenize_text_corpus(tok, Path(args.text_file)) if args.text_file else None

    def evaluate_model(model, label: str):
        raw_loader = None
        stride_ids = corpus_ids
        loader_mode = choose_eval_mode(corpus_ids is not None)
        if loader_mode == "bos_block":
            metrics = eval_ppl_bos_blocks(
                model=model,
                input_ids=stride_ids,
                seqlen=args.seqlen,
                bos_token_id=tok.bos_token_id,
                device=args.device,
                loss_dtype=loss_dtype,
                max_batches=args.max_batches,
            )
        elif loader_mode == "stride":
            metrics = eval_ppl_stride(
                model=model,
                input_ids=stride_ids,
                seqlen=args.seqlen,
                stride=args.stride,
                device=args.device,
                loss_dtype=loss_dtype,
                max_batches=args.max_batches,
            )
        else:
            raw_loader = make_raw_loader()
            if corpus_ids is None:
                corpus_ids_local = _extract_corpus_ids(raw_loader)
                local_mode = choose_eval_mode(corpus_ids_local is not None)
                if local_mode == "stride" and corpus_ids_local is not None:
                    stride_ids = corpus_ids_local
                    metrics = eval_ppl_stride(
                        model=model,
                        input_ids=stride_ids,
                        seqlen=args.seqlen,
                        stride=args.stride,
                        device=args.device,
                        loss_dtype=loss_dtype,
                        max_batches=args.max_batches,
                    )
                    loader_mode = "stride"
                elif local_mode == "bos_block" and corpus_ids_local is not None:
                    stride_ids = corpus_ids_local
                    metrics = eval_ppl_bos_blocks(
                        model=model,
                        input_ids=stride_ids,
                        seqlen=args.seqlen,
                        bos_token_id=tok.bos_token_id,
                        device=args.device,
                        loss_dtype=loss_dtype,
                        max_batches=args.max_batches,
                    )
                    loader_mode = "bos_block"
                else:
                    batches = list(
                        _normalize_loader_to_batches(
                            raw_loader=raw_loader,
                            seqlen=args.seqlen,
                            batch_size=args.batch_size,
                            device=args.device,
                            max_batches=args.max_batches,
                        )
                    )
                    metrics = eval_ppl(model, iter(batches), loss_dtype=loss_dtype)
            else:
                batches = list(
                    _normalize_loader_to_batches(
                        raw_loader=raw_loader,
                        seqlen=args.seqlen,
                        batch_size=args.batch_size,
                        device=args.device,
                        max_batches=args.max_batches,
                    )
                )
                metrics = eval_ppl(model, iter(batches), loss_dtype=loss_dtype)

        print(
            f"[{label}] BASE ppl={metrics['ppl']:.6f} | mean_nll={metrics['mean_nll']:.6f} | "
            f"tokens={metrics['tokens']} | mode={loader_mode}"
        )
        return metrics, loader_mode, stride_ids

    if args.baseline_model:
        baseline_model = _load_model(args.baseline_model, dtype=dtype, device=args.device)
        baseline_model.eval()
        _maybe_report_cuda_mem(args.report_cuda_mem, "BASELINE:after_load", args.device)
        _reset_cuda_peak(args.device)
        evaluate_model(baseline_model, "BASELINE")
        _maybe_report_cuda_mem(args.report_cuda_mem, "BASELINE:after_eval", args.device)
        del baseline_model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    stage_plan = [("A", "A"), ("AB", "AB"), ("FULL", "ABC")]
    if args.stage != "ALL":
        stage_plan = [(args.stage, "ABC" if args.stage == "FULL" else args.stage)]

    for stage, label in stage_plan:
        model = _load_model(args.base_model, dtype=dtype, device=args.device)
        model.eval()
        _maybe_report_cuda_mem(args.report_cuda_mem, f"{label}:after_load", args.device)

        mgr = DynamicStageManager(
            model=model,
            base_model_dir=Path(args.base_model),
            bundles_dir=Path(args.bundles_dir),
            device=args.device,
            dtype=dtype,
            passlayer_return_tuple=_detect_layer_return_tuple(model),
            restore_mode=args.restore_mode,
        )
        mgr.set_stage(stage)
        _maybe_report_cuda_mem(args.report_cuda_mem, f"{label}:after_set_stage", args.device)
        _reset_cuda_peak(args.device)

        if stage == "A":
            print("\n=== GROUP META ===")
            print(mgr.stage_meta())

        metrics, loader_mode, stride_ids = evaluate_model(model, label)
        _maybe_report_cuda_mem(args.report_cuda_mem, f"{label}:after_eval", args.device)

        if stage == "A" or args.stage != "ALL":
            stride_desc = f" stride={min(args.stride, args.seqlen)}" if loader_mode == "stride" else ""
            print(
                f"device={args.device} dtype={args.dtype} loss_dtype={args.loss_dtype} "
                f"restore_mode={args.restore_mode} stage={stage} ppl_mode={loader_mode}{stride_desc}\n"
            )

        for adapter_path in _parse_lora_list({"A": args.lora_A, "AB": args.lora_AB, "FULL": args.lora_FULL}[stage]):
            model_lora = _apply_lora_no_merge(model, adapter_path)
            _reset_cuda_peak(args.device)
            if loader_mode == "bos_block":
                metrics_lora = eval_ppl_bos_blocks(
                    model=model_lora,
                    input_ids=stride_ids,
                    seqlen=args.seqlen,
                    bos_token_id=tok.bos_token_id,
                    device=args.device,
                    loss_dtype=loss_dtype,
                    max_batches=args.max_batches,
                )
            elif loader_mode == "stride":
                metrics_lora = eval_ppl_stride(
                    model=model_lora,
                    input_ids=stride_ids,
                    seqlen=args.seqlen,
                    stride=args.stride,
                    device=args.device,
                    loss_dtype=loss_dtype,
                    max_batches=args.max_batches,
                )
            else:
                raw_loader = make_raw_loader()
                batches = list(
                    _normalize_loader_to_batches(
                        raw_loader=raw_loader,
                        seqlen=args.seqlen,
                        batch_size=args.batch_size,
                        device=args.device,
                        max_batches=args.max_batches,
                    )
                )
                metrics_lora = eval_ppl(model_lora, iter(batches), loss_dtype=loss_dtype)
            print(
                f"[{label}] LoRA({Path(adapter_path).name}) ppl={metrics_lora['ppl']:.6f} | "
                f"mean_nll={metrics_lora['mean_nll']:.6f} | tokens={metrics_lora['tokens']} | mode={loader_mode}"
            )
            _maybe_report_cuda_mem(args.report_cuda_mem, f"{label}:LoRA({Path(adapter_path).name})", args.device)
            del model_lora

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
