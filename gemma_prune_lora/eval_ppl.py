# gemma_prune_lora/eval_ppl.py
#
# Gemma 1 7B pruning stage(A / AB / FULL) corpus perplexity evaluator
# with optional LoRA adapters applied without merge.
#
# Example:
"""
CUDA_VISIBLE_DEVICES=2 DEVICE=cuda:0 \
python -m gemma_prune_lora.eval_ppl \
    --base_model ./gemma_7b_results/pruning/A \
    --bundles_dir ./gemma_7b_results/pruning/bundles \
    --text_file ./data/wikitext2_test.txt \
    --seqlen 1024 --batch_size 1 --max_batches 64 \
    --device cuda:0 --dtype bf16

CUDA_VISIBLE_DEVICES=6 DEVICE=cuda:0 \
python -m gemma_prune_lora.eval_ppl \
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
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        if not self.return_tuple:
            return hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_value,)
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
        bundles_dir: Path,
        device: str,
        dtype: torch.dtype,
        passlayer_return_tuple: bool,
    ):
        if GemmaDecoderLayer is None:
            raise RuntimeError("GemmaDecoderLayer import failed. Check transformers version.")

        self.model = model
        self.layers = _get_layers(model)
        self.device = device
        self.dtype = dtype
        self.hidden_size = getattr(model.config, "hidden_size", 0)
        self.return_tuple = passlayer_return_tuple

        self.num_layers = len(self.layers)
        B_raw = _build_layer_map(bundles_dir / "B")
        C_raw = _build_layer_map(bundles_dir / "C")
        self.B_map, self.C_map, self.index_shift = _maybe_shift_indices_to_zero_based(B_raw, C_raw, self.num_layers)
        self.B_idx = sorted(self.B_map)
        self.C_idx = sorted(self.C_map)
        self.removed = sorted(set(self.B_idx) | set(self.C_idx))

        self.set_stage("A")

    def stage_meta(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "index_shift_applied": self.index_shift,
            "B": self.B_idx,
            "C": self.C_idx,
            "removed": self.removed,
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
    tokens: List[int] = []
    made = 0
    batch_buf: List[torch.Tensor] = []

    with text_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = tok(line, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
            tokens.extend(ids)

            while len(tokens) >= seqlen:
                batch_buf.append(torch.tensor(tokens[:seqlen], dtype=torch.long))
                tokens = tokens[seqlen:]

                if len(batch_buf) == batch_size:
                    input_ids = torch.stack(batch_buf, dim=0).to(device)
                    yield {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids, dtype=torch.long)}
                    batch_buf = []
                    made += 1
                    if max_batches is not None and made >= max_batches:
                        return


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
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--dataset", default="wikitext2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--text_file", default=None)
    ap.add_argument("--lora_A", default=None)
    ap.add_argument("--lora_AB", default=None)
    ap.add_argument("--lora_FULL", default=None)
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

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

    for stage, label in [("A", "A"), ("AB", "AB"), ("FULL", "ABC")]:
        model = _load_model(args.base_model, dtype=dtype, device=args.device)
        model.eval()

        mgr = DynamicStageManager(
            model=model,
            bundles_dir=Path(args.bundles_dir),
            device=args.device,
            dtype=dtype,
            passlayer_return_tuple=_detect_layer_return_tuple(model),
        )
        mgr.set_stage(stage)

        if stage == "A":
            print("\n=== GROUP META ===")
            print(mgr.stage_meta())
            print(f"device={args.device} dtype={args.dtype}\n")

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

        metrics = eval_ppl(model, iter(batches))
        print(f"[{label}] BASE ppl={metrics['ppl']:.6f} | mean_nll={metrics['mean_nll']:.6f} | tokens={metrics['tokens']}")

        for adapter_path in _parse_lora_list({"A": args.lora_A, "AB": args.lora_AB, "FULL": args.lora_FULL}[stage]):
            model_lora = _apply_lora_no_merge(model, adapter_path)
            metrics_lora = eval_ppl(model_lora, iter(batches))
            print(
                f"[{label}] LoRA({Path(adapter_path).name}) ppl={metrics_lora['ppl']:.6f} | "
                f"mean_nll={metrics_lora['mean_nll']:.6f} | tokens={metrics_lora['tokens']}"
            )
            del model_lora

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
