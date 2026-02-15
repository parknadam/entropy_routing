# falcon_prune_lora/falcon_eval_ppl.py
# A / AB / ABC(FULL) stage별 "corpus perplexity" 평가 + (옵션) LoRA 어댑터(merge 없이) 얹어서 ppl 평가
# Falcon 모델 구조 대응
#
# Usage (텍스트 파일로 평가; 가장 안정적)
"""
CUDA_VISIBLE_DEVICES=6 \
python -m falcon_prune_lora.falcon_eval_ppl \
     --base_model ./falcon_results/pruning/A \
     --bundles_dir ./falcon_results/pruning/bundles \
     --text_file ./data/wikitext2_test.txt \
     --seqlen 1024 --batch_size 1 --max_batches 64 \
     --dtype bf16 \
     --lora_A   ./falcon_lora/adapters/stageA \
     --lora_AB  ./falcon_lora/adapters/stageAB \
     --lora_FULL ./falcon_lora/adapters/stageFULL
"""

from __future__ import annotations

import argparse
import inspect
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterator, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# PEFT (LoRA) optional import
# -----------------------------
try:
    from peft import PeftModel
except Exception:
    PeftModel = None


# -----------------------------
# Try import project get_loaders (optional)
# -----------------------------
def _try_import_get_loaders():
    cands = [
        "falcon_prune_lora.pruning.data",
        "pruning.data",
    ]
    for m in cands:
        try:
            mod = __import__(m, fromlist=["get_loaders"])
            if hasattr(mod, "get_loaders"):
                return getattr(mod, "get_loaders")
        except Exception:
            pass
    return None


GET_LOADERS = _try_import_get_loaders()


# -----------------------------
# Falcon DecoderLayer import
# -----------------------------
try:
    from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
except Exception:
    FalconDecoderLayer = None


def _get_falcon_layers(model) -> nn.ModuleList:
    """Falcon 모델의 디코더 레이어 리스트 반환 (다양한 wrapper 대응)"""
    # FalconForCausalLM → FalconModel.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # PeftModel wrapping
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model") and hasattr(base.model, "transformer"):
            return base.model.transformer.h
        if hasattr(base, "transformer"):
            return base.transformer.h
    # model.model.transformer.h (다중 래핑)
    if hasattr(model, "model"):
        if hasattr(model.model, "transformer") and hasattr(model.model.transformer, "h"):
            return model.model.transformer.h
    raise RuntimeError(
        "Falcon layers 경로를 찾지 못했습니다. "
        "(예상: model.transformer.h)"
    )


class FalconPassLayer(nn.Module):
    def __init__(self, return_tuple: bool = True):
        super().__init__()
        self.return_tuple = return_tuple

    def forward(
        self,
        hidden_states,
        alibi=None,
        attention_mask=None,
        position_ids=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        if not self.return_tuple:
            return hidden_states
        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (layer_past,)
        return outputs


def _detect_layer_return_tuple(model) -> bool:
    """
    transformers 버전에 따라 decoder_layer의 반환 형태가 달라질 수 있어,
    PassLayer도 동일한 형태로 맞춰주기 위한 휴리스틱.
    """
    try:
        core = model.transformer if hasattr(model, "transformer") else model
        src = inspect.getsource(core.forward)
        if "layer_outputs[0]" in src or "outputs = layer(" in src:
            return True
        if "hidden_states = layer(" in src and "layer_outputs" not in src:
            return False
    except Exception:
        pass
    return True


_LAYER_RE = re.compile(r"layer_(\d+)\.safetensors$")


def _build_layer_map(dir_path: Path) -> Dict[int, Path]:
    m: Dict[int, Path] = {}
    if not dir_path.exists():
        return m
    for p in dir_path.glob("layer_*.safetensors"):
        mm = _LAYER_RE.search(p.name)
        if mm:
            idx = int(mm.group(1))
            m[idx] = p
    return m


def _strip_layer_prefix(sd: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, torch.Tensor]:
    """번들 파일의 키에서 레이어 prefix를 제거"""
    out = {}
    needles = [
        f"transformer.h.{layer_idx}.",
        f"model.transformer.h.{layer_idx}.",
        f"h.{layer_idx}.",
    ]
    for k, v in sd.items():
        nk = k
        for nd in needles:
            if nd in nk:
                nk = nk.split(nd, 1)[1]
                break
        out[nk] = v
    return out


def _maybe_shift_indices_to_zero_based(
    B_map: Dict[int, Path],
    C_map: Dict[int, Path],
    num_layers: int,
) -> Tuple[Dict[int, Path], Dict[int, Path], int]:
    all_idx = sorted(set(B_map.keys()) | set(C_map.keys()))
    if not all_idx:
        return B_map, C_map, 0

    out_of_range = any((i < 0 or i >= num_layers) for i in all_idx)
    if not out_of_range:
        return B_map, C_map, 0

    one_based_ok = all((1 <= i <= num_layers) for i in all_idx)
    if one_based_ok:
        B2 = {i - 1: p for i, p in B_map.items()}
        C2 = {i - 1: p for i, p in C_map.items()}
        return B2, C2, -1

    raise ValueError(
        f"bundles layer index mismatch (num_layers={num_layers}, max={max(all_idx)}). "
        "base_model과 bundles가 같은 기반 모델인지 확인해 주세요."
    )


class DynamicStageManager:
    """
    A:    removed 모두 PassLayer
    AB:   C만 PassLayer (B 복구)
    FULL: B,C 모두 복구 (ABC)
    """

    def __init__(
        self,
        model,
        bundles_dir: Path,
        device: str,
        dtype: torch.dtype,
        passlayer_return_tuple: bool,
    ):
        if FalconDecoderLayer is None:
            raise RuntimeError(
                "FalconDecoderLayer import 실패. "
                "transformers 버전을 확인하세요. (pip install -U transformers)"
            )

        self.model = model
        self.layers = _get_falcon_layers(model)
        self.device = device
        self.dtype = dtype
        self.passlayer_return_tuple = passlayer_return_tuple

        self.num_layers = len(self.layers)
        self.B_dir = bundles_dir / "B"
        self.C_dir = bundles_dir / "C"

        B_raw = _build_layer_map(self.B_dir)
        C_raw = _build_layer_map(self.C_dir)
        self.B_map, self.C_map, self.index_shift = _maybe_shift_indices_to_zero_based(
            B_raw, C_raw, self.num_layers
        )

        self.B_idx = sorted(self.B_map.keys())
        self.C_idx = sorted(self.C_map.keys())
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

    def _bundle_path(self, layer_i: int) -> Optional[Path]:
        if layer_i in self.B_map:
            return self.B_map[layer_i]
        if layer_i in self.C_map:
            return self.C_map[layer_i]
        return None

    def _restore_one_layer(self, layer_i: int):
        p = self._bundle_path(layer_i)
        if p is None:
            raise FileNotFoundError(f"layer_{layer_i}.safetensors not found in B/C.")

        # FalconDecoderLayer 생성
        try:
            new_layer = FalconDecoderLayer(self.model.config, layer_i)
        except TypeError:
            new_layer = FalconDecoderLayer(self.model.config)

        new_layer = new_layer.to(self.device, dtype=self.dtype)

        sd = load_file(str(p), device="cpu")
        sd = _strip_layer_prefix(sd, layer_i)

        missing, unexpected = new_layer.load_state_dict(sd, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"[WARN] layer {layer_i}: missing={len(missing)} unexpected={len(unexpected)}")

        old = self.layers[layer_i]
        self.layers[layer_i] = new_layer
        del old

    def _pass_one_layer(self, layer_i: int):
        old = self.layers[layer_i]
        self.layers[layer_i] = FalconPassLayer(
            return_tuple=self.passlayer_return_tuple
        ).to(self.device)
        del old

    def set_stage(self, stage: str):
        stage = stage.upper()
        if stage not in ("A", "AB", "FULL"):
            raise ValueError("stage must be A / AB / FULL")

        if stage == "A":
            pass_set = set(self.removed)
        elif stage == "AB":
            pass_set = set(self.C_idx)
        else:
            pass_set = set()

        for i in self.removed:
            cur = self.layers[i]
            is_pass = isinstance(cur, FalconPassLayer)
            if i in pass_set:
                if not is_pass:
                    self._pass_one_layer(i)
            else:
                if is_pass:
                    self._restore_one_layer(i)

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


# -----------------------------
# Loader normalization (dict/tuple/tensor/str 모두 처리)
# -----------------------------
def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x


def _is_attn_mask_like(t: torch.Tensor) -> bool:
    if t.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8, torch.bool):
        return False
    u = torch.unique(t.detach().cpu())
    return all(int(v) in (0, 1) for v in u.tolist()[:10])


def _extract_batch(
    batch: Any,
    tok: AutoTokenizer,
    device: str,
    seqlen: int,
) -> Optional[Dict[str, torch.Tensor]]:
    # 1) dict
    if isinstance(batch, dict):
        if "input_ids" in batch:
            input_ids = batch["input_ids"]
            attn = batch.get("attention_mask", None)
            labels = batch.get("labels", None)

            if not torch.is_tensor(input_ids):
                return None
            input_ids = _ensure_2d(input_ids)

            if input_ids.size(1) > seqlen:
                input_ids = input_ids[:, :seqlen]

            input_ids = input_ids.to(device)

            if attn is None:
                attn = torch.ones_like(input_ids, dtype=torch.long)
            else:
                attn = _ensure_2d(attn)
                if attn.size(1) > input_ids.size(1):
                    attn = attn[:, : input_ids.size(1)]
                elif attn.size(1) < input_ids.size(1):
                    pad = torch.ones((attn.size(0), input_ids.size(1) - attn.size(1)), dtype=attn.dtype)
                    attn = torch.cat([attn, pad], dim=1)
                attn = attn.to(device)

            out = {"input_ids": input_ids, "attention_mask": attn}
            if labels is not None and torch.is_tensor(labels):
                labels = _ensure_2d(labels)
                if labels.size(1) > input_ids.size(1):
                    labels = labels[:, : input_ids.size(1)]
                elif labels.size(1) < input_ids.size(1):
                    pad = torch.full(
                        (labels.size(0), input_ids.size(1) - labels.size(1)), -100, dtype=labels.dtype
                    )
                    labels = torch.cat([labels, pad], dim=1)
                out["labels"] = labels.to(device)
            return out
        return None

    # 2) tuple/list
    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            return None
        if len(batch) == 1 and torch.is_tensor(batch[0]):
            input_ids = _ensure_2d(batch[0])
            if input_ids.size(1) > seqlen:
                input_ids = input_ids[:, :seqlen]
            input_ids = input_ids.to(device)
            attn = torch.ones_like(input_ids, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attn}

        if len(batch) >= 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]):
            x = _ensure_2d(batch[0])
            y = _ensure_2d(batch[1])

            if x.size(1) > seqlen:
                x = x[:, :seqlen]
                if y.size(1) >= seqlen:
                    y = y[:, :seqlen]

            x = x.to(device)
            y = y.to(device)

            if y.shape == x.shape and _is_attn_mask_like(y):
                return {"input_ids": x, "attention_mask": y}

            attn = torch.ones_like(x, dtype=torch.long)
            return {"input_ids": x, "attention_mask": attn, "labels": y}

        return None

    # 3) tensor
    if torch.is_tensor(batch):
        input_ids = _ensure_2d(batch)
        if input_ids.size(1) > seqlen:
            input_ids = input_ids[:, :seqlen]
        input_ids = input_ids.to(device)
        attn = torch.ones_like(input_ids, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attn}

    return None


def _pack_text_lines_to_batches(
    lines: Iterator[str],
    tok: AutoTokenizer,
    seqlen: int,
    batch_size: int,
    device: str,
    max_batches: Optional[int],
) -> Iterator[Dict[str, torch.Tensor]]:
    buf: List[int] = []
    made = 0
    cur_batch: List[torch.Tensor] = []

    for s in lines:
        if not isinstance(s, str):
            continue
        if not s.strip():
            continue
        ids = tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        if not ids:
            continue
        buf.extend(ids)

        while len(buf) >= seqlen:
            chunk = torch.tensor(buf[:seqlen], dtype=torch.long)
            buf = buf[seqlen:]

            cur_batch.append(chunk)
            if len(cur_batch) == batch_size:
                input_ids = torch.stack(cur_batch, dim=0).to(device)
                attn = torch.ones_like(input_ids, dtype=torch.long)
                yield {"input_ids": input_ids, "attention_mask": attn}
                cur_batch = []
                made += 1
                if max_batches is not None and made >= max_batches:
                    return


def _iter_textfile_batches(
    tok: AutoTokenizer,
    text_file: Path,
    seqlen: int,
    batch_size: int,
    device: str,
    max_batches: Optional[int],
) -> Iterator[Dict[str, torch.Tensor]]:
    def _lines():
        with text_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line.rstrip("\n")

    return _pack_text_lines_to_batches(_lines(), tok, seqlen, batch_size, device, max_batches)


def _normalize_loader_to_batches(
    raw_loader: Any,
    tok: AutoTokenizer,
    seqlen: int,
    batch_size: int,
    device: str,
    max_batches: Optional[int],
) -> Iterator[Dict[str, torch.Tensor]]:
    if isinstance(raw_loader, str):
        return _pack_text_lines_to_batches(iter([raw_loader]), tok, seqlen, batch_size, device, max_batches)

    if isinstance(raw_loader, list) and raw_loader and isinstance(raw_loader[0], str):
        return _pack_text_lines_to_batches(iter(raw_loader), tok, seqlen, batch_size, device, max_batches)

    if torch.is_tensor(raw_loader):
        ids = raw_loader
        if ids.dim() == 2:
            ids = ids[0]

        def _gen():
            made = 0
            cur: List[torch.Tensor] = []
            total = ids.numel()
            for start in range(0, total - seqlen + 1, seqlen):
                cur.append(ids[start : start + seqlen].clone().long())
                if len(cur) == batch_size:
                    x = torch.stack(cur, dim=0).to(device)
                    attn = torch.ones_like(x, dtype=torch.long)
                    yield {"input_ids": x, "attention_mask": attn}
                    cur = []
                    made += 1
                    if max_batches is not None and made >= max_batches:
                        return

        return _gen()

    it = iter(raw_loader)

    try:
        first = next(it)
    except StopIteration:
        return iter([])

    if isinstance(first, str):
        def _lines():
            yield first
            for x in it:
                if isinstance(x, str):
                    yield x
        return _pack_text_lines_to_batches(_lines(), tok, seqlen, batch_size, device, max_batches)

    def _batches():
        made = 0
        b0 = _extract_batch(first, tok, device, seqlen)
        if b0 is not None:
            yield b0
            made += 1
            if max_batches is not None and made >= max_batches:
                return

        for b in it:
            bb = _extract_batch(b, tok, device, seqlen)
            if bb is None:
                continue
            yield bb
            made += 1
            if max_batches is not None and made >= max_batches:
                return

    return _batches()


# -----------------------------
# PPL eval
# -----------------------------
@torch.no_grad()
def eval_ppl(model, loader: Iterator[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    sum_nll = 0.0
    sum_tok = 0

    for batch in loader:
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
            logits = out.logits
            V = logits.size(-1)

            loss_sum = F.cross_entropy(
                logits.float().view(-1, V),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            tok_cnt = (labels != -100).sum().item()
            sum_nll += float(loss_sum.item())
            sum_tok += int(tok_cnt)
            continue

        # case 2) no labels: next-token shift
        if input_ids.shape[1] < 2:
            continue

        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attn[:, 1:].contiguous().float()

        V = shift_logits.size(-1)
        loss_tok = F.cross_entropy(
            shift_logits.float().view(-1, V),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels).float()

        sum_nll += float((loss_tok * shift_mask).sum().item())
        sum_tok += int(shift_mask.sum().item())

    if sum_tok == 0:
        return {"mean_nll": float("nan"), "ppl": float("nan"), "tokens": 0}

    mean_nll = sum_nll / sum_tok
    ppl = math.exp(mean_nll)
    return {"mean_nll": mean_nll, "ppl": ppl, "tokens": sum_tok}


def _load_model(base_model: str, dtype: torch.dtype, device: str):
    """Falcon 모델 로드 (trust_remote_code 없이 native HF 사용)"""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    except TypeError:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=dtype, low_cpu_mem_usage=True
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, dtype=dtype, low_cpu_mem_usage=True
            )
    return model.to(device)


def _pick_split(obj: Any, split: str) -> Any:
    if isinstance(obj, dict):
        if split in obj:
            return obj[split]
        for k in (split, "test", "validation", "val", "train"):
            if k in obj:
                return obj[k]
        return next(iter(obj.values()))

    if isinstance(obj, tuple) and len(obj) >= 2:
        cand = obj[1] if split in ("test", "validation", "val") else obj[0]
        other = obj[0] if cand is obj[1] else obj[1]
        if isinstance(cand, str) and not isinstance(other, str):
            return other
        return cand

    return obj


def _apply_lora_no_merge(model, adapter_path: str):
    if adapter_path is None:
        return None
    if PeftModel is None:
        raise RuntimeError("peft import 실패. (pip install peft) 후 다시 실행해 주세요.")
    model_lora = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model_lora.eval()
    return model_lora


def _parse_lora_list(spec: Optional[str]) -> List[str]:
    if not spec:
        return []
    return [p.strip() for p in spec.split(",") if p.strip()]


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

    ap.add_argument("--text_file", default=None, help="txt 파일로 평가 (가장 안정적)")

    # LoRA (no-merge) 옵션
    ap.add_argument("--lora_A", default=None)
    ap.add_argument("--lora_AB", default=None)
    ap.add_argument("--lora_FULL", default=None)

    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def make_raw_loader():
        if args.text_file is not None:
            return _iter_textfile_batches(
                tok=tok,
                text_file=Path(args.text_file),
                seqlen=args.seqlen,
                batch_size=args.batch_size,
                device=args.device,
                max_batches=args.max_batches,
            )

        if GET_LOADERS is None:
            raise RuntimeError(
                "get_loaders import 실패. --text_file로 평가하거나 get_loaders 경로를 맞춰줘야 합니다."
            )

        try:
            raw = GET_LOADERS(
                args.dataset, tok, seqlen=args.seqlen,
                nsamples=args.nsamples, seed=args.seed, split=args.split,
            )
        except TypeError:
            try:
                raw = GET_LOADERS(args.dataset, tok, args.seqlen, args.nsamples, args.seed)
            except TypeError:
                raw = GET_LOADERS(
                    args.dataset, nsamples=args.nsamples, seed=args.seed,
                    seqlen=args.seqlen, tokenizer=tok,
                )

        return _pick_split(raw, args.split)

    # stage별 평가
    for stage, label in [("A", "A"), ("AB", "AB"), ("FULL", "ABC")]:
        # stage마다 fresh load (LoRA in-place 오염 방지)
        model = _load_model(args.base_model, dtype=dtype, device=args.device)
        model.eval()

        passlayer_return_tuple = _detect_layer_return_tuple(model)
        mgr = DynamicStageManager(
            model=model,
            bundles_dir=Path(args.bundles_dir),
            device=args.device,
            dtype=dtype,
            passlayer_return_tuple=passlayer_return_tuple,
        )
        mgr.set_stage(stage)

        if stage == "A":
            print("\n=== GROUP META ===")
            print(mgr.stage_meta())
            print(f"device={args.device} dtype={args.dtype}\n")

        # 배치 캐싱 (base vs lora 공정 비교)
        raw_loader = make_raw_loader()
        batches = list(
            _normalize_loader_to_batches(
                raw_loader=raw_loader,
                tok=tok,
                seqlen=args.seqlen,
                batch_size=args.batch_size,
                device=args.device,
                max_batches=args.max_batches,
            )
        )

        # 1) base ppl
        m_base = eval_ppl(model, iter(batches))
        print(
            f"[{label}] BASE ppl={m_base['ppl']:.6f} | "
            f"mean_nll={m_base['mean_nll']:.6f} | tokens={m_base['tokens']}"
        )

        # 2) lora ppl (no-merge)
        lora_spec = {"A": args.lora_A, "AB": args.lora_AB, "FULL": args.lora_FULL}[stage]
        lora_paths = _parse_lora_list(lora_spec)

        for p in lora_paths:
            model_lora = _apply_lora_no_merge(model, p)
            m_lora = eval_ppl(model_lora, iter(batches))
            print(
                f"[{label}] LoRA({Path(p).name}) ppl={m_lora['ppl']:.6f} | "
                f"mean_nll={m_lora['mean_nll']:.6f} | tokens={m_lora['tokens']}"
            )
            del model_lora

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()