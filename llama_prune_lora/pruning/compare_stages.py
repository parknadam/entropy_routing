# prune_lora/pruning/compare_stages.py
from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
except Exception:
    LlamaDecoderLayer = None


# -----------------------------
# PassLayer
# -----------------------------
class LlamaPassLayer(nn.Module):
    """
    LLaMA forward loop이 tuple을 기대하는 버전이 많아서 기본은 tuple 반환.
    (문제 생기면 --pass_return_tensor 옵션으로 Tensor만 반환하도록 전환)
    """
    def __init__(self, return_tensor: bool = False):
        super().__init__()
        self.return_tensor = return_tensor

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
        if self.return_tensor:
            return hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (None,)
        if use_cache:
            outputs = outputs + (past_key_value,)
        return outputs


def _get_llama_layers(model) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        return model.model.model.layers
    raise RuntimeError("LLaMA layers 경로를 찾지 못했어요. (예: model.model.layers)")


# -----------------------------
# bundles scan
# -----------------------------
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


def _maybe_shift_to_zero_based(B: Dict[int, Path], C: Dict[int, Path], num_layers: int) -> Tuple[Dict[int, Path], Dict[int, Path], int]:
    all_idx = sorted(set(B.keys()) | set(C.keys()))
    if not all_idx:
        return B, C, 0

    if all(0 <= i < num_layers for i in all_idx):
        return B, C, 0

    # 1-based (1..num_layers)로 보이면 -1 shift
    if all(1 <= i <= num_layers for i in all_idx):
        return {i - 1: p for i, p in B.items()}, {i - 1: p for i, p in C.items()}, -1

    raise RuntimeError(
        f"bundles index가 base_model 레이어 수와 불일치합니다. "
        f"(num_layers={num_layers}, min={min(all_idx)}, max={max(all_idx)})"
    )


def _strip_layer_prefix(sd: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, torch.Tensor]:
    needles = [
        f"model.layers.{layer_idx}.",
        f"model.model.layers.{layer_idx}.",
        f"layers.{layer_idx}.",
    ]
    out = {}
    for k, v in sd.items():
        nk = k
        for nd in needles:
            if nd in nk:
                nk = nk.split(nd, 1)[1]
                break
        out[nk] = v
    return out


# -----------------------------
# Stage manager: A / AB / FULL(ABC)
# -----------------------------
class DynamicStageManager:
    def __init__(self, model, bundles_dir: Path, device: str, dtype: torch.dtype, pass_return_tensor: bool):
        if LlamaDecoderLayer is None:
            raise RuntimeError("transformers에서 LlamaDecoderLayer import 실패")

        self.model = model
        self.layers = _get_llama_layers(model)
        self.device = device
        self.dtype = dtype
        self.num_layers = len(self.layers)
        self.pass_return_tensor = pass_return_tensor

        self.B_map_raw = _build_layer_map(bundles_dir / "B")
        self.C_map_raw = _build_layer_map(bundles_dir / "C")
        self.B_map, self.C_map, self.index_shift = _maybe_shift_to_zero_based(self.B_map_raw, self.C_map_raw, self.num_layers)

        self.B_idx = sorted(self.B_map.keys())
        self.C_idx = sorted(self.C_map.keys())
        self.removed = sorted(set(self.B_idx) | set(self.C_idx))

        # 레이어별 디바이스(단일 device면 다 같고, 나중에 확장 여지)
        self.layer_device: Dict[int, torch.device] = {}
        base_dev = torch.device(device) if device != "cpu" else torch.device("cpu")
        for i, layer in enumerate(self.layers):
            p = next(layer.parameters(), None)
            self.layer_device[i] = p.device if p is not None else base_dev

        # 현재 pass 상태 추적(클래스 isinstance에 의존 X)
        self.current_pass = set()

        # 초기 A로 세팅
        self.set_stage("A")

    def stage_meta(self) -> Dict:
        return {
            "num_layers": self.num_layers,
            "index_shift_applied": self.index_shift,
            "B": self.B_idx,
            "C": self.C_idx,
            "removed": self.removed,
        }

    def _bundle_path(self, layer_i: int) -> Optional[Path]:
        return self.B_map.get(layer_i) or self.C_map.get(layer_i)

    def _restore_one_layer(self, layer_i: int):
        p = self._bundle_path(layer_i)
        if p is None:
            raise FileNotFoundError(f"bundle layer_{layer_i}.safetensors not found in B/C")

        try:
            new_layer = LlamaDecoderLayer(self.model.config, layer_i)
        except TypeError:
            new_layer = LlamaDecoderLayer(self.model.config)

        dev = self.layer_device[layer_i]
        new_layer = new_layer.to(dev, dtype=self.dtype)

        sd = load_file(str(p), device="cpu")
        sd = _strip_layer_prefix(sd, layer_i)
        new_layer.load_state_dict(sd, strict=False)

        old = self.layers[layer_i]
        self.layers[layer_i] = new_layer
        del old

    def _pass_one_layer(self, layer_i: int):
        dev = self.layer_device[layer_i]
        old = self.layers[layer_i]
        self.layers[layer_i] = LlamaPassLayer(return_tensor=self.pass_return_tensor).to(dev)
        del old

    def set_stage(self, stage: str):
        stage = stage.upper()
        if stage not in ("A", "AB", "FULL"):
            raise ValueError("stage must be A / AB / FULL")

        if stage == "A":
            pass_set = set(self.removed)          # B,C 모두 pass
        elif stage == "AB":
            pass_set = set(self.C_idx)            # C만 pass, B는 restore
        else:
            pass_set = set()                      # B,C 모두 restore

        # removed에 대해서만 전환
        for i in self.removed:
            if i in pass_set:
                if i not in self.current_pass:
                    self._pass_one_layer(i)
                    self.current_pass.add(i)
            else:
                if i in self.current_pass:
                    self._restore_one_layer(i)
                    self.current_pass.remove(i)

        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()


# -----------------------------
# chat formatting
# -----------------------------
def _is_chat(tokenizer, mode: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    # auto
    return hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None


def _encode(tokenizer, prompt: str, device: str, max_length: int, chat: bool, system_prompt: str, add_generation_prompt: bool):
    if chat and hasattr(tokenizer, "apply_chat_template"):
        msgs = []
        if system_prompt.strip():
            msgs.append({"role": "system", "content": system_prompt.strip()})
        msgs.append({"role": "user", "content": prompt})
        input_ids = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        attn = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attn}

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def prompt_ppl(model, tok, prompt: str, device: str, max_length: int, chat: bool, system_prompt: str) -> float:
    enc = _encode(tok, prompt, device, max_length, chat, system_prompt, add_generation_prompt=False)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    if int(attn.sum().item()) < 2:
        return float("inf")

    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits
    logits_use = logits[:, :-1, :]
    target = input_ids[:, 1:]
    mask = attn[:, 1:].to(logits_use.dtype)
    denom = mask.sum().clamp_min(1.0)

    log_probs = F.log_softmax(logits_use, dim=-1)
    token_logp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    nll = (-(token_logp) * mask).sum() / denom
    nll = float(nll.item())
    return float(math.exp(nll)) if nll < 80 else float("inf")


@torch.no_grad()
def generate(model, tok, prompt: str, device: str, max_length: int, max_new_tokens: int, chat: bool, system_prompt: str) -> str:
    enc = _encode(tok, prompt, device, max_length, chat, system_prompt, add_generation_prompt=True)
    input_ids = enc["input_ids"]
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
        pad_token_id=tok.eos_token_id,
    )
    new_tokens = gen[0, input_ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def _load_model(base_model: str, dtype: torch.dtype, device: str):
    try:
        m = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, low_cpu_mem_usage=True)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, low_cpu_mem_usage=True)
    return m.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="A 모델 경로 (프루닝 결과 A)")
    ap.add_argument("--bundles_dir", required=True, help="bundles dir (B/, C/ 포함)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--chat", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--system_prompt", default="")
    ap.add_argument("--stages", default="A,AB,FULL", help="예: A,AB,FULL")
    ap.add_argument("--pass_return_tensor", action="store_true", help="PassLayer가 Tensor만 반환 (문제 생길 때만)")
    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = _load_model(args.base_model, dtype=dtype, device=args.device)
    model.eval()

    mgr = DynamicStageManager(
        model=model,
        bundles_dir=Path(args.bundles_dir),
        device=args.device,
        dtype=dtype,
        pass_return_tensor=args.pass_return_tensor,
    )

    chat = _is_chat(tok, args.chat)
    print("=== META ===")
    print(mgr.stage_meta())
    print(f"chat={chat} device={args.device} dtype={args.dtype}\n")

    if args.prompt is None:
        prompt = input("PROMPT> ").strip()
    else:
        prompt = args.prompt.strip()

    stages = [s.strip().upper() for s in args.stages.split(",") if s.strip()]
    for st in stages:
        print("\n" + "=" * 80)
        print(f"[STAGE {st}]")

        mgr.set_stage(st)

        t0 = time.perf_counter()
        ppl = prompt_ppl(model, tok, prompt, args.device, args.max_length, chat, args.system_prompt)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        out = generate(model, tok, prompt, args.device, args.max_length, args.max_new_tokens, chat, args.system_prompt)
        t3 = time.perf_counter()

        print(f"prompt_ppl: {ppl:.4f} | ppl_time: {(t1 - t0):.3f}s")
        print(f"gen_time: {(t3 - t2):.3f}s | max_new_tokens={args.max_new_tokens}")
        print("--- output (new tokens only) ---")
        print(out)

    print("\nDONE.")


if __name__ == "__main__":
    main()
