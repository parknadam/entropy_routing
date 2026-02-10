#!/usr/bin/env python3
"""
Mistral pruning 결과(A + bundles B/C)를 순차 로드하며 응답 비교.

Stage 매핑:
- A:   pruned A 모델만
- B:   A + B layers 복원 (internal stage: AB)
- C:   A + B + C layers 복원 (internal stage: FULL)

사용 예시:
python -m mistral_prune_lora.mistral_stage_chat \
  --base_model ./25_mistral_results/pruning/A \
  --bundles_dir ./25_mistral_results/pruning/bundles \
  --device cuda:0 \
  --dtype bf16 \
  --max_input_tokens 1024 \
  --max_new_tokens 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .mistral_eval_ppl import DynamicStageManager, _detect_layer_return_tuple
except Exception:
    from mistral_prune_lora.mistral_eval_ppl import DynamicStageManager, _detect_layer_return_tuple


def _to_dtype(dtype_str: str) -> torch.dtype:
    m = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return m[dtype_str]


@torch.no_grad()
def _generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    return tokenizer.decode(gen[0], skip_special_tokens=True)


def _run_one_prompt(
    mgr: DynamicStageManager,
    model,
    tokenizer,
    prompt: str,
    args,
):
    stage_order = [("A", "A"), ("B", "AB"), ("C", "FULL")]
    print("\n" + "=" * 72)
    print(f"PROMPT: {prompt}")
    print("=" * 72)

    for label, internal_stage in stage_order:
        mgr.set_stage(internal_stage)
        out = _generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=args.device,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\n[{label} Stage Response]")
        print(out)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="Pruned A model path")
    ap.add_argument("--bundles_dir", required=True, help="Bundles root path (contains B/ and C/)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_input_tokens", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--prompt", default=None, help="If set, run once and exit")
    return ap.parse_args()


def main():
    args = parse_args()
    dtype = _to_dtype(args.dtype)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    passlayer_return_tuple = _detect_layer_return_tuple(model)
    mgr = DynamicStageManager(
        model=model,
        bundles_dir=Path(args.bundles_dir),
        device=args.device,
        dtype=dtype,
        passlayer_return_tuple=passlayer_return_tuple,
    )

    print("[Loaded Stage Metadata]")
    print(mgr.stage_meta())

    if args.prompt:
        _run_one_prompt(mgr, model, tok, args.prompt, args)
        return

    print("\n[REPL] 프롬프트를 입력하면 A->B->C 순서로 응답을 출력합니다. 종료: Ctrl+C")
    while True:
        prompt = input("\nPROMPT> ").strip()
        if not prompt:
            continue
        _run_one_prompt(mgr, model, tok, prompt, args)


if __name__ == "__main__":
    main()
