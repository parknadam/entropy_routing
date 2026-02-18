#!/usr/bin/env python3
"""
Progressive Model Loading Test (Falcon)
======================================
Stage A:     Kept layers only, dropped layers are PassLayer
Stage A+B:   + Bundle B loaded
Stage A+B+C: + Bundle C loaded -> full model

Each stage generates a response for "Tell me about Stuttgart."

# 명령어
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,3,2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m falcon_prune_lora.test_model
"""

import json
import time

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
"""
BASE = "/home/devewha/entropy_routing/falcon_results/pruning"
A_DIR = f"{BASE}/A"
B_DIR = f"{BASE}/bundles/B"
C_DIR = f"{BASE}/bundles/C"
"""
BASE = "/home/devewha/entropy_routing/merged_models_mistral_7b"
A_DIR = f"{BASE}/A_merged"
BASE = "/home/devewha/entropy_routing/falcon_results/pruning"
B_DIR = f"{BASE}/bundles/B"
C_DIR = f"{BASE}/bundles/C"


PROMPT = "Tell me about Stuttgart."
MAX_NEW_TOKENS = 128


class PassLayer(nn.Module):
    """Falcon decoder-compatible identity layer."""

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
        if use_cache:
            return (hidden_states, layer_past)
        return (hidden_states,)


def load_bundle_into_layer(model, bundle_dir, layer_indices, device):
    """
    Load per-layer safetensors from bundle directory into Falcon layers.
    Bundle keys should match layer state_dict keys.
    """
    for idx in layer_indices:
        sf_path = f"{bundle_dir}/layer_{idx:03d}.safetensors"
        weights = load_file(sf_path, device=str(device))

        layer = model.transformer.h[idx]
        missing, unexpected = layer.load_state_dict(weights, strict=False)
        if missing:
            print(f"  [!] Layer {idx}: missing keys {missing}")
        if unexpected:
            print(f"  [!] Layer {idx}: unexpected keys {unexpected}")
        print(f"  Loaded layer {idx} weights from {sf_path}")


@torch.no_grad()
def generate_text(model, tokenizer, prompt, device, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy decoding with use_cache=False for PassLayer compatibility."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.time() - t0

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    n_tokens = len(new_ids)
    return text, elapsed, n_tokens


def print_response(stage_name, text, elapsed, n_tokens):
    print(f"\n{'=' * 70}")
    print(f"  {stage_name}")
    print(f"{'=' * 70}")
    print(text)
    print(f"{'=' * 70}")
    if elapsed > 0:
        print(f"  tokens={n_tokens}  time={elapsed:.2f}s  tok/s={n_tokens/elapsed:.1f}")
    print()


def main():
    with open(f"{A_DIR}/manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    a_kept = manifest["stages"]["A"]["kept_layers"]
    a_dropped = manifest["stages"]["A"]["dropped_layers"]
    b_layers = manifest["stages"]["B"]["removed_layers"]
    c_layers = manifest["stages"]["C"]["removed_layers"]

    print("=" * 70)
    print("  Progressive Model Loading Test (Falcon)")
    print("=" * 70)
    print(f"  Kept layers (A):   {a_kept}")
    print(f"  Dropped layers:    {a_dropped}")
    print(f"  Bundle B layers:   {b_layers}")
    print(f"  Bundle C layers:   {c_layers}")
    print(f"  Prompt: {PROMPT}")
    print("=" * 70)

    print("\n[1/6] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(A_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("[2/6] Loading model from Stage A checkpoint ...")
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        A_DIR,
        dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"  Model loaded on {device} (layers={model.config.num_hidden_layers})")

    print("[3/6] Replacing dropped layers with PassLayer ...")
    original_layers = {}
    for idx in a_dropped:
        original_layers[idx] = model.transformer.h[idx]
        model.transformer.h[idx] = PassLayer().to(device)
    print(f"  Layers {a_dropped} -> PassLayer")

    print("\n" + "=" * 70)
    print("  Stage A")
    print("=" * 70)
    text_a, time_a, ntok_a = generate_text(model, tokenizer, PROMPT, device)
    print_response("Stage A Response", text_a, time_a, ntok_a)

    print("=" * 70)
    print("  Stage A+B : Loading Bundle B")
    print("=" * 70)
    print("[4/6] Restoring B layers and loading Bundle B weights ...")
    for idx in b_layers:
        model.transformer.h[idx] = original_layers[idx]
    load_bundle_into_layer(model, B_DIR, b_layers, device)

    text_ab, time_ab, ntok_ab = generate_text(model, tokenizer, PROMPT, device)
    print_response("Stage A+B Response", text_ab, time_ab, ntok_ab)

    print("=" * 70)
    print("  Stage A+B+C : Loading Bundle C -> Full Model")
    print("=" * 70)
    print("[5/6] Restoring C layers and loading Bundle C weights ...")
    for idx in c_layers:
        model.transformer.h[idx] = original_layers[idx]
    load_bundle_into_layer(model, C_DIR, c_layers, device)

    text_abc, time_abc, ntok_abc = generate_text(model, tokenizer, PROMPT, device)
    print_response("Stage A+B+C Response", text_abc, time_abc, ntok_abc)

    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'Stage':<20} {'Layers':>8} {'Tokens':>8} {'Time(s)':>10} {'Tok/s':>8}")
    print(f"  {'-' * 56}")
    print(f"  {'A (pruned)':<20} {len(a_kept):>8} {ntok_a:>8} {time_a:>10.2f} {ntok_a/time_a:>8.1f}")
    print(f"  {'A+B':<20} {len(a_kept) + len(b_layers):>8} {ntok_ab:>8} {time_ab:>10.2f} {ntok_ab/time_ab:>8.1f}")
    print(
        f"  {'A+B+C (full)':<20} "
        f"{len(a_kept) + len(b_layers) + len(c_layers):>8} "
        f"{ntok_abc:>8} {time_abc:>10.2f} {ntok_abc/time_abc:>8.1f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
