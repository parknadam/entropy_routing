#!/usr/bin/env python3
"""
Progressive Model Loading Test (Merged Version)
=================================================
Stage A:     Full 32-layer skeleton loaded, layers 21-28 replaced with PassLayer
Stage A+B:   + B_merged layers (21-24) restored
Stage A+B+C: + C_merged layers (25-28) restored → full 32-layer model

Each stage generates a response for "Tell me about Stuttgart."
"""

import torch
import torch.nn as nn
import json
import time
from transformers import AutoModelForCausalLM, LlamaTokenizerFast
from safetensors.torch import load_file

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

BASE = "/home/devewha/entropy_routing/merged_models_llama_7b"
A_DIR = f"{BASE}/A_merged"
B_DIR = f"{BASE}/B_merged"
C_DIR = f"{BASE}/C_merged"

PROMPT = "Tell me about Paris."
MAX_NEW_TOKENS = 128


# ──────────────────────────────────────────────
# PassLayer: identity (skip) layer
# ──────────────────────────────────────────────
class PassLayer(nn.Module):
    """
    Identity layer — passes hidden_states through unchanged.
    Replaces pruned decoder layers so the model can still run
    without those layers contributing to the computation.
    """

    def forward(self, hidden_states, *args, **kwargs):
        # transformers 4.57+: decoder_layer returns tensor directly, not tuple
        return hidden_states


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_bundle_into_layer(model, bundle_dir, layer_indices, device):
    """
    Load individual layer safetensors from a bundle directory
    into the corresponding model layers.
    """
    for idx in layer_indices:
        sf_path = f"{bundle_dir}/layer_{int(idx):03d}.safetensors"
        weights = load_file(sf_path, device=str(device))

        layer = model.model.layers[idx]
        missing, unexpected = layer.load_state_dict(weights, strict=False)
        if unexpected:
            print(f"  [!] Layer {idx}: unexpected keys {unexpected}")
        print(f"  Loaded layer {idx} weights from {sf_path}")


@torch.no_grad()
def generate_text(model, tokenizer, prompt, device, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy decoding with use_cache=False to avoid KV-cache issues with PassLayer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
    )
    elapsed = time.time() - t0

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    n_tokens = len(new_ids)
    return text, elapsed, n_tokens


def print_response(stage_name, text, elapsed, n_tokens):
    print(f"\n{'─'*70}")
    print(f"  {stage_name}")
    print(f"{'─'*70}")
    print(text)
    print(f"{'─'*70}")
    print(f"  tokens={n_tokens}  time={elapsed:.2f}s  "
          f"tok/s={n_tokens/elapsed:.1f}" if elapsed > 0 else "")
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    # ── Load layer indices from bundle_meta.json ──
    with open(f"{B_DIR}/bundle_meta.json") as f:
        b_layers = json.load(f)["layer_indices"]
    with open(f"{C_DIR}/bundle_meta.json") as f:
        c_layers = json.load(f)["layer_indices"]
    dropped = sorted(b_layers + c_layers)

    print("=" * 70)
    print("  Progressive Model Loading Test (Merged)")
    print("=" * 70)
    print(f"  Dropped layers:    {dropped}")
    print(f"  Bundle B layers:   {b_layers}")
    print(f"  Bundle C layers:   {c_layers}")
    print(f"  Prompt: {PROMPT}")
    print("=" * 70)

    # ── 1. Load tokenizer ──────────────────────
    print("\n[1/5] Loading tokenizer ...")
    tokenizer = LlamaTokenizerFast.from_pretrained(A_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Load model (full 32 layers, layers 21-28 are init weights) ──
    print("[2/5] Loading model from A_merged ...")
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        A_DIR,
        dtype=torch.float16,
    ).to(device)
    model.eval()
    print(f"  Model loaded on {device}  "
          f"(layers={model.config.num_hidden_layers})")

    # ── 3. Save original layers 21-28, then replace with PassLayer ──
    print("[3/5] Replacing dropped layers with PassLayer ...")
    original_layers = {}
    for idx in dropped:
        original_layers[idx] = model.model.layers[idx]
        model.model.layers[idx] = PassLayer().to(device)
    print(f"  Layers {dropped} → PassLayer")

    # ═══════════════════════════════════════════
    # Stage A  (24 active layers)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Stage A : 24 active layers + 8 PassLayers")
    print("=" * 70)

    text_a, time_a, ntok_a = generate_text(model, tokenizer, PROMPT, device)
    print_response("Stage A Response", text_a, time_a, ntok_a)

    # ═══════════════════════════════════════════
    # Stage A+B  (28 active layers)
    # ═══════════════════════════════════════════
    print("=" * 70)
    print("  Stage A+B : Loading B_merged (layers 21-24)")
    print("=" * 70)

    print("[4/5] Restoring layers 21-24 and loading B_merged weights ...")
    for idx in b_layers:
        model.model.layers[idx] = original_layers[idx]
    load_bundle_into_layer(model, B_DIR, b_layers, device)

    text_ab, time_ab, ntok_ab = generate_text(model, tokenizer, PROMPT, device)
    print_response("Stage A+B Response", text_ab, time_ab, ntok_ab)

    # ═══════════════════════════════════════════
    # Stage A+B+C  (32 active layers — full model)
    # ═══════════════════════════════════════════
    print("=" * 70)
    print("  Stage A+B+C : Loading C_merged (layers 25-28) → Full Model")
    print("=" * 70)

    print("[5/5] Restoring layers 25-28 and loading C_merged weights ...")
    for idx in c_layers:
        model.model.layers[idx] = original_layers[idx]
    load_bundle_into_layer(model, C_DIR, c_layers, device)

    text_abc, time_abc, ntok_abc = generate_text(model, tokenizer, PROMPT, device)
    print_response("Stage A+B+C Response", text_abc, time_abc, ntok_abc)

    # ═══════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'Stage':<20} {'Layers':>8} {'Tokens':>8} {'Time(s)':>10} {'Tok/s':>8}")
    print(f"  {'-'*56}")
    print(f"  {'A (pruned)':<20} {'24':>8} {ntok_a:>8} {time_a:>10.2f} {ntok_a/time_a:>8.1f}")
    print(f"  {'A+B':<20} {'28':>8} {ntok_ab:>8} {time_ab:>10.2f} {ntok_ab/time_ab:>8.1f}")
    print(f"  {'A+B+C (full)':<20} {'32':>8} {ntok_abc:>8} {time_abc:>10.2f} {ntok_abc/time_abc:>8.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
