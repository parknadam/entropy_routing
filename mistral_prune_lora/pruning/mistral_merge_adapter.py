#!/usr/bin/env python3
"""
LoRA 어댑터를 Mistral 베이스 모델에 머지하는 유틸리티.

사용법:
# A 단일 머지
python -m mistral_prune_lora.pruning.mistral_merge_adapter \
  --base_model ./25_mistral_results/pruning/A \
  --adapter_path ./mistral_kd_lora_results/adapters/A_lora/stageA/stageA \
  --output_dir ./merged_models_mistral_7b/A_merged \
  --device cuda:0

# 여러 어댑터 순차 머지 (예: A_merged + B_lora + C_lora)
python -m mistral_prune_lora.pruning.mistral_merge_adapter \
  --base_model ./25_mistral_results/pruning/A \
  --adapter_paths ./mistral_results/adapters/A_lora/stageA ./mistral_results/adapters/B_lora/stageB \
  --output_dir ./25_mistral_results/merged_models/AB_merged \
  --device cuda:0
"""


import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _device_map_from_arg(device: str):
    if device in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return device
    return {"": device}


def merge_single_adapter(base_model_path, adapter_path, output_dir, device="cuda:0"):
    """단일 어댑터를 베이스 모델에 머지."""
    print(f"\n{'=' * 60}")
    print("Merging Adapter into Base Model (Mistral)")
    print(f"{'=' * 60}")
    print(f"Base Model: {base_model_path}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_dir}")

    print("\n[1/4] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=_device_map_from_arg(device),
        trust_remote_code=True,
    )

    print("[2/4] Loading adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("[3/4] Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    print(f"[4/4] Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer saved.")
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")

    print(f"\n✓ Merge completed: {output_dir}")
    return output_dir


def merge_multiple_adapters(base_model_path, adapter_paths, output_dir, device="cuda:0"):
    """여러 어댑터를 순차적으로 머지."""
    print(f"\n{'=' * 60}")
    print("Sequential Adapter Merging (Mistral)")
    print(f"{'=' * 60}")
    print(f"Base Model: {base_model_path}")
    print(f"Adapters: {adapter_paths}")
    print(f"Final Output: {output_dir}")

    current_model_path = base_model_path

    for i, adapter_path in enumerate(adapter_paths, 1):
        print(f"\n--- Stage {i}/{len(adapter_paths)} ---")

        print(f"Loading model from {current_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            current_model_path,
            torch_dtype=torch.float16,
            device_map=_device_map_from_arg(device),
            trust_remote_code=True,
        )

        print(f"Loading and merging adapter: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        merged_model = model.merge_and_unload()

        if i < len(adapter_paths):
            temp_dir = f"{output_dir}_temp_stage{i}"
            print(f"Saving intermediate model to {temp_dir}...")
            merged_model.save_pretrained(temp_dir)
            current_model_path = temp_dir
        else:
            print(f"Saving final merged model to {output_dir}...")
            os.makedirs(output_dir, exist_ok=True)
            merged_model.save_pretrained(output_dir)

            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model_path)
                tokenizer.save_pretrained(output_dir)
                print("Tokenizer saved.")
            except Exception as e:
                print(f"Warning: Could not save tokenizer: {e}")

        del model, merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n✓ All adapters merged successfully: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter(s) into Mistral base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter_path", type=str, help="Path to single adapter (for single merge)")
    parser.add_argument(
        "--adapter_paths",
        type=str,
        nargs="+",
        help="Paths to multiple adapters (for sequential merge)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for merged model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (default: cuda:0)")

    args = parser.parse_args()

    if args.adapter_path and args.adapter_paths:
        raise ValueError("Specify either --adapter_path OR --adapter_paths, not both")
    if not args.adapter_path and not args.adapter_paths:
        raise ValueError("Must specify either --adapter_path or --adapter_paths")

    if args.adapter_path:
        merge_single_adapter(args.base_model, args.adapter_path, args.output_dir, args.device)
    else:
        merge_multiple_adapters(args.base_model, args.adapter_paths, args.output_dir, args.device)


if __name__ == "__main__":
    main()
