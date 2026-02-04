#!/usr/bin/env python3
"""
LoRA 어댑터를 베이스 모델에 머지하는 유틸리티

사용법:
# A
python -m prune_lora.pruning.merge_adapter \
  --base_model ./7b_results/pruning/A \
  --adapter_path ./kd_lora_results/adapters/A_lora/stageA/stageA \
  --output_dir ./merged_models/A_merged \
  --device cuda:0

# bundles
python -m prune_lora.pruning.merge_adapter \
  --base_model ./7b_results/pruning/bundles/B \
  --adapter_path ./kd_lora_results/adapters/B_lora/stageB \
  --output_dir ./merged_models/B_merged \
  --device cuda:0

또는 여러 어댑터 순차 머지:
python merge_adapter.py \
  --base_model ./7b_results/pruning/A \
  --adapter_paths ./adapters/A_lora/stageA ./adapters/B_lora/stageB \
  --output_dir ./merged_models/AB_merged
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_single_adapter(base_model_path, adapter_path, output_dir, device="cuda:0"):
    """단일 어댑터를 베이스 모델에 머지"""
    print(f"\n{'='*60}")
    print(f"Merging Adapter into Base Model")
    print(f"{'='*60}")
    print(f"Base Model: {base_model_path}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_dir}")
    
    # 베이스 모델 로드
    print("\n[1/4] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # 어댑터 로드
    print("[2/4] Loading adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # 어댑터 머지
    print("[3/4] Merging adapter into base model...")
    merged_model = model.merge_and_unload()
    
    # 저장
    print(f"[4/4] Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    
    # 토크나이저도 함께 저장
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer saved.")
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")
    
    print(f"\n✓ Merge completed: {output_dir}")
    return output_dir


def merge_multiple_adapters(base_model_path, adapter_paths, output_dir, device="cuda:0"):
    """여러 어댑터를 순차적으로 머지"""
    print(f"\n{'='*60}")
    print(f"Sequential Adapter Merging")
    print(f"{'='*60}")
    print(f"Base Model: {base_model_path}")
    print(f"Adapters: {adapter_paths}")
    print(f"Final Output: {output_dir}")
    
    current_model_path = base_model_path
    
    for i, adapter_path in enumerate(adapter_paths, 1):
        print(f"\n--- Stage {i}/{len(adapter_paths)} ---")
        
        # 베이스 모델 로드
        print(f"Loading model from {current_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            current_model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        # 어댑터 로드 및 머지
        print(f"Loading and merging adapter: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        merged_model = model.merge_and_unload()
        
        # 임시 또는 최종 저장
        if i < len(adapter_paths):
            # 중간 단계는 임시 디렉토리에 저장
            temp_dir = f"{output_dir}_temp_stage{i}"
            print(f"Saving intermediate model to {temp_dir}...")
            merged_model.save_pretrained(temp_dir)
            current_model_path = temp_dir
        else:
            # 마지막 단계는 최종 출력 디렉토리에 저장
            print(f"Saving final merged model to {output_dir}...")
            os.makedirs(output_dir, exist_ok=True)
            merged_model.save_pretrained(output_dir)
            
            # 토크나이저 저장
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model_path)
                tokenizer.save_pretrained(output_dir)
                print("Tokenizer saved.")
            except Exception as e:
                print(f"Warning: Could not save tokenizer: {e}")
        
        # 메모리 정리
        del model, merged_model
        torch.cuda.empty_cache()
    
    print(f"\n✓ All adapters merged successfully: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter(s) into base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter_path", type=str, help="Path to single adapter (for single merge)")
    parser.add_argument("--adapter_paths", type=str, nargs="+", help="Paths to multiple adapters (for sequential merge)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for merged model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (default: cuda:0)")
    
    args = parser.parse_args()
    
    # 입력 검증
    if args.adapter_path and args.adapter_paths:
        raise ValueError("Specify either --adapter_path OR --adapter_paths, not both")
    if not args.adapter_path and not args.adapter_paths:
        raise ValueError("Must specify either --adapter_path or --adapter_paths")
    
    # 머지 실행
    if args.adapter_path:
        # 단일 어댑터 머지
        merge_single_adapter(args.base_model, args.adapter_path, args.output_dir, args.device)
    else:
        # 여러 어댑터 순차 머지
        merge_multiple_adapters(args.base_model, args.adapter_paths, args.output_dir, args.device)


if __name__ == "__main__":
    main()