#!/usr/bin/env python3
"""
LoRA 어댑터를 Llama 베이스 모델에 머지하는 유틸리티.

핵심 동작:
  1. pruned A 모델 로드 (config: 32 layers, 실제 가중치: 일부 레이어 드롭 가능)
  2. LoRA 어댑터 적용 & 머지
  3. manifest.json에서 dropped layer 인덱스를 읽어 PassLayer로 복원
     → 드롭된 레이어의 랜덤 가중치가 저장되지 않아 모델 크기 유지
  4. tokenizer + config + generation_config 포함 완전한 모델로 저장

사용법:
# A 단일 머지
python -m llama_prune_lora.pruning.llama_merge_adapter \
  --base_model ./7b_results/pruning/A \
  --adapter_path ./kd_lora_results/adapters/A_lora/stageA/stageA \
  --output_dir ./merged_models_llama_7b/A_merged \
  --device cuda:0

# 여러 어댑터 순차 머지 (예: A_merged + B_lora + C_lora)
python -m llama_prune_lora.pruning.llama_merge_adapter \
  --base_model ./7b_results/pruning/A \
  --adapter_paths ./kd_lora_results/adapters/A_lora/stageA ./kd_lora_results/adapters/B_lora/stageB \
  --output_dir ./merged_models_llama_7b/AB_merged \
  --device cuda:0

# 원본 모델 토크나이저를 사용하고 싶을 때
python -m llama_prune_lora.pruning.llama_merge_adapter \
  --base_model ./7b_results/pruning/A \
  --adapter_path ./adapters/A_lora/stageA \
  --output_dir ./merged_models_llama_7b/A_merged \
  --tokenizer_path meta-llama/Llama-2-7b-hf
"""

import argparse
import json
import os
import shutil

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class PassLayer(nn.Module):
    """
    드롭된 레이어를 대체하는 파라미터-프리 패스 레이어.
    forward 시그니처는 LlamaDecoderLayer/MistralDecoderLayer와 호환.
    """

    def __init__(self, hidden_size: int = 0):
        super().__init__()
        self.hidden_size = hidden_size

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
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs


def _read_dropped_layers_from_manifest(base_model_path: str):
    """
    manifest.json에서 프루닝으로 제거된 레이어 인덱스를 읽습니다.
    Returns: sorted list of dropped layer indices, or empty list
    """
    manifest_path = os.path.join(base_model_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"  ⚠ manifest.json not found in {base_model_path}")
        return []

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"  ⚠ manifest.json 파싱 실패: {e}")
        return []

    stages = manifest.get("stages", {})
    dropped = stages.get("A", {}).get("dropped_layers", [])

    if not dropped:
        b_removed = stages.get("B", {}).get("removed_layers", [])
        c_removed = stages.get("C", {}).get("removed_layers", [])
        dropped = sorted(set(b_removed + c_removed))

    if not dropped:
        dropped = manifest.get("simdrop", {}).get("removed_layers", [])

    dropped = sorted(set(int(i) for i in dropped))
    return dropped


def _get_model_layers(model) -> nn.ModuleList:
    """모델의 decoder layers에 접근합니다."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        return model.model.decoder.layers
    raise RuntimeError("Cannot find model layers (expected model.model.layers)")


def _restore_passlayers(model, dropped_indices: list):
    """
    머지 후 드롭된 레이어 위치에 PassLayer를 복원합니다.
    save_pretrained() 시 해당 위치의 가중치가 저장되지 않도록 합니다.
    """
    if not dropped_indices:
        return model

    layers = _get_model_layers(model)
    hidden_size = model.config.hidden_size
    restored = []

    for idx in dropped_indices:
        if 0 <= idx < len(layers):
            old = layers[idx]
            layers[idx] = PassLayer(hidden_size)
            del old
            restored.append(idx)

    if restored:
        print(f"  ✓ PassLayer 복원: {len(restored)}개 레이어 {restored}")
    return model


def _resolve_tokenizer(*candidate_paths: str, trust_remote_code: bool = True) -> AutoTokenizer:
    """여러 후보 경로를 순서대로 시도하여 토크나이저를 로드합니다."""
    errors = []
    for path in candidate_paths:
        if path is None:
            continue
        try:
            resolved = os.path.abspath(path) if os.path.exists(path) else path
            tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=trust_remote_code)
            print(f"  ✓ Tokenizer loaded from: {path}")
            return tok
        except Exception as e:
            errors.append((path, str(e)))
            continue

    msg_lines = ["Tokenizer를 로드할 수 없습니다. 시도한 경로:"]
    for p, err in errors:
        msg_lines.append(f"  - {p}: {err}")
    msg_lines.append("--tokenizer_path 로 토크나이저 경로를 명시해 주세요.")
    raise RuntimeError("\n".join(msg_lines))


def _find_tokenizer_candidates(base_model_path: str, adapter_path: str = None, tokenizer_path: str = None):
    """토크나이저를 찾기 위한 후보 경로 목록을 생성합니다."""
    candidates = []

    if tokenizer_path:
        candidates.append(tokenizer_path)

    candidates.append(base_model_path)

    orig_cfg_dir = os.path.join(base_model_path, "original_config")
    if os.path.isdir(orig_cfg_dir):
        candidates.append(orig_cfg_dir)

    if adapter_path:
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.isfile(adapter_config_path):
            try:
                with open(adapter_config_path, "r") as f:
                    adapter_cfg = json.load(f)
                base_name = adapter_cfg.get("base_model_name_or_path", None)
                if base_name:
                    candidates.append(base_name)
            except Exception:
                pass

    manifest_path = os.path.join(base_model_path, "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            original_model = manifest.get("base_model", None)
            if original_model:
                candidates.append(original_model)
        except Exception:
            pass

    return candidates


def _device_map_from_arg(device: str):
    if device in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return device
    return {"": device}


def _save_complete_model(model, tokenizer, output_dir: str, base_model_path: str = None):
    """모델, 토크나이저, config, generation_config를 모두 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)

    print("  Saving model weights...")
    model.save_pretrained(output_dir, safe_serialization=True)

    print("  Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(output_dir)
            print("  Saved generation_config.")
    except Exception:
        if base_model_path:
            gen_cfg_file = os.path.join(base_model_path, "generation_config.json")
            if os.path.isfile(gen_cfg_file):
                shutil.copy2(gen_cfg_file, os.path.join(output_dir, "generation_config.json"))
                print("  Copied generation_config from base model.")

    if base_model_path:
        manifest_src = os.path.join(base_model_path, "manifest.json")
        if os.path.isfile(manifest_src):
            shutil.copy2(manifest_src, os.path.join(output_dir, "manifest.json"))
            print("  Copied manifest.json (pruning info).")

    saved_files = sorted(os.listdir(output_dir))
    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in saved_files
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"  Saved files: {saved_files}")
    print(f"  Total size: {total_bytes / (1024**3):.2f} GB")


def _verify_merged_model(output_dir: str):
    """머지된 모델이 정상적으로 로드되는지 검증합니다."""
    print(f"\n[Verify] Checking merged model at {output_dir}...")

    abs_dir = os.path.abspath(output_dir)
    files = os.listdir(abs_dir) if os.path.isdir(abs_dir) else []

    tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
    weight_patterns = ["model.safetensors", "model-00001-of-"]

    has_config = "config.json" in files
    has_tokenizer = any(tf in files for tf in tokenizer_files)
    has_weights = any(any(wp in f for wp in weight_patterns) for f in files)
    has_manifest = "manifest.json" in files

    print(f"  config.json:   {'✓' if has_config else '✗ MISSING'}")
    print(f"  tokenizer:     {'✓' if has_tokenizer else '✗ MISSING'}")
    print(f"  weights:       {'✓' if has_weights else '✗ MISSING'}")
    print(f"  manifest.json: {'✓' if has_manifest else '- (optional)'}")

    if has_config and has_tokenizer and has_weights:
        print("  ✓ 모든 필수 파일 존재. from_pretrained() 로 바로 로드 가능.")
        return True

    print("  ✗ 일부 파일 누락. 확인 필요.")
    return False


def merge_single_adapter(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    device: str = "cuda:0",
    tokenizer_path: str = None,
    verify: bool = True,
):
    """단일 LoRA 어댑터를 베이스 모델에 머지하고 완전한 모델로 저장합니다."""
    print(f"\n{'=' * 60}")
    print("Merging Adapter into Base Model (Llama)")
    print(f"{'=' * 60}")
    print(f"Base Model:  {base_model_path}")
    print(f"Adapter:     {adapter_path}")
    print(f"Output:      {output_dir}")
    if tokenizer_path:
        print(f"Tokenizer:   {tokenizer_path}")

    print("\n[1/6] Reading manifest for dropped layers...")
    dropped_layers = _read_dropped_layers_from_manifest(base_model_path)
    if dropped_layers:
        print(f"  Dropped layers: {dropped_layers} ({len(dropped_layers)}개)")
    else:
        print("  No dropped layers found (non-pruned model or missing manifest)")

    print("\n[2/6] Loading tokenizer...")
    tok_candidates = _find_tokenizer_candidates(base_model_path, adapter_path, tokenizer_path)
    tokenizer = _resolve_tokenizer(*tok_candidates)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token = eos_token ({tokenizer.eos_token})")

    print("\n[3/6] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=_device_map_from_arg(device),
        trust_remote_code=True,
    )
    n_layers = base_model.config.num_hidden_layers
    print(f"  Loaded: {n_layers} layers, {sum(p.numel() for p in base_model.parameters())/1e6:.1f}M params")

    print("\n[4/6] Loading and merging adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()
    print("  ✓ LoRA weights fused into base model")

    print("\n[5/6] Restoring PassLayers for dropped layers...")
    merged_model = _restore_passlayers(merged_model, dropped_layers)

    print(f"\n[6/6] Saving complete merged model to {output_dir}...")
    _save_complete_model(merged_model, tokenizer, output_dir, base_model_path)

    if verify:
        del merged_model, model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _verify_merged_model(output_dir)

    print(f"\n{'=' * 60}")
    print(f"✓ Merge completed: {output_dir}")
    print(f"{'=' * 60}")
    return output_dir


def merge_multiple_adapters(
    base_model_path: str,
    adapter_paths: list,
    output_dir: str,
    device: str = "cuda:0",
    tokenizer_path: str = None,
    verify: bool = True,
):
    """여러 LoRA 어댑터를 순차적으로 머지합니다."""
    print(f"\n{'=' * 60}")
    print("Sequential Adapter Merging (Llama)")
    print(f"{'=' * 60}")
    print(f"Base Model:  {base_model_path}")
    print(f"Adapters:    {adapter_paths}")
    print(f"Final Output: {output_dir}")

    print("\n[0a] Reading manifest for dropped layers...")
    dropped_layers = _read_dropped_layers_from_manifest(base_model_path)
    if dropped_layers:
        print(f"  Dropped layers: {dropped_layers} ({len(dropped_layers)}개)")

    print("\n[0b] Loading tokenizer...")
    first_adapter = adapter_paths[0] if adapter_paths else None
    tok_candidates = _find_tokenizer_candidates(base_model_path, first_adapter, tokenizer_path)
    tokenizer = _resolve_tokenizer(*tok_candidates)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token = eos_token ({tokenizer.eos_token})")

    current_model_path = base_model_path
    temp_dirs = []

    for i, adapter_path in enumerate(adapter_paths, 1):
        is_last = i == len(adapter_paths)
        print(f"\n{'─' * 40}")
        print(f"Stage {i}/{len(adapter_paths)}: {os.path.basename(adapter_path)}")
        print(f"{'─' * 40}")

        print(f"  Loading model from {current_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            current_model_path,
            torch_dtype=torch.float16,
            device_map=_device_map_from_arg(device),
            trust_remote_code=True,
        )

        print(f"  Loading adapter: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        merged_model = model.merge_and_unload()
        print(f"  ✓ Merged stage {i}")

        print("  Restoring PassLayers...")
        merged_model = _restore_passlayers(merged_model, dropped_layers)

        if is_last:
            print(f"\n  Saving final merged model to {output_dir}...")
            _save_complete_model(merged_model, tokenizer, output_dir, base_model_path)
        else:
            temp_dir = f"{output_dir}_temp_stage{i}"
            print(f"  Saving intermediate model to {temp_dir}...")
            _save_complete_model(merged_model, tokenizer, temp_dir, base_model_path)
            current_model_path = temp_dir
            temp_dirs.append(temp_dir)

        del model, merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for td in temp_dirs:
        try:
            shutil.rmtree(td)
            print(f"  Cleaned up temp dir: {td}")
        except Exception as e:
            print(f"  Warning: Could not remove temp dir {td}: {e}")

    if verify:
        _verify_merged_model(output_dir)

    print(f"\n{'=' * 60}")
    print(f"✓ All {len(adapter_paths)} adapters merged: {output_dir}")
    print(f"{'=' * 60}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter(s) into Llama base model with tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 단일 머지
  python -m llama_prune_lora.pruning.llama_merge_adapter \\
    --base_model ./pruning/A \\
    --adapter_path ./adapters/A_lora/stageA \\
    --output_dir ./merged/A_merged

  # 순차 머지 (A+B)
  python -m llama_prune_lora.pruning.llama_merge_adapter \\
    --base_model ./pruning/A \\
    --adapter_paths ./adapters/A_lora/stageA ./adapters/B_lora/stageB \\
    --output_dir ./merged/AB_merged

  # 원본 모델 토크나이저 명시
  python -m llama_prune_lora.pruning.llama_merge_adapter \\
    --base_model ./pruning/A \\
    --adapter_path ./adapters/A_lora/stageA \\
    --output_dir ./merged/A_merged \\
    --tokenizer_path meta-llama/Llama-2-7b-hf
        """,
    )
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model (pruned A model)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to single adapter (for single merge)")
    parser.add_argument(
        "--adapter_paths",
        type=str,
        nargs="+",
        default=None,
        help="Paths to multiple adapters (for sequential merge)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for merged model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (default: cuda:0)")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Explicit tokenizer path (e.g., meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument("--no_verify", action="store_true", help="Skip post-merge verification")

    args = parser.parse_args()

    if args.adapter_path and args.adapter_paths:
        raise ValueError("Specify either --adapter_path OR --adapter_paths, not both")
    if not args.adapter_path and not args.adapter_paths:
        raise ValueError("Must specify either --adapter_path or --adapter_paths")

    if args.adapter_path:
        merge_single_adapter(
            base_model_path=args.base_model,
            adapter_path=args.adapter_path,
            output_dir=args.output_dir,
            device=args.device,
            tokenizer_path=args.tokenizer_path,
            verify=not args.no_verify,
        )
    else:
        merge_multiple_adapters(
            base_model_path=args.base_model,
            adapter_paths=args.adapter_paths,
            output_dir=args.output_dir,
            device=args.device,
            tokenizer_path=args.tokenizer_path,
            verify=not args.no_verify,
        )


if __name__ == "__main__":
    main()
