# qwen_prune_lora/pruning/qwen_layeronly_drop.py
# Qwen 모델용 완전한 레이어 드랍 구현 (토크나이저 오류 대응 버전)
"""
사용 예시:

python -m qwen_prune_lora.pruning.qwen_layeronly_drop \
  --model Qwen/Qwen-7B \
  --device cuda:0 \
  --drop_frac 0.10 \
  --keep_last_layer \
  --nsamples 64 \
  --seqlen 1024 \
  --max_batches 32 \
  --save_dir ./qwen_results/pruning/A \
  --save_removed_dir ./qwen_results/pruning/bundles
"""

import argparse
import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 상대 import 사용
from .qwen_simdrop import choose_block_to_drop, drop_consecutive_layers
from .qwen_bundler import export_two_bundles
from .data import get_loaders


def _check_dependencies():
    """필수 의존성 확인"""
    missing = []
    
    try:
        import tiktoken
    except ImportError:
        missing.append("tiktoken")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import safetensors
    except ImportError:
        missing.append("safetensors")
    
    if missing:
        print("❌ Missing dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall with:")
        print(f"   pip install {' '.join(missing)}")
        sys.exit(1)


def _resolve_embed_device(model, fallback_device: str):
    """Qwen 모델의 임베딩 레이어 디바이스 확인"""
    embed_key = "transformer.wte"
    if (
        hasattr(model, "hf_device_map")
        and isinstance(model.hf_device_map, dict)
        and embed_key in model.hf_device_map
    ):
        dev = model.hf_device_map[embed_key]
        return torch.device(dev) if not isinstance(dev, torch.device) else dev
    return torch.device(fallback_device)


def _load_model(model_name: str, seqlen: int, device_str: str):
    """Qwen 모델 로드 - 개선된 버전"""
    
    # tiktoken 확인
    try:
        import tiktoken
        print("✓ tiktoken found")
    except ImportError:
        raise ImportError(
            "\n❌ tiktoken is required for Qwen models.\n"
            "Install: pip install tiktoken\n"
        )
    
    # 토크나이저 로드 (여러 방법 시도)
    print(f"Loading tokenizer from {model_name}...")
    tokenizer_loaded = False
    tok = None
    
    # 방법 1: 기본 AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=False,  # slow tokenizer 사용
            trust_remote_code=True,
            revision="main",
        )
        tokenizer_loaded = True
        print("✓ Tokenizer loaded (method 1: slow tokenizer)")
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        
        # 방법 2: use_fast=True
        try:
            tok = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=True,
                trust_remote_code=True,
                revision="main",
            )
            tokenizer_loaded = True
            print("✓ Tokenizer loaded (method 2: fast tokenizer)")
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            
            # 방법 3: PreTrainedTokenizerFast
            try:
                from transformers import PreTrainedTokenizerFast
                tok = PreTrainedTokenizerFast.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
                tokenizer_loaded = True
                print("✓ Tokenizer loaded (method 3: PreTrainedTokenizerFast)")
            except Exception as e3:
                print(f"Method 3 failed: {e3}")
                raise RuntimeError(
                    f"\n❌ All tokenizer loading methods failed.\n"
                    f"Please check:\n"
                    f"1. pip install tiktoken transformers-stream-generator einops\n"
                    f"2. pip install --upgrade transformers\n"
                    f"3. rm -rf ~/.cache/huggingface/modules/transformers_modules/Qwen\n"
                )
    
    if not tokenizer_loaded or tok is None:
        raise RuntimeError("Failed to load tokenizer")
    
    # 토크나이저 설정
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print("✓ Set pad_token = eos_token")
    tok.padding_side = "left"

    # 모델 로드
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=True,
    ).to(device_str)
    
    print("✓ Model loaded")
    
    model.seqlen = seqlen
    model.config.use_cache = False
    model.eval()
    return model, tok


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def _count_params_of_layers(m, idxs):
    layers_ = m.transformer.h
    s = 0
    for i_ in idxs:
        for p_ in layers_[i_].parameters(recurse=True):
            s += p_.numel()
    return s


def build_layers_map(model, out_path: str):
    """레이어 인덱스 ↔ 파라미터 키 매핑 저장"""
    layers = model.transformer.h
    sd_keys = list(model.state_dict().keys())
    prefix = "transformer.h."
    L = len(layers)

    layer_map = {str(i): [] for i in range(L)}
    non_layer_keys = []

    for k in sd_keys:
        if k.startswith(prefix):
            lid = int(k[len(prefix):].split(".", 1)[0])
            layer_map[str(lid)].append(k)
        else:
            non_layer_keys.append(k)

    payload = {
        "num_layers": L,
        "layer_prefix": prefix,
        "layers": layer_map,
        "non_layer_keys": non_layer_keys,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Qwen 모델 레이어 프루닝")
    ap.add_argument("--model", type=str, default="Qwen/Qwen-7B", help="Qwen 모델 경로")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--drop_frac", type=float, default=0.25, help="제거할 레이어 비율")
    ap.add_argument("--keep_last_layer", action="store_true", help="마지막 레이어 유지")
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max_batches", type=int, default=64)
    ap.add_argument("--save_dir", type=str, default="./A", help="A 모델 저장 경로")
    ap.add_argument("--save_removed_dir", type=str, default="./bundles", help="B/C 번들 저장 경로")
    ap.add_argument("--split_policy", type=str, choices=["half", "ratio"], default="half")
    ap.add_argument("--split_ratio", type=float, default=0.5)
    ap.add_argument("--prune_log", type=str, default="prune_log.json")
    ap.add_argument("--pass_return_tuple", dest='pass_return_tuple', action='store_true',
                    default=True, help="PassLayer tuple 반환 (기본 True)")
    ap.add_argument("--no_pass_return_tuple", dest='pass_return_tuple', action='store_false',
                    help="PassLayer tensor만 반환")
    args = ap.parse_args()

    # 의존성 확인
    print("=" * 60)
    print("Checking dependencies...")
    print("=" * 60)
    _check_dependencies()
    print("✓ All dependencies found\n")

    torch.manual_seed(args.seed)

    # [1] 모델 로드
    print("=" * 60)
    print("[1/7] Loading Qwen model")
    print("=" * 60)
    model, tokenizer = _load_model(args.model, args.seqlen, args.device)
    embed_dev = _resolve_embed_device(model, args.device)
    print(f"Model loaded on {embed_dev}\n")

    # [2] 원본 설정 저장
    print("=" * 60)
    print("[2/7] Saving original config")
    print("=" * 60)
    orig_dir = os.path.join(args.save_dir, "original_config")
    os.makedirs(orig_dir, exist_ok=True)

    try:
        model.config.to_json_file(os.path.join(orig_dir, "config.json"))
        print("✓ config.json saved")
    except Exception as e:
        print(f"⚠ config save failed: {e}")
    
    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.to_json_file(os.path.join(orig_dir, "generation_config.json"))
            print("✓ generation_config.json saved")
    except Exception as e:
        print(f"⚠ generation_config save failed: {e}")

    tokenizer.save_pretrained(orig_dir)
    print("✓ tokenizer saved")

    with open(os.path.join(orig_dir, "source.json"), "w", encoding="utf-8") as f:
        json.dump({"base_model": args.model}, f, ensure_ascii=False, indent=2)
    print("✓ source.json saved\n")

    # [3] 캘리브레이션 데이터 로드
    print("=" * 60)
    print("[3/7] Loading calibration data (C4)")
    print("=" * 60)
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        tokenizer=tokenizer,
    )
    print(f"✓ Loaded {args.nsamples} samples\n")


    # [4] 프루닝 수행
    print("=" * 60)
    print("[4/7] Finding optimal layers to prune")
    print("=" * 60)
    L_full = len(model.transformer.h)
    n = int(round(args.drop_frac * L_full))
    print(f"Total layers: {L_full}")
    print(f"Layers to remove: {n} ({args.drop_frac*100:.1f}%)")

    if n <= 0:
        print("\n⚠ drop_frac <= 0, saving original model as A")
        # ... (기존 코드)
        return

    # Angular distance 계산
    print("\nComputing angular distances...")
    
    try:
        best_ell, best_d, _L = choose_block_to_drop(
            model,
            dataloader,
            embed_dev,
            n=n,
            keep_last_layer=args.keep_last_layer,
            max_batches=args.max_batches,
        )
    except Exception as e:
        print(f"\n❌ Error during layer selection: {e}")
        print("\nPossible causes:")
        print("1. Dataloader issue - check if C4 dataset is accessible")
        print("2. Model forward pass failed")
        print("3. Not enough samples captured")
        raise
    
    # None 체크 추가
    if best_ell is None:
        raise RuntimeError(
            "choose_block_to_drop returned None. "
            "This usually means no valid layer positions were found. "
            f"Model has {L_full} layers, trying to drop {n} layers."
        )
    
    print(f"\n✓ Selected starting layer: {best_ell}")
    print(f"✓ Angular distance: {best_d:.4f}")

    # 범위 검증
    if args.keep_last_layer and best_ell + n > L_full - 1:
        old_ell = best_ell
        best_ell = max(0, L_full - 1 - n)
        print(f"⚠ Adjusted start layer: {old_ell} → {best_ell} (keep_last_layer constraint)")

    removed_indices = list(range(best_ell, best_ell + n))

    if n > 0 and args.keep_last_layer:
        assert removed_indices[-1] < (L_full - 1), f"keep_last_layer violation: removing layer {removed_indices[-1]}, but last layer is {L_full-1}"


    # 파라미터 카운트
    P_total = _count_params(model)
    P_drop = _count_params_of_layers(model, removed_indices)

    print(f"\n✓ Selected block: layers {best_ell} to {best_ell+n-1}")
    print(f"✓ Angular distance: {best_d:.4f}")
    print(f"✓ Removing {P_drop:,} / {P_total:,} parameters ({100*P_drop/P_total:.2f}%)\n")

    # [5] 번들 저장
    print("=" * 60)
    print("[5/7] Exporting bundles (B/C split)")
    print("=" * 60)
    os.makedirs(args.save_removed_dir, exist_ok=True)
    
    B_idx, C_idx = export_two_bundles(
        model=model,
        removed_indices=removed_indices,
        out_root=args.save_removed_dir,
        config=model.config,
        split_policy=args.split_policy,
        split_ratio=args.split_ratio,
    )
    print(f"✓ Bundle B: {len(B_idx)} layers → {args.save_removed_dir}/B")
    print(f"✓ Bundle C: {len(C_idx)} layers → {args.save_removed_dir}/C\n")

    # [6] 레이어 드랍 적용
    print("=" * 60)
    print("[6/7] Applying layer drop")
    print("=" * 60)
    model, _ = drop_consecutive_layers(
        model, best_ell, n, return_tuple=args.pass_return_tuple
    )
    model = model.to(embed_dev)
    if torch.cuda.is_available() and ("cuda" in str(embed_dev)):
        torch.cuda.empty_cache()

    new_depth = len(model.transformer.h)
    print(f"✓ Model depth: {L_full} → {new_depth}\n")

    # [7] A 모델 저장
    print("=" * 60)
    print("[7/7] Saving pruned model (A)")
    print("=" * 60)
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.save_dir)
    print(f"✓ Model saved to {args.save_dir}")

    # 레이어 맵 저장
    build_layers_map(model, os.path.join(args.save_dir, "layers_map.json"))
    print("✓ layers_map.json saved")

    kept_idx = [i for i in range(L_full) if i not in removed_indices]

    # 프루닝 로그
    prune_log = {
        "model": args.model,
        "arch": "qwen",
        "seed": args.seed,
        "seqlen": args.seqlen,
        "drop_frac": args.drop_frac,
        "keep_last_layer": args.keep_last_layer,
        "selected_block": {
            "start": best_ell,
            "n": n,
            "indices": removed_indices,
            "angular_distance": float(best_d),
        },
        "split": {
            "policy": args.split_policy,
            "ratio": args.split_ratio,
            "B": B_idx,
            "C": C_idx,
        },
        "params": {
            "P_total": int(P_total),
            "P_drop": int(P_drop),
            "drop_percentage": float(100 * P_drop / P_total),
        },
    }
    with open(os.path.join(args.save_dir, args.prune_log), "w", encoding="utf-8") as f:
        json.dump(prune_log, f, ensure_ascii=False, indent=2)
    print("✓ prune_log.json saved")

    # 매니페스트
    manifest = {
        "version": "1.0",
        "base_model": args.model,
        "arch": "qwen",
        "counts": {
            "L_full": int(L_full),
            "A_kept": len(kept_idx),
            "removed_total": len(removed_indices),
            "B": len(B_idx),
            "C": len(C_idx),
        },
        "selection": {
            "method": "angular_distance",
            "block": {"start": int(best_ell), "n": int(n)},
            "angular_distance": float(best_d),
            "keep_last_layer": args.keep_last_layer,
        },
        "stages": {
            "A": {"kept_layers": kept_idx, "dropped_layers": removed_indices},
            "B": {"removed_layers": B_idx},
            "C": {"removed_layers": C_idx},
        },
        "artifacts": {
            "A": {"dir": os.path.abspath(args.save_dir)},
            "B": {"dir": os.path.abspath(os.path.join(args.save_removed_dir, "B"))},
            "C": {"dir": os.path.abspath(os.path.join(args.save_removed_dir, "C"))},
            "original_config": {"dir": os.path.abspath(orig_dir)},
        },
    }
    with open(os.path.join(args.save_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("✓ manifest.json saved")

    print("\n" + "=" * 60)
    print("✅ PRUNING COMPLETE!")
    print("=" * 60)
    print(f"A model: {args.save_dir}")
    print(f"Bundles: {args.save_removed_dir}")
    print(f"Stats: {len(removed_indices)}/{L_full} layers removed ({args.drop_frac*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()