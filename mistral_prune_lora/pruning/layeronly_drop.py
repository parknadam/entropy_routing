# layeronly_drop.py
# 레이어 드랍 전용 엔트리: 유사도 기반 연속 블록 제거 + B/C 번들 저장 + A 저장(토크나이저 포함)
# Mistral/LLaMA 모두 지원
"""
# Mistral 7B 예시
python -m mistral_prune_lora.pruning.layeronly_drop \
  --model mistralai/Mistral-7B-v0.1 \
  --device cuda:2 \
  --drop_frac 0.25 \
  --keep_last_layer \
  --nsamples 64 \
  --seqlen 1024 \
  --max_batches 32 \
  --save_dir ./25_mistral_results/pruning/A \
  --save_removed_dir ./25_mistral_results/pruning/bundles

# LLaMA-2 7B 예시 (기존 호환성 유지)
python -m pruning.layeronly_drop \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda:0 \
  --drop_frac 0.25 \
  --keep_last_layer \
  --nsamples 64 \
  --seqlen 1024 \
  --max_batches 32 \
  --save_dir ./7b_results/pruning/A \
  --save_removed_dir ./7b_results/pruning/bundles
"""

import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .data import get_loaders
from .simdrop import choose_block_to_drop, drop_consecutive_layers, _detect_model_type, _get_layers
from .bundler import export_two_bundles, split_indices


def _resolve_embed_device(model, fallback_device: str):
    """임베딩 레이어의 디바이스 확인"""
    embed_key = "model.embed_tokens"
    if (
        hasattr(model, "hf_device_map")
        and isinstance(model.hf_device_map, dict)
        and embed_key in model.hf_device_map
    ):
        dev = model.hf_device_map[embed_key]
        return torch.device(dev) if not isinstance(dev, torch.device) else dev
    return torch.device(fallback_device)


def _load_model(model_name: str, seqlen: int, device_str: str):
    """모델 로드 (Mistral/LLaMA 자동 감지)"""
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only 안전

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
        attn_implementation="eager",  # CUDA ext 없어도 동작
    ).to(device_str)
    model.seqlen = seqlen
    model.config.use_cache = False
    model.eval()
    
    # 모델 타입 출력
    model_type = _detect_model_type(model)
    print(f"✓ Loaded model type: {model_type}")
    
    return model, tok


def _count_params(m):
    """총 파라미터 수"""
    return sum(p.numel() for p in m.parameters())


def _count_params_of_layers(m, idxs, model_type: str):
    """특정 레이어들의 파라미터 수"""
    layers = _get_layers(m, model_type)
    s = 0
    for i in idxs:
        for p in layers[i].parameters(recurse=True):
            s += p.numel()
    return s


def build_layers_map(model, model_type: str, out_path: str):
    """
    레이어 인덱스 ↔ 파라미터 키 목록 매핑을 저장
    Mistral/LLaMA 모두 지원
    """
    layers = _get_layers(model, model_type)
    sd_keys = list(model.state_dict().keys())
    
    if model_type == "opt":
        prefix = "model.decoder.layers."
    else:  # mistral, llama
        prefix = "model.layers."
    
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
        "model_type": model_type,
        "layers": layer_map,
        "non_layer_keys": non_layer_keys,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                    help="Model name (supports Mistral/LLaMA)")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--drop_frac", type=float, default=0.30)
    ap.add_argument("--keep_last_layer", action="store_true")
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max_batches", type=int, default=64)
    ap.add_argument("--save_dir", type=str, default="./A")
    ap.add_argument("--save_removed_dir", type=str, default="./bundles")
    ap.add_argument("--split_policy", type=str, choices=["half", "ratio"], default="half")
    ap.add_argument("--split_ratio", type=float, default=0.5)
    ap.add_argument("--prune_log", type=str, default="prune_log.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # [1] 모델/토크나이저 로드
    print("=" * 60)
    print("Layer Pruning for Mistral/LLaMA")
    print("=" * 60)
    model, tokenizer = _load_model(args.model, args.seqlen, args.device)
    model_type = _detect_model_type(model)
    embed_dev = _resolve_embed_device(model, args.device)

    # original_config/ 저장
    orig_dir = os.path.join(args.save_dir, "original_config")
    os.makedirs(orig_dir, exist_ok=True)

    # 모델 원본 설정 저장
    try:
        model.config.to_json_file(os.path.join(orig_dir, "config.json"))
    except Exception:
        pass
    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.to_json_file(os.path.join(orig_dir, "generation_config.json"))
    except Exception:
        pass

    # 토크나이저 저장
    tokenizer.save_pretrained(orig_dir)

    # 출처 메타
    with open(os.path.join(orig_dir, "source.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_model": args.model,
            "model_type": model_type,
        }, f, ensure_ascii=False, indent=2)

    # [2] 캘리브레이션 로더
    print("\n[1/3] Loading calibration data (C4)")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        tokenizer=tokenizer,
    )

    # [3] 연속 블록 드랍 선택
    print("\n[2/3] Selecting layers to drop (angular distance)")
    L_full = len(_get_layers(model, model_type))
    n = int(round(args.drop_frac * L_full))

    orig_cfg_dir = os.path.join(args.save_dir, "original_config")

    if n <= 0:
        print("→ drop_frac <= 0, nothing to drop. Saving model as-is.")
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.save_dir)

        # manifest.json
        kept_idx = list(range(L_full))
        removed_indices, B_idx, C_idx = [], [], []

        manifest = {
            "version": "1.0",
            "base_model": args.model,
            "model_type": model_type,
            "counts": {
                "L_full": int(L_full),
                "A_kept": len(kept_idx),
                "removed_total": 0,
                "B": 0,
                "C": 0,
            },
            "selection": {
                "method": "angular_distance",
                "block": {"start": None, "n": 0},
                "angular_distance": None,
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
                "original_config": {"dir": os.path.abspath(orig_cfg_dir)},
            },
        }
        with open(os.path.join(args.save_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print("[3/3] Done.")
        return

    best_ell, best_d, _L_old = choose_block_to_drop(
        model,
        dataloader,
        embed_dev,
        n=n,
        model_type=model_type,
        keep_last_layer=args.keep_last_layer,
        max_batches=args.max_batches,
    )

    # 드랍 구간 재검증
    if args.keep_last_layer and best_ell + n > L_full - 1:
        best_ell = max(0, L_full - 1 - n)

    removed_indices = list(range(best_ell, best_ell + n))

    if n > 0 and args.keep_last_layer:
        assert removed_indices[-1] < (L_full - 1), "keep_last_layer 위반"

    # 파라미터 수 집계
    P_total = _count_params(model)
    P_drop = _count_params_of_layers(model, removed_indices, model_type)

    # B/C 번들 저장
    print(f"\n→ Saving bundles (B/C split)")
    os.makedirs(args.save_removed_dir, exist_ok=True)
    B_idx, C_idx = export_two_bundles(
        model=model,
        removed_indices=removed_indices,
        out_root=args.save_removed_dir,
        model_type=model_type,
        config=model.config,
        split_policy=args.split_policy,
        split_ratio=args.split_ratio,
    )
    print(f"→ Bundles saved: B({len(B_idx)}), C({len(C_idx)}) → {args.save_removed_dir}")

    # 드랍 적용
    model, _ = drop_consecutive_layers(model, best_ell, n, model_type=model_type)
    model = model.to(embed_dev)
    if torch.cuda.is_available() and ("cuda" in str(embed_dev)):
        torch.cuda.empty_cache()

    new_depth = len(_get_layers(model, model_type))
    print(f"→ Depth: {L_full} → {new_depth} (angular_dist={best_d:.4f}, n={n}, start={best_ell})")

    # [4] 저장
    print("\n[3/3] Saving pruned model (A)")
    os.makedirs(args.save_dir, exist_ok=True)

    model.save_pretrained(args.save_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.save_dir)

    # 레이어-키 매핑 저장
    build_layers_map(model, model_type, os.path.join(args.save_dir, "layers_map.json"))

    kept_idx = [i for i in range(L_full) if i not in removed_indices]

    # 프루닝 로그
    prune_log = {
        "model": args.model,
        "model_type": model_type,
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
        "params": {"P_total": int(P_total), "P_drop": int(P_drop)},
    }
    with open(os.path.join(args.save_dir, args.prune_log), "w", encoding="utf-8") as f:
        json.dump(prune_log, f, ensure_ascii=False, indent=2)

    # Manifest
    manifest = {
        "version": "1.0",
        "base_model": args.model,
        "model_type": model_type,
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
            "original_config": {"dir": os.path.abspath(orig_cfg_dir)},
        },
    }
    with open(os.path.join(args.save_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("✓ Pruning completed successfully!")
    print("=" * 60)
    print(f"Model type:      {model_type}")
    print(f"Original depth:  {L_full} layers")
    print(f"Pruned depth:    {new_depth} layers")
    print(f"Removed:         {len(removed_indices)} layers ({args.drop_frac*100:.1f}%)")
    print(f"Angular dist:    {best_d:.4f}")
    print(f"\nA model saved:   {args.save_dir}")
    print(f"B/C bundles:     {args.save_removed_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
