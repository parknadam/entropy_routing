# 레이어 드랍만 하는 코드 (Falcon Instruct 모델 구조 대응)
# 드랍 전용 엔트리: 유사도 기반 연속 블록 제거 + B/C 번들 저장 + A 저장(토크나이저 포함)
"""
사용 예시:

python -m falcon_prune_lora.pruning.layeronly_drop \
  --model tiiuae/falcon-7b-instruct \
  --device cuda:2 \
  --drop_frac 0.25 \
  --keep_last_layer \
  --nsamples 64 \
  --seqlen 1024 \
  --max_batches 32 \
  --save_dir ./falcon_results/pruning/A \
  --save_removed_dir ./falcon_results/pruning/bundles

# Falcon-40B-Instruct 예시
python -m falcon_prune_lora.pruning.layeronly_drop \
  --model tiiuae/falcon-40b-instruct \
  --device cuda:0 \
  --drop_frac 0.25 \
  --keep_last_layer \
  --nsamples 64 \
  --seqlen 1024 \
  --max_batches 32 \
  --save_dir ./falcon40b_results/pruning/A \
  --save_removed_dir ./falcon40b_results/pruning/bundles
"""

import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .data import get_loaders
from .simdrop import choose_block_to_drop, drop_consecutive_layers
from .bundler import export_two_bundles, split_indices


def _get_falcon_layers(model):
    """Falcon 모델의 디코더 레이어 리스트 반환"""
    return model.transformer.h


def _resolve_embed_device(model, fallback_device: str):
    """
    Falcon의 임베딩 레이어 디바이스 확인.
    Falcon: model.transformer.word_embeddings
    """
    embed_key = "transformer.word_embeddings"
    if (
        hasattr(model, "hf_device_map")
        and isinstance(model.hf_device_map, dict)
        and embed_key in model.hf_device_map
    ):
        dev = model.hf_device_map[embed_key]
        return torch.device(dev) if not isinstance(dev, torch.device) else dev
    return torch.device(fallback_device)


def _load_model(model_name: str, seqlen: int, device_str: str):
    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,  # Falcon 모델은 trust_remote_code 필요할 수 있음
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only 안전

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=True,  # Falcon 모델 필수
        attn_implementation="eager",  # CUDA ext 없어도 동작
    ).to(device_str)
    model.seqlen = seqlen
    model.config.use_cache = False
    model.eval()

    # Falcon의 max_position_embeddings에 seqlen 클램프
    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos and seqlen > max_pos:
        print(f"[WARN] seqlen({seqlen}) > max_position_embeddings({max_pos}), clamping to {max_pos}")
        model.seqlen = max_pos

    return model, tok


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def _count_params_of_layers(m, idxs):
    layers_ = _get_falcon_layers(m)
    s = 0
    for i_ in idxs:
        for p_ in layers_[i_].parameters(recurse=True):
            s += p_.numel()
    return s


def build_layers_map(model, out_path: str):
    """
    레이어 인덱스 ↔ 파라미터 키 목록 매핑을 저장.
    Falcon의 레이어 prefix: "transformer.h."
    """
    layers = _get_falcon_layers(model)
    sd_keys = list(model.state_dict().keys())
    prefix = "transformer.h."
    L = len(layers)

    layer_map = {str(i): [] for i in range(L)}
    non_layer_keys = []

    for k in sd_keys:
        if k.startswith(prefix):
            # e.g. "transformer.h.12.self_attention.query_key_value.weight"
            lid = int(k[len(prefix):].split(".", 1)[0])
            layer_map[str(lid)].append(k)
        else:
            non_layer_keys.append(k)

    payload = {
        "num_layers": L,
        "layer_prefix": prefix,
        "layers": layer_map,
        "non_layer_keys": non_layer_keys,  # word_embeddings, ln_f, lm_head 등
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="tiiuae/falcon-7b-instruct")
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
    model, tokenizer = _load_model(args.model, args.seqlen, args.device)
    embed_dev = _resolve_embed_device(model, args.device)

    # original_config/ 저장
    orig_dir = os.path.join(args.save_dir, "original_config")
    os.makedirs(orig_dir, exist_ok=True)

    try:
        model.config.to_json_file(os.path.join(orig_dir, "config.json"))
    except Exception:
        pass
    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.to_json_file(os.path.join(orig_dir, "generation_config.json"))
    except Exception:
        pass

    tokenizer.save_pretrained(orig_dir)

    with open(os.path.join(orig_dir, "source.json"), "w", encoding="utf-8") as f:
        json.dump({"base_model": args.model}, f, ensure_ascii=False, indent=2)

    # [2] 캘리브레이션 로더 (클램프된 seqlen 사용)
    actual_seqlen = model.seqlen  # max_position_embeddings에 클램프된 값
    print(f"[1/3] Calibration loader (C4), seqlen={actual_seqlen}")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=actual_seqlen,
        tokenizer=tokenizer,
    )

    # [3] 연속 블록 드랍 선택 → 번들 저장 → 드랍 적용
    print("[2/3] Angular-distance block selection & export bundles")
    L_full = len(_get_falcon_layers(model))
    n = int(round(args.drop_frac * L_full))

    orig_cfg_dir = os.path.join(args.save_dir, "original_config")

    if n <= 0:
        print("→ drop_frac <= 0, nothing to drop. Just saving A (no changes).")
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.save_dir)

        kept_idx = list(range(L_full))
        removed_indices, B_idx, C_idx = [], [], []

        manifest = {
            "version": "1.0",
            "base_model": args.model,
            "arch": "falcon",
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
                "keep_last_layer": True,
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
        keep_last_layer=args.keep_last_layer,
        max_batches=args.max_batches,
    )

    # 드랍 구간 재검증(범위 클램프)
    if args.keep_last_layer and best_ell + n > L_full - 1:
        best_ell = max(0, L_full - 1 - n)

    removed_indices = list(range(best_ell, best_ell + n))

    if n > 0:
        assert removed_indices[-1] < (L_full - 1), "keep_last_layer 위반"

    # 드랍 전 파라미터 수 집계
    P_total = _count_params(model)
    P_drop = _count_params_of_layers(model, removed_indices)

    # B/C로 분할 + 저장
    os.makedirs(args.save_removed_dir, exist_ok=True)
    B_idx, C_idx = export_two_bundles(
        model=model,
        removed_indices=removed_indices,
        out_root=args.save_removed_dir,
        config=model.config,
        split_policy=args.split_policy,
        split_ratio=args.split_ratio,
    )
    print(f"→ Saved bundles atomically: B({len(B_idx)}), C({len(C_idx)}) → {args.save_removed_dir}")

    # 드랍 적용
    model, _ = drop_consecutive_layers(model, best_ell, n)
    model = model.to(embed_dev)
    if torch.cuda.is_available() and ("cuda" in str(embed_dev)):
        torch.cuda.empty_cache()

    new_depth = len(_get_falcon_layers(model))
    print(f"→ Depth: {L_full} → {new_depth} (d_min={best_d:.4f}, n={n}, start={best_ell})")

    # [4] 저장(A/토크나이저/로그/매니페스트)
    print("[3/3] Saving A & manifest")
    os.makedirs(args.save_dir, exist_ok=True)

    model.save_pretrained(args.save_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.save_dir)

    build_layers_map(model, os.path.join(args.save_dir, "layers_map.json"))

    kept_idx = [i for i in range(L_full) if i not in removed_indices]

    prune_log = {
        "model": args.model,
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

    manifest = {
        "version": "1.0",
        "base_model": args.model,
        "arch": "falcon",
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
            "keep_last_layer": True,
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

    print("Done.")
    print(f"→ A saved to: {args.save_dir}")
    print(f"→ B/C bundles: {args.save_removed_dir}")


if __name__ == "__main__":
    main()