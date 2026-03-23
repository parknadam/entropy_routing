# layeronly_drop.py  (Phi-3 전용)
# 드랍 전용 엔트리: 유사도 기반 연속 블록 제거 + B/C 번들 저장 + A 저장
# ─────────────────────────────────────────────────────────────
# 변경 사유:
#   1. 모델 로드 시 trust_remote_code=True 추가
#      → Phi-3는 일부 transformers 버전에서 커스텀 코드가 필요할 수 있음
#   2. is_opt 관련 로직 전면 제거
#      → Phi-3는 OPT 아키텍처와 무관
#      → model.model.layers 직접 사용
#   3. 기본 모델 경로를 microsoft/Phi-3-mini-4k-instruct로 변경
#   4. 기본 seqlen을 2048로 유지 (Phi-3-mini 기본 컨텍스트)
#   5. simdrop, bundler 호출 시 is_opt 인자 제거
#   6. arch 필드를 "phi3"으로 설정
#   7. attn_implementation="eager" 유지
#      → Flash Attention 없는 환경에서도 동작하도록
# ─────────────────────────────────────────────────────────────
"""
# Phi-3-mini (3.8B) 25% 프루닝 예시
python -m phi3_prune_lora.pruning.layeronly_drop \
  --model microsoft/Phi-3-mini-4k-instruct \
  --device cuda:0 \
  --drop_frac 0.25 \
  --keep_last_layer \
  --nsamples 64 \
  --seqlen 1024 \
  --max_batches 32 \
  --save_dir ./phi3_results/pruning/A \
  --save_removed_dir ./phi3_results/pruning/bundles

# Phi-3-small (7B) 30% 프루닝 예시
python -m phi_prune_lora.pruning.layeronly_drop \
  --model microsoft/Phi-3-small-8k-instruct \
  --device cuda:0 \
  --drop_frac 0.25 \
  --keep_last_layer \
  --nsamples 64 \
  --seqlen 2048 \
  --max_batches 32 \
  --save_dir ./phi3_small_results/pruning/A \
  --save_removed_dir ./phi3_small_results/pruning/bundles
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .bundler import export_two_bundles
from .data import get_loaders
from .simdrop import choose_block_to_drop, drop_consecutive_layers


def _resolve_embed_device(model, fallback_device: str):
    """임베딩 레이어의 실제 디바이스를 감지 (device_map 사용 시 필요)."""
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
    """
    변경점:
      - trust_remote_code=True 추가 (Phi-3 호환)
      - torch_dtype=torch.bfloat16 사용
        → Phi-3는 bfloat16으로 학습되었으므로 fp16보다 bfloat16이 수치적으로 안전
        → GPU가 bf16 미지원 시 fp16으로 폴백
      - attn_implementation="eager" 유지 (CUDA 확장 없이도 동작)
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only 안전

    # bf16 가능 여부 확인
    dtype = torch.bfloat16
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
        print(f"[warn] bfloat16 not supported on this GPU, falling back to float16")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=True,       # Phi-3 필수
        attn_implementation="eager",  # CUDA ext 없어도 동작
    ).to(device_str)
    model.seqlen = seqlen
    model.config.use_cache = False
    model.eval()
    return model, tok


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def _count_params_of_layers(m, idxs):
    """
    변경점: is_opt 분기 제거 → model.model.layers 직접 사용
    """
    layers_ = m.model.layers
    s = 0
    for i_ in idxs:
        for p_ in layers_[i_].parameters(recurse=True):
            s += p_.numel()
    return s


def build_layers_map(model, out_path: str):
    """
    레이어 인덱스 ↔ 파라미터 키 목록 매핑을 저장.

    변경점:
      - is_opt 분기 제거
      - prefix를 "model.layers."로 고정 (Phi-3 구조)
    """
    layers = model.model.layers
    sd_keys = list(model.state_dict().keys())
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
        "layers": layer_map,
        "non_layer_keys": non_layer_keys,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    # 변경: 기본 모델을 Phi-3-mini로 설정
    ap.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--drop_frac", type=float, default=0.25)
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

    # [2] 캘리브레이션 로더
    print("[1/3] Calibration loader (C4)")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        tokenizer=tokenizer,
    )

    # [3] 연속 블록 드랍 선택 → 번들 저장 → 드랍 적용
    print("[2/3] Angular-distance block selection & export bundles")
    L_full = len(model.model.layers)  # 변경: is_opt 분기 제거
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
            "arch": "phi3",  # 변경: "llama"/"opt" → "phi3"
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

    # 변경: is_opt=False 인자 제거
    best_ell, best_d, _L_old = choose_block_to_drop(
        model,
        dataloader,
        embed_dev,
        n=n,
        keep_last_layer=args.keep_last_layer,
        max_batches=args.max_batches,
    )

    if args.keep_last_layer and best_ell + n > L_full - 1:
        best_ell = max(0, L_full - 1 - n)

    removed_indices = list(range(best_ell, best_ell + n))

    if n > 0:
        assert removed_indices[-1] < (L_full - 1), "keep_last_layer 위반"

    P_total = _count_params(model)
    P_drop = _count_params_of_layers(model, removed_indices)

    # 변경: is_opt 인자 제거
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

    # 변경: is_opt 인자 제거
    model, _ = drop_consecutive_layers(model, best_ell, n)
    model = model.to(embed_dev)
    if torch.cuda.is_available() and ("cuda" in str(embed_dev)):
        torch.cuda.empty_cache()

    new_depth = len(model.model.layers)  # 변경: is_opt 분기 제거
    print(f"→ Depth: {L_full} → {new_depth} (d_min={best_d:.4f}, n={n}, start={best_ell})")

    # [4] 저장
    print("[3/3] Saving A & manifest")
    os.makedirs(args.save_dir, exist_ok=True)

    model.save_pretrained(args.save_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.save_dir)

    # 변경: is_opt 인자 제거
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
        "arch": "phi3",  # 변경: "phi3"
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