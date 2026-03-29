# layeronly_drop.py  (Phi-3 전용)
# 드랍 전용 엔트리: 유사도 기반 연속 블록 제거 + B/C 번들 저장 + A 저장
# ─────────────────────────────────────────────────────────────
# 변경 사유:
#   1. 모델 로드 시 trust_remote_code=True 추가
#   2. is_opt 관련 로직 전면 제거
#   3. 기본 모델 경로를 microsoft/Phi-3-mini-4k-instruct로 변경
#   4. 기본 seqlen을 2048로 유지
#   5. simdrop, bundler 호출 시 is_opt 인자 제거
#   6. arch 필드를 "phi3"으로 설정
#   7. attn_implementation="eager" 유지
#   8. untied lm_head 저장 보정 (범용)
#      → config.tie_word_embeddings 값 대신 state_dict에 lm_head.weight가
#        실제로 존재하는지를 기준으로 판단
#      → Phi3Small처럼 config에 tie_word_embeddings가 명시되지 않은
#        모델에서도 정확하게 동작
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

# Phi-3-small (7B) 25% 프루닝 예시
python -m phi_prune_lora.pruning.layeronly_drop \
  --model microsoft/Phi-3-small-8k-instruct \
  --device cuda:0 \
  --drop_frac 0.25 \
  --keep_last_layer \
  --nsamples 64 \
  --seqlen 2048 \
  --max_batches 32 \
  --save_dir ./new_phi3_small_results/pruning/A \
  --save_removed_dir ./new_phi3_small_results/pruning/bundles
"""

import argparse
import json
import os

import torch
from safetensors import safe_open
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
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.bfloat16
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
        print("[warn] bfloat16 not supported on this GPU, falling back to float16")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device_str)
    model.seqlen = seqlen
    model.config.use_cache = False
    model.eval()
    return model, tok


def _count_params(m):
    return sum(p.numel() for p in m.parameters())


def _count_params_of_layers(m, idxs):
    layers_ = m.model.layers
    s = 0
    for i_ in idxs:
        for p_ in layers_[i_].parameters(recurse=True):
            s += p_.numel()
    return s


def build_layers_map(model, out_path: str):
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


# ─────────────────────────────────────────────────────────────
# lm_head.weight 누락 방지 (범용)
# ─────────────────────────────────────────────────────────────
# 판단 기준: config.tie_word_embeddings 값이 아니라,
# state_dict에 "lm_head.weight" 키가 실제로 존재하는지 여부.
# → Phi3Small처럼 config에 tie_word_embeddings가 명시되지 않아
#   HF 기본값(True)으로 잘못 판단되는 문제를 회피.
# ─────────────────────────────────────────────────────────────

def _has_lm_head_in_state_dict(model) -> bool:
    """model의 state_dict에 lm_head.weight가 독립 키로 존재하는지 확인."""
    return "lm_head.weight" in model.state_dict()


def _prepare_untied_lm_head_save(model):
    """
    lm_head.weight가 state_dict에 존재하는데 _tied_weights_keys에 등록되어 있으면
    save_pretrained()가 해당 텐서를 누락시킨다.
    저장 직전에 alias를 끊고 tied 선언을 제거하여 완전하게 저장되도록 한다.
    """
    if not _has_lm_head_in_state_dict(model):
        return  # lm_head.weight가 state_dict에 없으면 실제로 tied → 건드리지 않음

    # embed_tokens와 lm_head가 메모리를 공유하면 alias를 끊음
    output_emb = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    input_emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if (
        output_emb is not None
        and input_emb is not None
        and hasattr(output_emb, "weight")
        and hasattr(input_emb, "weight")
        and output_emb.weight.data_ptr() == input_emb.weight.data_ptr()
    ):
        output_emb.weight = torch.nn.Parameter(output_emb.weight.detach().clone())
        print("  [fix] untied save: cloned lm_head.weight to break alias with embeddings")

    # _tied_weights_keys에서 lm_head.weight 제거
    tied_keys = list(getattr(model, "_tied_weights_keys", None) or [])
    if "lm_head.weight" in tied_keys:
        kept_keys = [k for k in tied_keys if k != "lm_head.weight"]
        model._tied_weights_keys = kept_keys or None
        print("  [fix] untied save: removed lm_head.weight from _tied_weights_keys")


def _find_tensor_shard(save_dir: str, tensor_key: str):
    """save_dir 내 safetensors shard 파일들을 순회하며 tensor_key가 들어있는 파일명을 반환."""
    shard_files = sorted(
        f for f in os.listdir(save_dir)
        if f.endswith(".safetensors") and os.path.isfile(os.path.join(save_dir, f))
    )
    for fname in shard_files:
        shard_path = os.path.join(save_dir, fname)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            if tensor_key in handle.keys():
                return fname
    return None


def _ensure_lm_head_in_index(model, save_dir: str):
    """
    저장 후 model.safetensors.index.json에 lm_head.weight 매핑이 있는지 확인하고,
    누락되었으면 실제 shard를 찾아 매핑을 추가한다.
    """
    if not _has_lm_head_in_state_dict(model):
        return

    index_path = os.path.join(save_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        # 단일 shard인 경우 index 파일 자체가 없음 → 별도 검증 불필요
        return

    with open(index_path, "r", encoding="utf-8") as f:
        index_payload = json.load(f)

    weight_map = index_payload.setdefault("weight_map", {})
    mapped_shard = weight_map.get("lm_head.weight")

    # 이미 올바르게 매핑되어 있으면 패스
    if mapped_shard is not None:
        mapped_path = os.path.join(save_dir, mapped_shard)
        if os.path.isfile(mapped_path):
            with safe_open(mapped_path, framework="pt", device="cpu") as handle:
                if "lm_head.weight" in handle.keys():
                    return

    # 실제 shard를 찾아서 매핑 추가
    actual_shard = _find_tensor_shard(save_dir, "lm_head.weight")
    if actual_shard is None:
        raise RuntimeError(
            f"untied save is incomplete: lm_head.weight missing from saved shards in {save_dir}"
        )

    weight_map["lm_head.weight"] = actual_shard
    index_payload["weight_map"] = dict(sorted(weight_map.items()))
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, ensure_ascii=False, indent=2)
    print(f"  [fix] untied save: added lm_head.weight to index.json -> {actual_shard}")


def _assert_lm_head_saved(model, save_dir: str):
    """
    lm_head.weight가 state_dict에 존재하는 모델이라면,
    실제로 safetensors에 저장되었는지 최종 검증한다.
    """
    if not _has_lm_head_in_state_dict(model):
        return

    index_path = os.path.join(save_dir, "model.safetensors.index.json")

    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_payload = json.load(f)
        shard_name = (index_payload.get("weight_map") or {}).get("lm_head.weight")
        if shard_name is not None:
            shard_path = os.path.join(save_dir, shard_name)
            if os.path.isfile(shard_path):
                with safe_open(shard_path, framework="pt", device="cpu") as handle:
                    if "lm_head.weight" in handle.keys():
                        return
            raise RuntimeError(
                f"untied save is incomplete: lm_head.weight is mapped to {shard_name}, "
                f"but not present in the shard file"
            )
        # index.json은 있는데 lm_head.weight 매핑이 없음
        raise RuntimeError(
            f"untied save is incomplete: lm_head.weight missing from {index_path}"
        )

    # index.json이 없는 경우 (단일 shard) → shard 파일에서 직접 확인
    shard_name = _find_tensor_shard(save_dir, "lm_head.weight")
    if shard_name is not None:
        return
    raise RuntimeError(
        f"untied save is incomplete: lm_head.weight missing from saved shards in {save_dir}"
    )


def _save_model_pretrained(model, save_dir: str):
    """save_pretrained 래퍼: lm_head.weight 누락 방지."""
    _prepare_untied_lm_head_save(model)
    model.save_pretrained(save_dir, safe_serialization=True)
    _ensure_lm_head_in_index(model, save_dir)
    _assert_lm_head_saved(model, save_dir)


def main():
    ap = argparse.ArgumentParser()
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
    L_full = len(model.model.layers)
    n = int(round(args.drop_frac * L_full))

    orig_cfg_dir = os.path.join(args.save_dir, "original_config")

    if n <= 0:
        print("→ drop_frac <= 0, nothing to drop. Just saving A (no changes).")
        os.makedirs(args.save_dir, exist_ok=True)
        _save_model_pretrained(model, args.save_dir)
        tokenizer.save_pretrained(args.save_dir)

        kept_idx = list(range(L_full))
        removed_indices, B_idx, C_idx = [], [], []

        manifest = {
            "version": "1.0",
            "base_model": args.model,
            "arch": "phi3",
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

    if args.keep_last_layer and best_ell + n > L_full - 1:
        best_ell = max(0, L_full - 1 - n)

    removed_indices = list(range(best_ell, best_ell + n))

    if n > 0:
        assert removed_indices[-1] < (L_full - 1), "keep_last_layer 위반"

    P_total = _count_params(model)
    P_drop = _count_params_of_layers(model, removed_indices)

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

    model, _ = drop_consecutive_layers(model, best_ell, n)
    model = model.to(embed_dev)
    if torch.cuda.is_available() and ("cuda" in str(embed_dev)):
        torch.cuda.empty_cache()

    new_depth = len(model.model.layers)
    print(f"→ Depth: {L_full} → {new_depth} (d_min={best_d:.4f}, n={n}, start={best_ell})")

    # [4] 저장
    print("[3/3] Saving A & manifest")
    os.makedirs(args.save_dir, exist_ok=True)

    _save_model_pretrained(model, args.save_dir)
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
        "arch": "phi3",
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