#!/usr/bin/env python3
"""
LoRA 어댑터를 Mistral 베이스 모델에 머지하는 유틸리티.

핵심 동작:
  1. pruned A 모델 로드 (config: 32 layers, 실제 가중치: ~24 layers)
  2. LoRA 어댑터 적용 & 머지
  3. manifest.json에서 dropped layer 인덱스를 읽어 PassLayer로 복원
     → 드롭된 레이어의 랜덤 가중치가 저장되지 않아 모델 크기 유지 (~9GB)
  4. tokenizer + config + generation_config 포함 완전한 모델로 저장

사용법:
# A 단일 머지
python -m mistral_prune_lora.pruning.mistral_merge_adapter \
  --base_model ./25_mistral_results/pruning/A \
  --adapter_path ./mistral_kd_lora_results/adapters/A_lora/stageA/stageA \
  --output_dir ./merged_models_mistral_7b/A_merged \
  --device cuda:0

python -m mistral_prune_lora.pruning.mistral_merge_adapter \
  --base_model ./25_mistral_results/pruning/bundles/B \
  --adapter_path ./mistral_kd_lora_results/adapters/B_lora/stageB \
  --output_dir ./merged_models_mistral_7b/B_merged \
  --device cuda:0

# 여러 어댑터 순차 머지 (예: A_merged + B_lora + C_lora)
python -m mistral_prune_lora.pruning.mistral_merge_adapter \
  --base_model ./25_mistral_results/pruning/A \
  --adapter_paths ./mistral_results/adapters/A_lora/stageA ./mistral_results/adapters/B_lora/stageB \
  --output_dir ./25_mistral_results/merged_models/AB_merged \
  --device cuda:0

# 원본 모델 토크나이저를 사용하고 싶을 때
python -m mistral_prune_lora.pruning.mistral_merge_adapter \
  --base_model ./25_mistral_results/pruning/A \
  --adapter_path ./adapters/A_lora/stageA \
  --output_dir ./merged_models/A_merged \
  --tokenizer_path mistralai/Mistral-7B-v0.1
"""

import argparse
import json
import os
import re
import shutil

import torch
import torch.nn as nn
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# PassLayer: 프루닝된 레이어 자리를 차지하는 초경량 모듈 (파라미터 0)
# ============================================================
class PassLayer(nn.Module):
    """
    드롭된 레이어를 대체하는 파라미터-프리 패스 레이어.
    forward 시그니처는 MistralDecoderLayer/LlamaDecoderLayer와 호환.
    파라미터가 없으므로 save_pretrained() 시 가중치가 저장되지 않음.
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


# ============================================================
# manifest.json에서 dropped layer 인덱스 읽기
# ============================================================
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_BUNDLE_LAYER_FILE_RE = re.compile(r"^layer_(\d+)\.safetensors$")


def _unique_sorted_ints(indices):
    """정수 인덱스 리스트를 중복 제거 + 정렬하여 반환."""
    if not indices:
        return []
    return sorted(set(int(i) for i in indices))


def _read_stage_layers_from_manifest(base_model_path: str):
    """manifest.json에서 A/B/C 스테이지 레이어 정보를 읽습니다."""
    manifest_path = os.path.join(base_model_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"  ⚠ manifest.json not found in {base_model_path}")
        return {
            "A_kept": [],
            "A_dropped": [],
            "B_removed": [],
            "C_removed": [],
            "simdrop_removed": [],
        }

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"  ⚠ manifest.json 파싱 실패: {e}")
        return {
            "A_kept": [],
            "A_dropped": [],
            "B_removed": [],
            "C_removed": [],
            "simdrop_removed": [],
        }

    stages = manifest.get("stages", {}) or {}
    return {
        "A_kept": _unique_sorted_ints(stages.get("A", {}).get("kept_layers", [])),
        "A_dropped": _unique_sorted_ints(stages.get("A", {}).get("dropped_layers", [])),
        "B_removed": _unique_sorted_ints(stages.get("B", {}).get("removed_layers", [])),
        "C_removed": _unique_sorted_ints(stages.get("C", {}).get("removed_layers", [])),
        "simdrop_removed": _unique_sorted_ints(
            (manifest.get("simdrop", {}) or {}).get("removed_layers", [])
        ),
    }


def _infer_adapter_stage(adapter_path: str):
    """어댑터 경로 이름으로 stage(A/B/C)를 추론."""
    if not adapter_path:
        return None

    p = adapter_path.replace("\\", "/").lower()
    if "stageb" in p or "/b_lora/" in p:
        return "B"
    if "stagec" in p or "/c_lora/" in p:
        return "C"
    if "stagea" in p or "/a_lora/" in p:
        return "A"
    return None


def _read_adapter_layers_to_transform(adapter_path: str):
    """adapter_config.json의 layers_to_transform를 읽습니다."""
    if not adapter_path:
        return []

    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return []

    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        return _unique_sorted_ints(cfg.get("layers_to_transform", []))
    except Exception:
        return []


def _read_adapter_base_model_path(adapter_path: str):
    """adapter_config.json의 base_model_name_or_path를 읽습니다."""
    if not adapter_path:
        return None

    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return None

    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        return cfg.get("base_model_name_or_path")
    except Exception:
        return None


def _is_hf_model_dir(path: str) -> bool:
    """HF from_pretrained 가능한 모델 디렉터리인지 검사."""
    return os.path.isfile(os.path.join(path, "config.json"))


def _list_bundle_layer_files(bundle_dir: str):
    """bundle dir에서 layer_{idx}.safetensors 파일 목록 반환."""
    if not os.path.isdir(bundle_dir):
        return []
    files = []
    for fname in sorted(os.listdir(bundle_dir)):
        if _BUNDLE_LAYER_FILE_RE.match(fname):
            files.append(os.path.join(bundle_dir, fname))
    return files


def _load_bundle_indices(bundle_dir: str):
    """
    bundle dir에서 레이어 인덱스를 로드합니다.
    우선순위: bundle_meta.json(indices) -> 파일명(layer_XXX.safetensors)
    """
    meta_path = os.path.join(bundle_dir, "bundle_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            indices = meta.get("indices", []) or meta.get("layer_indices", [])
            if indices:
                return _unique_sorted_ints(indices)
        except Exception:
            pass

    idxs = []
    for layer_file in _list_bundle_layer_files(bundle_dir):
        m = _BUNDLE_LAYER_FILE_RE.match(os.path.basename(layer_file))
        if m:
            idxs.append(int(m.group(1)))
    return _unique_sorted_ints(idxs)


def _pick_bundle_layer_file(bundle_dir: str, idx: int):
    """bundle dir에서 idx에 해당하는 safetensors 파일을 찾습니다."""
    candidates = [f"layer_{idx:03d}.safetensors", f"layer_{idx}.safetensors"]
    for fname in candidates:
        p = os.path.join(bundle_dir, fname)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Layer file not found for idx={idx} in bundle: {bundle_dir}")


def _inject_bundle_layers_into_model(model, bundle_dir: str, indices: list):
    """
    bundle의 per-layer safetensors를 model.model.layers[idx]로 주입합니다.
    """
    if not indices:
        return

    layers = _get_model_layers(model)
    print(f"  Injecting bundle layers from: {bundle_dir}")
    for idx in indices:
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"Bundle layer idx {idx} is out of range [0, {len(layers)-1}]")

        sf_path = _pick_bundle_layer_file(bundle_dir, idx)
        sd = load_file(sf_path)

        # 모델 레이어 dtype/device에 맞춰 주입
        try:
            ref_param = next(layers[idx].parameters())
            dev, dtype = ref_param.device, ref_param.dtype
            sd = {k: v.to(device=dev, dtype=dtype) for k, v in sd.items()}
        except StopIteration:
            pass

        layers[idx].load_state_dict(sd, strict=True)
        print(f"    ✓ layer {idx} <- {os.path.basename(sf_path)}")


def _clear_output_dir(output_dir: str):
    """출력 디렉터리를 비웁니다."""
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        p = os.path.join(output_dir, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except Exception as e:
            raise RuntimeError(f"Failed to clean output dir entry: {p} ({e})")


def _save_bundle_only(model, output_dir: str, layer_indices: list, source_bundle_dir: str = None):
    """
    모델에서 지정 레이어만 bundle(layer_XXX.safetensors) 형태로 저장합니다.
    """
    if not layer_indices:
        raise ValueError("Bundle save requires non-empty layer_indices")

    _clear_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    layers = _get_model_layers(model)
    saved = []
    for idx in sorted(set(int(i) for i in layer_indices)):
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"Bundle save layer idx {idx} is out of range [0, {len(layers)-1}]")
        sd = {
            k: v.detach().to("cpu")
            for k, v in layers[idx].state_dict().items()
        }
        out_f = os.path.join(output_dir, f"layer_{idx:03d}.safetensors")
        save_file(sd, out_f)
        saved.append(os.path.basename(out_f))

    # bundle_meta 보존/갱신
    meta = {}
    meta_src = os.path.join(source_bundle_dir, "bundle_meta.json") if source_bundle_dir else None
    if meta_src and os.path.isfile(meta_src):
        try:
            with open(meta_src, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    meta["layer_indices"] = sorted(set(int(i) for i in layer_indices))
    meta["indices"] = sorted(set(int(i) for i in layer_indices))
    meta["merged"] = True
    with open(os.path.join(output_dir, "bundle_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"  Saved bundle files: {saved} + bundle_meta.json")
    print(f"  Total bundle size: {total_bytes / (1024**3):.2f} GB")


def _resolve_target_layers(base_model_path: str, adapter_path: str, stage_info: dict = None):
    """
    어댑터가 영향을 줄 레이어 인덱스를 결정합니다.
    우선순위:
      1) stageB/C 추론 시 manifest의 B/C 레이어
      2) adapter_config.layers_to_transform
    """
    if stage_info is None:
        stage_info = _read_stage_layers_from_manifest(base_model_path)

    stage = _infer_adapter_stage(adapter_path)
    adapter_layers = _read_adapter_layers_to_transform(adapter_path)

    target_layers = []
    source = None

    if stage == "B" and stage_info["B_removed"]:
        target_layers = stage_info["B_removed"]
        source = "manifest.stages.B.removed_layers"
    elif stage == "C" and stage_info["C_removed"]:
        target_layers = stage_info["C_removed"]
        source = "manifest.stages.C.removed_layers"
    elif adapter_layers:
        target_layers = adapter_layers
        source = "adapter_config.layers_to_transform"

    if target_layers:
        print(f"  Target layers ({source}): {target_layers}")

    if adapter_layers and target_layers and set(adapter_layers) != set(target_layers):
        print(
            "  ⚠ adapter_config.layers_to_transform와 manifest 타겟이 다릅니다. "
            "머지 범위는 manifest 기준으로 강제합니다."
        )

    return stage, target_layers


def _read_dropped_layers_from_manifest(base_model_path: str, merge_stage: str = None, stage_info: dict = None):
    """
    merge stage에 맞춰 PassLayer로 복원할 레이어를 결정합니다.
      - stage A: A dropped (기존 동작)
      - stage B: C removed만 복원 (B 레이어 머지 결과 보존)
      - stage C: 복원 없음 (FULL)
      - 기타/미지정: 기존 fallback 동작
    """
    if stage_info is None:
        stage_info = _read_stage_layers_from_manifest(base_model_path)

    A_dropped = stage_info["A_dropped"]
    B_removed = stage_info["B_removed"]
    C_removed = stage_info["C_removed"]
    simdrop_removed = stage_info["simdrop_removed"]

    if merge_stage == "B":
        if C_removed:
            return C_removed
        if A_dropped and B_removed:
            return _unique_sorted_ints(set(A_dropped) - set(B_removed))
        return []

    if merge_stage == "C":
        return []

    # 기본/Stage A fallback
    if A_dropped:
        return A_dropped
    if B_removed or C_removed:
        return _unique_sorted_ints(B_removed + C_removed)
    if simdrop_removed:
        return simdrop_removed
    return []


def _extract_layer_idx(name: str):
    """파라미터/모듈 이름에서 decoder layer index를 추출."""
    m = _LAYER_INDEX_RE.search(name)
    return int(m.group(1)) if m else None


def _collect_lora_layers(model):
    """모델 내 LoRA 파라미터가 존재하는 레이어 인덱스 목록."""
    layers = set()
    for name, _ in model.named_parameters():
        if "lora_" not in name.lower():
            continue
        idx = _extract_layer_idx(name)
        if idx is not None:
            layers.add(idx)
    return sorted(layers)


def _zero_lora_weights_outside_layers(model, target_layers: list):
    """
    target 외 레이어의 LoRA 가중치를 0으로 만들어 merge 영향이 없게 합니다.
    (adapter가 전 레이어로 저장돼도 target 레이어에만 반영되도록 강제)
    """
    if not target_layers:
        return

    target_set = set(target_layers)
    zeroed_layers = set()
    zeroed_tensors = 0

    with torch.no_grad():
        for module_name, module in model.named_modules():
            layer_idx = _extract_layer_idx(module_name)
            if layer_idx is None or layer_idx in target_set:
                continue

            lora_A = getattr(module, "lora_A", None)
            lora_B = getattr(module, "lora_B", None)
            if lora_A is not None and lora_B is not None and hasattr(lora_A, "keys"):
                for adapter_name in list(lora_A.keys()):
                    if adapter_name in lora_A and hasattr(lora_A[adapter_name], "weight"):
                        lora_A[adapter_name].weight.zero_()
                        zeroed_tensors += 1
                        zeroed_layers.add(layer_idx)
                    if adapter_name in lora_B and hasattr(lora_B[adapter_name], "weight"):
                        lora_B[adapter_name].weight.zero_()
                        zeroed_tensors += 1
                        zeroed_layers.add(layer_idx)

            lora_emb_A = getattr(module, "lora_embedding_A", None)
            lora_emb_B = getattr(module, "lora_embedding_B", None)
            if lora_emb_A is not None and lora_emb_B is not None and hasattr(lora_emb_A, "keys"):
                for adapter_name in list(lora_emb_A.keys()):
                    if adapter_name in lora_emb_A:
                        lora_emb_A[adapter_name].zero_()
                        zeroed_tensors += 1
                        zeroed_layers.add(layer_idx)
                    if adapter_name in lora_emb_B:
                        lora_emb_B[adapter_name].zero_()
                        zeroed_tensors += 1
                        zeroed_layers.add(layer_idx)

    if zeroed_tensors:
        print(
            f"  ✓ LoRA scope enforced: zeroed {zeroed_tensors} tensors "
            f"outside target layers {sorted(zeroed_layers)}"
        )


def _enforce_adapter_scope(model, target_layers: list):
    """
    LoRA merge 범위를 target_layers로 강제합니다.
    """
    lora_layers = _collect_lora_layers(model)
    if not lora_layers:
        print("  ⚠ No LoRA parameters found in adapter.")
        return

    print(f"  LoRA layers found: {lora_layers}")
    if not target_layers:
        return

    target_set = set(target_layers)
    scoped = sorted(i for i in lora_layers if i in target_set)
    if not scoped:
        raise RuntimeError(
            f"Adapter has no LoRA params on target layers {target_layers}. "
            f"Found layers: {lora_layers}"
        )

    unexpected = sorted(i for i in lora_layers if i not in target_set)
    if unexpected:
        print(f"  ⚠ Non-target LoRA layers detected: {unexpected}")
        _zero_lora_weights_outside_layers(model, target_layers)

    print(f"  ✓ Effective merge target layers: {target_layers}")


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
    이렇게 하면 save_pretrained() 시 해당 위치의 가중치가 저장되지 않아
    모델 크기가 pruned 모델과 동일하게 유지됩니다.
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


# ============================================================
# 토크나이저 로드 (다중 경로 폴백)
# ============================================================
def _resolve_tokenizer(*candidate_paths: str, trust_remote_code: bool = True) -> AutoTokenizer:
    """여러 후보 경로를 순서대로 시도하여 토크나이저를 로드합니다."""
    errors = []
    for path in candidate_paths:
        if path is None:
            continue
        try:
            # ★ 로컬 경로는 절대경로로 변환 (transformers가 ./를 HF repo ID로 오인하는 버그 방지)
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

    # 1) 사용자 명시 경로 (최우선)
    if tokenizer_path:
        candidates.append(tokenizer_path)

    # 2) base_model 자체
    candidates.append(base_model_path)

    # 3) base_model/original_config/ (layeronly_drop이 저장한 원본 설정)
    orig_cfg_dir = os.path.join(base_model_path, "original_config")
    if os.path.isdir(orig_cfg_dir):
        candidates.append(orig_cfg_dir)

    # 4) 어댑터의 config에서 base_model_name_or_path 추출
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

    # 5) manifest.json에서 원본 모델명 추출
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


# ============================================================
# 디바이스 매핑
# ============================================================
def _device_map_from_arg(device: str):
    if device in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return device
    return {"": device}


# ============================================================
# 모델 저장 (model + tokenizer + config + generation_config)
# ============================================================
def _save_complete_model(model, tokenizer, output_dir: str, base_model_path: str = None):
    """모델, 토크나이저, config, generation_config를 모두 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)

    # 1) 모델 가중치 + config.json
    print(f"  Saving model weights...")
    model.save_pretrained(output_dir, safe_serialization=True)

    # 2) 토크나이저
    print(f"  Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    # 3) generation_config (있으면)
    try:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(output_dir)
            print(f"  Saved generation_config.")
    except Exception:
        if base_model_path:
            gen_cfg_file = os.path.join(base_model_path, "generation_config.json")
            if os.path.isfile(gen_cfg_file):
                shutil.copy2(gen_cfg_file, os.path.join(output_dir, "generation_config.json"))
                print(f"  Copied generation_config from base model.")

    # 4) manifest.json 복사 (프루닝 정보 보존 — 이후 단계에서 dropped layers 식별에 필요)
    if base_model_path:
        manifest_src = os.path.join(base_model_path, "manifest.json")
        if os.path.isfile(manifest_src):
            shutil.copy2(manifest_src, os.path.join(output_dir, "manifest.json"))
            print(f"  Copied manifest.json (pruning info).")

    # 5) 저장된 파일 + 크기 출력
    saved_files = sorted(os.listdir(output_dir))
    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in saved_files
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"  Saved files: {saved_files}")
    print(f"  Total size: {total_bytes / (1024**3):.2f} GB")


# ============================================================
# 머지 후 검증
# ============================================================
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
        print(f"  ✓ 모든 필수 파일 존재. from_pretrained() 로 바로 로드 가능.")
        return True
    else:
        print(f"  ✗ 일부 파일 누락. 확인 필요.")
        return False


# ============================================================
# 단일 어댑터 머지
# ============================================================
def merge_single_adapter(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    device: str = "cuda:0",
    tokenizer_path: str = None,
    verify: bool = True,
    save_bundle_only: bool = False,
):
    """단일 LoRA 어댑터를 베이스 모델에 머지하고 완전한 모델로 저장합니다."""
    print(f"\n{'=' * 60}")
    print("Merging Adapter into Base Model (Mistral)")
    print(f"{'=' * 60}")
    print(f"Base Model:  {base_model_path}")
    print(f"Adapter:     {adapter_path}")
    print(f"Output:      {output_dir}")
    if tokenizer_path:
        print(f"Tokenizer:   {tokenizer_path}")

    # base_model이 HF dir이 아니면 bundle(B/C)로 간주하고 skeleton 모델을 자동 결정
    bundle_mode = False
    bundle_dir = None
    bundle_indices = []
    effective_base_model_path = base_model_path

    if not _is_hf_model_dir(base_model_path):
        bundle_indices = _load_bundle_indices(base_model_path)
        if bundle_indices:
            bundle_mode = True
            bundle_dir = base_model_path
            adapter_base = _read_adapter_base_model_path(adapter_path)
            if not adapter_base:
                raise RuntimeError(
                    "base_model이 HF 모델 디렉터리(config.json)도 아니고 "
                    "bundle_dir로 보이지만 adapter_config.json에 "
                    "base_model_name_or_path가 없습니다."
                )

            # adapter_config의 상대경로를 현재 작업 디렉터리 기준으로 해석
            resolved_adapter_base = os.path.abspath(adapter_base) if os.path.exists(adapter_base) else adapter_base
            if not _is_hf_model_dir(resolved_adapter_base):
                raise RuntimeError(
                    f"Bundle merge requires a skeleton HF model dir. "
                    f"Not found/invalid: {resolved_adapter_base}"
                )
            effective_base_model_path = resolved_adapter_base
            print(f"  Detected bundle base: {bundle_dir}")
            print(f"  Using skeleton model: {effective_base_model_path}")
            print(f"  Bundle layer indices: {bundle_indices}")
        else:
            raise RuntimeError(
                f"--base_model 경로에 config.json이 없고 bundle 레이어 파일도 없습니다: {base_model_path}"
            )

    # [1/6] manifest + adapter에서 레이어 라우팅 정보 읽기
    print("\n[1/6] Reading manifest/adapter for layer routing...")
    stage_info = _read_stage_layers_from_manifest(effective_base_model_path)
    adapter_stage, target_layers = _resolve_target_layers(
        effective_base_model_path, adapter_path, stage_info
    )
    dropped_layers = _read_dropped_layers_from_manifest(
        effective_base_model_path,
        merge_stage=adapter_stage,
        stage_info=stage_info,
    )

    if adapter_stage:
        print(f"  Adapter stage inferred: {adapter_stage}")
    if dropped_layers:
        print(f"  PassLayer restore targets: {dropped_layers} ({len(dropped_layers)}개)")
    else:
        print(f"  No PassLayer restore targets for this stage")

    # [2/6] 토크나이저 로드
    print("\n[2/6] Loading tokenizer...")
    tok_candidates = _find_tokenizer_candidates(effective_base_model_path, adapter_path, tokenizer_path)
    tokenizer = _resolve_tokenizer(*tok_candidates)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token = eos_token ({tokenizer.eos_token})")

    # [3/6] 베이스 모델 로드
    print("\n[3/6] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        effective_base_model_path,
        torch_dtype=torch.float16,
        device_map=_device_map_from_arg(device),
        trust_remote_code=True,
    )
    n_layers = base_model.config.num_hidden_layers
    print(f"  Loaded: {n_layers} layers, {sum(p.numel() for p in base_model.parameters())/1e6:.1f}M params")
    if bundle_mode:
        _inject_bundle_layers_into_model(base_model, bundle_dir, bundle_indices)

    # [4/6] 어댑터 로드 & 머지
    print("\n[4/6] Loading and merging adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    _enforce_adapter_scope(model, target_layers)
    merged_model = model.merge_and_unload()
    print(f"  ✓ LoRA weights fused into base model")

    # [5/6] ★ 드롭된 레이어에 PassLayer 복원 (크기 보존 핵심!)
    if save_bundle_only:
        print("\n[5/6] Skipping PassLayer restore (bundle-only save mode)...")
    else:
        print("\n[5/6] Restoring PassLayers for dropped layers...")
        merged_model = _restore_passlayers(merged_model, dropped_layers)

    # [6/6] 저장
    if save_bundle_only:
        print(f"\n[6/6] Saving merged bundle-only output to {output_dir}...")
    else:
        print(f"\n[6/6] Saving complete merged model to {output_dir}...")
    if save_bundle_only:
        bundle_save_indices = bundle_indices if bundle_indices else target_layers
        if not bundle_save_indices:
            raise RuntimeError("save_bundle_only=True but no bundle/target layer indices were resolved.")
        _save_bundle_only(
            merged_model,
            output_dir=output_dir,
            layer_indices=bundle_save_indices,
            source_bundle_dir=bundle_dir,
        )
    else:
        _save_complete_model(merged_model, tokenizer, output_dir, effective_base_model_path)

    if verify and not save_bundle_only:
        del merged_model, model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _verify_merged_model(output_dir)
    else:
        del merged_model, model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"✓ Merge completed: {output_dir}")
    print(f"{'=' * 60}")
    return output_dir


# ============================================================
# 다중 어댑터 순차 머지
# ============================================================
def merge_multiple_adapters(
    base_model_path: str,
    adapter_paths: list,
    output_dir: str,
    device: str = "cuda:0",
    tokenizer_path: str = None,
    verify: bool = True,
    save_bundle_only: bool = False,
):
    """여러 LoRA 어댑터를 순차적으로 머지합니다."""
    if save_bundle_only:
        raise ValueError("--save_bundle_only is supported only with --adapter_path (single merge).")
    print(f"\n{'=' * 60}")
    print("Sequential Adapter Merging (Mistral)")
    print(f"{'=' * 60}")
    print(f"Base Model:  {base_model_path}")
    print(f"Adapters:    {adapter_paths}")
    print(f"Final Output: {output_dir}")

    # manifest에서 stage 레이어 정보 읽기
    print("\n[0a] Reading manifest for stage layers...")
    stage_info = _read_stage_layers_from_manifest(base_model_path)
    if stage_info["A_dropped"]:
        print(f"  A dropped layers: {stage_info['A_dropped']}")
    if stage_info["B_removed"]:
        print(f"  B layers:         {stage_info['B_removed']}")
    if stage_info["C_removed"]:
        print(f"  C layers:         {stage_info['C_removed']}")

    # 토크나이저: 최초 1회만 로드
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
        is_last = (i == len(adapter_paths))
        print(f"\n{'─' * 40}")
        print(f"Stage {i}/{len(adapter_paths)}: {os.path.basename(adapter_path)}")
        print(f"{'─' * 40}")

        adapter_stage, target_layers = _resolve_target_layers(base_model_path, adapter_path, stage_info)
        dropped_layers = _read_dropped_layers_from_manifest(
            base_model_path,
            merge_stage=adapter_stage,
            stage_info=stage_info,
        )
        if adapter_stage:
            print(f"  Adapter stage inferred: {adapter_stage}")
        if dropped_layers:
            print(f"  PassLayer restore targets: {dropped_layers}")
        else:
            print(f"  PassLayer restore targets: []")

        print(f"  Loading model from {current_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            current_model_path,
            torch_dtype=torch.float16,
            device_map=_device_map_from_arg(device),
            trust_remote_code=True,
        )

        print(f"  Loading adapter: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        _enforce_adapter_scope(model, target_layers)
        merged_model = model.merge_and_unload()
        print(f"  ✓ Merged stage {i}")

        # ★ PassLayer 복원
        print(f"  Restoring PassLayers...")
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

    # 임시 디렉터리 정리
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


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter(s) into Mistral base model with tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 단일 머지
  python -m mistral_prune_lora.pruning.mistral_merge_adapter \\
    --base_model ./pruning/A \\
    --adapter_path ./adapters/A_lora/stageA \\
    --output_dir ./merged/A_merged

  # 순차 머지 (A+B)
  python -m mistral_prune_lora.pruning.mistral_merge_adapter \\
    --base_model ./pruning/A \\
    --adapter_paths ./adapters/A_lora/stageA ./adapters/B_lora/stageB \\
    --output_dir ./merged/AB_merged

  # 원본 모델 토크나이저 명시
  python -m mistral_prune_lora.pruning.mistral_merge_adapter \\
    --base_model ./pruning/A \\
    --adapter_path ./adapters/A_lora/stageA \\
    --output_dir ./merged/A_merged \\
    --tokenizer_path mistralai/Mistral-7B-v0.1
        """,
    )
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base model (pruned A model)")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to single adapter (for single merge)")
    parser.add_argument("--adapter_paths", type=str, nargs="+", default=None,
                        help="Paths to multiple adapters (for sequential merge)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for merged model")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Explicit tokenizer path (e.g., mistralai/Mistral-7B-v0.1)")
    parser.add_argument("--no_verify", action="store_true",
                        help="Skip post-merge verification")
    parser.add_argument("--save_bundle_only", action="store_true",
                        help="Save only target layers as bundle (layer_XXX.safetensors). Single-merge only.")

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
            save_bundle_only=args.save_bundle_only,
        )
    else:
        merge_multiple_adapters(
            base_model_path=args.base_model,
            adapter_paths=args.adapter_paths,
            output_dir=args.output_dir,
            device=args.device,
            tokenizer_path=args.tokenizer_path,
            verify=not args.no_verify,
            save_bundle_only=args.save_bundle_only,
        )


if __name__ == "__main__":
    main()
