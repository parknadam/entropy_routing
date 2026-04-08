# model_utils.py
# Gemma / LLaMA / OPT 공용 아키텍처 헬퍼

import inspect

from transformers import PretrainedConfig


def detect_arch(model_name_or_config) -> str:
    """
    모델 이름 또는 config 객체로부터 아키텍처를 판별합니다.
    반환: "gemma" | "llama" | "opt" | "unknown"
    """
    if isinstance(model_name_or_config, str):
        name = model_name_or_config.lower()
        if "gemma" in name:
            return "gemma"
        if "opt" in name:
            return "opt"
        if "llama" in name or "vicuna" in name or "alpaca" in name:
            return "llama"
        return "unknown"

    # PretrainedConfig 객체
    if isinstance(model_name_or_config, PretrainedConfig):
        mt = getattr(model_name_or_config, "model_type", "").lower()
        if "gemma" in mt:
            return "gemma"
        if "opt" in mt:
            return "opt"
        if "llama" in mt:
            return "llama"
        return "unknown"

    return "unknown"


def is_opt_arch(arch: str) -> bool:
    return arch == "opt"


def get_layers(model, arch: str):
    """
    아키텍처에 맞는 디코더 레이어 리스트(nn.ModuleList)를 반환합니다.
    - OPT: model.model.decoder.layers
    - Gemma / LLaMA: model.model.layers
    """
    if arch == "opt":
        return model.model.decoder.layers
    # gemma, llama, 기타 HF CausalLM 호환 모델
    return model.model.layers


def get_num_layers(model, arch: str) -> int:
    return len(get_layers(model, arch))


def get_layer_prefix(arch: str) -> str:
    if arch == "opt":
        return "model.decoder.layers."
    return "model.layers."


def detect_layer_return_tuple(model) -> bool:
    """
    transformers 버전에 따라 decoder layer가 tensor 또는 tuple을 반환할 수 있어
    PassLayer도 같은 계약을 따르도록 휴리스틱으로 판별합니다.
    Gemma 현재 구현은 tensor 반환이 기본이므로 fallback도 False로 둡니다.
    """
    try:
        core = model.model if hasattr(model, "model") else model
        src = inspect.getsource(core.forward)
        if "layer_outputs[0]" in src or "layer_outputs = decoder_layer" in src:
            return True
        if "hidden_states = decoder_layer" in src and "layer_outputs[0]" not in src:
            return False
    except Exception:
        pass
    return False
