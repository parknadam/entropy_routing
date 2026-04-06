# model_utils.py
# Gemma / LLaMA / OPT 공용 아키텍처 헬퍼

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
