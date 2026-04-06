# identity.py
# Mistral/LLaMA 레이어 보관을 위한 PassLayer 코드
import torch
import torch.nn as nn

class PassLayer(nn.Module):
    """
    Mistral과 LLaMA 디코더 레이어를 대체하는 초경량 패스 레이어.
    - Mistral: MistralDecoderLayer
    - LLaMA: LlamaDecoderLayer
    동일한 forward 시그니처를 유지하여 호환성 보장
    """
    def __init__(self, hidden_size: int):
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
        **kwargs
    ):
        # HF Mistral/LlamaDecoderLayer는 보통 tuple을 반환
        # use_cache=False면 (hidden_states,) 형태로 반환
        if use_cache:
            # present_key_value를 past_key_value 그대로 넘기는 "더미 패스"
            return (hidden_states, None, past_key_value)
        return (hidden_states,)


# Backward compatibility alias
LlamaPassLayer = PassLayer
MistralPassLayer = PassLayer