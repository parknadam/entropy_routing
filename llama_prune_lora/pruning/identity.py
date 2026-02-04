# 레이어 보관 위한 코드
import torch
import torch.nn as nn

""" class LlamaPassLayer(nn.Module):
    #LLaMA 디코더 레이어와 동일한 forward 시그니처를 가지는 초경량 패스.
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
        **kwargs
    ):
        # use_cache=False로 운용하므로 hidden_states만 반환해도 OK (HF LLaMA 호환)
        return hidden_states """

class LlamaPassLayer(nn.Module):
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
        **kwargs
    ):
        # HF LlamaDecoderLayer는 보통 tuple을 반환
        # use_cache=False면 (hidden_states,) 형태로 반환해도 호환이 좋아짐
        if use_cache:
            # present_key_value를 past_key_value 그대로 넘기는 "더미 패스"
            # (정확한 KV 캐시 보존까지는 별도 설계가 필요)
            return (hidden_states, None, past_key_value)
        return (hidden_states,)