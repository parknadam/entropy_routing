# 레이어 보관 위한 코드 (Falcon 구조 대응)
import torch
import torch.nn as nn


class FalconPassLayer(nn.Module):
    """
    Falcon 디코더 레이어와 동일한 forward 시그니처를 가지는 초경량 패스.
    FalconDecoderLayer의 forward는 다음을 반환:
      - use_cache=False: (hidden_states, )  또는  (hidden_states, attn_weights)
      - use_cache=True:  (hidden_states, present_kv) 또는 (hidden_states, attn_weights, present_kv)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        hidden_states,
        alibi=None,
        attention_mask=None,
        position_ids=None,
        layer_past=None,           # Falcon은 past_key_value 대신 layer_past 사용
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        # HF FalconDecoderLayer 반환 형식에 맞춤
        if use_cache:
            # present_key_value를 layer_past 그대로 넘기는 더미 패스
            return (hidden_states, layer_past)
        return (hidden_states,)