# 레이어 보관 위한 코드 (Gemma 중심)
import torch.nn as nn


class PassLayer(nn.Module):
    """
    디코더 레이어와 동일한 forward 시그니처를 가지는 초경량 패스스루 레이어.
    Gemma, LLaMA, OPT 모두 호환.
    """

    def __init__(self, hidden_size: int, return_tuple: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.return_tuple = return_tuple

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if not self.return_tuple:
            return hidden_states

        cache_obj = past_key_values if past_key_values is not None else past_key_value
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (cache_obj,)
        return outputs


# 하위 호환 별칭
LlamaPassLayer = PassLayer
GemmaPassLayer = PassLayer
