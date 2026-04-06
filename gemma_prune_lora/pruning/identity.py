# 레이어 보관 위한 코드 (Gemma / LLaMA / OPT 공용)
import torch.nn as nn


class PassLayer(nn.Module):
    """
    디코더 레이어와 동일한 forward 시그니처를 가지는 초경량 패스스루 레이어.
    Gemma, LLaMA, OPT 모두 호환.
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
        **kwargs,
    ):
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs


# 하위 호환 별칭
LlamaPassLayer = PassLayer
GemmaPassLayer = PassLayer
