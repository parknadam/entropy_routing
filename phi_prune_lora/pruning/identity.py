# Phi-3 용 항등(pass-through) 레이어
# ─────────────────────────────────────────────────────────────
# 변경 사유:
#   1. 클래스명을 Phi3PassLayer로 변경 (모델 아키텍처 명확화)
#   2. forward 시그니처를 Phi3DecoderLayer에 맞춤:
#      - position_embeddings 파라미터 추가 (Phi-3는 RoPE를 position_embeddings tuple로 전달)
#      - cache_position 파라미터 추가 (Phi-3 추론 시 사용)
#      - output_attentions 파라미터 유지 (HF 내부 호출 호환)
#   3. 반환 형식을 최신 transformers와 구 버전 모두 호환하도록
#      tuple 반환 유지 (Phi3Model의 내부 루프가 layer_outputs[0]으로 접근)
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn


class Phi3PassLayer(nn.Module):
    """
    Phi-3 디코더 레이어와 동일한 forward 시그니처를 가지는 초경량 항등 패스.
    프루닝된 위치에 삽입되어 hidden_states를 그대로 통과시킵니다.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,  # Phi-3 전용: (cos, sin) tuple
        **kwargs,
    ):
        # Phi3DecoderLayer는 transformers 버전에 따라 반환 형식이 다름:
        #   - 구 버전 (4.40~4.44): tuple (hidden_states, self_attn_weights, present_key_value)
        #   - 최신 버전 (4.45+):   torch.Tensor (hidden_states만)
        # 안전하게 tuple 반환을 유지하면 양쪽 모두 호환됨
        # (Phi3Model 내부 루프가 layer_outputs[0] 또는 직접 사용)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)          # attn_weights placeholder
        if use_cache:
            outputs += (past_key_value,)  # present_key_value placeholder
        return outputs