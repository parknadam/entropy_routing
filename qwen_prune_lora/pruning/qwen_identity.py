# qwen_prune_lora/pruning/qwen_identity.py
# Qwen 모델용 PassLayer

import torch
import torch.nn as nn


class QwenPassLayer(nn.Module):
    """
    Qwen 디코더 레이어와 동일한 forward 시그니처를 가지는 패스 레이어.
    QWenBlock의 forward와 호환되도록 설계.
    """
    def __init__(self, hidden_size: int, return_tuple: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.return_tuple = return_tuple

    def forward(
        self,
        hidden_states,
        rotary_pos_emb=None,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs
    ):
        """
        Qwen QWenBlock의 forward 시그니처와 일치.
        hidden_states를 그대로 통과시킴.
        """
        if not self.return_tuple:
            return hidden_states
        
        # Qwen은 보통 (hidden_states, present) 형태로 반환
        # present는 KV cache를 의미
        outputs = (hidden_states,)
        
        if use_cache:
            # layer_past를 그대로 present로 반환 (더미)
            outputs = outputs + (layer_past,)
        
        if output_attentions:
            # attention weights는 None으로
            outputs = outputs + (None,)
        
        return outputs