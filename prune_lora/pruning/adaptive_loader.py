# adaptive_loader.py
# 번들 로더 + 레이어 복구 유틸
import os, json
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

from Code.PruningAndLoRA.lib.identity import LlamaPassLayer

def _get_layers(model, is_opt: bool):
    return model.model.decoder.layers if is_opt else model.model.layers

def patch_passlayers(model, dropped_indices, is_opt: bool):
    """A 로드 후, dropped 레이어를 PassLayer로 바꿔서 '진짜 A'를 만든다."""
    layers = _get_layers(model, is_opt)
    hs = model.config.hidden_size
    for i in dropped_indices:
        layers[i] = LlamaPassLayer(hs)
    return model

def make_decoder_layer(config, is_opt: bool):
    """번들로부터 복구할 때 사용할 레이어 클래스 생성."""
    if is_opt:
        from transformers.models.opt.modeling_opt import OPTDecoderLayer
        return OPTDecoderLayer(config)
    else:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        return LlamaDecoderLayer(config)

@torch.no_grad()
def rehydrate_layers_from_bundle(
    model,
    bundle_dir: str,
    indices: list[int],
    is_opt: bool,
    device: torch.device,
    dtype: torch.dtype,
    strict: bool = True,
):
    """
    bundle_dir/B 또는 bundle_dir/C 안의 layer_XXX.safetensors를 읽어서
    해당 레이어를 '진짜 디코더 레이어'로 교체 + weight 로드.
    """
    layers = _get_layers(model, is_opt)

    for idx in indices:
        path = os.path.join(bundle_dir, f"layer_{idx:03d}.safetensors")
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        # 1) 새 레이어 생성
        new_layer = make_decoder_layer(model.config, is_opt=is_opt)

        # 2) 가중치 로드 (번들은 layer.state_dict() 기반)
        sd = load_file(path, device="cpu")
        missing, unexpected = new_layer.load_state_dict(sd, strict=strict)

        if strict and (missing or unexpected):
            raise RuntimeError(f"Layer {idx}: missing={missing}, unexpected={unexpected}")

        # 3) 디바이스/타입 이동
        new_layer.to(device=device, dtype=dtype)
        new_layer.eval()

        # 4) 모델에 교체
        layers[idx] = new_layer

    return model

def load_manifest(manifest_path: str):
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_stageA_model(A_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        A_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)
    tok = AutoTokenizer.from_pretrained(A_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.eval()
    return model, tok
