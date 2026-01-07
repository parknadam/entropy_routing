# eval_script.py
import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# 1) PassLayer (프루닝된 레이어 자리 유지용)
# -----------------------------
class LlamaPassLayer(nn.Module):
    """
    HuggingFace LlamaDecoderLayer와 호환되도록 forward 반환 형태를 맞춘 pass layer.
    (hidden_states 그대로 통과)
    """
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
        present = past_key_value if use_cache else None
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (None,)
        if use_cache:
            outputs = outputs + (present,)
        return outputs


def _get_llama_layers(model) -> nn.ModuleList:
    # transformers 버전 차이 대응
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        return model.model.model.layers
    raise RuntimeError("LLaMA layers 경로를 찾지 못했어요. model.model.layers 형태인지 확인해줘.")


# -----------------------------
# 2) bundles/B,C에서 레이어 인덱스 파싱
# -----------------------------
_LAYER_RE = re.compile(r"layer_(\d+)\.safetensors$")

def _collect_layer_indices(bundle_dir: Path) -> List[int]:
    idxs = []
    for p in bundle_dir.glob("layer_*.safetensors"):
        m = _LAYER_RE.search(p.name)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def _strip_layer_prefix(sd: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, torch.Tensor]:
    """
    bundle에 저장된 key가 다음 중 어떤 형태든 layer module에 로드 가능하게 변환:
    - "model.layers.{i}.self_attn.q_proj.weight"
    - "model.model.layers.{i}...."
    - 이미 "self_attn.q_proj.weight" 형태(접두어 없음)
    """
    out = {}
    needle1 = f"layers.{layer_idx}."
    for k, v in sd.items():
        if needle1 in k:
            out[k.split(needle1, 1)[1]] = v
        else:
            out[k] = v
    return out


# -----------------------------
# 3) Stage(A/AB/FULL) 모델 구성 (서버리스 없이 로컬에서)
# -----------------------------
def build_stage_model(
    base_model: str,
    bundles_dir: Optional[str],
    stage: str,
    device: str,
    dtype: torch.dtype,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Dict]:
    """
    - base_model: 원본 LLaMA2-7B 경로(or HF id)
    - bundles_dir: layeronly_drop의 --save_removed_dir (그 안에 B/, C/가 있다고 가정)
    - stage:
        A    : B,C 모두 PassLayer
        AB   : B는 복구(원래 레이어), C는 PassLayer
        FULL : B,C 모두 복구(=원본에 가까움)
    """
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    meta = {"stage": stage, "B": [], "C": [], "removed": []}

    if bundles_dir is None:
        # bundles 없이 FULL처럼 사용
        return model, tok, meta

    bundles_dir = Path(bundles_dir)
    B_dir = bundles_dir / "B"
    C_dir = bundles_dir / "C"

    B_idx = _collect_layer_indices(B_dir) if B_dir.exists() else []
    C_idx = _collect_layer_indices(C_dir) if C_dir.exists() else []
    removed = sorted(set(B_idx) | set(C_idx))

    meta.update({"B": B_idx, "C": C_idx, "removed": removed})

    layers = _get_llama_layers(model)

    # A/AB에서 PassLayer로 치환할 인덱스 결정
    if stage.upper() == "A":
        pass_indices = removed
        restore_indices = []  # 따로 복구 없음
    elif stage.upper() == "AB":
        pass_indices = C_idx
        restore_indices = B_idx
    elif stage.upper() == "FULL":
        pass_indices = []
        restore_indices = B_idx + C_idx
    else:
        raise ValueError("stage는 A / AB / FULL 중 하나여야 해요.")

    # PassLayer 삽입
    for i in pass_indices:
        if 0 <= i < len(layers):
            layers[i] = LlamaPassLayer().to(device)

    # 복구 레이어는 bundle의 safetensors로 weights를 다시 로드(선택적으로 정확성↑)
    # (base_model이 이미 동일 weights면 사실상 동일하지만, bundle 기반 “정확한 복구”를 위해 로드)
    def _load_layer_weights(layer_i: int, fpath: Path):
        sd = load_file(str(fpath), device="cpu")
        sd2 = _strip_layer_prefix(sd, layer_i)
        missing, unexpected = layers[layer_i].load_state_dict(sd2, strict=False)
        # strict=True로 하고 싶으면 여기서 바꿔도 됨. (버전차 키 이름이 다르면 strict=False가 안전)
        return missing, unexpected

    for i in restore_indices:
        # B 또는 C에 layer_i 파일이 있을 수 있음
        fB = B_dir / f"layer_{i}.safetensors"
        fC = C_dir / f"layer_{i}.safetensors"
        if fB.exists():
            _load_layer_weights(i, fB)
        elif fC.exists():
            _load_layer_weights(i, fC)

    return model, tok, meta


# -----------------------------
# 4) 프롬프트 엔트로피 / PPL 계산
# -----------------------------
@torch.no_grad()
def prompt_metrics(model, tokenizer, prompt: str, max_length: int, device: str) -> Dict[str, float]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)      # [1, L]
    attn = enc["attention_mask"].to(device)      # [1, L]

    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits                           # [1, L, V]

    # shift: logits[:, i] predicts token i+1
    logits_use = logits[:, :-1, :]               # [1, L-1, V]
    target = input_ids[:, 1:]                    # [1, L-1]
    mask = attn[:, 1:].to(logits_use.dtype)      # [1, L-1]

    # (A) Prompt NLL / PPL (정답 토큰 기반)
    log_probs = F.log_softmax(logits_use, dim=-1)
    token_logp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
    token_nll = -token_logp * mask
    denom = mask.sum().clamp_min(1.0)
    nll_mean = (token_nll.sum() / denom).item()
    ppl = float(math.exp(nll_mean))

    # (B) Prompt Predictive Entropy (모델 분포 엔트로피)
    probs = F.softmax(logits_use, dim=-1)
    ent = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)          # [1, L-1]
    ent_mean = ((ent * mask).sum() / denom).item()

    # (C) 프롬프트 마지막 위치에서의 next-token 엔트로피 (라우팅에 유용)
    last_logits = logits[:, -1, :]                                      # [1, V]
    last_probs = F.softmax(last_logits, dim=-1)
    last_ent = float((-(last_probs * last_probs.clamp_min(1e-12).log()).sum()).item())

    return {
        "prompt_nll": nll_mean,
        "prompt_ppl": ppl,
        "prompt_pred_entropy_mean": ent_mean,
        "prompt_last_entropy": last_ent,
        "prompt_tokens_scored": int(denom.item()),
        "prompt_len_tokens": int(attn.sum().item()),
    }


# -----------------------------
# 5) 응답 생성 + 생성 토큰 엔트로피(선택)
# -----------------------------
@torch.no_grad()
def generate_with_entropy(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> Dict:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    # output_scores=True로 step별 logits 얻기
    """ gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature if temperature > 0.0 else 1.0,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    ) """
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.12,
        no_repeat_ngram_size=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True,
    )


    seq = gen.sequences[0]  # [L + new]
    text = tokenizer.decode(seq, skip_special_tokens=True)

    # 생성 step logits: gen.scores는 길이 = 생성 토큰 수, 각 원소 shape [1, V]
    ent_list = []
    for step_logits in gen.scores:
        probs = F.softmax(step_logits[0], dim=-1)
        ent = float((-(probs * probs.clamp_min(1e-12).log()).sum()).item())
        ent_list.append(ent)

    gen_ent_mean = float(sum(ent_list) / len(ent_list)) if ent_list else 0.0
    return {
        "text": text,
        "gen_entropy_mean": gen_ent_mean,
        "gen_entropy_per_step": ent_list,
        "gen_tokens": len(ent_list),
    }


# -----------------------------
# 6) main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="원본 모델 경로 or HF id (예: meta-llama/Llama-2-7b-hf 또는 로컬 캐시 경로)")
    ap.add_argument("--bundles_dir", required=True, help="layeronly_drop.py의 --save_removed_dir (B/, C/ 포함)")
    ap.add_argument("--stage", default="A", choices=["A", "AB", "FULL"], help="복구 단계")
    ap.add_argument("--prompt", required=True, help="입력 프롬프트")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    model, tok, meta = build_stage_model(
        base_model=args.base_model,
        bundles_dir=args.bundles_dir,
        stage=args.stage,
        device=args.device,
        dtype=dtype,
    )

    pm = prompt_metrics(model, tok, args.prompt, max_length=args.max_length, device=args.device)
    gm = generate_with_entropy(
        model, tok, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device
    )

    print("\n=== STAGE META ===")
    print(meta)

    print("\n=== PROMPT METRICS ===")
    for k, v in pm.items():
        print(f"{k}: {v}")

    print("\n=== GENERATION ===")
    print(f"gen_tokens: {gm['gen_tokens']}")
    print(f"gen_entropy_mean: {gm['gen_entropy_mean']}")
    print("\n--- output text ---")
    print(gm["text"])


if __name__ == "__main__":
    main()
