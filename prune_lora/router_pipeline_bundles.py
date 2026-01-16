# router_pipeline_bundles.py
import argparse
import json
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# 1) PassLayer (드랍 레이어 자리 유지용)
# -----------------------------
class LlamaPassLayer(nn.Module):
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


# -----------------------------
# 2) bundle 메타 읽기 & 레이어 인덱스 추출
# -----------------------------
def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_bundle_indices(bundle_dir: str) -> List[int]:
    """
    bundles/B, bundles/C 아래에 layer_{idx}.safetensors가 있다고 가정.
    혹시 bundle_meta.json이 있으면 거기서도 indices를 찾도록 함(여러 키 시도).
    """
    meta_path = os.path.join(bundle_dir, "bundle_meta.json")
    if os.path.exists(meta_path):
        meta = _read_json(meta_path)
        # 다양한 키 케이스 대응
        for k in ["indices", "layer_indices", "layers", "restored_indices", "kept_indices"]:
            if k in meta and isinstance(meta[k], list):
                return sorted([int(x) for x in meta[k]])
        # 혹시 {"B":[...]} 같은 구조일 수도
        if "B" in meta and isinstance(meta["B"], list):
            return sorted([int(x) for x in meta["B"]])
        if "C" in meta and isinstance(meta["C"], list):
            return sorted([int(x) for x in meta["C"]])

    # fallback: 파일명에서 파싱
    indices = []
    for fn in os.listdir(bundle_dir):
        if fn.startswith("layer_") and fn.endswith(".safetensors"):
            idx = int(fn[len("layer_") : -len(".safetensors")])
            indices.append(idx)
    return sorted(indices)


# -----------------------------
# 3) A 디렉토리에서 "드랍된 레이어 인덱스" 찾기
# -----------------------------
def get_removed_indices(a_dir: str, bundles_dir: str) -> List[int]:
    """
    A/manifest.json 또는 A/prune_log.json 또는 A/layers_map.json 등에서
    removed indices를 찾으려고 시도. 없으면 B∪C로 추정.
    """
    candidates = [
        os.path.join(a_dir, "manifest.json"),
        os.path.join(a_dir, "prune_log.json"),
        os.path.join(a_dir, "layers_map.json"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            meta = _read_json(path)
            for k in ["removed", "removed_indices", "dropped", "drop_indices"]:
                if k in meta and isinstance(meta[k], list):
                    return sorted([int(x) for x in meta[k]])
            # layers_map.json이 {"removed":[...]} 형태가 아닐 수도 있어 fallback
        except Exception:
            pass

    # fallback: bundles/B,C의 합집합 = 제거 대상 레이어라고 가정
    b_idx = get_bundle_indices(os.path.join(bundles_dir, "B"))
    c_idx = get_bundle_indices(os.path.join(bundles_dir, "C"))
    return sorted(list(set(b_idx + c_idx)))


def ensure_passlayers(model: AutoModelForCausalLM, removed: List[int]):
    """
    save_pretrained로 저장된 A를 다시 로드하면, 드랍 레이어가 PassLayer로 남아있지 않고
    LlamaDecoderLayer가 랜덤 init 상태일 수 있음.
    그래서 removed 레이어는 강제로 PassLayer로 치환해 '진짜 A'를 재현.
    """
    layers = model.model.layers
    for idx in removed:
        layers[idx] = LlamaPassLayer()


# -----------------------------
# 4) bundle의 layer_{idx}.safetensors를 모델 레이어로 복원
# -----------------------------
def _strip_prefix_for_layer(sd: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, torch.Tensor]:
    """
    layer 파일에 저장된 key가
      - 'model.layers.{idx}.xxx'
      - 'layers.{idx}.xxx'
      - 'xxx'(이미 레이어 내부키)
    등 다양할 수 있어서, 가능한 prefix를 제거해서 레이어 module.load_state_dict에 맞춤.
    """
    prefixes = [
        f"model.layers.{layer_idx}.",
        f"model.model.layers.{layer_idx}.",
        f"layers.{layer_idx}.",
    ]
    # 가장 많이 걸리는 prefix를 찾아 제거
    for pref in prefixes:
        if any(k.startswith(pref) for k in sd.keys()):
            return {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
    return sd  # 이미 레이어 내부키라고 가정


def restore_layers_from_bundle(
    model: AutoModelForCausalLM,
    bundle_dir: str,
    indices: List[int],
):
    """
    model.model.layers[idx]가 PassLayer(또는 random layer)인 상태에서,
    해당 idx를 진짜 LlamaDecoderLayer로 교체하고 weights를 로드.
    """
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer  # transformers 버전에 따라 경로 동일

    layers = model.model.layers
    config = model.config

    for idx in indices:
        layer_path = os.path.join(bundle_dir, f"layer_{idx}.safetensors")
        if not os.path.exists(layer_path):
            raise FileNotFoundError(f"missing bundle layer file: {layer_path}")

        # 1) 새 decoder layer 인스턴스 생성
        try:
            new_layer = LlamaDecoderLayer(config, layer_idx=idx)
        except TypeError:
            # 구버전 transformers 호환
            new_layer = LlamaDecoderLayer(config)

        # device/dtype 맞추기
        new_layer.to(next(model.parameters()).device)
        new_layer.to(dtype=next(model.parameters()).dtype)

        # 2) safetensors 로드
        sd = load_file(layer_path)  # CPU tensors
        sd = _strip_prefix_for_layer(sd, idx)

        # 3) dtype/device 정렬 후 로드
        for k in list(sd.keys()):
            sd[k] = sd[k].to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

        missing, unexpected = new_layer.load_state_dict(sd, strict=False)
        # 너무 시끄러우면 아래 출력 주석처리
        if len(unexpected) > 0:
            print(f"[restore] layer {idx}: unexpected keys (show 5) = {unexpected[:5]}")
        if len(missing) > 0:
            print(f"[restore] layer {idx}: missing keys (show 5) = {missing[:5]}")

        # 4) 모델에 삽입
        layers[idx] = new_layer


# -----------------------------
# 5) completion NLL (품질 지표) 계산
# -----------------------------
@torch.no_grad()
def completion_nll(model, tok, prompts, completions, device, max_len=2048, batch_size=2):
    model.eval()
    losses = []

    for i in range(0, len(prompts), batch_size):
        p_batch = prompts[i:i+batch_size]
        c_batch = completions[i:i+batch_size]

        full = [p + c for p, c in zip(p_batch, c_batch)]
        enc_full = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        input_ids = enc_full["input_ids"].to(device)
        attn = enc_full["attention_mask"].to(device)

        enc_p = tok(p_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        p_lens = enc_p["attention_mask"].sum(dim=1).to(device)  # [B]

        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits  # [B, L, V]

        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)   # [B, L-1, V]
        target = input_ids[:, 1:]                                # [B, L-1]

        B, Lm1 = target.shape
        idx = torch.arange(Lm1, device=device).unsqueeze(0).expand(B, -1)
        comp_mask = idx >= (p_lens.unsqueeze(1) - 1)
        comp_mask = comp_mask & attn[:, 1:].bool()

        lp_true = logp.gather(2, target.unsqueeze(-1)).squeeze(-1)
        nll = -lp_true

        cm = comp_mask.float()
        denom = cm.sum(dim=1).clamp_min(1.0)
        loss = (nll * cm).sum(dim=1) / denom
        losses.append(loss.detach().cpu())

    return torch.cat(losses, dim=0)


# -----------------------------
# 6) A에서 뽑는 라우터 피처(예시: 6차원)
# -----------------------------
@torch.no_grad()
def extract_prompt_features_A(model, tok, prompts, device, max_prompt_len=1024, batch_size=8):
    model.eval()
    feats = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, output_hidden_states=True)
        logits = out.logits              # [B,L,V]
        hlast = out.hidden_states[-1]    # [B,L,H]

        valid = attn[:, :-1].bool() & attn[:, 1:].bool()
        probs = torch.softmax(logits[:, :-1, :].float(), dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]

        vf = valid.float()
        denom = vf.sum(dim=1).clamp_min(1.0)
        ent_mean = (ent * vf).sum(dim=1) / denom
        margin_mean = (margin * vf).sum(dim=1) / denom

        last_idx = (vf.sum(dim=1).long() - 1).clamp_min(0)
        ent_last = ent.gather(1, last_idx.view(-1,1)).squeeze(1)
        margin_last = margin.gather(1, last_idx.view(-1,1)).squeeze(1)

        len_tokens = attn.sum(dim=1).float().clamp_max(float(max_prompt_len))

        last_tok_idx = (attn.sum(dim=1).long() - 1).clamp_min(0)
        last_h = hlast.gather(1, last_tok_idx.view(-1,1,1).expand(-1,1,hlast.size(-1))).squeeze(1)
        last_h_norm = last_h.float().norm(dim=-1)

        f = torch.stack([len_tokens, ent_mean, ent_last, margin_mean, margin_last, last_h_norm], dim=1)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)


# -----------------------------
# 7) 라벨: tau 통과하는 가장 싼 stage 선택
# -----------------------------
def make_labels(loss_A, loss_AB, loss_ABC, tau: float):
    N = loss_A.numel()
    y = torch.full((N,), 2, dtype=torch.long)  # default ABC
    y[loss_AB <= tau] = 1
    y[loss_A <= tau] = 0
    return y


# -----------------------------
# 8) build-data (A + bundles/B,C로 AB/ABC 구성)
# -----------------------------
def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if max_samples is not None and len(items) >= max_samples:
                break
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", required=True)
    ap.add_argument("--out_pt", required=True)

    ap.add_argument("--model_A", required=True, help="e.g., ./results/pruning/A")
    ap.add_argument("--bundles_dir", required=True, help="e.g., ./results/pruning/bundles (contains B/ and C/)")

    ap.add_argument("--tau", type=float, required=True, help="loss <= tau 를 '통과'로 간주")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--max_len", type=int, default=2048)

    ap.add_argument("--bs_feat", type=int, default=8)
    ap.add_argument("--bs_q", type=int, default=2)

    args = ap.parse_args()

    device = args.device
    items = read_jsonl(args.data_jsonl, args.max_samples)
    prompts = [it["prompt"] for it in items]
    completions = [it["completion"] for it in items]

    tok = AutoTokenizer.from_pretrained(args.model_A)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.float16 if args.fp16 else None
    model = AutoModelForCausalLM.from_pretrained(args.model_A, torch_dtype=dtype)
    model.to(device)

    # (중요) A 재현: removed 레이어를 PassLayer로 강제 치환
    removed = get_removed_indices(args.model_A, args.bundles_dir)
    ensure_passlayers(model, removed)

    # 1) 피처: A로만 뽑기
    X = extract_prompt_features_A(model, tok, prompts, device=device,
                                  max_prompt_len=args.max_prompt_len, batch_size=args.bs_feat)

    # 2) 품질(loss) 계산: A
    loss_A = completion_nll(model, tok, prompts, completions, device=device,
                            max_len=args.max_len, batch_size=args.bs_q)

    # 3) AB: B 번들 복원 후 loss 계산
    b_dir = os.path.join(args.bundles_dir, "B")
    b_idx = get_bundle_indices(b_dir)
    restore_layers_from_bundle(model, b_dir, b_idx)
    loss_AB = completion_nll(model, tok, prompts, completions, device=device,
                             max_len=args.max_len, batch_size=args.bs_q)

    # 4) ABC: C 번들 복원 후 loss 계산
    c_dir = os.path.join(args.bundles_dir, "C")
    c_idx = get_bundle_indices(c_dir)
    restore_layers_from_bundle(model, c_dir, c_idx)
    loss_ABC = completion_nll(model, tok, prompts, completions, device=device,
                              max_len=args.max_len, batch_size=args.bs_q)

    # 5) 라벨 생성
    y = make_labels(loss_A, loss_AB, loss_ABC, tau=args.tau)

    out = {
        "X": X,
        "y": y,
        "loss_A": loss_A,
        "loss_AB": loss_AB,
        "loss_ABC": loss_ABC,
        "tau": args.tau,
        "meta": {
            "model_A": args.model_A,
            "bundles_dir": args.bundles_dir,
            "removed_indices": removed,
            "B_indices": b_idx,
            "C_indices": c_idx,
        },
    }
    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    torch.save(out, args.out_pt)
    print(f"[build-data] saved: {args.out_pt}")


if __name__ == "__main__":
    main()
