# prompt_router.py
# 프롬프트 수준을 점수로 판단
import re
import torch

CODE_HINT = re.compile(r"```|def |class |import |SELECT |INSERT |UPDATE |FROM |JOIN |\{|\}|\(|\)|;")
MATH_HINT = re.compile(r"\b(prove|derive|theorem|lemma|증명|유도|정리|정확히|수식)\b", re.IGNORECASE)

@torch.no_grad()
def compute_prompt_metrics(model, tokenizer, prompt: str, device: torch.device, max_len: int = 2048):
    """
    A로 prefill 1회 돌려서 불확실성 지표를 계산.
    반환:
      avg_nll: 프롬프트 토큰 평균 NLL (낮을수록 쉬운 편)
      last_entropy: 마지막 위치의 next-token 엔트로피 (낮을수록 확신)
      top1_margin: top1-top2 확률 차 (클수록 확신)
      ntoks: 토큰 수
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # use_cache는 일단 False (PassLayer/서빙 구조 안정성을 위해)
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits  # [B, T, V]
    B, T, V = logits.shape
    ntoks = int(attn.sum().item())

    # ---- avg prompt NLL ----
    # predict token t+1 from logits at t
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attn[:, 1:].float()

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    nll_tok = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    denom = shift_mask.sum().clamp_min(1.0)
    avg_nll = float((nll_tok * shift_mask).sum().item() / denom.item())

    # ---- last token entropy & margin ----
    last_logits = logits[:, -1, :]
    p = torch.softmax(last_logits, dim=-1)

    entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(dim=-1).mean()
    last_entropy = float(entropy.item())

    top2 = torch.topk(p, k=2, dim=-1).values
    top1_margin = float((top2[:, 0] - top2[:, 1]).mean().item())

    return {
        "avg_nll": avg_nll,
        "last_entropy": last_entropy,
        "top1_margin": top1_margin,
        "ntoks": ntoks,
    }

def heuristic_force_deep(prompt: str) -> bool:
    """프롬프트만 보고도 '깊게 가는 게 안전'한 케이스를 먼저 거른다."""
    if len(prompt) > 4000:  # 매우 긴 입력
        return True
    if CODE_HINT.search(prompt):
        return True
    if MATH_HINT.search(prompt):
        return True
    return False

class PromptDepthRouter:
    """
    score가 낮으면 A, 중간이면 AB, 높으면 ABC.
    - avg_nll, entropy는 높을수록 어려움
    - margin은 높을수록 쉬움(확신)
    """
    def __init__(
        self,
        t_A: float = 3.2,
        t_AB: float = 4.2,
        w_nll: float = 1.0,
        w_ent: float = 0.7,
        w_margin: float = 1.2,
    ):
        self.t_A = t_A
        self.t_AB = t_AB
        self.w_nll = w_nll
        self.w_ent = w_ent
        self.w_margin = w_margin

    def score(self, m: dict) -> float:
        return self.w_nll * m["avg_nll"] + self.w_ent * m["last_entropy"] - self.w_margin * m["top1_margin"]

    def route(self, prompt: str, metrics: dict) -> tuple[str, float]:
        if heuristic_force_deep(prompt):
            return "ABC", float("inf")

        s = self.score(metrics)
        if s <= self.t_A:
            return "A", s
        elif s <= self.t_AB:
            return "AB", s
        else:
            return "ABC", s
