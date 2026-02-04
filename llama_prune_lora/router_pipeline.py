# router_pipeline.py
# ProgressiveServe-style stage router (A/AB/ABC) with:
# 1) offline labeling via NLL on completion
# 2) feature extraction from A only
# 3) router training (logistic / MLP)
# 4) temperature scaling calibration
# 5) threshold grid-search under quality constraint (fail-rate or regret)

"""
#데이터 생성(A 피처 + A/AB/ABC completion NLL + 라벨)
python prune_lora.router_pipeline.py build-data \
  --data_jsonl ./data/prompt_completion.jsonl \
  --out_pt ./router_data.pt \
  --model_A ./results/pruning/A \
  --model_AB ./results/pruning/AB \
  --model_ABC ./results/pruning/ABC \
  --tau 2.5 \
  --device cuda \
  --fp16

"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def parse_floats_csv(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_ints_csv(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# -------------------------
# StandardScaler (torch-only)
# -------------------------

class StandardScaler:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, X: torch.Tensor):
        # X: [N, m]
        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0).clamp_min(self.eps)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None
        return (X - self.mean) / self.std

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    def load_state_dict(self, sd: Dict):
        self.mean = sd["mean"]
        self.std = sd["std"]
        self.eps = sd.get("eps", 1e-6)
        return self


# -------------------------
# Router models
# -------------------------

class LogisticRouter(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)  # logits


class MLPRouter(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # logits


# -------------------------
# Temperature scaling (calibration)
# -------------------------

class TemperatureScaler(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        # optimize logT to keep T>0
        self.logT = nn.Parameter(torch.tensor(math.log(init_T), dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.logT).clamp(1e-3, 1e3)
        return logits / T

    @torch.no_grad()
    def temperature(self) -> float:
        return float(torch.exp(self.logT).item())


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, max_iter: int = 50) -> float:
    """
    logits: [N, C] (un-calibrated)
    y: [N]
    """
    scaler = TemperatureScaler(init_T=1.0)
    scaler.train()

    optimizer = torch.optim.LBFGS([scaler.logT], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits)
        loss = criterion(scaled, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.temperature()


# -------------------------
# Feature extraction from A (prompt only)
# -------------------------

@torch.no_grad()
def extract_prompt_features_A(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    device: str,
    max_prompt_len: int = 1024,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Returns features tensor [N, m] computed using A only.
    Features (m=6):
      0 len_tokens (clipped)
      1 entropy_mean over prompt positions (next-token dist)
      2 entropy_last
      3 margin_mean (top1 - top2 prob)
      4 margin_last
      5 last_hidden_norm (norm of last token hidden state)
    """
    model.eval()
    feats = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        # We want logits and hidden states for last hidden norm.
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            use_cache=False,
            output_hidden_states=True,
        )
        logits = out.logits  # [B, L, V]
        hidden_last = out.hidden_states[-1]  # [B, L, H]

        B, L, V = logits.shape

        # next-token distribution for positions 0..L-2 (since logits[t] predicts token t+1)
        # mask out padding positions:
        # valid positions for next-token: where attention_mask at t==1 and t+1 exists
        valid = attn[:, :-1].bool() & attn[:, 1:].bool()  # [B, L-1]
        logits_nt = logits[:, :-1, :]  # [B, L-1, V]

        # softmax probs
        probs = torch.softmax(logits_nt.float(), dim=-1)  # float32 for stability

        # entropy per position
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)  # [B, L-1]

        # margin per position: top1 - top2 probability
        top2 = torch.topk(probs, k=2, dim=-1).values  # [B, L-1, 2]
        margin = (top2[..., 0] - top2[..., 1])  # [B, L-1]

        # reduce with masks
        valid_f = valid.float()
        denom = valid_f.sum(dim=1).clamp_min(1.0)  # [B]

        ent_mean = (ent * valid_f).sum(dim=1) / denom
        margin_mean = (margin * valid_f).sum(dim=1) / denom

        # last valid next-token position index per sample
        # If prompt length is 1 token, valid will be all false -> handle safely
        last_idx = (valid_f.sum(dim=1).long() - 1).clamp_min(0)  # [B]
        # gather ent_last/margin_last: we need index in [0, L-2]
        ent_last = ent.gather(1, last_idx.view(-1,1)).squeeze(1)
        margin_last = margin.gather(1, last_idx.view(-1,1)).squeeze(1)

        # length tokens = sum(attn)
        len_tokens = attn.sum(dim=1).float().clamp_max(float(max_prompt_len))

        # last hidden norm: take last non-pad token hidden state
        last_tok_idx = (attn.sum(dim=1).long() - 1).clamp_min(0)  # [B]
        last_h = hidden_last.gather(
            1, last_tok_idx.view(-1,1,1).expand(-1,1,hidden_last.size(-1))
        ).squeeze(1)  # [B, H]
        last_h_norm = last_h.float().norm(dim=-1)

        f = torch.stack([len_tokens, ent_mean, ent_last, margin_mean, margin_last, last_h_norm], dim=1)  # [B,6]
        feats.append(f.cpu())

    return torch.cat(feats, dim=0)


# -------------------------
# NLL on completion (quality) for a given stage model
# -------------------------

@torch.no_grad()
def completion_nll(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    completions: List[str],
    device: str,
    max_len: int = 2048,
    batch_size: int = 4,
) -> torch.Tensor:
    """
    Computes mean NLL over completion tokens only.
    Returns loss [N] (lower is better).
    """
    model.eval()
    losses = []

    for i in range(0, len(prompts), batch_size):
        p_batch = prompts[i:i+batch_size]
        c_batch = completions[i:i+batch_size]

        # build full text and get token boundaries to mask completion tokens
        full = [p + c for p, c in zip(p_batch, c_batch)]
        enc_full = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        input_ids = enc_full["input_ids"].to(device)
        attn = enc_full["attention_mask"].to(device)

        # prompt lengths (tokenized separately to find boundary)
        enc_p = tok(p_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        p_lens = enc_p["attention_mask"].sum(dim=1).to(device)  # [B]

        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits  # [B, L, V]

        # next-token logprobs for positions 0..L-2
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)  # [B, L-1, V]
        target = input_ids[:, 1:]  # [B, L-1]

        # mask for completion tokens in target positions:
        # token position t in target corresponds to original token index t+1 in input_ids
        # completion starts at index p_len (0-based) in input_ids, so in target it's at t >= p_len-1
        B, Lm1 = target.shape
        idx = torch.arange(Lm1, device=device).unsqueeze(0).expand(B, -1)  # [B, L-1]
        comp_mask = idx >= (p_lens.unsqueeze(1) - 1)  # [B, L-1]
        comp_mask = comp_mask & attn[:, 1:].bool()  # exclude padding

        # gather logp of true targets
        lp_true = logp.gather(2, target.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        nll = -lp_true

        comp_mask_f = comp_mask.float()
        denom = comp_mask_f.sum(dim=1).clamp_min(1.0)
        loss = (nll * comp_mask_f).sum(dim=1) / denom  # [B]
        losses.append(loss.detach().cpu())

    return torch.cat(losses, dim=0)


# -------------------------
# Label policy: choose cheapest stage that passes tau
# -------------------------

def make_labels_min_cost_pass_tau(
    loss_A: torch.Tensor,
    loss_AB: torch.Tensor,
    loss_ABC: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    loss_*: [N] (lower is better)
    label: 0=A, 1=AB, 2=ABC
    """
    N = loss_A.numel()
    y = torch.full((N,), 2, dtype=torch.long)  # default ABC
    pass_A = loss_A <= tau
    pass_AB = loss_AB <= tau

    y[pass_AB] = 1
    y[pass_A] = 0
    return y


# -------------------------
# Train router
# -------------------------

@dataclass
class TrainConfig:
    model_type: str  # "logistic" or "mlp"
    hidden: int = 128
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 256
    seed: int = 0

def train_router(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    cfg: TrainConfig
) -> Tuple[nn.Module, StandardScaler, float]:
    set_seed(cfg.seed)

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xva = scaler.transform(X_val)

    in_dim = Xtr.size(1)
    if cfg.model_type == "logistic":
        model = LogisticRouter(in_dim)
    elif cfg.model_type == "mlp":
        model = MLPRouter(in_dim, hidden=cfg.hidden, dropout=cfg.dropout)
    else:
        raise ValueError("model_type must be logistic or mlp")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    Xtr = Xtr.to(device); y_train = y_train.to(device)
    Xva = Xva.to(device); y_val = y_val.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None

    for ep in range(1, cfg.epochs + 1):
        model.train()
        # shuffle
        idx = torch.randperm(Xtr.size(0), device=device)
        for s in range(0, Xtr.size(0), cfg.batch_size):
            b = idx[s:s+cfg.batch_size]
            xb, yb = Xtr[b], y_train[b]
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(Xva)
            pred = logits_val.argmax(dim=-1)
            acc = (pred == y_val).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[train] epoch={ep} val_acc={acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Fit temperature scaling on val logits (CPU ok)
    model.eval()
    with torch.no_grad():
        logits_val = model(Xva).detach().cpu()
        yv_cpu = y_val.detach().cpu()
    T = fit_temperature(logits_val, yv_cpu)
    print(f"[calib] temperature T={T:.4f}")

    # move model back to cpu for saving
    model.cpu()
    return model, scaler, T


# -------------------------
# Threshold selection (grid search)
# -------------------------

def decide_stage_from_probs(
    probs: torch.Tensor,  # [N,3]
    tA: float, tAB: float,
    dA: float, dAB: float,
) -> torch.Tensor:
    """
    probs: calibrated probabilities
    returns chosen stage [N] in {0,1,2}
    """
    p_sorted, idx_sorted = torch.sort(probs, dim=1, descending=True)
    p1 = p_sorted[:, 0]
    p2 = p_sorted[:, 1]

    pA = probs[:, 0]
    pAB = probs[:, 1]

    choose = torch.full((probs.size(0),), 2, dtype=torch.long)

    condA = (pA >= tA) & ((pA - p2) >= dA)
    choose[condA] = 0

    # for remaining samples only
    remaining = choose == 2
    condAB = remaining & (pAB >= tAB) & ((pAB - p2) >= dAB)
    choose[condAB] = 1

    return choose


def eval_policy(
    choose: torch.Tensor,           # [N]
    loss_by_stage: torch.Tensor,    # [N,3] (lower better)
    costs: torch.Tensor,            # [3]
    constraint: str,                # "failrate" or "regret"
    tau: float,
    eps: float,
) -> Dict[str, float]:
    """
    - failrate: fail if chosen loss > tau. constraint: fail_rate <= eps
    - regret: regret = max(0, chosen_loss - abc_loss). constraint: mean_regret <= eps
    """
    N = choose.numel()
    chosen_loss = loss_by_stage[torch.arange(N), choose]
    avg_cost = costs[choose].mean().item()
    avg_loss = chosen_loss.mean().item()

    if constraint == "failrate":
        fail = (chosen_loss > tau).float()
        fail_rate = fail.mean().item()
        ok = fail_rate <= eps
        return {
            "avg_cost": avg_cost,
            "avg_loss": avg_loss,
            "fail_rate": fail_rate,
            "ok": float(ok),
        }
    elif constraint == "regret":
        abc_loss = loss_by_stage[:, 2]
        regret = (chosen_loss - abc_loss).clamp_min(0.0)
        mean_regret = regret.mean().item()
        ok = mean_regret <= eps
        return {
            "avg_cost": avg_cost,
            "avg_loss": avg_loss,
            "mean_regret": mean_regret,
            "ok": float(ok),
        }
    else:
        raise ValueError("constraint must be failrate or regret")


def grid_search_thresholds(
    probs: torch.Tensor,            # [N,3] calibrated
    loss_by_stage: torch.Tensor,    # [N,3]
    costs: List[float],
    constraint: str,
    tau: float,
    eps: float,
    tA_list: List[float],
    tAB_list: List[float],
    dA_list: List[float],
    dAB_list: List[float],
) -> Dict:
    costs_t = torch.tensor(costs, dtype=torch.float32)

    best = None
    best_metrics = None

    for tA in tA_list:
        for dA in dA_list:
            for tAB in tAB_list:
                for dAB in dAB_list:
                    choose = decide_stage_from_probs(probs, tA=tA, tAB=tAB, dA=dA, dAB=dAB)
                    metrics = eval_policy(
                        choose=choose,
                        loss_by_stage=loss_by_stage,
                        costs=costs_t,
                        constraint=constraint,
                        tau=tau,
                        eps=eps,
                    )
                    if metrics["ok"] < 0.5:
                        continue

                    # minimize avg_cost; tie-break by avg_loss
                    if best is None or (metrics["avg_cost"] < best_metrics["avg_cost"] - 1e-12) or (
                        abs(metrics["avg_cost"] - best_metrics["avg_cost"]) <= 1e-12 and metrics["avg_loss"] < best_metrics["avg_loss"]
                    ):
                        best = {"tA": tA, "tAB": tAB, "dA": dA, "dAB": dAB}
                        best_metrics = metrics

    if best is None:
        return {"found": False, "reason": "No feasible thresholds under constraint.", "best": None}
    return {"found": True, "best": best, "metrics": best_metrics}


# -------------------------
# Save/Load bundles
# -------------------------

def save_pt(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)

def load_pt(path: str) -> Dict:
    return torch.load(path, map_location="cpu")


# -------------------------
# Subcommands
# -------------------------

def cmd_build_data(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    items = read_jsonl(args.data_jsonl, max_samples=args.max_samples)
    prompts = [it["prompt"] for it in items]
    completions = [it["completion"] for it in items]

    # Load tokenizer once (assume compatible across stages)
    tok = AutoTokenizer.from_pretrained(args.model_A)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Load A for features
    print("[build] loading A for features...")
    model_A = AutoModelForCausalLM.from_pretrained(args.model_A, torch_dtype=torch.float16 if args.fp16 else None)
    model_A.to(device)

    print("[build] extracting features from A...")
    X = extract_prompt_features_A(
        model=model_A,
        tok=tok,
        prompts=prompts,
        device=device,
        max_prompt_len=args.max_prompt_len,
        batch_size=args.batch_size_feat,
    )
    del model_A
    torch.cuda.empty_cache()

    # Load stage models for quality (NLL on completion)
    def load_model(path: str):
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16 if args.fp16 else None)
        m.to(device)
        return m

    print("[build] computing completion NLL for A/AB/ABC ...")
    modelA_q = load_model(args.model_A)
    loss_A = completion_nll(modelA_q, tok, prompts, completions, device, max_len=args.max_len, batch_size=args.batch_size_q)
    del modelA_q
    torch.cuda.empty_cache()

    modelAB_q = load_model(args.model_AB)
    loss_AB = completion_nll(modelAB_q, tok, prompts, completions, device, max_len=args.max_len, batch_size=args.batch_size_q)
    del modelAB_q
    torch.cuda.empty_cache()

    modelABC_q = load_model(args.model_ABC)
    loss_ABC = completion_nll(modelABC_q, tok, prompts, completions, device, max_len=args.max_len, batch_size=args.batch_size_q)
    del modelABC_q
    torch.cuda.empty_cache()

    # Make labels by min-cost stage passing tau
    y = make_labels_min_cost_pass_tau(loss_A, loss_AB, loss_ABC, tau=args.tau)

    out = {
        "X": X,  # [N,m]
        "y": y,  # [N]
        "loss_A": loss_A,
        "loss_AB": loss_AB,
        "loss_ABC": loss_ABC,
        "tau": args.tau,
        "meta": {
            "feature_dim": X.size(1),
            "max_prompt_len": args.max_prompt_len,
            "max_len": args.max_len,
            "model_A": args.model_A,
            "model_AB": args.model_AB,
            "model_ABC": args.model_ABC,
        },
    }
    save_pt(out, args.out_pt)
    print(f"[build] saved dataset to: {args.out_pt}")


def cmd_train(args):
    data = load_pt(args.data_pt)
    X = data["X"].float()
    y = data["y"].long()

    set_seed(args.seed)
    N = X.size(0)
    idx = torch.randperm(N)
    n_val = int(N * args.val_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    cfg = TrainConfig(
        model_type=args.model_type,
        hidden=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model, scaler, T = train_router(X_train, y_train, X_val, y_val, cfg)

    bundle = {
        "router_state": model.state_dict(),
        "scaler": scaler.state_dict(),
        "T": T,
        "model_type": args.model_type,
        "in_dim": X.size(1),
        "hidden": args.hidden,
        "dropout": args.dropout,
        "meta": {
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        },
        "splits": {
            "train_idx": tr_idx.cpu(),
            "val_idx": val_idx.cpu(),
        },
    }
    save_pt(bundle, args.out_router_pt)
    print(f"[train] saved router bundle to: {args.out_router_pt}")


def _load_router_bundle(router_pt: str) -> Tuple[nn.Module, StandardScaler, float, Dict]:
    b = load_pt(router_pt)
    in_dim = int(b["in_dim"])
    model_type = b["model_type"]
    if model_type == "logistic":
        model = LogisticRouter(in_dim)
    else:
        model = MLPRouter(in_dim, hidden=int(b["hidden"]), dropout=float(b["dropout"]))
    model.load_state_dict(b["router_state"])
    model.eval()

    scaler = StandardScaler().load_state_dict(b["scaler"])
    T = float(b["T"])
    return model, scaler, T, b


def cmd_tune(args):
    data = load_pt(args.data_pt)
    X = data["X"].float()
    loss_A = data["loss_A"].float()
    loss_AB = data["loss_AB"].float()
    loss_ABC = data["loss_ABC"].float()

    model, scaler, T, b = _load_router_bundle(args.router_pt)
    val_idx = b["splits"]["val_idx"].long()

    X_val = X[val_idx]
    loss_val = torch.stack([loss_A[val_idx], loss_AB[val_idx], loss_ABC[val_idx]], dim=1)  # [Nv,3]

    # router probs (calibrated)
    Xs = scaler.transform(X_val)
    with torch.no_grad():
        logits = model(Xs)  # [Nv,3]
        logits = logits / max(T, 1e-6)
        probs = torch.softmax(logits, dim=-1)

    # grid ranges
    tA_list = parse_floats_csv(args.tA_list)
    tAB_list = parse_floats_csv(args.tAB_list)
    dA_list = parse_floats_csv(args.dA_list)
    dAB_list = parse_floats_csv(args.dAB_list)
    costs = parse_floats_csv(args.costs)
    assert len(costs) == 3

    res = grid_search_thresholds(
        probs=probs,
        loss_by_stage=loss_val,
        costs=costs,
        constraint=args.constraint,
        tau=args.tau,
        eps=args.eps,
        tA_list=tA_list,
        tAB_list=tAB_list,
        dA_list=dA_list,
        dAB_list=dAB_list,
    )

    out = {
        "found": res["found"],
        "result": res,
        "constraint": args.constraint,
        "tau": args.tau,
        "eps": args.eps,
        "costs": costs,
        "grid": {
            "tA_list": tA_list,
            "tAB_list": tAB_list,
            "dA_list": dA_list,
            "dAB_list": dAB_list,
        },
        "router_pt": args.router_pt,
        "data_pt": args.data_pt,
    }
    save_pt(out, args.out_tune_pt)
    print(f"[tune] saved tuning result to: {args.out_tune_pt}")
    print(json.dumps(res, ensure_ascii=False, indent=2))


def cmd_predict(args):
    # runtime demo: prompt -> features(A) -> router -> calibrated probs -> stage decision
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    router, scaler, T, _ = _load_router_bundle(args.router_pt)
    tune = load_pt(args.tune_pt)
    if not tune["result"]["found"]:
        raise RuntimeError("No feasible thresholds found in tune file.")

    best = tune["result"]["best"]
    tA, tAB, dA, dAB = best["tA"], best["tAB"], best["dA"], best["dAB"]

    # load A only for feature extraction
    tok = AutoTokenizer.from_pretrained(args.model_A)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model_A = AutoModelForCausalLM.from_pretrained(args.model_A, torch_dtype=torch.float16 if args.fp16 else None)
    model_A.to(device)

    prompts = [args.prompt]
    X = extract_prompt_features_A(model_A, tok, prompts, device=device, max_prompt_len=args.max_prompt_len, batch_size=1)
    Xs = scaler.transform(X.float())
    with torch.no_grad():
        logits = router(Xs)
        logits = logits / max(T, 1e-6)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    # apply decision
    p_sorted, _ = torch.sort(probs, descending=True)
    p2 = p_sorted[1].item()
    pA = probs[0].item()
    pAB = probs[1].item()

    if (pA >= tA) and ((pA - p2) >= dA):
        stage = "A"
    elif (pAB >= tAB) and ((pAB - p2) >= dAB):
        stage = "AB"
    else:
        stage = "ABC"

    print(f"[predict] probs(A,AB,ABC) = {probs.tolist()}")
    print(f"[predict] chosen_stage = {stage}  (tA={tA}, tAB={tAB}, dA={dA}, dAB={dAB}, T={T:.4f})")


# -------------------------
# Main
# -------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # build-data
    p1 = sub.add_parser("build-data")
    p1.add_argument("--data_jsonl", required=True)
    p1.add_argument("--out_pt", required=True)
    p1.add_argument("--model_A", required=True)
    p1.add_argument("--model_AB", required=True)
    p1.add_argument("--model_ABC", required=True)
    p1.add_argument("--tau", type=float, required=True, help="pass threshold for loss (lower better).")
    p1.add_argument("--device", default="auto")
    p1.add_argument("--fp16", action="store_true")
    p1.add_argument("--max_samples", type=int, default=None)
    p1.add_argument("--max_prompt_len", type=int, default=1024)
    p1.add_argument("--max_len", type=int, default=2048)
    p1.add_argument("--batch_size_feat", type=int, default=8)
    p1.add_argument("--batch_size_q", type=int, default=4)
    p1.set_defaults(func=cmd_build_data)

    # train
    p2 = sub.add_parser("train")
    p2.add_argument("--data_pt", required=True)
    p2.add_argument("--out_router_pt", required=True)
    p2.add_argument("--model_type", choices=["logistic", "mlp"], default="logistic")
    p2.add_argument("--hidden", type=int, default=128)
    p2.add_argument("--dropout", type=float, default=0.2)
    p2.add_argument("--lr", type=float, default=1e-3)
    p2.add_argument("--weight_decay", type=float, default=1e-4)
    p2.add_argument("--epochs", type=int, default=20)
    p2.add_argument("--batch_size", type=int, default=256)
    p2.add_argument("--val_ratio", type=float, default=0.2)
    p2.add_argument("--seed", type=int, default=0)
    p2.set_defaults(func=cmd_train)

    # tune thresholds
    p3 = sub.add_parser("tune")
    p3.add_argument("--data_pt", required=True)
    p3.add_argument("--router_pt", required=True)
    p3.add_argument("--out_tune_pt", required=True)
    p3.add_argument("--constraint", choices=["failrate", "regret"], default="failrate")
    p3.add_argument("--tau", type=float, required=True, help="for failrate constraint: fail if loss > tau")
    p3.add_argument("--eps", type=float, required=True, help="failrate<=eps or mean_regret<=eps")
    p3.add_argument("--costs", required=True, help='CSV: "costA,costAB,costABC"')
    # grid ranges (CSV lists)
    p3.add_argument("--tA_list", default="0.70,0.75,0.80,0.85,0.90,0.93,0.95,0.97")
    p3.add_argument("--tAB_list", default="0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    p3.add_argument("--dA_list", default="0.00,0.02,0.05,0.08,0.10")
    p3.add_argument("--dAB_list", default="0.00,0.02,0.05,0.08,0.10")
    p3.set_defaults(func=cmd_tune)

    # predict
    p4 = sub.add_parser("predict")
    p4.add_argument("--router_pt", required=True)
    p4.add_argument("--tune_pt", required=True)
    p4.add_argument("--model_A", required=True)
    p4.add_argument("--prompt", required=True)
    p4.add_argument("--device", default="auto")
    p4.add_argument("--fp16", action="store_true")
    p4.add_argument("--max_prompt_len", type=int, default=1024)
    p4.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
