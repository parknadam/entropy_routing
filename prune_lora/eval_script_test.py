import math, torch, random
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
from lib.eval import eval_ppl
from lib.data import get_loaders
from peft import PeftModel


PRUNED_DIR = "llama2_7b_hybrid_sgpt_auto"
#PRUNED_DIR = "meta-llama/Llama-2-7b-hf"
BASE_NAME  = "meta-llama/Llama-2-7b-hf" 
ADAPTER_DIR = "healed_qa_model"

MAX_SAMPLES_PER_TASK = 200   # 빠르게 보고 싶으면 100~200, 정식은 더 크게
SEQLEN = 1024                # 프루닝 시 사용 길이와 일치 


print("loading pruned model...")
""" model = AutoModelForCausalLM.from_pretrained(
    PRUNED_DIR, torch_dtype=torch.float16, device_map="auto"
).eval() """

base = AutoModelForCausalLM.from_pretrained(
    PRUNED_DIR, torch_dtype=torch.float16, device_map="auto", local_files_only=True
)

model = base
print("num_layers:", model.config.num_hidden_layers)
with torch.no_grad():
    total, zeros = 0, 0
    for n,p in model.named_parameters():
        if p.ndim >= 2:
            total += p.numel()
            zeros += (p==0).sum().item()
    print("global sparsity:", zeros/total)



if ADAPTER_DIR:  # 힐링(QLoRA) 결과가 여기 들어있음
    print("attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    # 선택: 속도용 머지 (VRAM 여유 있으면 추천)
    # model = model.merge_and_unload()

model.eval()

# 임베딩이 올라간 디바이스(샤딩 환경에서도 안전)
try:
    embed_dev = next(model.model.embed_tokens.parameters()).device
except AttributeError:
    embed_dev = next(model.parameters()).device


try:
    tok = AutoTokenizer.from_pretrained(PRUNED_DIR, use_fast=True)
except Exception:
    if ADAPTER_DIR:
        try:
            tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(BASE_NAME, use_fast=True)
    else:
        tok = AutoTokenizer.from_pretrained(BASE_NAME, use_fast=True)




model.eval()

model.seqlen = SEQLEN
model.config.use_cache = False

# ✅ 임베딩이 올라간 디바이스 찾아서 embed_dev 설정 (device_map="auto" 대비)
def _find_embed_device(m):
    for path in ["model.embed_tokens", "model.model.embed_tokens", "base_model.model.model.embed_tokens"]:
        cur = m
        try:
            for attr in path.split("."):
                cur = getattr(cur, attr)
            return next(cur.parameters()).device
        except Exception:
            pass
    return next(m.parameters()).device

embed_dev = _find_embed_device(model)
print("embed_dev:", embed_dev)

# 토크나이저: 베이스 디렉터리에 없을 수도 있어서 베이스→어댑터→원본 순으로 폴백
try:
    tok = AutoTokenizer.from_pretrained(PRUNED_DIR, use_fast=True)
except Exception:
    try:
        tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(BASE_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"


# 프루닝 시 사용한 길이와 일치
model.seqlen = SEQLEN
model.config.use_cache = False

# ------------------------
# option_nll_batch: 여러 선택지를 한 번에 처리하여 각 옵션별 (NLL, opt_len) 반환
# inputs: model, tok, prompt (str), options (list of str), device (torch.device)
# 반환: list of tuples [(nll0, len0), (nll1, len1), ...]
# ------------------------
@torch.no_grad()
def option_nll_batch(model, tok, prompt, options, device):
    # build full strings
    inputs = [prompt + " " + o.strip() for o in options]
    toks = tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=SEQLEN)
    input_ids = toks["input_ids"].to(device)
    attn = toks.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(device)

    # labels: copy of input_ids, but mask prompt tokens with -100
    labels = input_ids.clone()
    # compute option token lengths (without special tokens)
    opt_lens = []
    for o in options:
        opt_ids = tok(" " + o.strip(), add_special_tokens=False)["input_ids"]
        opt_lens.append(len(opt_ids))

    # compute actual lengths (non-pad) per example
    if tok.pad_token_id is not None:
        nonpad_lens = (input_ids != tok.pad_token_id).sum(dim=1).tolist()
    else:
        nonpad_lens = [input_ids.size(1)] * input_ids.size(0)

    # mask prompt tokens: for each example, keep only the final opt_len tokens
    for i in range(input_ids.size(0)):
        full_len = nonpad_lens[i]
        opt_len = opt_lens[i]
        prompt_part = full_len - opt_len
        if prompt_part > 0:
            labels[i, :prompt_part] = -100

    # forward once to get logits
    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits  # [B, T, V]
    # shift for causal LM: predict token t from logits at t-1
    shift_logits = logits[:, :-1, :].float()  # to float for numeric stability
    shift_labels = labels[:, 1:].contiguous()

    vocab = shift_logits.size(-1)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    flat_logits = shift_logits.view(-1, vocab)          # [B*(T-1), V]
    flat_labels = shift_labels.view(-1)                 # [B*(T-1)]
    flat_loss = loss_fct(flat_logits, flat_labels)      # [B*(T-1)]
    per_token_loss = flat_loss.view(shift_labels.size())  # [B, T-1]

    # sum over tokens (ignore -100 positions) to get per-example NLL and counts
    nlls = per_token_loss.sum(dim=1).cpu().tolist()
    opt_token_counts = (shift_labels != -100).sum(dim=1).cpu().tolist()

    results = [(float(nlls[i]), int(opt_token_counts[i])) for i in range(len(options))]
    return results

# ------------------------
# 평가 1: HellaSwag (4지 선다)
# ------------------------
@torch.no_grad()
def eval_hellaswag(model, tok, device, max_samples=MAX_SAMPLES_PER_TASK):
    ds = load_dataset("hellaswag", split="validation")
    n = min(len(ds), max_samples)
    correct = 0
    for i in range(n):
        r = ds[i]
        ctx = r.get("ctx", None) or r.get("context", "")
        endings = r["endings"] if "endings" in r else [r[f"ending{i}"] for i in range(4)]
        label = int(r["label"])
        # 간단 프롬프트
        prompt = f"{ctx.strip()}\n\nComplete the sentence:\n"
        nlls = option_nll_batch(model, tok, prompt, [" " + e.strip() for e in endings], device)
        # nlls: list of (nll, count)
        scores = [nll / max(cnt, 1) for (nll, cnt) in nlls]
        pred = min(range(len(scores)), key=lambda k: scores[k])
        correct += int(pred == label)
    acc = correct / n if n > 0 else 0.0
    return acc

# ------------------------
# 평가 2: BoolQ (Yes/No)
#  - 답을 ' yes' vs ' no' 문자열의 로그우도로 판별
# ------------------------
@torch.no_grad()
def eval_boolq(model, tok, device, max_samples=MAX_SAMPLES_PER_TASK):
    ds = load_dataset("boolq", split="validation")
    n = min(len(ds), max_samples)
    correct = 0
    for i in range(n):
        r = ds[i]
        passage = r["passage"].strip()
        question = r["question"].strip()
        gold = bool(r["answer"])
        prompt = (
            "Answer the question with a single word: yes or no.\n\n"
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            "Answer:"
        )
        pairs = option_nll_batch(model, tok, prompt, [" yes", " no"], device)
        nll_yes, Ly = pairs[0]
        nll_no,  Ln = pairs[1]
        pred_yes = nll_yes < nll_no
        correct += int(pred_yes == gold)
    acc = correct / n if n > 0 else 0.0
    return acc

# ------------------------
# 평가 3: MMLU (lukaemon/mmlu, 4지선다)
#  - validation split 사용(없는 경우 test로 폴백)
#  - 과목 섞어서 전체 평균
# ------------------------
@torch.no_grad()
def eval_mmlu(model, tokenizer, device, max_samples=100):
    subjects = get_dataset_config_names("lukaemon/mmlu")
    print(f"[MMLU] {len(subjects)} subjects")

    def letter_id(letter: str):
        for s in [f" {letter}", letter]:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) >= 1:
                return ids[-1]
        return tokenizer.encode(letter, add_special_tokens=False)[-1]

    A_id, B_id, C_id, D_id = map(letter_id, ["A", "B", "C", "D"])
    choice_ids = torch.tensor([A_id, B_id, C_id, D_id], device=device)

    def score_prompt(q, choices):
        prompt = (
            f"{q}\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n"
            f"Answer:"
        )
        toks = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inp = toks["input_ids"].to(device)
        attn = toks["attention_mask"].to(device)

        if inp.shape[1] > getattr(model, "seqlen", 2048):
            inp = inp[:, -model.seqlen:]
            attn = attn[:, -model.seqlen:]

        out = model(inp, attention_mask=attn).logits
        last = out[:, -1, :].squeeze(0)
        cand_logits = last.index_select(dim=-1, index=choice_ids)
        pred = torch.argmax(cand_logits).item()
        return pred

    subject_acc = []
    for si, subject in enumerate(subjects, 1):
        try:
            ds = load_dataset("lukaemon/mmlu", subject, split="validation", trust_remote_code=True)
        except Exception:
            ds = load_dataset("lukaemon/mmlu", subject, split="test", trust_remote_code=True)

        idxs = list(range(len(ds)))
        if max_samples is not None and len(idxs) > max_samples:
            random.seed(0)
            idxs = random.sample(idxs, max_samples)

        correct = 0
        for i in idxs:
            ex = ds[i]
            q, choices, ans = ex["question"], ex["choices"], ex["answer"]
            pred_idx = score_prompt(q, choices)
            gold_idx = "ABCD".index(ans.strip())
            correct += int(pred_idx == gold_idx)

        acc = correct / len(idxs) if idxs else 0.0
        subject_acc.append(acc)
        print(f"[MMLU] {si:2d}/{len(subjects)} {subject:35s} acc={acc:.3f}")

    macro = sum(subject_acc) / len(subject_acc) if subject_acc else float("nan")
    print(f"[MMLU] macro-avg acc = {macro:.4f}")
    return macro

# ------------------------
# 0) NaN/Inf 스캔
# ------------------------
bad = []
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.numel() == 0:
            continue
        if torch.isnan(p).any() or torch.isinf(p).any():
            bad.append(n)
print("NaN/Inf params:", len(bad), "->", bad[:5])

# ------------------------
# 1) 언어모델링 PPL (WikiText-2)
# ------------------------
ppl = eval_ppl(model, tok, device=embed_dev, dataset="wikitext2")
print(f"[PPL] WikiText-2: {ppl:.4f}")

# ------------------------
# 2) 스모크 테스트 (NLL/PPL)
# ------------------------
_, testenc = get_loaders("wikitext2", seed=0, seqlen=model.seqlen, tokenizer=tok)
ids = testenc.input_ids[:, :SEQLEN].to(embed_dev)
am  = torch.ones_like(ids, device=embed_dev)
with torch.no_grad():
    out = model(ids, attention_mask=am).logits
    shift_logits = out[:, :-1, :].contiguous().float()
    shift_labels = ids[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    print(f"[smoke] NLL={loss.item():.4f}  PPL={math.exp(loss.item()):.2f}")

# ------------------------
# 3) 제로샷 벤치마크 (HellaSwag / BoolQ / MMLU)
# ------------------------
print("[Eval] HellaSwag running...")
hs_acc = eval_hellaswag(model, tok, embed_dev, max_samples=MAX_SAMPLES_PER_TASK)
print(f"[Acc] HellaSwag: {hs_acc*100:.2f}%")

print("[Eval] BoolQ running...")
bq_acc = eval_boolq(model, tok, embed_dev, max_samples=MAX_SAMPLES_PER_TASK)
print(f"[Acc] BoolQ: {bq_acc*100:.2f}%")

print("[Eval] MMLU running...")
mmlu_acc = eval_mmlu(model, tok, embed_dev, max_samples=MAX_SAMPLES_PER_TASK)
print(f"[Acc] MMLU (overall subset): {mmlu_acc*100:.2f}%")
