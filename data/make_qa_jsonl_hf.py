# make_qa_jsonl_hf.py
import argparse, json, os, random
from transformers import AutoTokenizer

def build_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

def truncate_context_by_tokens(tok, context: str, question: str, max_prompt_tokens: int) -> str:
    prefix = "Context:\n"
    suffix = f"\n\nQuestion: {question}\nAnswer:"
    fixed = tok(prefix + suffix, add_special_tokens=False).input_ids
    budget = max_prompt_tokens - len(fixed)
    if budget <= 0:
        return ""
    ctx_ids = tok(context, add_special_tokens=False).input_ids
    if len(ctx_ids) <= budget:
        return context
    return tok.decode(ctx_ids[:budget], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="squad", choices=["squad","squad_v2"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--model_for_tokenizer", required=True)
    ap.add_argument("--max_prompt_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    # datasets import는 여기서 (설치 안 돼 있으면 에러가 더 명확)
    from datasets import load_dataset

    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_for_tokenizer, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    ds = load_dataset(args.dataset, split=args.split)

    idxs = list(range(len(ds)))
    if args.shuffle:
        random.shuffle(idxs)
    idxs = idxs[: args.max_samples]

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    n = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as w:
        for i in idxs:
            ex = ds[i]
            context = ex["context"]
            question = ex["question"]

            # squad_v2는 answers가 비어있을 수 있음(불가능 질문) → skip
            answers = ex.get("answers", {})
            texts = answers.get("text", [])
            if not texts:
                continue
            answer = texts[0].strip()
            if not answer:
                continue

            context = truncate_context_by_tokens(tok, context, question, args.max_prompt_tokens)
            prompt = build_prompt(context, question)
            completion = " " + answer  # 앞 공백 1개(토큰 경계 안정)

            w.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
            n += 1

    print(f"saved: {args.out_jsonl} (n={n})")

if __name__ == "__main__":
    main()
