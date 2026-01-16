# make_qa_jsonl.py
import argparse
import json
import os
import random
from typing import Dict, Any, List, Optional

from transformers import AutoTokenizer


def iter_squad_examples(squad: Dict[str, Any], allow_impossible: bool = False):
    """
    Yields (context, question, answer_text) from SQuAD v1/v2 format JSON.
    For SQuAD v2: skips impossible questions unless allow_impossible=True.
    """
    for article in squad.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                question = qa.get("question", "")

                # SQuAD v2
                if qa.get("is_impossible", False):
                    if allow_impossible:
                        # impossible은 completion이 없어서 NLL 기반 라벨링이 애매해짐 → 보통은 skip 추천
                        yield context, question, ""
                    continue

                answers = qa.get("answers", [])
                if not answers:
                    continue
                answer_text = answers[0].get("text", "")
                if not answer_text:
                    continue
                yield context, question, answer_text


def build_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def truncate_context_by_tokens(
    tok,
    context: str,
    question: str,
    max_prompt_tokens: int,
) -> str:
    """
    Token-budget 기반으로 context를 잘라서 prompt 전체 토큰 수를 max_prompt_tokens 이하로 맞춤.
    (QA는 보통 context 앞부분이 중요하니 앞쪽을 남김)
    """
    prefix = "Context:\n"
    suffix = f"\n\nQuestion: {question}\nAnswer:"
    # fixed tokens excluding context
    fixed_ids = tok(prefix + suffix, add_special_tokens=False).input_ids
    budget_for_ctx = max_prompt_tokens - len(fixed_ids)
    if budget_for_ctx <= 0:
        return ""  # 질문이 너무 길면 context를 버림

    ctx_ids = tok(context, add_special_tokens=False).input_ids
    if len(ctx_ids) <= budget_for_ctx:
        return context
    ctx_ids = ctx_ids[:budget_for_ctx]
    return tok.decode(ctx_ids, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad_json", required=True, help="path to SQuAD v1/v2 json")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--model_for_tokenizer", default=None, help="e.g., ./results/pruning/A (recommended)")
    ap.add_argument("--max_prompt_tokens", type=int, default=1024)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.squad_json, "r", encoding="utf-8") as f:
        squad = json.load(f)

    tok = None
    if args.model_for_tokenizer is not None:
        tok = AutoTokenizer.from_pretrained(args.model_for_tokenizer, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

    examples = list(iter_squad_examples(squad, allow_impossible=False))
    if args.shuffle:
        random.shuffle(examples)

    if args.max_samples is not None:
        examples = examples[: args.max_samples]

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    n_written = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as w:
        for context, question, answer in examples:
            if tok is not None:
                context = truncate_context_by_tokens(tok, context, question, args.max_prompt_tokens)
            prompt = build_prompt(context, question)
            # completion은 NLL 계산 시 토큰 경계가 잘 잡히도록 앞에 공백 하나 두는 편이 안전
            completion = " " + answer.strip()

            w.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"saved: {args.out_jsonl} (n={n_written})")


if __name__ == "__main__":
    main()
