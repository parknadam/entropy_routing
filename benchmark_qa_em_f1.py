#!/usr/bin/env python3
"""
QA EM/F1 benchmark for measurement_suite baseline/stage1/stage2/stage3 engines.

Design goals:
- Official SQuAD-style EM/F1 normalization and max-over-ground-truth scoring.
- Generic dataset loading from HuggingFace, json/jsonl, csv, or directories.
- measurement_suite-native engine initialization via benchmark_transition.
- Reproducible outputs: JSON + CSV + Markdown + per-example prediction logs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import string
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from vllm import LLM, SamplingParams

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_transition import MODELS, ProgressiveChatbotMeasured  # noqa: E402


ENGINE_BASELINE = "baseline"
ENGINE_STAGE1 = "stage1"
ENGINE_STAGE2 = "stage2"
ENGINE_STAGE3 = "stage3"
ALL_ENGINES = (ENGINE_BASELINE, ENGINE_STAGE1, ENGINE_STAGE2, ENGINE_STAGE3)

HF_DATASET_ALIASES = {
    "squad": "rajpurkar/squad",
    "squad_v2": "rajpurkar/squad_v2",
    "newsqa": "legacy107/newsqa",
}
HF_DEFAULT_SPLITS = {
    "squad": "validation",
    "squad_v2": "validation",
    "newsqa": "validation",
}

PROMPT_STYLE_AUTO = "auto"
PROMPT_STYLE_GENERIC = "generic"
PROMPT_STYLE_NEWSQA = "newsqa"
PROMPT_STYLE_CHOICES = (PROMPT_STYLE_AUTO, PROMPT_STYLE_GENERIC, PROMPT_STYLE_NEWSQA)

NO_ANSWER_ALIASES = {
    "",
    "unanswerable",
    "no answer",
    "no valid answer",
    "not answerable",
    "cannot be answered",
    "cant be answered",
    "not in the context",
    "not enough information",
}

ANSWER_PREFIX_PATTERNS = (
    re.compile(r"^\s*(?:final\s+answer|short\s+answer|answer|prediction|response)\s*[:：-]\s*(.+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:the\s+answer\s+is|it\s+is|it's)\s+(.+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:정답|답변|답)\s*[:：-]\s*(.+)\s*$", re.IGNORECASE),
)

REASONING_PREFIXES = (
    "let's think",
    "let us think",
    "step by step",
    "analysis:",
    "reasoning:",
    "based on the context",
    "from the context",
    "to answer this",
    "we need to",
    "first,",
    "first ",
    "the context says",
)

EXPLANATORY_SPLITTER_PATTERNS = (
    re.compile(r"\s+(?:because|since|as|therefore|thus|which means)\b", re.IGNORECASE),
    re.compile(r"\s*[.;]\s+"),
    re.compile(r"\s+[-–]\s+"),
)

SUMMARY_FIELDS = [
    "dataset",
    "engine",
    "n",
    "answerable_count",
    "unanswerable_count",
    "em",
    "em_ci95_low",
    "em_ci95_high",
    "f1",
    "f1_ci95_low",
    "f1_ci95_high",
    "raw_empty_rate",
    "prediction_empty_rate",
    "elapsed_s",
]

PAIRWISE_FIELDS = [
    "dataset",
    "baseline_engine",
    "candidate_engine",
    "delta_em",
    "delta_em_ci95_low",
    "delta_em_ci95_high",
    "delta_em_p_two_sided",
    "delta_f1",
    "delta_f1_ci95_low",
    "delta_f1_ci95_high",
    "delta_f1_p_two_sided",
]


@dataclass
class QAExample:
    example_id: str
    prompt: str
    question: str
    context: str
    answers: List[str]
    is_impossible: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


def _cleanup_distributed_state() -> None:
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass


def _trim(text: str, max_chars: int = 220) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _deep_get(obj: Any, dotted_key: str) -> Any:
    if not dotted_key:
        return None

    cur = obj
    for part in dotted_key.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            if not part.isdigit():
                return None
            idx = int(part)
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            return None
        if cur is None:
            return None
    return cur


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = _normalize_space(value).lower()
    if not text:
        return False
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return text in {"impossible", "unanswerable"}


def _parse_jsonish_string(value: str) -> Any:
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _flatten_answers_obj(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = _parse_jsonish_string(value)
        if parsed is not value:
            return _flatten_answers_obj(parsed)
        if "|||" in value:
            return [part.strip() for part in value.split("|||") if part.strip()]
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, dict):
        for key in ("text", "answers", "answer", "spans", "aliases", "value"):
            if key in value:
                return _flatten_answers_obj(value.get(key))
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(_flatten_answers_obj(item))
        return out
    return [str(value)]


def _normalize_answer_official(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(text or "")))))


def _official_exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer_official(prediction) == _normalize_answer_official(ground_truth))


def _official_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer_official(prediction).split()
    gold_tokens = _normalize_answer_official(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return float((2 * precision * recall) / (precision + recall))


def _metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: Sequence[str]) -> float:
    if not ground_truths:
        ground_truths = [""]
    return max(float(metric_fn(prediction, truth)) for truth in ground_truths)


def _best_matching_reference(prediction: str, ground_truths: Sequence[str]) -> str:
    if not ground_truths:
        return ""

    best_answer = ground_truths[0]
    best_key = (-1.0, -1.0)
    for answer in ground_truths:
        key = (
            _official_exact_match(prediction, answer),
            _official_f1(prediction, answer),
        )
        if key > best_key:
            best_answer = answer
            best_key = key
    return best_answer


def _looks_like_reasoning_text(text: str) -> bool:
    normalized = _normalize_space(text).lower()
    if not normalized:
        return False
    if any(normalized.startswith(prefix) for prefix in REASONING_PREFIXES):
        return True
    token_count = len(normalized.split())
    if token_count > 12:
        return True
    if any(marker in normalized for marker in (" because ", " therefore ", " thus ", " so that ", " which means ")):
        return True
    if normalized.endswith((".", "!", "?")) and token_count > 6:
        return True
    return False


def _is_plausible_short_answer_span(text: str) -> bool:
    value = _normalize_space(text)
    if not value:
        return False
    if len(value) > 160:
        return False
    token_count = len(value.split())
    if token_count == 0 or token_count > 12:
        return False
    lowered = value.lower()
    if lowered.startswith(("context:", "question:", "analysis:", "reasoning:", "assistant:", "model:")):
        return False
    if _looks_like_reasoning_text(value):
        return False
    return True


def _trim_explanatory_tail(text: str) -> str:
    value = _normalize_space(text)
    if not value:
        return ""

    candidates = [value]
    for pattern in EXPLANATORY_SPLITTER_PATTERNS:
        left = pattern.split(value, maxsplit=1)[0].strip()
        if left and left != value:
            candidates.append(left)

    for candidate in candidates:
        cleaned = candidate.strip(" \t\r\n`'\"*_")
        if _is_plausible_short_answer_span(cleaned):
            return cleaned
    return candidates[0]


def _extract_prediction_details(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").replace("\x00", " ").strip()
    if not text:
        return {
            "text": "",
            "strategy": "empty_raw_output",
            "contract_followed": False,
        }

    text = re.sub(r"</s>|<\|assistant\|>|<\|endoftext\|>", " ", text, flags=re.IGNORECASE)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {
            "text": "",
            "strategy": "empty_cleaned_output",
            "contract_followed": False,
        }

    for line in lines:
        candidate = re.sub(r"^\s*(?:assistant|model)\s*[:：]\s*", "", line, flags=re.IGNORECASE).strip()
        for pattern in ANSWER_PREFIX_PATTERNS:
            match = pattern.match(candidate)
            if match:
                value = match.group(1).strip()
                if value:
                    return {
                        "text": _clean_extracted_prediction(value),
                        "strategy": "explicit_answer_prefix",
                        "contract_followed": True,
                    }

    if len(lines) == 1:
        candidate = _clean_extracted_prediction(lines[0])
        if _is_plausible_short_answer_span(candidate):
            return {
                "text": candidate,
                "strategy": "single_short_line_fallback",
                "contract_followed": False,
            }

    for line in reversed(lines[-3:]):
        candidate = re.sub(r"^\s*(?:assistant|model)\s*[:：]\s*", "", line, flags=re.IGNORECASE).strip()
        candidate = _clean_extracted_prediction(candidate)
        if _is_plausible_short_answer_span(candidate):
            return {
                "text": candidate,
                "strategy": "last_short_line_fallback",
                "contract_followed": False,
            }

    return {
        "text": "",
        "strategy": "rejected_non_contract_output",
        "contract_followed": False,
    }


def _clean_extracted_prediction(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^\s*(?:[-*>]+|\d+[\.\)])\s*", "", value)
    value = re.sub(r"^\s*(?:assistant|model)\s*[:：]\s*", "", value, flags=re.IGNORECASE)
    value = value.strip(" \t\r\n`'\"*_")
    value = re.sub(r"^(?:the\s+answer\s+is|answer\s+is|it\s+is|it's)\s+", "", value, flags=re.IGNORECASE)
    value = _trim_explanatory_tail(value)
    return value.strip()


def _canonicalize_prediction(prediction: str, enable_no_answer_aliases: bool) -> str:
    value = _normalize_space(prediction)
    if not enable_no_answer_aliases:
        return value
    normalized = _normalize_answer_official(value)
    if normalized in NO_ANSWER_ALIASES:
        return ""
    return value


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * q
    floor_idx = int(math.floor(k))
    ceil_idx = int(math.ceil(k))
    if floor_idx == ceil_idx:
        return float(sorted_values[floor_idx])
    lower = sorted_values[floor_idx] * (ceil_idx - k)
    upper = sorted_values[ceil_idx] * (k - floor_idx)
    return float(lower + upper)


def _bootstrap_ci_mean(values: Sequence[float], samples: int, seed: int) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        value = float(values[0])
        return value, value

    rng = random.Random(seed)
    means: List[float] = []
    rounds = max(1, int(samples))
    for _ in range(rounds):
        total = 0.0
        for _ in range(n):
            total += float(values[rng.randrange(n)])
        means.append(total / n)
    means.sort()
    return _quantile(means, 0.025), _quantile(means, 0.975)


def _bootstrap_paired_mean_delta(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    samples: int,
    seed: int,
) -> Dict[str, float]:
    if len(baseline_values) != len(candidate_values):
        raise ValueError("Paired arrays must have the same length.")

    n = len(baseline_values)
    if n == 0:
        return {
            "delta": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "p_two_sided": 1.0,
        }

    diffs = [float(candidate) - float(base) for base, candidate in zip(baseline_values, candidate_values)]
    point = float(sum(diffs) / n)
    if n == 1:
        return {
            "delta": point,
            "ci95_low": point,
            "ci95_high": point,
            "p_two_sided": 1.0,
        }

    rng = random.Random(seed)
    means: List[float] = []
    rounds = max(1, int(samples))
    for _ in range(rounds):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = _quantile(means, 0.025)
    hi = _quantile(means, 0.975)

    le_zero = 0
    ge_zero = 0
    for value in means:
        if value <= 0.0:
            le_zero += 1
        if value >= 0.0:
            ge_zero += 1
    p_two = min(1.0, 2.0 * min(le_zero / len(means), ge_zero / len(means)))
    return {
        "delta": point,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_two_sided": float(p_two),
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, default=str)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_partial_summary_csv(out_dir: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    rows = sorted(summary_rows, key=lambda row: str(row.get("engine", "")))
    _write_csv(out_dir / "qa_em_f1_partial_summary.csv", rows, SUMMARY_FIELDS)


def _hash_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _hash_json_obj(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _example_snapshot_row(example: QAExample) -> Dict[str, Any]:
    row = {
        "id": example.example_id,
        "question": example.question,
        "context": example.context,
        "answers": list(example.answers),
        "is_impossible": bool(example.is_impossible),
        "prompt": example.prompt,
        "metadata": dict(example.metadata),
    }
    row["prompt_sha256"] = _hash_text(example.prompt)
    row["context_sha256"] = _hash_text(example.context)
    row["record_sha256"] = _hash_json_obj(row)
    return row


def _write_examples_snapshot(path: Path, examples: Sequence[QAExample]) -> Dict[str, Any]:
    rows = [_example_snapshot_row(example) for example in examples]
    _write_jsonl(path, rows)
    record_hashes = [row["record_sha256"] for row in rows]
    joined = "\n".join(record_hashes)
    return {
        "path": str(path),
        "n": len(rows),
        "example_ids_sha256": _hash_text("\n".join(example.example_id for example in examples)),
        "records_sha256": _hash_text(joined),
    }


def _format_chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        return user_prompt


def _is_placeholder_source(source: str) -> bool:
    s = (source or "").strip().lower()
    if not s:
        return False
    if s.startswith("/abs/path/") or s.startswith("/실제경로/"):
        return True
    if s.startswith("jsonl:/abs/path/") or s.startswith("json:/abs/path/") or s.startswith("csv:/abs/path/"):
        return True
    if s.startswith("jsonl:/실제경로/") or s.startswith("json:/실제경로/") or s.startswith("csv:/실제경로/"):
        return True
    if "<" in s or ">" in s:
        return True
    if "/path/to/" in s or "your/path" in s:
        return True
    return False


def _parse_source_spec(source: str) -> Tuple[str, str]:
    text = source.strip()
    if not text:
        raise FileNotFoundError("Empty dataset source.")
    if _is_placeholder_source(text):
        raise FileNotFoundError(
            f"Dataset source looks like a placeholder path: {source}. "
            "Replace it with a real file/directory path or hf:<dataset>:<split>."
        )
    if text.startswith("hf:"):
        return "hf", text
    if text.startswith("jsonl:"):
        return "jsonl", text[len("jsonl:") :]
    if text.startswith("json:"):
        return "json", text[len("json:") :]
    if text.startswith("csv:"):
        return "csv", text[len("csv:") :]

    path = Path(text)
    if path.exists():
        if path.is_dir():
            return "dir", text
        if path.suffix.lower() == ".jsonl":
            return "jsonl", text
        if path.suffix.lower() == ".json":
            return "json", text
        if path.suffix.lower() == ".csv":
            return "csv", text

    raise FileNotFoundError(
        f"Unsupported or missing dataset source: {source}. "
        "Use hf:..., jsonl:/path, json:/path, csv:/path, or an existing file/dir path."
    )


def _iter_records_from_json_obj(obj: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(obj, dict):
        raise ValueError("Unsupported JSON root type for QA records.")

    data_items = obj.get("data")
    if isinstance(data_items, list):
        yielded_nested = False
        for article_idx, article in enumerate(data_items):
            if not isinstance(article, dict):
                continue
            paragraphs = article.get("paragraphs")
            if not isinstance(paragraphs, list):
                continue
            yielded_nested = True
            for para_idx, paragraph in enumerate(paragraphs):
                if not isinstance(paragraph, dict):
                    continue
                context = paragraph.get("context")
                qas = paragraph.get("qas") or paragraph.get("questions") or []
                if not isinstance(qas, list):
                    continue
                for qa_idx, qa in enumerate(qas):
                    if not isinstance(qa, dict):
                        continue
                    row = dict(qa)
                    if context is not None and row.get("context") is None:
                        row["context"] = context
                    if article.get("title") is not None and row.get("title") is None:
                        row["title"] = article.get("title")
                    row.setdefault("_source_article_index", article_idx)
                    row.setdefault("_source_paragraph_index", para_idx)
                    row.setdefault("_source_qa_index", qa_idx)
                    yield row
        if yielded_nested:
            return

        for item in data_items:
            if isinstance(item, dict):
                yield item
        return

    for key in ("examples", "items", "records"):
        values = obj.get(key)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    yield item
            return

    raise ValueError("Unsupported JSON structure for QA dataset.")


def _iter_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
                if isinstance(obj, dict):
                    yield obj
        return

    if path.suffix.lower() == ".json":
        with path.open() as handle:
            obj = json.load(handle)
        yield from _iter_records_from_json_obj(obj)
        return

    raise ValueError(f"Unsupported JSON extension for QA dataset: {path}")


def _iter_csv_records(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=1):
            if not row:
                continue
            item = {str(key): value for key, value in row.items()}
            item.setdefault("_source_row", row_idx)
            item.setdefault("_source_file", str(path))
            yield item


def _load_hf_records(spec: str) -> List[Dict[str, Any]]:
    body = spec[len("hf:") :]
    parts = body.split(":")
    if len(parts) < 2:
        raise ValueError("HF source spec must be hf:<dataset_name>:<split> or hf:<dataset_name>:<subset>:<split>")

    dataset_name = parts[0]
    if len(parts) == 2:
        subset_name = None
        split = parts[1]
    else:
        subset_name = parts[1]
        split = parts[2]

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "datasets package is required for hf: sources. Install with: pip install datasets"
        ) from exc

    if subset_name:
        ds = load_dataset(dataset_name, subset_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    return [dict(item) for item in ds]


def _iter_dir_records(path: Path) -> Iterator[Dict[str, Any]]:
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl", ".csv"}
    )
    for file_path in files:
        if file_path.suffix.lower() in {".json", ".jsonl"}:
            yield from _iter_json_records(file_path)
        elif file_path.suffix.lower() == ".csv":
            yield from _iter_csv_records(file_path)


def _resolve_dataset_source(args: argparse.Namespace) -> str:
    if args.dataset_source:
        return args.dataset_source

    dataset_name = str(args.dataset).strip()
    if not dataset_name:
        raise ValueError("Dataset name is empty.")
    if dataset_name.startswith("hf:"):
        return dataset_name

    split = args.dataset_split or HF_DEFAULT_SPLITS.get(dataset_name.lower(), "validation")
    resolved_name = HF_DATASET_ALIASES.get(dataset_name.lower(), dataset_name)
    return f"hf:{resolved_name}:{split}"


def _infer_prompt_style(dataset_name: str, source: str) -> str:
    hints = [str(dataset_name or "").strip().lower(), str(source or "").strip().lower()]
    if any("newsqa" in hint for hint in hints):
        return PROMPT_STYLE_NEWSQA
    return PROMPT_STYLE_GENERIC


def _resolve_prompt_style(args: argparse.Namespace, source: str) -> str:
    requested = str(getattr(args, "prompt_style", PROMPT_STYLE_AUTO) or PROMPT_STYLE_AUTO).strip().lower()
    if requested and requested != PROMPT_STYLE_AUTO:
        return requested
    return _infer_prompt_style(str(getattr(args, "dataset", "")), source)


def _build_qa_prompt(
    context: str,
    question: str,
    no_answer_text: str,
    prompt_style: str = PROMPT_STYLE_GENERIC,
    title: str = "",
) -> str:
    normalized_style = str(prompt_style or PROMPT_STYLE_GENERIC).strip().lower()
    if normalized_style == PROMPT_STYLE_NEWSQA:
        instructions = [
            "You are answering a question about a news article.",
            "Copy the shortest exact span from the article that answers the question.",
            "Do not explain or restate the question.",
            "Return exactly one line in the format: Answer: <short answer>.",
        ]
        if no_answer_text:
            instructions.append(
                f"If the article does not contain the answer, return exactly: Answer: {no_answer_text}."
            )

        parts = [" ".join(instructions), ""]
        if title:
            parts.append("Headline:")
            parts.append(title)
            parts.append("")
        if context:
            parts.append("Article:")
            parts.append(context)
            parts.append("")
        parts.append("Question:")
        parts.append(question)
        parts.append("")
        parts.append("Answer:")
        return "\n".join(parts)

    instructions = [
        "Answer the question using only the provided context.",
        "Return exactly one line in the format: Answer: <short answer>.",
    ]
    if no_answer_text:
        instructions.append(f"If the answer is not stated in the context, return exactly: Answer: {no_answer_text}.")

    parts = [" ".join(instructions), ""]
    if context:
        parts.append("Context:")
        parts.append(context)
        parts.append("")
    parts.append("Question:")
    parts.append(question)
    parts.append("")
    parts.append("Answer:")
    return "\n".join(parts)


def _extract_answers(record: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[str], bool]:
    answers: List[str] = []
    field_candidates = [
        args.answers_field,
        "answers.text",
        "answers",
        "answer",
        "reference_answers",
        "references",
        "gold_answers",
        "gold",
        "target",
        "output",
    ]

    for field_name in field_candidates:
        value = _deep_get(record, field_name)
        if value is None:
            continue
        answers.extend(_flatten_answers_obj(value))

    cleaned_answers = _dedupe_preserve_order(_normalize_space(value) for value in answers if _normalize_space(value))
    is_impossible = _coerce_bool(_deep_get(record, args.is_impossible_field))
    if not cleaned_answers and str(args.dataset).strip().lower() == "squad_v2":
        is_impossible = True
    return cleaned_answers, is_impossible


def load_qa_examples(args: argparse.Namespace) -> Tuple[List[QAExample], Dict[str, Any], str]:
    source = _resolve_dataset_source(args)
    prompt_style = _resolve_prompt_style(args, source)
    kind, payload = _parse_source_spec(source)

    if kind == "hf":
        records = _load_hf_records(payload)
    elif kind in {"json", "jsonl"}:
        records = list(_iter_json_records(Path(payload)))
    elif kind == "csv":
        records = list(_iter_csv_records(Path(payload)))
    elif kind == "dir":
        records = list(_iter_dir_records(Path(payload)))
    else:
        raise ValueError(f"Unsupported source kind: {kind}")

    stats = Counter()
    examples: List[QAExample] = []

    for idx, record in enumerate(records):
        stats["raw_records"] += 1

        question = ""
        for field_name in (
            args.question_field,
            "question",
            "query",
            "input",
            "prompt",
            "problem",
        ):
            question = _normalize_space(_deep_get(record, field_name))
            if question:
                break
        if not question:
            stats["skipped_missing_question"] += 1
            continue

        context = ""
        for field_name in (
            args.context_field,
            "context",
            "passage",
            "document",
            "article",
        ):
            context = _normalize_space(_deep_get(record, field_name))
            if context:
                break
        if not context:
            stats["examples_without_context"] += 1

        answers, is_impossible = _extract_answers(record, args)
        if not answers and not is_impossible:
            stats["skipped_missing_answers"] += 1
            continue

        example_id = _normalize_space(_deep_get(record, args.id_field))
        if not example_id:
            example_id = _normalize_space(_deep_get(record, "id")) or f"qa_{idx:06d}"

        title = _normalize_space(_deep_get(record, args.title_field)) or _normalize_space(_deep_get(record, "title"))
        metadata = {
            "title": title,
            "source_file": _deep_get(record, "_source_file"),
            "source_row": _deep_get(record, "_source_row"),
            "article_index": _deep_get(record, "_source_article_index"),
            "paragraph_index": _deep_get(record, "_source_paragraph_index"),
            "qa_index": _deep_get(record, "_source_qa_index"),
            "prompt_style": prompt_style,
            "prompt_contract": "Answer: <short answer>",
        }
        prompt = _build_qa_prompt(
            context=context,
            question=question,
            no_answer_text=args.no_answer_text,
            prompt_style=prompt_style,
            title=title,
        )
        examples.append(
            QAExample(
                example_id=example_id,
                prompt=prompt,
                question=question,
                context=context,
                answers=answers,
                is_impossible=is_impossible,
                metadata=metadata,
            )
        )

    if not examples:
        raise RuntimeError(f"No valid QA examples loaded from source={source}")

    if args.limit > 0 and len(examples) > args.limit:
        rng = random.Random(args.seed)
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        indices = sorted(indices[: args.limit])
        examples = [examples[index] for index in indices]
        stats["sampled_examples"] = len(examples)
    else:
        stats["sampled_examples"] = len(examples)

    stats["valid_examples"] = len(examples)
    stats["answerable_examples"] = sum(1 for example in examples if not example.is_impossible)
    stats["unanswerable_examples"] = sum(1 for example in examples if example.is_impossible)
    stats["prompt_style"] = prompt_style
    return examples, dict(stats), source


class EngineBase:
    engine_name: str

    def generate_batch(self, user_prompts: Sequence[str]) -> List[str]:
        raise NotImplementedError

    def diagnostics(self) -> Dict[str, Any]:
        return {}

    def close(self) -> None:
        return None


class BaselineEngine(EngineBase):
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        enforce_eager: bool,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int,
    ):
        cfg = MODELS[model_name]
        self.engine_name = ENGINE_BASELINE
        self.model_path = cfg["baseline_path"]
        self.trust_remote_code = bool(cfg.get("trust_remote_code", True))
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=self.trust_remote_code,
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len),
            tensor_parallel_size=max(1, int(tensor_parallel_size)),
            enforce_eager=bool(enforce_eager),
            enable_prefix_caching=bool(cfg.get("enable_prefix_caching", True)),
            disable_sliding_window=bool(cfg.get("disable_sliding_window", False)),
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)

    def generate_batch(self, user_prompts: Sequence[str]) -> List[str]:
        formatted = [_format_chat_prompt(self.tokenizer, prompt) for prompt in user_prompts]
        outputs = self.llm.generate(formatted, self.params)
        texts: List[str] = []
        for output in outputs:
            texts.append(output.outputs[0].text.strip() if output.outputs else "")
        return texts

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "engine": self.engine_name,
            "model_path": self.model_path,
            "trust_remote_code": self.trust_remote_code,
        }

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        try:
            self.tokenizer = None
            self.params = None
            self.llm = None
            if llm is not None:
                del llm
        finally:
            _cleanup_distributed_state()
            torch.cuda.empty_cache()
            gc.collect()


class ProgressiveStageEngine(EngineBase):
    def __init__(
        self,
        model_name: str,
        target_stage: int,
        max_new_tokens: int,
        enforce_eager: bool,
        transition_timeout_s: float,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int,
    ):
        if target_stage not in (1, 2, 3):
            raise ValueError(f"target_stage must be 1/2/3, got {target_stage}")

        self.engine_name = f"stage{target_stage}"
        MODELS[model_name]["gpu_memory_utilization"] = float(gpu_memory_utilization)
        MODELS[model_name]["max_model_len"] = int(max_model_len)
        MODELS[model_name]["tensor_parallel_size"] = max(1, int(tensor_parallel_size))
        self.chatbot = ProgressiveChatbotMeasured(
            model_name=model_name,
            fixed_max_tokens=max_new_tokens,
            overlap_rounds=0,
            throughput_requests=1,
            transition_mode="clean",
            reconcile_mode="auto",
            forward_variant="dualpath_inplace",
            instant_streams=4,
            alpha_update_mode="inplace",
            transition_window_samples=32,
            consistency_probe=False,
            enforce_eager=bool(enforce_eager),
        )
        self.chatbot.sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
        self.tokenizer = self.chatbot.tokenizer

        if target_stage >= 2:
            ok = self.chatbot.transition_to_stage(2, timeout_s=transition_timeout_s)
            if not ok:
                raise RuntimeError("Failed to transition progressive engine to stage2")
        if target_stage >= 3:
            ok = self.chatbot.transition_to_stage(3, timeout_s=transition_timeout_s)
            if not ok:
                raise RuntimeError("Failed to transition progressive engine to stage3")

    def generate_batch(self, user_prompts: Sequence[str]) -> List[str]:
        formatted = [_format_chat_prompt(self.tokenizer, prompt) for prompt in user_prompts]
        outputs = self.chatbot.llm.generate(formatted, self.chatbot.sampling_params)
        texts: List[str] = []
        for output in outputs:
            texts.append(output.outputs[0].text.strip() if output.outputs else "")
        return texts

    def diagnostics(self) -> Dict[str, Any]:
        metrics = getattr(self.chatbot, "metrics", {})
        return {
            "engine": self.engine_name,
            "current_stage": int(getattr(self.chatbot, "current_stage", 1)),
            "stages": metrics.get("stages", {}),
            "stage_transition_times": metrics.get("stage_transition_times", {}),
        }

    def close(self) -> None:
        chatbot = getattr(self, "chatbot", None)
        try:
            self.tokenizer = None
            if chatbot is not None:
                chatbot.cleanup()
            self.chatbot = None
            if chatbot is not None:
                del chatbot
        finally:
            _cleanup_distributed_state()
            torch.cuda.empty_cache()
            gc.collect()


def _effective_gpu_memory_utilization(requested_utilization: float) -> float:
    return max(0.05, min(0.99, float(requested_utilization)))


def _is_memory_related_engine_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    patterns = [
        "no available memory for the cache blocks",
        "cuda out of memory",
        "outofmemoryerror",
        "torch.outofmemoryerror",
        "memory reserved for kv cache is -",
        "cuda error: out of memory",
    ]
    return any(pattern in message for pattern in patterns)


def _engine_attempt_plan(base_util: float, base_len: int) -> List[Tuple[float, int]]:
    plan: List[Tuple[float, int]] = []

    def add(utilization: float, length: int) -> None:
        candidate = (max(0.05, min(0.99, float(utilization))), max(16, int(length)))
        if candidate not in plan:
            plan.append(candidate)

    add(base_util, base_len)
    add(min(0.92, base_util + 0.06), base_len)
    add(min(0.88, base_util + 0.03), min(base_len, 768))
    add(max(0.05, base_util - 0.05), min(base_len, 512))
    add(max(0.05, base_util - 0.10), min(base_len, 384))
    return plan


def _construct_engine(
    engine_name: str,
    args: argparse.Namespace,
    gpu_memory_utilization: float,
    max_model_len: int,
    tensor_parallel_size: int,
) -> EngineBase:
    if engine_name == ENGINE_BASELINE:
        return BaselineEngine(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            enforce_eager=args.enforce_eager,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
        )
    if engine_name == ENGINE_STAGE1:
        return ProgressiveStageEngine(
            model_name=args.model,
            target_stage=1,
            max_new_tokens=args.max_new_tokens,
            enforce_eager=args.enforce_eager,
            transition_timeout_s=args.stage_transition_timeout_s,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
        )
    if engine_name == ENGINE_STAGE2:
        return ProgressiveStageEngine(
            model_name=args.model,
            target_stage=2,
            max_new_tokens=args.max_new_tokens,
            enforce_eager=args.enforce_eager,
            transition_timeout_s=args.stage_transition_timeout_s,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
        )
    if engine_name == ENGINE_STAGE3:
        return ProgressiveStageEngine(
            model_name=args.model,
            target_stage=3,
            max_new_tokens=args.max_new_tokens,
            enforce_eager=args.enforce_eager,
            transition_timeout_s=args.stage_transition_timeout_s,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
        )
    raise ValueError(f"Unsupported engine: {engine_name}")


def _build_engine(engine_name: str, args: argparse.Namespace) -> EngineBase:
    cfg = MODELS[args.model]
    requested_util = (
        float(args.gpu_memory_utilization)
        if float(args.gpu_memory_utilization) > 0
        else float(cfg.get("gpu_memory_utilization", 0.4))
    )
    effective_util = _effective_gpu_memory_utilization(requested_util)

    progressive_max_len = (
        int(args.max_model_len)
        if int(args.max_model_len) > 0
        else int(cfg.get("max_model_len", 2048))
    )
    baseline_max_len = (
        int(args.baseline_max_model_len)
        if int(args.baseline_max_model_len) > 0
        else progressive_max_len
    )
    tensor_parallel_size = (
        int(args.tensor_parallel_size)
        if int(args.tensor_parallel_size) > 0
        else int(cfg.get("tensor_parallel_size", 1))
    )
    tensor_parallel_size = max(1, tensor_parallel_size)

    desired_len = baseline_max_len if engine_name == ENGINE_BASELINE else progressive_max_len
    attempts = _engine_attempt_plan(effective_util, desired_len)

    last_exc: Optional[Exception] = None
    for attempt_idx, (utilization, max_len) in enumerate(attempts, start=1):
        print(
            f"[engine-config] model={args.model} engine={engine_name} "
            f"attempt={attempt_idx}/{len(attempts)} util={utilization:.3f} "
            f"max_model_len={max_len} tp={tensor_parallel_size}"
        )
        try:
            return _construct_engine(
                engine_name=engine_name,
                args=args,
                gpu_memory_utilization=utilization,
                max_model_len=max_len,
                tensor_parallel_size=tensor_parallel_size,
            )
        except Exception as exc:
            last_exc = exc
            if not _is_memory_related_engine_error(exc):
                raise
            print(f"[engine-config][warn] memory-related init failure on attempt {attempt_idx}: {type(exc).__name__}: {exc}")
            _cleanup_distributed_state()
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.3)

    assert last_exc is not None
    raise RuntimeError(
        f"Failed to initialize engine={engine_name} after memory-tuned retries. "
        "Try fewer engines, lower --max-model-len, or use a less-loaded GPU."
    ) from last_exc


def _generate_with_auto_batch_shrink(engine: EngineBase, prompts: Sequence[str]) -> List[str]:
    try:
        return engine.generate_batch(prompts)
    except Exception as exc:
        if not _is_memory_related_engine_error(exc):
            raise
        size = len(prompts)
        if size <= 1:
            raise RuntimeError(
                "OOM even at batch_size=1. Try a less busy GPU, lower --max-model-len, "
                "or reduce --gpu-memory-utilization."
            ) from exc
        mid = size // 2
        print(f"[eval][warn] OOM on batch_size={size}; retrying with {mid}+{size - mid}")
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.2)
        return _generate_with_auto_batch_shrink(engine, prompts[:mid]) + _generate_with_auto_batch_shrink(engine, prompts[mid:])


def evaluate_dataset(
    dataset_name: str,
    engine: EngineBase,
    examples: Sequence[QAExample],
    batch_size: int,
    bootstrap_samples: int,
    seed: int,
    enable_no_answer_aliases: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prediction_rows: List[Dict[str, Any]] = []

    batch_size = max(1, int(batch_size))
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [example.prompt for example in batch]
        outputs = _generate_with_auto_batch_shrink(engine, prompts)

        for example, raw_output in zip(batch, outputs):
            raw_text = str(raw_output or "")
            extracted = _extract_prediction_details(raw_text)
            extracted_prediction = str(extracted["text"])
            final_prediction = _canonicalize_prediction(extracted_prediction, enable_no_answer_aliases)
            gold_answers = example.answers or [""]

            em = _metric_max_over_ground_truths(_official_exact_match, final_prediction, gold_answers)
            f1 = _metric_max_over_ground_truths(_official_f1, final_prediction, gold_answers)
            best_reference = _best_matching_reference(final_prediction, gold_answers)

            prediction_rows.append(
                {
                    "id": example.example_id,
                    "question": example.question,
                    "context_preview": _trim(example.context, max_chars=320),
                    "answers": gold_answers,
                    "is_impossible": bool(example.is_impossible),
                    "prediction": final_prediction,
                    "prediction_extracted": extracted_prediction,
                    "prediction_normalized": _normalize_answer_official(final_prediction),
                    "prediction_extraction_strategy": extracted["strategy"],
                    "prediction_contract_followed": bool(extracted["contract_followed"]),
                    "best_reference": best_reference,
                    "best_reference_normalized": _normalize_answer_official(best_reference),
                    "raw_output": raw_text,
                    "raw_output_preview": _trim(raw_text),
                    "raw_output_empty": not bool(raw_text.strip()),
                    "prediction_empty": not bool(final_prediction.strip()),
                    "em": float(em),
                    "f1": float(f1),
                    "metadata": example.metadata,
                }
            )

    em_values = [float(row["em"]) for row in prediction_rows]
    f1_values = [float(row["f1"]) for row in prediction_rows]
    raw_empty_values = [1.0 if row["raw_output_empty"] else 0.0 for row in prediction_rows]
    prediction_empty_values = [1.0 if row["prediction_empty"] else 0.0 for row in prediction_rows]

    em_score = float(sum(em_values) / len(em_values)) if em_values else 0.0
    f1_score = float(sum(f1_values) / len(f1_values)) if f1_values else 0.0
    raw_empty_rate = float(sum(raw_empty_values) / len(raw_empty_values)) if raw_empty_values else 0.0
    prediction_empty_rate = float(sum(prediction_empty_values) / len(prediction_empty_values)) if prediction_empty_values else 0.0
    em_lo, em_hi = _bootstrap_ci_mean(em_values, samples=bootstrap_samples, seed=seed)
    f1_lo, f1_hi = _bootstrap_ci_mean(f1_values, samples=bootstrap_samples, seed=seed + 17)

    summary = {
        "dataset": dataset_name,
        "engine": engine.engine_name,
        "n": len(prediction_rows),
        "answerable_count": sum(1 for example in examples if not example.is_impossible),
        "unanswerable_count": sum(1 for example in examples if example.is_impossible),
        "em": em_score,
        "em_ci95_low": em_lo,
        "em_ci95_high": em_hi,
        "f1": f1_score,
        "f1_ci95_low": f1_lo,
        "f1_ci95_high": f1_hi,
        "raw_empty_rate": raw_empty_rate,
        "prediction_empty_rate": prediction_empty_rate,
    }
    return summary, prediction_rows


def _build_markdown_summary(
    dataset_name: str,
    source: str,
    summary_rows: Sequence[Dict[str, Any]],
    pairwise_rows: Sequence[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# QA EM/F1 Summary")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_name}`")
    lines.append(f"- Source: `{source}`")
    lines.append(f"- Generated: `{datetime.now().isoformat()}`")
    lines.append("")
    lines.append("## Metrics by Engine")
    lines.append("")
    lines.append("| Engine | N | Answerable | Unanswerable | EM | F1 | EM 95% CI | F1 95% CI | Raw empty | Pred empty |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {engine} | {n} | {answerable} | {unanswerable} | {em:.4f} | {f1:.4f} | "
            "[{em_lo:.4f}, {em_hi:.4f}] | [{f1_lo:.4f}, {f1_hi:.4f}] | {raw_empty:.4f} | {pred_empty:.4f} |".format(
                engine=row["engine"],
                n=row["n"],
                answerable=row["answerable_count"],
                unanswerable=row["unanswerable_count"],
                em=row["em"],
                f1=row["f1"],
                em_lo=row["em_ci95_low"],
                em_hi=row["em_ci95_high"],
                f1_lo=row["f1_ci95_low"],
                f1_hi=row["f1_ci95_high"],
                raw_empty=row["raw_empty_rate"],
                pred_empty=row["prediction_empty_rate"],
            )
        )

    if pairwise_rows:
        lines.append("")
        lines.append("## Paired Delta vs Baseline")
        lines.append("")
        lines.append("| Candidate | Delta EM | EM 95% CI | p(two-sided) | Delta F1 | F1 95% CI | p(two-sided) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in pairwise_rows:
            lines.append(
                "| {candidate} | {dem:.4f} | [{dem_lo:.4f}, {dem_hi:.4f}] | {dem_p:.4f} | "
                "{df1:.4f} | [{df1_lo:.4f}, {df1_hi:.4f}] | {df1_p:.4f} |".format(
                    candidate=row["candidate_engine"],
                    dem=row["delta_em"],
                    dem_lo=row["delta_em_ci95_low"],
                    dem_hi=row["delta_em_ci95_high"],
                    dem_p=row["delta_em_p_two_sided"],
                    df1=row["delta_f1"],
                    df1_lo=row["delta_f1_ci95_low"],
                    df1_hi=row["delta_f1_ci95_high"],
                    df1_p=row["delta_f1_p_two_sided"],
                )
            )
    lines.append("")
    return "\n".join(lines)


def _apply_model_path_overrides(args: argparse.Namespace) -> None:
    cfg = MODELS[args.model]
    overrides = {
        "baseline_path": args.baseline_model_path,
        "progressive_path": args.progressive_model_path,
        "stage_b_checkpoint": args.stage_b_checkpoint,
        "stage_c_checkpoint": args.stage_c_checkpoint,
    }
    for field_name, override_path in overrides.items():
        if not override_path:
            continue
        path_obj = Path(override_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Override path not found for {field_name}: {override_path}")
        cfg[field_name] = str(path_obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA EM/F1 benchmark with measurement_suite stage-wise execution")
    parser.add_argument("--model", choices=sorted(MODELS.keys()), default="llama")
    parser.add_argument("--dataset", type=str, default="squad", help="Dataset alias or HuggingFace dataset name.")
    parser.add_argument("--dataset-source", type=str, default="", help="hf:..., jsonl:/path, json:/path, csv:/path, or a real file/dir path.")
    parser.add_argument("--dataset-split", type=str, default="", help="Split used when dataset source is derived from --dataset.")
    parser.add_argument("--question-field", type=str, default="question")
    parser.add_argument("--context-field", type=str, default="context")
    parser.add_argument("--answers-field", type=str, default="answers")
    parser.add_argument("--id-field", type=str, default="id")
    parser.add_argument("--title-field", type=str, default="title")
    parser.add_argument("--is-impossible-field", type=str, default="is_impossible")
    parser.add_argument("--engines", nargs="+", default=list(ALL_ENGINES), choices=list(ALL_ENGINES))
    parser.add_argument("--limit", type=int, default=0, help="Optional deterministic sample limit (0 = full dataset).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--baseline-max-model-len", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=0)
    parser.add_argument("--stage-transition-timeout-s", type=float, default=300.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--baseline-model-path", type=str, default="")
    parser.add_argument("--progressive-model-path", type=str, default="")
    parser.add_argument("--stage-b-checkpoint", type=str, default="")
    parser.add_argument("--stage-c-checkpoint", type=str, default="")
    parser.add_argument("--no-answer-text", type=str, default="unanswerable")
    parser.add_argument(
        "--prompt-style",
        type=str,
        default=PROMPT_STYLE_AUTO,
        choices=list(PROMPT_STYLE_CHOICES),
        help="Prompt template style. 'auto' infers a dataset-specific template when supported.",
    )
    parser.add_argument(
        "--disable-no-answer-alias-canonicalization",
        action="store_true",
        help="Disable canonicalizing common no-answer strings to the official null prediction.",
    )
    parser.add_argument("--out-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    _apply_model_path_overrides(args)
    examples, loader_stats, resolved_source = load_qa_examples(args)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(args.dataset).strip()) or "qa"
        out_dir = SCRIPT_DIR.parent / "results" / "raw" / f"qa_em_f1_{dataset_slug}_{args.model}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = out_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    examples_snapshot = _write_examples_snapshot(out_dir / "qa_em_f1_examples.jsonl", examples)

    print(f"[load] dataset={args.dataset} source={resolved_source} n={len(examples)}")
    print(f"[load] answerable={loader_stats.get('answerable_examples', 0)} unanswerable={loader_stats.get('unanswerable_examples', 0)}")
    print(f"[load] prompt_style={loader_stats.get('prompt_style', PROMPT_STYLE_GENERIC)}")
    print(f"[load] snapshot={examples_snapshot['path']}")

    summary_rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []
    predictions_by_engine: Dict[str, List[Dict[str, Any]]] = {}
    diagnostics_by_engine: Dict[str, Dict[str, Any]] = {}

    enable_no_answer_aliases = not bool(args.disable_no_answer_alias_canonicalization)
    selected_engines = list(dict.fromkeys(args.engines))
    engine_order = {engine_name: idx for idx, engine_name in enumerate(selected_engines)}

    for engine_idx, engine_name in enumerate(selected_engines, start=1):
        print(f"\n[engine] initializing {engine_name} ...")
        engine: Optional[EngineBase] = None
        try:
            engine = _build_engine(engine_name, args)
            start_time = time.time()
            summary, predictions = evaluate_dataset(
                dataset_name=str(args.dataset),
                engine=engine,
                examples=examples,
                batch_size=args.batch_size,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + 1000 * engine_idx,
                enable_no_answer_aliases=enable_no_answer_aliases,
            )
            summary["elapsed_s"] = time.time() - start_time
            summary_rows.append(summary)
            predictions_by_engine[engine_name] = predictions
            diagnostics_by_engine[engine_name] = engine.diagnostics()
            _write_jsonl(prediction_dir / f"{engine_name}.jsonl", predictions)
            _write_partial_summary_csv(out_dir, summary_rows)
            print(
                "  -> em={:.4f} f1={:.4f} raw_empty={:.4f} pred_empty={:.4f} n={} ({:.1f}s)".format(
                    summary["em"],
                    summary["f1"],
                    summary["raw_empty_rate"],
                    summary["prediction_empty_rate"],
                    summary["n"],
                    summary["elapsed_s"],
                )
            )
        finally:
            if engine is not None:
                engine.close()
            engine = None
            _cleanup_distributed_state()
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.5)

    summary_rows.sort(key=lambda row: engine_order.get(str(row.get("engine")), len(engine_order)))
    if ENGINE_BASELINE in predictions_by_engine:
        baseline_predictions = predictions_by_engine[ENGINE_BASELINE]
        baseline_ids = [row["id"] for row in baseline_predictions]
        baseline_em = [float(row["em"]) for row in baseline_predictions]
        baseline_f1 = [float(row["f1"]) for row in baseline_predictions]

        for engine_name in selected_engines:
            if engine_name == ENGINE_BASELINE:
                continue
            candidate_predictions = predictions_by_engine.get(engine_name)
            if not candidate_predictions:
                continue
            candidate_ids = [row["id"] for row in candidate_predictions]
            if candidate_ids != baseline_ids:
                raise RuntimeError(f"Prediction id order mismatch between baseline and {engine_name}")

            candidate_em = [float(row["em"]) for row in candidate_predictions]
            candidate_f1 = [float(row["f1"]) for row in candidate_predictions]
            delta_em = _bootstrap_paired_mean_delta(
                baseline_values=baseline_em,
                candidate_values=candidate_em,
                samples=args.bootstrap_samples,
                seed=args.seed + 700 + selected_engines.index(engine_name),
            )
            delta_f1 = _bootstrap_paired_mean_delta(
                baseline_values=baseline_f1,
                candidate_values=candidate_f1,
                samples=args.bootstrap_samples,
                seed=args.seed + 900 + selected_engines.index(engine_name),
            )
            pairwise_rows.append(
                {
                    "dataset": str(args.dataset),
                    "baseline_engine": ENGINE_BASELINE,
                    "candidate_engine": engine_name,
                    "delta_em": delta_em["delta"],
                    "delta_em_ci95_low": delta_em["ci95_low"],
                    "delta_em_ci95_high": delta_em["ci95_high"],
                    "delta_em_p_two_sided": delta_em["p_two_sided"],
                    "delta_f1": delta_f1["delta"],
                    "delta_f1_ci95_low": delta_f1["ci95_low"],
                    "delta_f1_ci95_high": delta_f1["ci95_high"],
                    "delta_f1_p_two_sided": delta_f1["p_two_sided"],
                }
            )

    pairwise_rows.sort(key=lambda row: engine_order.get(str(row.get("candidate_engine")), len(engine_order)))
    markdown = _build_markdown_summary(str(args.dataset), resolved_source, summary_rows, pairwise_rows)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset": str(args.dataset),
        "source": resolved_source,
        "model": args.model,
        "engines": selected_engines,
        "config": {
            "limit": args.limit,
            "batch_size": args.batch_size,
            "bootstrap_samples": args.bootstrap_samples,
            "max_new_tokens": args.max_new_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "baseline_max_model_len": args.baseline_max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size,
            "seed": args.seed,
            "stage_transition_timeout_s": args.stage_transition_timeout_s,
            "enforce_eager": bool(args.enforce_eager),
            "no_answer_text": args.no_answer_text,
            "prompt_style": args.prompt_style,
            "enable_no_answer_alias_canonicalization": enable_no_answer_aliases,
            "question_field": args.question_field,
            "context_field": args.context_field,
            "answers_field": args.answers_field,
            "id_field": args.id_field,
            "title_field": args.title_field,
            "is_impossible_field": args.is_impossible_field,
            "baseline_model_path": args.baseline_model_path,
            "progressive_model_path": args.progressive_model_path,
            "stage_b_checkpoint": args.stage_b_checkpoint,
            "stage_c_checkpoint": args.stage_c_checkpoint,
        },
        "resolved_model_config": {
            "baseline_path": MODELS[args.model].get("baseline_path"),
            "progressive_path": MODELS[args.model].get("progressive_path"),
            "stage_b_checkpoint": MODELS[args.model].get("stage_b_checkpoint"),
            "stage_c_checkpoint": MODELS[args.model].get("stage_c_checkpoint"),
            "gpu_memory_utilization": MODELS[args.model].get("gpu_memory_utilization"),
            "max_model_len": MODELS[args.model].get("max_model_len"),
            "tensor_parallel_size": MODELS[args.model].get("tensor_parallel_size"),
            "trust_remote_code": MODELS[args.model].get("trust_remote_code", True),
            "enable_prefix_caching": MODELS[args.model].get("enable_prefix_caching", True),
            "disable_sliding_window": MODELS[args.model].get("disable_sliding_window", False),
        },
        "loader_stats": loader_stats,
        "dataset_snapshot": examples_snapshot,
        "metric_definition": {
            "em": "Official SQuAD exact match with lower/punctuation/article/whitespace normalization.",
            "f1": "Official SQuAD token-level F1 with the same normalization and max-over-ground-truth scoring.",
        },
        "prediction_contract": {
            "target_format": "Answer: <short answer>",
            "fallback_policy": "Only accept short, non-reasoning fallback lines; otherwise score as empty prediction.",
            "prompt_style_effective": loader_stats.get("prompt_style", PROMPT_STYLE_GENERIC),
        },
        "summary_rows": summary_rows,
        "pairwise_rows": pairwise_rows,
        "engine_diagnostics": diagnostics_by_engine,
        "prediction_dir": str(prediction_dir),
    }

    results_json = out_dir / "qa_em_f1_results.json"
    summary_csv = out_dir / "qa_em_f1_summary.csv"
    pairwise_csv = out_dir / "qa_em_f1_pairwise.csv"
    summary_md = out_dir / "qa_em_f1_summary.md"

    _write_json(results_json, payload)
    _write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    _write_csv(pairwise_csv, pairwise_rows, PAIRWISE_FIELDS)
    summary_md.write_text(markdown)

    print("\n[done]")
    print(f"results_json: {results_json}")
    print(f"summary_csv : {summary_csv}")
    print(f"pairwise_csv: {pairwise_csv}")
    print(f"summary_md  : {summary_md}")
    print(f"examples    : {examples_snapshot['path']}")
    print(f"predictions : {prediction_dir}")


if __name__ == "__main__":
    main()
