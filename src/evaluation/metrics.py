"""Self-contained benchmark metrics for Vietnamese VLM evaluation."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable


def normalize_text(text: str) -> str:
    """Normalize answer text for exact-match style VQA metrics."""

    text = unicodedata.normalize("NFC", str(text)).casefold()
    chars = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] in {"P", "S"}:
            chars.append(" ")
        else:
            chars.append(char)
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def as_references(references: str | Iterable[str]) -> list[str]:
    if isinstance(references, str):
        refs = [references]
    else:
        refs = [str(ref) for ref in references]
    return [ref for ref in refs if normalize_text(ref)]


def exact_match(prediction: str, references: str | Iterable[str]) -> float:
    pred = normalize_text(prediction)
    return 1.0 if any(pred == normalize_text(ref) for ref in as_references(references)) else 0.0


def token_f1(prediction: str, references: str | Iterable[str]) -> float:
    pred_tokens = tokenize(prediction)
    if not pred_tokens:
        return 0.0
    return max((_token_f1_single(pred_tokens, tokenize(ref)) for ref in as_references(references)), default=0.0)


def anls(prediction: str, references: str | Iterable[str], threshold: float = 0.5) -> float:
    pred = normalize_text(prediction)
    if not pred:
        return 0.0
    scores = []
    for ref in as_references(references):
        ref_norm = normalize_text(ref)
        max_len = max(len(pred), len(ref_norm))
        if max_len == 0:
            scores.append(1.0)
            continue
        score = 1.0 - (_levenshtein(pred, ref_norm) / max_len)
        scores.append(score if score >= threshold else 0.0)
    return max(scores, default=0.0)


def summarize_vqa_scores(rows: Iterable[dict]) -> dict[str, float]:
    totals = {"exact_match": 0.0, "token_f1": 0.0, "anls": 0.0}
    count = 0
    for row in rows:
        prediction = str(row.get("prediction", ""))
        refs = as_references(row.get("references", []))
        totals["exact_match"] += exact_match(prediction, refs)
        totals["token_f1"] += token_f1(prediction, refs)
        totals["anls"] += anls(prediction, refs)
        count += 1
    if count == 0:
        return {"count": 0, **totals}
    return {"count": count, **{name: value / count for name, value in totals.items()}}


def caption_metrics(predictions: list[str], references: list[list[str]]) -> dict[str, float]:
    """Compute corpus-level caption metrics without external packages.

    CIDEr here follows the standard TF-IDF n-gram cosine idea and is intended for
    local model selection. For leaderboard-grade reporting, run the official
    COCO-caption evaluator as a secondary check.
    """

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")
    cleaned_refs = [as_references(refs) for refs in references]
    return {
        "metric_backend": "local_approx",
        "count": len(predictions),
        "bleu1": _corpus_bleu(predictions, cleaned_refs, max_n=1),
        "bleu2": _corpus_bleu(predictions, cleaned_refs, max_n=2),
        "bleu3": _corpus_bleu(predictions, cleaned_refs, max_n=3),
        "bleu4": _corpus_bleu(predictions, cleaned_refs, max_n=4),
        "meteor": _meteor(predictions, cleaned_refs),
        "cider": _cider(predictions, cleaned_refs),
    }


def _token_f1_single(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (left_char != right_char)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _corpus_bleu(predictions: list[str], references: list[list[str]], max_n: int) -> float:
    clipped = [0] * max_n
    totals = [0] * max_n
    pred_len = 0
    ref_len = 0

    for prediction, refs in zip(predictions, references, strict=True):
        pred_tokens = tokenize(prediction)
        ref_tokens = [tokenize(ref) for ref in refs]
        pred_len += len(pred_tokens)
        ref_len += _closest_ref_len(len(pred_tokens), ref_tokens)
        for n in range(1, max_n + 1):
            pred_counts = _ngrams(pred_tokens, n)
            totals[n - 1] += sum(pred_counts.values())
            max_ref_counts: Counter[tuple[str, ...]] = Counter()
            for ref in ref_tokens:
                for gram, count in _ngrams(ref, n).items():
                    max_ref_counts[gram] = max(max_ref_counts[gram], count)
            clipped[n - 1] += sum(min(count, max_ref_counts[gram]) for gram, count in pred_counts.items())

    if pred_len == 0:
        return 0.0
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1 - (ref_len / max(pred_len, 1)))
    log_precision = 0.0
    for idx in range(max_n):
        if totals[idx] == 0:
            return 0.0
        precision = clipped[idx] / totals[idx] if clipped[idx] else 1e-9
        log_precision += math.log(precision) / max_n
    return brevity_penalty * math.exp(log_precision)


def _closest_ref_len(pred_len: int, ref_tokens: list[list[str]]) -> int:
    if not ref_tokens:
        return 0
    return min((len(ref) for ref in ref_tokens), key=lambda length: (abs(length - pred_len), length))


def _meteor(predictions: list[str], references: list[list[str]]) -> float:
    if not predictions:
        return 0.0
    scores = []
    for prediction, refs in zip(predictions, references, strict=True):
        pred_tokens = tokenize(prediction)
        if not pred_tokens:
            scores.append(0.0)
            continue
        best = 0.0
        for ref in refs:
            ref_tokens = tokenize(ref)
            overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
            if overlap == 0:
                continue
            precision = overlap / len(pred_tokens)
            recall = overlap / len(ref_tokens)
            score = (10 * precision * recall) / (recall + 9 * precision)
            best = max(best, score)
        scores.append(best)
    return sum(scores) / len(scores)


def _cider(predictions: list[str], references: list[list[str]]) -> float:
    if not predictions:
        return 0.0
    doc_freq: dict[int, Counter[tuple[str, ...]]] = {n: Counter() for n in range(1, 5)}
    for refs in references:
        for n in range(1, 5):
            unique_grams = set()
            for ref in refs:
                unique_grams.update(_ngrams(tokenize(ref), n))
            doc_freq[n].update(unique_grams)

    scores = []
    for prediction, refs in zip(predictions, references, strict=True):
        n_scores = []
        for n in range(1, 5):
            pred_vec = _tfidf_vector(tokenize(prediction), n, doc_freq[n], len(references))
            ref_scores = [
                _cosine(pred_vec, _tfidf_vector(tokenize(ref), n, doc_freq[n], len(references)))
                for ref in refs
            ]
            n_scores.append(sum(ref_scores) / len(ref_scores) if ref_scores else 0.0)
        scores.append(10.0 * sum(n_scores) / len(n_scores))
    return sum(scores) / len(scores)


def _tfidf_vector(
    tokens: list[str], n: int, doc_freq: Counter[tuple[str, ...]], document_count: int
) -> dict[tuple[str, ...], float]:
    counts = _ngrams(tokens, n)
    total = sum(counts.values())
    if total == 0:
        return {}
    vector = {}
    for gram, count in counts.items():
        tf = count / total
        idf = math.log((document_count + 1.0) / (doc_freq.get(gram, 0) + 1.0)) + 1.0
        vector[gram] = tf * idf
    return vector


def _cosine(left: dict, right: dict) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
