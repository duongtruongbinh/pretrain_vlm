"""Benchmark evaluation entry point.

Usage:
    python scripts/evaluate.py ktvic --annotations ... --checkpoint ...
    python scripts/evaluate.py viet-cultural-vqa --annotations ... --image-root ... --checkpoint ...
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from tqdm.auto import tqdm

from src.inference import (
    DEFAULT_SYSTEM_PROMPT,
    generate_answer,
    generate_caption,
    load_stage1_model,
    load_stage2_model,
    read_json_or_jsonl,
    write_json,
    write_jsonl,
)
from src.metrics import caption_metrics, summarize_vqa_scores
from src.runtime import render


# ---------------------------------------------------------------------------
# KTVIC
# ---------------------------------------------------------------------------

def _add_ktvic_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--predictions-jsonl", default=None)
    parser.add_argument("--output-dir", default="outputs/benchmarks/ktvic")
    parser.add_argument("--config-section", default="train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=None)


def _load_ktvic_records(annotation_path: str | Path, image_root: str | Path | None, limit: int | None) -> list[dict]:
    path = Path(annotation_path).expanduser().resolve()
    root = Path(image_root).expanduser().resolve() if image_root else path.parent
    payload = read_json_or_jsonl(path)
    if not isinstance(payload, dict) or "images" not in payload or "annotations" not in payload:
        raise ValueError("KTVIC annotations must use COCO-style images/annotations JSON.")

    images = {item["id"]: item for item in payload["images"]}
    grouped: OrderedDict[str, dict] = OrderedDict()
    for annotation in payload["annotations"]:
        image_id = annotation["image_id"]
        file_name = images[image_id]["file_name"]
        raw = Path(str(file_name).strip()).expanduser()
        image_path = str(raw.resolve()) if raw.is_absolute() else str((root / raw).resolve())
        record = grouped.setdefault(image_path, {"id": image_id, "image": image_path, "references": []})
        record["references"].append(str(annotation["caption"]).strip())

    records = list(grouped.values())
    return records[:limit] if limit is not None else records


def _run_ktvic(args: argparse.Namespace) -> tuple[list[dict], dict]:
    from PIL import Image

    prompt = args.prompt or render("caption_prompt.j2")

    if args.predictions_jsonl:
        rows = read_json_or_jsonl(args.predictions_jsonl)
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
    else:
        if not args.annotations:
            raise ValueError("--annotations is required unless --predictions-jsonl is provided.")
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --predictions-jsonl is provided.")
        records = _load_ktvic_records(args.annotations, args.image_root, args.max_samples)
        model, processor, _ = load_stage1_model(args.checkpoint, args.config_section, args.device)
        rows = []
        for record in tqdm(records, desc="ktvic", dynamic_ncols=True):
            with Image.open(record["image"]) as image:
                prediction = generate_caption(model, processor, image, prompt, args.max_new_tokens)
            rows.append({
                "id": record["id"],
                "image": record["image"],
                "prediction": prediction,
                "references": record["references"],
            })

    scores = caption_metrics(
        [row.get("prediction", "") for row in rows],
        [row.get("references", []) for row in rows],
    )
    return rows, scores


def cmd_ktvic(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows, scores = _run_ktvic(args)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_json(output_dir / "metrics.json", scores)
    print(json.dumps(scores, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Viet Cultural VQA
# ---------------------------------------------------------------------------

def _add_vcvqa_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--predictions-jsonl", default=None)
    parser.add_argument("--output-dir", default="outputs/benchmarks/viet_cultural_vqa")
    parser.add_argument("--config-section", default="instruction_train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-text-tokens", type=int, default=None)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)


def _load_vcvqa_records(annotation_path: str | Path, image_root: str | Path, limit: int | None) -> list[dict]:
    path = Path(annotation_path).expanduser().resolve()
    root = Path(image_root).expanduser().resolve()
    samples = read_json_or_jsonl(path)
    if not isinstance(samples, list):
        raise ValueError("Viet Cultural VQA split must be a JSON list.")

    records = []
    for sample in samples:
        raw = Path(str(sample["image_path"]).strip()).expanduser()
        parts = raw.parts
        if parts[:2] == ("data", "images"):
            raw = Path(*parts[1:])
        image_path = str((root / raw).resolve())
        image_id = str(sample["image_id"])
        category = str(sample["category"]).strip()
        keyword = str(sample["keyword"]).strip()
        for question_item in sample["questions"]:
            question = str(question_item["question"]).strip()
            answer = str(question_item["answer"]).strip()
            question_id = str(question_item["question_id"])
            explanation = str(question_item.get("detailed_explanation", "")).strip()
            records.append({
                "id": f"{image_id}_{question_id}",
                "image": image_path,
                "question": question,
                "references": [answer],
                "explanations": [explanation] if explanation else [],
                "category": category,
                "keyword": keyword,
                "question_type": str(question_item.get("question_type", "")).strip(),
                "difficulty": str(question_item.get("difficulty", "")).strip(),
                "cognitive_level": str(question_item.get("cognitive_level", "")).strip(),
            })
            if limit is not None and len(records) >= limit:
                return records
    return records


def _run_vcvqa(args: argparse.Namespace) -> tuple[list[dict], dict]:
    from PIL import Image

    if args.predictions_jsonl:
        rows = read_json_or_jsonl(args.predictions_jsonl)
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
    else:
        if not args.annotations:
            raise ValueError("--annotations is required unless --predictions-jsonl is provided.")
        if not args.image_root:
            raise ValueError("--image-root is required unless --predictions-jsonl is provided.")
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --predictions-jsonl is provided.")
        records = _load_vcvqa_records(args.annotations, args.image_root, args.max_samples)
        model, collator, _ = load_stage2_model(
            args.checkpoint, args.config_section, args.device, args.max_text_tokens
        )
        rows = []
        for record in tqdm(records, desc="viet-cultural-vqa", dynamic_ncols=True):
            with Image.open(record["image"]) as image:
                prediction, raw_prediction = generate_answer(
                    model, collator, image, record["question"],
                    system_prompt=args.system_prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            rows.append({
                "id": record["id"],
                "image": record["image"],
                "question": record["question"],
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "references": record["references"],
                "explanations": record["explanations"],
                "category": record["category"],
                "keyword": record["keyword"],
                "question_type": record["question_type"],
                "difficulty": record["difficulty"],
                "cognitive_level": record["cognitive_level"],
            })

    scores = summarize_vqa_scores(rows)
    return rows, scores


def cmd_vcvqa(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows, scores = _run_vcvqa(args)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_json(output_dir / "metrics.json", scores)
    print(json.dumps(scores, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    _add_ktvic_args(subparsers.add_parser("ktvic", help="Stage-1 KTVIC captioning benchmark."))
    _add_vcvqa_args(subparsers.add_parser("viet-cultural-vqa", help="Stage-2 Viet Cultural VQA benchmark."))

    args = parser.parse_args()
    if args.cmd == "ktvic":
        cmd_ktvic(args)
    elif args.cmd == "viet-cultural-vqa":
        cmd_vcvqa(args)


if __name__ == "__main__":
    main()
