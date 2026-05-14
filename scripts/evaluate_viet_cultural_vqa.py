"""Evaluate a Stage-2 instruction checkpoint on Viet Cultural VQA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.io import read_json_or_jsonl, write_json, write_jsonl  # noqa: E402
from src.evaluation.metrics import summarize_vqa_scores  # noqa: E402
from src.evaluation.vqa_runtime import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    generate_answer,
    load_stage2_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-2 Vietnamese VQA on Viet Cultural VQA.")
    parser.add_argument("--annotations", default=None, help="Viet Cultural VQA split JSON/JSONL path.")
    parser.add_argument("--image-root", default=None, help="Directory containing downloaded dataset images.")
    parser.add_argument("--checkpoint", default=None, help="Stage-2 checkpoint directory.")
    parser.add_argument("--predictions-jsonl", default=None, help="Existing predictions JSONL; skips generation.")
    parser.add_argument("--output-dir", default="outputs/benchmarks/viet_cultural_vqa")
    parser.add_argument("--config-section", default="instruction_train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of QA rows to evaluate.")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-text-tokens", type=int, default=None)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser.parse_args()


def load_viet_cultural_vqa_records(
    annotation_path: str | Path, image_root: str | Path, limit: int | None = None
) -> list[dict]:
    path = Path(annotation_path).expanduser().resolve()
    root = Path(image_root).expanduser().resolve()
    samples = read_json_or_jsonl(path)
    if not isinstance(samples, list):
        raise ValueError("Viet Cultural VQA split must be a JSON list.")

    records = []
    for sample in samples:
        image_path = _resolve_image_path(sample["image_path"], root)
        image_id = str(sample["image_id"])
        category = str(sample["category"]).strip()
        keyword = str(sample["keyword"]).strip()
        for question_item in sample["questions"]:
            question = str(question_item["question"]).strip()
            answer = str(question_item["answer"]).strip()
            question_id = str(question_item["question_id"])
            explanation = str(question_item.get("detailed_explanation", "")).strip()
            records.append(
                {
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
                }
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def _resolve_image_path(image_value: str, image_root: Path) -> str:
    raw = Path(str(image_value).strip()).expanduser()
    parts = raw.parts
    if parts[:2] == ("data", "images"):
        raw = Path(*parts[1:])
    return str((image_root / raw).resolve())


def evaluate_records(args: argparse.Namespace) -> tuple[list[dict], dict]:
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
        records = load_viet_cultural_vqa_records(args.annotations, args.image_root, limit=args.max_samples)
        model, collator, _state = load_stage2_model(
            args.checkpoint, args.config_section, args.device, args.max_text_tokens
        )
        rows = []
        for record in tqdm(records, desc="viet-cultural-vqa", dynamic_ncols=True):
            with Image.open(record["image"]) as image:
                prediction, raw_prediction = generate_answer(
                    model,
                    collator,
                    image,
                    record["question"],
                    system_prompt=args.system_prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            rows.append(
                {
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
                }
            )

    scores = summarize_vqa_scores(rows)
    return rows, scores


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows, scores = evaluate_records(args)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_json(output_dir / "metrics.json", scores)
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
