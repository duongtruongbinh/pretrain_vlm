"""Benchmark evaluation entry point.

Usage:
    python scripts/evaluate.py ktvic --annotations ... --checkpoint ...
    python scripts/evaluate.py vista-conversation --annotations ... --image-root ... --checkpoint ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (
    generate_answer,
    generate_caption,
    load_stage1_model,
    load_stage2_model,
    read_json_or_jsonl,
    write_json,
    write_jsonl,
)
from src.metrics import caption_metrics, summarize_instruction_scores
from src.runtime import render


VISTA_SYSTEM_PROMPT = (
    "Bạn là trợ lý thị giác tiếng Việt. "
    "Trả lời tự nhiên, đúng trọng tâm, dựa trên ảnh và lịch sử hội thoại nếu có."
)
VISTA_QUESTION_TEMPLATE = "{question}\nTrả lời:"


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


def _add_vista_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--predictions-jsonl", default=None)
    parser.add_argument("--output-dir", default="outputs/benchmarks/vista_conversation")
    parser.add_argument("--config-section", default="instruction_train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-text-tokens", type=int, default=None)
    parser.add_argument("--system-prompt", default=VISTA_SYSTEM_PROMPT)
    parser.add_argument("--question-template", default=VISTA_QUESTION_TEMPLATE)


def _load_vista_records(annotation_path: str | Path, image_root: str | Path, limit: int | None) -> list[dict]:
    path = Path(annotation_path).expanduser().resolve()
    root = Path(image_root).expanduser().resolve()
    rows = _read_vista_rows(path)
    records = []
    for row in rows:
        image_path = str((root / str(row["file_name"]).strip()).resolve())
        history: list[dict] = []
        turn_number = 0
        for turn in row["conversation"]:
            role = str(turn["role"]).strip()
            content = str(turn["content"]).strip()
            if role == "assistant":
                user_turn = _last_user_turn(history)
                if user_turn:
                    turn_number += 1
                    records.append(
                        {
                            "id": f"{row['id']}_turn{turn_number}",
                            "image": image_path,
                            "image_id": str(row["id"]),
                            "file_name": str(row["file_name"]).strip(),
                            "question": _format_question_with_history(history[:-1], user_turn),
                            "references": [content],
                            "captions": [str(caption).strip() for caption in row.get("captions", [])],
                            "turn_index": turn_number,
                        }
                    )
                    if limit is not None and len(records) >= limit:
                        return records
            history.append({"role": role, "content": content})
    return records


def _read_vista_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    rows = read_json_or_jsonl(path)
    if not isinstance(rows, list):
        raise ValueError("Vista conversation annotations must be a JSON list, JSONL rows, or parquet table.")
    return rows


def _last_user_turn(history: list[dict]) -> str:
    for turn in reversed(history):
        if turn["role"] == "user":
            return turn["content"]
    return ""


def _format_question_with_history(history: list[dict], question: str) -> str:
    if not history:
        return question
    lines = ["Lịch sử hội thoại:"]
    for turn in history:
        speaker = "Người dùng" if turn["role"] == "user" else "Trợ lý"
        lines.append(f"{speaker}: {turn['content']}")
    lines.extend(["", f"Câu hỏi hiện tại: {question}"])
    return "\n".join(lines)


def _run_vista(args: argparse.Namespace) -> tuple[list[dict], dict]:
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
        records = _load_vista_records(args.annotations, args.image_root, args.max_samples)
        model, collator, _ = load_stage2_model(
            args.checkpoint, args.config_section, args.device, args.max_text_tokens
        )
        rows = []
        for record in tqdm(records, desc="vista-conversation", dynamic_ncols=True):
            question_text = args.question_template.format(question=record["question"])
            with Image.open(record["image"]) as image:
                _, raw_prediction = generate_answer(
                    model,
                    collator,
                    image,
                    record["question"],
                    system_prompt=args.system_prompt,
                    max_new_tokens=args.max_new_tokens,
                    question_text=question_text,
                )
            rows.append({**record, "prediction": raw_prediction.strip(), "raw_prediction": raw_prediction})

    return rows, summarize_instruction_scores(rows)


def cmd_vista(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows, scores = _run_vista(args)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_json(output_dir / "metrics.json", scores)
    print(json.dumps(scores, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    _add_ktvic_args(subparsers.add_parser("ktvic", help="Stage-1 KTVIC captioning benchmark."))
    _add_vista_args(subparsers.add_parser("vista-conversation", help="Stage-2 Vista conversation benchmark."))

    args = parser.parse_args()
    if args.cmd == "ktvic":
        cmd_ktvic(args)
    elif args.cmd == "vista-conversation":
        cmd_vista(args)


if __name__ == "__main__":
    main()
