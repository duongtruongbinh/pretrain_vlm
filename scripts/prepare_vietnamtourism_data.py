"""Convert raw crawl + generated conversations to project instruction-tuning JSONL."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from src.runtime import hash_split, load_config


def normalize_conversation_messages(raw_messages) -> list[dict[str, str]]:
    if raw_messages is None:
        return []
    if not isinstance(raw_messages, list):
        raise ValueError("conversation must be a list.")
    if len(raw_messages) % 2 != 0:
        raw_messages = raw_messages[:-1]

    normalized_messages: list[dict[str, str]] = []
    expected_role = "user"
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise ValueError(f"conversation message #{index} must be a dict.")

        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role != expected_role:
            raise ValueError(
                f"expected role '{expected_role}' at conversation message #{index}, got '{role}'."
            )
        if not content:
            raise ValueError(f"conversation message #{index} has empty content.")

        normalized_messages.append({"role": role, "content": content})
        expected_role = "assistant" if expected_role == "user" else "user"

    return normalized_messages


def qa_pairs_to_conversation(qa_pairs: list[dict]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for qa in qa_pairs:
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()
        if not question or not answer:
            continue
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    return normalize_conversation_messages(messages)


def conversation_from_record(record: dict) -> list[dict[str, str]]:
    conversation = normalize_conversation_messages(record.get("conversation"))
    if conversation:
        return conversation
    return qa_pairs_to_conversation(record.get("qa_pairs", []))


def build_instruction_entry(
    *,
    image_path: Path,
    conversation: list[dict[str, str]],
    system_prompt: str,
    image_id: str,
    source: str,
    title: str = "",
    caption: str = "",
    description: str = "",
    article_url: str = "",
    date: str = "",
    post_id: str = "",
) -> dict:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation)
    return {
        "id": f"vntour_{image_id}",
        "image": str(image_path),
        "messages": messages,
        "sample_type": "conversation",
        "source_dataset": source,
        "image_id": image_id,
        "title": title,
        "caption": caption,
        "description": description,
        "article_url": article_url,
        "date": date,
        "post_id": post_id,
    }


def main() -> None:
    cfg = load_config("prepare_vietnamtourism")
    raw_dir = Path(cfg["raw_dir"]).expanduser().resolve()
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    system_prompt = str(cfg["system_prompt"])
    seed = int(cfg.get("seed", 42))
    val_ratio = float(cfg.get("val_ratio", 0.01))
    test_ratio = float(cfg.get("test_ratio", 0.01))
    source = "vietnamtourism"

    batch_results = raw_dir / "batch_results.jsonl"
    if not batch_results.exists():
        raise FileNotFoundError(f"Run generate script first: {batch_results}")

    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    counters: Counter = Counter()
    split_image_ids: dict[str, set] = defaultdict(set)

    try:
        with batch_results.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                image_id = rec["image_id"]
                image_path = Path(rec["image_path"])

                if not image_path.exists():
                    counters["skipped_missing_image"] += 1
                    continue

                split = assign_split(image_id, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
                split_image_ids[split].add(image_id)

                try:
                    conversation = conversation_from_record(rec)
                except ValueError as exc:
                    counters["skipped_invalid_conversation"] += 1
                    logger.warning("skipping {}: {}", image_id, exc)
                    continue
                if not conversation:
                    counters["skipped_empty_conversation"] += 1
                    continue

                entry = build_instruction_entry(
                    image_path=image_path,
                    conversation=conversation,
                    system_prompt=system_prompt,
                    image_id=image_id,
                    source=source,
                    title=rec.get("title", ""),
                    caption=rec.get("caption", ""),
                    description=rec.get("description", ""),
                    article_url=rec.get("article_url", ""),
                    date=rec.get("date", ""),
                    post_id=rec.get("post_id", ""),
                )
                handles[split].write(json.dumps(entry, ensure_ascii=False) + "\n")
                counters[f"{split}_images"] += 1
                counters["total_images"] += 1
    finally:
        for h in handles.values():
            h.close()

    logger.info("[prepare] done")
    for split in ("train", "val", "test"):
        logger.info("  {}: {} images → {}.jsonl", split, counters[f"{split}_images"], split)
    logger.info("  skipped missing images: {}", counters["skipped_missing_image"])
    logger.info("  skipped empty conv:     {}", counters["skipped_empty_conversation"])
    logger.info("  skipped invalid conv:   {}", counters["skipped_invalid_conversation"])

    (output_dir / "prepare_report.json").write_text(
        json.dumps(
            {
                "source": source,
                "counts": dict(counters),
                "split_images": {k: len(v) for k, v in split_image_ids.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
