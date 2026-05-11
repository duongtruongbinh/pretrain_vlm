"""Convert raw_crawl.jsonl + batch_results.jsonl to project instruction-tuning JSONL."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.runtime import load_config


def assign_split(image_id: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.sha1(f"{seed}:{image_id}".encode()).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    if score < test_ratio:
        return "test"
    if score < test_ratio + val_ratio:
        return "val"
    return "train"


def build_instruction_entry(
    *,
    image_path: Path,
    qa_pairs: list[dict],
    system_prompt: str,
    image_id: str,
    source: str,
) -> dict:
    messages = [{"role": "system", "content": system_prompt}]
    for qa in qa_pairs:
        messages.append({"role": "user", "content": qa["question"]})
        messages.append({"role": "assistant", "content": qa["answer"]})
    return {
        "id": f"vntour_{image_id}",
        "image": str(image_path),
        "messages": messages,
        "sample_type": "qa",
        "source_dataset": source,
        "image_id": image_id,
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

                qa_pairs = [
                    qa for qa in rec.get("qa_pairs", [])
                    if qa.get("question") and qa.get("answer")
                ]
                if not qa_pairs:
                    counters["skipped_empty_qa"] += 1
                    continue

                entry = build_instruction_entry(
                    image_path=image_path,
                    qa_pairs=qa_pairs,
                    system_prompt=system_prompt,
                    image_id=image_id,
                    source=source,
                )
                handles[split].write(json.dumps(entry, ensure_ascii=False) + "\n")
                counters[f"{split}_images"] += 1
                counters["total_images"] += 1
    finally:
        for h in handles.values():
            h.close()

    print("[prepare] done")
    for split in ("train", "val", "test"):
        print(f"  {split}: {counters[f'{split}_images']} images → {split}.jsonl")
    print(f"  skipped missing images: {counters['skipped_missing_image']}")
    print(f"  skipped empty qa:       {counters['skipped_empty_qa']}")

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
