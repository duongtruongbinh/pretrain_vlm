"""Convert raw_crawl.jsonl + batch_results.jsonl to project instruction-tuning JSONL."""
from __future__ import annotations

import hashlib
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


def build_instruction_sample(
    *,
    image_path: Path,
    qa_pair: dict,
    system_prompt: str,
    image_id: str,
    source: str,
    sample_index: int,
) -> dict:
    return {
        "id": f"vntour_{image_id}_qa_{sample_index + 1:03d}",
        "image": str(image_path),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": qa_pair["question"]},
            {"role": "assistant", "content": qa_pair["answer"]},
        ],
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

                for idx, qa_pair in enumerate(rec.get("qa_pairs", [])):
                    if not qa_pair.get("question") or not qa_pair.get("answer"):
                        counters["skipped_empty_qa"] += 1
                        continue
                    sample = build_instruction_sample(
                        image_path=image_path,
                        qa_pair=qa_pair,
                        system_prompt=system_prompt,
                        image_id=image_id,
                        source=source,
                        sample_index=idx,
                    )
                    handles[split].write(json.dumps(sample, ensure_ascii=False) + "\n")
                    counters[f"{split}_samples"] += 1
                    counters["total_samples"] += 1
    finally:
        for h in handles.values():
            h.close()

    print("[prepare] done")
    for split in ("train", "val", "test"):
        print(
            f"  {split}: {counters[f'{split}_samples']} samples "
            f"from {len(split_image_ids[split])} images"
        )
    print(f"  skipped missing images: {counters['skipped_missing_image']}")

    (output_dir / "prepare_report.json").write_text(
        json.dumps(
            {
                "source": source,
                "counts": dict(counters),
                "split_unique_images": {k: len(v) for k, v in split_image_ids.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
