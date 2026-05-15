from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from collections import Counter
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare DavidPhilips/coco2017 as local image-caption JSONL."
    )
    parser.add_argument("--config-section", default="prepare_coco")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def sanitize_id(value: str, fallback: str) -> str:
    text = str(value or "").strip() or fallback
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or fallback


def split_mapping(config: dict) -> dict[str, str]:
    raw = config.get("split_map") or {"train": "train", "validation": "val"}
    if not isinstance(raw, dict) or not raw:
        raise ValueError("prepare_coco.split_map must be a non-empty mapping.")
    return {str(hf_split): str(local_split) for hf_split, local_split in raw.items()}


def choose_caption(row: dict, caption_field: str) -> str:
    return str(row.get(caption_field, "") or "").strip()


def save_image(image_value, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    should_close = False
    if isinstance(image_value, Image.Image):
        image = image_value
    elif isinstance(image_value, dict) and image_value.get("bytes") is not None:
        image = Image.open(BytesIO(image_value["bytes"]))
        should_close = True
    elif isinstance(image_value, dict) and image_value.get("path"):
        image = Image.open(image_value["path"])
        should_close = True
    else:
        raise TypeError(f"Unsupported image payload type: {type(image_value)!r}")

    try:
        image.convert("RGB").save(destination, format="JPEG", quality=95)
    finally:
        if should_close:
            image.close()


def prepare_split(config: dict, hf_split: str, local_split: str, output_dir: Path) -> None:
    dataset_name = str(config["dataset_name"]).strip()
    dataset_config = config.get("dataset_config")
    caption_field = str(config.get("caption_field", "caption_vi")).strip()
    image_field = str(config.get("image_field", "image")).strip()
    image_id_field = str(config.get("image_id_field", "image_id")).strip()
    caption_id_field = str(config.get("caption_id_field", "caption_id")).strip()
    streaming = bool(config.get("streaming", True))
    overwrite = bool(config.get("overwrite", False))
    max_rows = config.get("max_rows_per_split")
    max_rows = int(max_rows) if max_rows is not None else None

    output_jsonl = output_dir / f"{local_split}.jsonl"
    if output_jsonl.exists() and not overwrite:
        print(f"Skip existing {output_jsonl}. Set overwrite=true to rebuild.")
        return

    print(
        f"[coco] loading {dataset_name} config={dataset_config or 'default'} split={hf_split} streaming={streaming}"
    )
    dataset = load_dataset(dataset_name, dataset_config, split=hf_split, streaming=streaming)

    images_dir = output_dir / "images" / local_split
    stats = Counter()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(dataset):
            if max_rows is not None and row_index >= max_rows:
                break

            image_id = sanitize_id(row.get(image_id_field), f"{local_split}_{row_index:08d}")
            caption_id = sanitize_id(row.get(caption_id_field), f"{row_index:08d}")
            caption = choose_caption(row, caption_field)
            if not caption:
                stats["skipped_empty_caption"] += 1
                continue

            image_path = images_dir / f"{image_id}.jpg"
            try:
                save_image(row[image_field], image_path)
            except Exception as error:
                stats["skipped_bad_image"] += 1
                print(f"[coco][warn] skip image_id={image_id} row={row_index}: {error}")
                continue

            json.dump(
                {
                    "image": str(image_path),
                    "caption": caption,
                    "source_dataset": dataset_name,
                    "source_split": hf_split,
                    "image_id": image_id,
                    "caption_id": caption_id,
                },
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")
            stats["rows_written"] += 1

    print(
        f"[coco] wrote {output_jsonl} | "
        + " | ".join(f"{key}={value}" for key, value in sorted(stats.items()))
    )


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config_section)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_rows_per_split is not None:
        config["max_rows_per_split"] = args.max_rows_per_split
    if args.overwrite:
        config["overwrite"] = True
    if args.inspect_only:
        config["max_rows_per_split"] = 3
        config["overwrite"] = True

    output_dir = Path(config["output_dir"]).expanduser().resolve()
    for hf_split, local_split in split_mapping(config).items():
        prepare_split(config, hf_split, local_split, output_dir)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    sys.stdout.flush()
    sys.stderr.flush()
    # Streaming image datasets can abort during native teardown in this env after
    # all Python file handles are closed. Exit directly once preparation succeeds.
    os._exit(0)
