from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Image as HfImage
from datasets import load_dataset
from loguru import logger
from PIL import Image

from src.runtime import load_config


IMAGE_FIELD_CANDIDATES = ("image", "Image", "img")
DESCRIPTION_FIELD_CANDIDATES = ("Description", "description", "caption", "Caption")
QNA_FIELD_CANDIDATES = ("QnA", "qna", "messages", "conversations", "conversation")
DEFAULT_CONFIG_SECTION = "instruction_data_gpt"
SPLIT_NAME_MAP = {
    "train": "train",
    "training": "train",
    "validation": "val",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def log_check(name: str, condition: bool, detail: str) -> None:
    if not condition:
        logger.error("[error] {}: {}", name, detail)
        raise RuntimeError(detail)
    logger.info("[check] {}: {}", name, detail)


def sanitize_image_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "image"


def select_field(field_names, *, override, candidates, field_label: str) -> str:
    if override is not None:
        override = str(override).strip()
        log_check(
            f"{field_label}_override",
            override in field_names,
            f"Using configured {field_label} field '{override}'.",
        )
        return override

    for candidate in candidates:
        if candidate in field_names:
            log_check(
                f"{field_label}_auto_select", True, f"Selected field '{candidate}' from {list(field_names)}."
            )
            return candidate

    raise KeyError(f"Could not infer {field_label} field from available columns: {list(field_names)}")


def summarize_value(value):
    if isinstance(value, dict):
        return {key: summarize_value(sub_value) for key, sub_value in list(value.items())[:4]}
    if isinstance(value, list):
        return [summarize_value(item) for item in value[:2]]
    if isinstance(value, str):
        return value[:160]
    return str(type(value).__name__)


def normalize_qna_messages(raw_messages) -> list[dict[str, str]]:
    if raw_messages is None:
        return []
    if isinstance(raw_messages, str):
        raw_messages = json.loads(raw_messages)
    if not isinstance(raw_messages, list):
        raise ValueError("QnA field must be a list or a JSON-encoded list.")
    if len(raw_messages) % 2 != 0:
        raw_messages = raw_messages[:-1]  # drop trailing unpaired user message

    normalized_messages = []
    expected_role = "user"
    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("Each QnA message must be a mapping.")

        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role != expected_role:
            raise ValueError(f"Expected QnA role '{expected_role}' but received '{role}'.")
        if not content:
            raise ValueError(f"Encountered an empty '{role}' message.")

        normalized_messages.append({"role": role, "content": content})
        expected_role = "assistant" if expected_role == "user" else "user"

    return normalized_messages


def stable_split_for_image(image_key: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.sha1(f"{seed}:{image_key}".encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    if score < float(test_ratio):
        return "test"
    if score < float(test_ratio) + float(val_ratio):
        return "val"
    return "train"


def determine_output_split(
    raw_split: str, *, split_mode: str, image_key: str, seed: int, val_ratio: float, test_ratio: float
) -> str:
    normalized_mode = str(split_mode).strip().lower()
    normalized_raw_split = SPLIT_NAME_MAP.get(str(raw_split).strip().lower())

    if normalized_mode == "source":
        if normalized_raw_split is None:
            raise ValueError(f"Source split '{raw_split}' cannot be mapped to train/val/test.")
        return normalized_raw_split

    if normalized_mode == "image_level":
        return stable_split_for_image(image_key, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

    if normalized_mode != "auto":
        raise ValueError(f"Unsupported split_mode '{split_mode}'.")

    return stable_split_for_image(image_key, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)


def infer_image_key(row: dict, image_field: str, row_index: int, raw_split: str) -> str:
    for candidate in ("image_id", "id", "file_name", "filename", "name"):
        candidate_value = row.get(candidate)
        if candidate_value is None:
            continue
        candidate_text = str(candidate_value).strip()
        if candidate_text:
            return sanitize_image_key(Path(candidate_text).stem or candidate_text)

    image_value = row[image_field]
    if isinstance(image_value, dict):
        image_path = image_value.get("path")
        if image_path:
            return sanitize_image_key(Path(str(image_path)).stem)

        image_bytes = image_value.get("bytes")
        if image_bytes:
            return hashlib.sha1(image_bytes).hexdigest()[:16]

    return f"{raw_split}_{row_index:07d}"


def infer_image_extension(image_value, image_key: str) -> str:
    if isinstance(image_value, dict):
        image_path = image_value.get("path")
        if image_path:
            suffix = Path(str(image_path)).suffix.lower()
            if suffix:
                return suffix
    if isinstance(image_value, Image.Image):
        return ".jpg"
    return ".png"


def save_image_asset(image_value, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image_value, dict):
        image_path = image_value.get("path")
        image_bytes = image_value.get("bytes")
        if image_path and Path(str(image_path)).exists():
            shutil.copy2(str(image_path), destination)
        elif image_bytes is not None:
            destination.write_bytes(image_bytes)
        else:
            raise ValueError("Image payload does not provide a valid path or bytes.")
    elif isinstance(image_value, Image.Image):
        image = image_value
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(destination, format="JPEG", quality=95)
    else:
        raise TypeError(f"Unsupported image value type: {type(image_value)!r}")

    with Image.open(destination) as image:
        if image.mode != "RGB":
            image.convert("RGB").save(destination)


def build_description_sample(
    *,
    sample_id: str,
    image_path: Path,
    image_key: str,
    system_prompt: str,
    user_prompt: str,
    assistant_text: str,
    source_dataset: str,
) -> dict:
    return {
        "id": sample_id,
        "image": str(image_path.resolve()),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ],
        "sample_type": "description",
        "source_dataset": source_dataset,
        "image_id": image_key,
    }


def build_qna_samples(
    *,
    image_path: Path,
    image_key: str,
    qna_messages: list[dict[str, str]],
    system_prompt: str,
    sample_id_prefix: str,
    source_dataset: str,
) -> list[dict]:
    return [
        {
            "id": f"{sample_id_prefix}_qa",
            "image": str(image_path.resolve()),
            "messages": [{"role": "system", "content": system_prompt}, *qna_messages],
            "sample_type": "qa",
            "source_dataset": source_dataset,
            "image_id": image_key,
        }
    ]


def parse_args(argv: list[str] | None = None, *, default_config_section: str = DEFAULT_CONFIG_SECTION):
    parser = argparse.ArgumentParser(
        description="Prepare image instruction-tuning JSONL from a Hugging Face dataset."
    )
    parser.add_argument("--config-section", default=default_config_section)
    return parser.parse_args(argv)


def load_dataset_from_config(config: dict):
    dataset_name = str(config["dataset_name"]).strip()
    dataset_config = config.get("dataset_config")
    kwargs = {"streaming": bool(config.get("streaming", False))}
    if dataset_config is not None:
        kwargs["name"] = str(dataset_config).strip()
    return load_dataset(dataset_name, **kwargs)


def get_first_row(dataset_dict, first_split: str, *, streaming: bool):
    first_dataset = dataset_dict[first_split]
    if streaming:
        return next(iter(first_dataset))
    return first_dataset[0]


def maybe_cast_images(dataset_dict, image_field: str, *, streaming: bool):
    if streaming:
        return dataset_dict
    return {
        split_name: split_dataset.cast_column(image_field, HfImage(decode=False))
        for split_name, split_dataset in dataset_dict.items()
    }


def dataset_len_text(dataset) -> str:
    try:
        return str(len(dataset))
    except TypeError:
        return "unknown (streaming)"


def run(config_section: str) -> None:
    config = load_config(config_section)
    dataset_name = str(config["dataset_name"]).strip()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    images_root = output_dir / "images"
    streaming = bool(config.get("streaming", False))

    logger.info("[instruction-data] loading dataset '{}' from config section '{}'...", dataset_name, config_section)
    dataset_dict = load_dataset_from_config(config)
    raw_splits = list(dataset_dict.keys())
    log_check("dataset_load", bool(raw_splits), f"Loaded raw splits: {raw_splits}")

    first_split = raw_splits[0]
    first_features = list(dataset_dict[first_split].features.keys())
    image_field = select_field(
        first_features,
        override=config.get("image_field"),
        candidates=IMAGE_FIELD_CANDIDATES,
        field_label="image",
    )
    description_field = select_field(
        first_features,
        override=config.get("description_field"),
        candidates=DESCRIPTION_FIELD_CANDIDATES,
        field_label="description",
    )
    qna_field = select_field(
        first_features, override=config.get("qna_field"), candidates=QNA_FIELD_CANDIDATES, field_label="qna"
    )

    logger.info("[instruction-data] schema summary:")
    for raw_split in raw_splits:
        logger.info("  - {}: columns={}", raw_split, list(dataset_dict[raw_split].features.keys()))

    preview_row = get_first_row(dataset_dict, first_split, streaming=streaming)
    preview_summary = {
        image_field: summarize_value(preview_row[image_field]),
        description_field: summarize_value(preview_row.get(description_field)),
        qna_field: summarize_value(preview_row.get(qna_field)),
    }
    logger.info("[instruction-data] first-row preview: {}", json.dumps(preview_summary, ensure_ascii=False))

    dataset_dict = maybe_cast_images(dataset_dict, image_field, streaming=streaming)

    configured_split_mode = str(config.get("split_mode", "auto")).strip().lower()
    mapped_raw_splits = {SPLIT_NAME_MAP.get(str(split_name).strip().lower()) for split_name in raw_splits}
    if configured_split_mode == "auto":
        effective_split_mode = (
            "source" if {"train", "val", "test"}.issubset(mapped_raw_splits) else "image_level"
        )
    else:
        effective_split_mode = configured_split_mode
    log_check(
        "split_mode",
        True,
        f"Resolved split_mode='{effective_split_mode}' from configured value '{configured_split_mode}'.",
    )

    if bool(config.get("inspect_only", False)):
        logger.info("[instruction-data] inspect_only=true, stopping after schema checks.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    output_handles = {
        "train": (output_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (output_dir / "val.jsonl").open("w", encoding="utf-8"),
        "test": (output_dir / "test.jsonl").open("w", encoding="utf-8"),
    }

    counters = Counter()
    split_to_image_ids = defaultdict(set)
    image_id_to_split = {}
    written_images = set()
    approx_token_lengths = []
    max_rows = config.get("max_rows")
    if max_rows is not None:
        max_rows = int(max_rows)

    try:
        processed_rows = 0
        for raw_split in raw_splits:
            raw_dataset = dataset_dict[raw_split]
            logger.info(
                "[instruction-data] processing raw split '{}' with {} rows...",
                raw_split, dataset_len_text(raw_dataset),
            )

            for row_index, row in enumerate(raw_dataset):
                if max_rows is not None and processed_rows >= max_rows:
                    logger.info("[instruction-data] reached max_rows={}, stopping early.", max_rows)
                    break

                processed_rows += 1
                counters["rows_total"] += 1
                try:
                    image_key = infer_image_key(
                        row, image_field=image_field, row_index=row_index, raw_split=raw_split
                    )
                    output_split = determine_output_split(
                        raw_split,
                        split_mode=effective_split_mode,
                        image_key=image_key,
                        seed=int(config.get("split_seed", 42)),
                        val_ratio=float(config.get("val_ratio", 0.01)),
                        test_ratio=float(config.get("test_ratio", 0.01)),
                    )

                    previous_split = image_id_to_split.setdefault(image_key, output_split)
                    if previous_split != output_split:
                        raise RuntimeError(
                            f"Image '{image_key}' was assigned to both "
                            f"'{previous_split}' and '{output_split}'."
                        )

                    image_value = row[image_field]
                    image_extension = infer_image_extension(image_value, image_key=image_key)
                    image_output_path = images_root / output_split / f"{image_key}{image_extension}"
                    if image_output_path not in written_images:
                        save_image_asset(image_value, image_output_path)
                        written_images.add(image_output_path)
                        counters[f"{output_split}_images_written"] += 1

                    built_samples = []

                    description_text = str(row.get(description_field, "") or "").strip()
                    if bool(config.get("use_description_samples", True)):
                        if description_text:
                            built_samples.append(
                                build_description_sample(
                                    sample_id=f"{output_split}_{image_key}_desc",
                                    image_path=image_output_path,
                                    image_key=image_key,
                                    system_prompt=str(config["system_prompt"]).strip(),
                                    user_prompt=str(config["description_user_prompt"]).strip(),
                                    assistant_text=description_text,
                                    source_dataset=dataset_name,
                                )
                            )
                        else:
                            counters["rows_empty_description"] += 1

                    if bool(config.get("use_qna_samples", True)):
                        raw_qna = row.get(qna_field)
                        if raw_qna is not None and raw_qna != "":
                            qna_messages = normalize_qna_messages(raw_qna)
                            if qna_messages:
                                built_samples.extend(
                                    build_qna_samples(
                                        image_path=image_output_path,
                                        image_key=image_key,
                                        qna_messages=qna_messages,
                                        system_prompt=str(config["system_prompt"]).strip(),
                                        sample_id_prefix=f"{output_split}_{image_key}",
                                        source_dataset=dataset_name,
                                    )
                                )
                            else:
                                counters["rows_empty_qna"] += 1
                    if not built_samples:
                        counters["rows_skipped_no_samples"] += 1
                        continue

                    split_to_image_ids[output_split].add(image_key)
                    for sample in built_samples:
                        json.dump(sample, output_handles[output_split], ensure_ascii=False)
                        output_handles[output_split].write("\n")
                        counters[f"{output_split}_samples_written"] += 1
                        counters["samples_written"] += 1
                        approx_token_lengths.append(
                            sum(len(str(message["content"]).split()) for message in sample["messages"])
                        )
                except Exception as error:
                    error_text = str(error).lower()
                    if (
                        "image payload" in error_text
                        or "unsupported image" in error_text
                        or "cannot identify image file" in error_text
                    ):
                        counters["rows_missing_or_bad_image"] += 1
                    elif "qna" in error_text or "assistant" in error_text or "user" in error_text:
                        counters["rows_malformed_qna"] += 1
                    else:
                        counters["rows_other_errors"] += 1
                    counters["rows_skipped"] += 1
                    logger.warning("[instruction-data] skipped row split={} index={}: {}", raw_split, row_index, error)

            if max_rows is not None and processed_rows >= max_rows:
                break
    finally:
        for handle in output_handles.values():
            handle.close()

    log_check(
        "samples_written",
        counters["samples_written"] > 0,
        f"Wrote {counters['samples_written']} instruction samples.",
    )
    leakage_free = (
        split_to_image_ids["train"].isdisjoint(split_to_image_ids["val"])
        and split_to_image_ids["train"].isdisjoint(split_to_image_ids["test"])
        and split_to_image_ids["val"].isdisjoint(split_to_image_ids["test"])
    )
    log_check("image_level_split", leakage_free, "Verified that train/val/test image sets are disjoint.")

    report = {
        "dataset_name": dataset_name,
        "raw_splits": raw_splits,
        "selected_fields": {"image": image_field, "description": description_field, "qna": qna_field},
        "config": {
            "split_mode": effective_split_mode,
            "split_seed": int(config.get("split_seed", 42)),
            "val_ratio": float(config.get("val_ratio", 0.01)),
            "test_ratio": float(config.get("test_ratio", 0.01)),
            "use_description_samples": bool(config.get("use_description_samples", True)),
            "use_qna_samples": bool(config.get("use_qna_samples", True)),
        },
        "counts": {
            "rows_total": counters["rows_total"],
            "rows_skipped": counters["rows_skipped"],
            "rows_skipped_no_samples": counters["rows_skipped_no_samples"],
            "rows_empty_description": counters["rows_empty_description"],
            "rows_empty_qna": counters["rows_empty_qna"],
            "rows_missing_or_bad_image": counters["rows_missing_or_bad_image"],
            "rows_malformed_qna": counters["rows_malformed_qna"],
            "rows_other_errors": counters["rows_other_errors"],
            "samples_written": counters["samples_written"],
            "train_samples_written": counters["train_samples_written"],
            "val_samples_written": counters["val_samples_written"],
            "test_samples_written": counters["test_samples_written"],
            "train_images_written": counters["train_images_written"],
            "val_images_written": counters["val_images_written"],
            "test_images_written": counters["test_images_written"],
            "train_unique_images": len(split_to_image_ids["train"]),
            "val_unique_images": len(split_to_image_ids["val"]),
            "test_unique_images": len(split_to_image_ids["test"]),
        },
        "approx_message_word_count": {
            "min": min(approx_token_lengths) if approx_token_lengths else 0,
            "max": max(approx_token_lengths) if approx_token_lengths else 0,
            "mean": round(sum(approx_token_lengths) / len(approx_token_lengths), 2)
            if approx_token_lengths
            else 0.0,
        },
    }
    report_path = output_dir / "prepare_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    logger.info("[instruction-data] done")
    logger.info("[instruction-data] report written to {}", report_path)


def main(argv: list[str] | None = None, *, default_config_section: str = DEFAULT_CONFIG_SECTION) -> None:
    args = parse_args(argv, default_config_section=default_config_section)
    run(args.config_section)


if __name__ == "__main__":
    main()
