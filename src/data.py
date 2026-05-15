"""Dataset definitions for caption pretraining and instruction tuning."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.runtime import resolve_record_image_path


class ImageCaptionDataset(Dataset):
    def __init__(self, jsonl_path: str | Path | list[str | Path]):
        paths = [jsonl_path] if isinstance(jsonl_path, (str, Path)) else jsonl_path
        self.jsonl_paths = [Path(path).expanduser().resolve() for path in paths]
        self.records = []
        self.bad_indices = set()

        self.source_indices: list[int] = []

        for source_idx, p in enumerate(self.jsonl_paths):
            with p.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    record = json.loads(line)
                    if "image" not in record or "caption" not in record:
                        raise ValueError(
                            f"Each JSONL line must contain 'image' and 'caption'. Bad line {line_number} in {p}."
                        )
                    record["image"] = resolve_record_image_path(record["image"], jsonl_path=p)
                    self.records.append(record)
                    self.source_indices.append(source_idx)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        n = len(self.records)
        for offset in range(n):
            current_index = (index + offset) % n
            if current_index in self.bad_indices:
                continue

            record = self.records[current_index]
            image_path = record["image"]

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
            except Exception as error:
                warnings.warn(f"Skipping corrupt image at {image_path}: {error}")
                self.bad_indices.add(current_index)
                continue

            return {"image": image, "caption": record["caption"]}

        raise RuntimeError(f"All images at index {index} are invalid or unreadable.")


def _validate_messages(messages, *, sample_id: str) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Sample '{sample_id}' must contain a non-empty 'messages' list.")

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Sample '{sample_id}' message #{message_index} must be a mapping.")

        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Sample '{sample_id}' message #{message_index} has unsupported role '{role}'.")
        if not content:
            raise ValueError(f"Sample '{sample_id}' message #{message_index} has empty content.")

    if messages[-1]["role"] != "assistant":
        raise ValueError(f"Sample '{sample_id}' must end with an assistant message.")


class ImageInstructionDataset(Dataset):
    def __init__(self, jsonl_path: str | Path | list[str | Path]):
        paths = [jsonl_path] if isinstance(jsonl_path, (str, Path)) else jsonl_path
        self.jsonl_paths = [Path(p).expanduser().resolve() for p in paths]
        self.records = []
        self.source_indices: list[int] = []
        self.bad_indices = set()

        for source_idx, p in enumerate(self.jsonl_paths):
            with p.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    record = json.loads(line)
                    sample_id = str(record.get("id", f"line_{line_number}"))
                    image_path = str(record.get("image", "")).strip()
                    messages = record.get("messages")

                    if not image_path:
                        raise ValueError(f"Sample '{sample_id}' on line {line_number} is missing 'image'.")
                    _validate_messages(messages, sample_id=sample_id)
                    image_path = resolve_record_image_path(image_path, jsonl_path=p)

                    self.records.append(
                        {
                            "id": sample_id,
                            "image": image_path,
                            "messages": messages,
                            "sample_type": str(record.get("sample_type", "unknown")),
                            "image_id": str(record.get("image_id", "")),
                            "source_dataset": str(record.get("source_dataset", "")),
                        }
                    )
                    self.source_indices.append(source_idx)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        n = len(self.records)
        for offset in range(n):
            current_index = (index + offset) % n
            if current_index in self.bad_indices:
                continue

            record = self.records[current_index]
            image_path = record["image"]

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
            except Exception as error:
                warnings.warn(f"Skipping corrupt instruction image at {image_path}: {error}")
                self.bad_indices.add(current_index)
                continue

            return {
                "id": record["id"],
                "image": image,
                "image_path": image_path,
                "messages": record["messages"],
                "sample_type": record["sample_type"],
                "image_id": record["image_id"],
                "source_dataset": record["source_dataset"],
            }

        paths = ", ".join(str(path) for path in self.jsonl_paths)
        raise RuntimeError(f"All images in {paths} are invalid or unreadable.")
