from __future__ import annotations

import json
import warnings
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.paths import resolve_record_image_path


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
    def __init__(self, jsonl_path: str | Path):
        self.jsonl_path = Path(jsonl_path).expanduser().resolve()
        self.records = []
        self.bad_indices = set()

        with self.jsonl_path.open("r", encoding="utf-8") as handle:
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
                image_path = resolve_record_image_path(image_path, jsonl_path=self.jsonl_path)

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

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        num_records = len(self.records)
        if num_records == 0:
            raise RuntimeError(f"No records were loaded from {self.jsonl_path}.")

        for offset in range(num_records):
            current_index = (index + offset) % num_records
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

        raise RuntimeError(f"All images in {self.jsonl_path} are invalid or unreadable.")
