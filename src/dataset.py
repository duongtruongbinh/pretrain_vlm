from __future__ import annotations

import json
import warnings
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.paths import resolve_record_image_path


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
                            f"Each JSONL line must contain 'image' and 'caption'. "
                            f"Bad line {line_number} in {p}."
                        )
                    record["image"] = resolve_record_image_path(
                        record["image"], jsonl_path=p
                    )
                    self.records.append(record)
                    self.source_indices.append(source_idx)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        num_records = len(self.records)
        if num_records == 0:
            loaded_paths = ", ".join(str(path) for path in self.jsonl_paths)
            raise RuntimeError(f"No records were loaded from: {loaded_paths}")

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
                warnings.warn(f"Skipping corrupt image at {image_path}: {error}")
                self.bad_indices.add(current_index)
                continue

            return {"image": image, "caption": record["caption"]}

        raise RuntimeError(f"All images at index {index} are invalid or unreadable.")
