"""Cross-cutting runtime utilities used by scripts and training entrypoints."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
import yaml
from loguru import logger
from torch.utils.data import Sampler, WeightedRandomSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(section_name: str) -> dict:
    """Load a named config section from `config.yaml`."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if section_name not in config:
        raise KeyError(f"Missing '{section_name}' section in {CONFIG_PATH}")

    section = config[section_name]
    if not isinstance(section, dict):
        raise TypeError(f"Config section '{section_name}' must be a mapping.")
    return section


def resolve_config_path(value: str | Path) -> str:
    """Resolve a single path-like config value to an absolute string path."""

    return str(Path(value).expanduser().resolve())


def resolve_config_paths(value: str | Path | list[str | Path]) -> str | list[str]:
    """Resolve one or many path-like config values to absolute string paths."""

    if isinstance(value, list):
        return [resolve_config_path(item) for item in value]
    return resolve_config_path(value)


def resolve_record_image_path(image_value: str | Path, *, jsonl_path: Path) -> str:
    """Resolve an image path stored in a JSONL record."""

    raw_path = Path(str(image_value).strip()).expanduser()
    if raw_path.is_absolute():
        return str(raw_path)

    candidates = (PROJECT_ROOT / raw_path, Path.cwd() / raw_path, jsonl_path.parent / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str((PROJECT_ROOT / raw_path).resolve())


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs for reproducible runs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: str | Path, accelerator):
    """Configure Loguru sinks for the main process only."""

    logger.remove()
    if not accelerator.is_main_process:
        return logger

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.add(_tqdm_sink, level="INFO", 
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <6}</level> | {message}",
        colorize=True)
    logger.add(
        output_path / "train.log", level="INFO",
        encoding="utf-8", rotation="50 MB",
        retention=5, enqueue=True,
    )
    return logger


def _tqdm_sink(message) -> None:
    tqdm.write(str(message), end="")


class EpochShuffleSampler(Sampler[int]):
    """Shuffle indices deterministically per epoch."""

    def __init__(self, dataset, seed: int):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


def build_weighted_sampler(dataset, seed: int) -> WeightedRandomSampler:
    """Equal-contribution sampler: each source jsonl contributes 1/n of samples."""

    n_sources = len(dataset.jsonl_paths)
    counts = [0] * n_sources
    for src_idx in dataset.source_indices:
        counts[src_idx] += 1

    source_weight = [1.0 / c if c > 0 else 0.0 for c in counts]
    weights = [source_weight[src_idx] for src_idx in dataset.source_indices]

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=weights, num_samples=len(dataset), replacement=True, generator=generator
    )


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record to a JSONL file."""

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def current_lr(scheduler, optimizer) -> float:
    """Return current learning rate from scheduler, falling back to optimizer."""
    try:
        return float(scheduler.get_last_lr()[0])
    except Exception:
        return float(optimizer.param_groups[0]["lr"])

