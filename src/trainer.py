from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import Sampler, WeightedRandomSampler


class EpochShuffleSampler(Sampler[int]):
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
    """Equal-contribution sampler: each source jsonl contributes 1/n_sources of samples."""
    n_sources = len(dataset.jsonl_paths)
    counts = [0] * n_sources
    for src_idx in dataset.source_indices:
        counts[src_idx] += 1

    source_weight = [1.0 / c if c > 0 else 0.0 for c in counts]
    weights = [source_weight[src_idx] for src_idx in dataset.source_indices]

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def log_message(message: str, accelerator: Accelerator, log_path: Path) -> None:
    accelerator.print(message)
    if accelerator.is_main_process:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_loss(model, eval_loader, accelerator) -> float:
    """Compute token-weighted eval loss over the full eval split."""
    total_loss, total_tokens = 0.0, 0.0
    for batch in eval_loader:
        with torch.no_grad(), accelerator.autocast():
            outputs = model(**batch)
        sup_tokens = (batch["labels"] != -100).sum()
        stats = torch.stack(
            [outputs.loss.detach().float() * sup_tokens.float(), sup_tokens.float()]
        )
        gathered = accelerator.gather_for_metrics(stats.unsqueeze(0))
        total_loss += gathered[:, 0].sum().item()
        total_tokens += gathered[:, 1].sum().item()
    return total_loss / total_tokens if total_tokens > 0 else float("nan")
