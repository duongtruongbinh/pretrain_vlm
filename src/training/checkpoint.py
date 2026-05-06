from __future__ import annotations

import json
import platform
import random
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import yaml


def _remap_projector_state(state_dict: dict) -> dict:
    """Map legacy sequential projector keys to HF LLaVA projector keys."""

    mapping = {
        "0.weight": "linear_1.weight",
        "0.bias": "linear_1.bias",
        "2.weight": "linear_2.weight",
        "2.bias": "linear_2.bias",
    }
    return {mapping.get(k, k): v for k, v in state_dict.items()}


def save_training_checkpoint(
    *,
    path: str | Path,
    model,
    processor,
    tokenizer,
    optimizer,
    scheduler,
    training_config: dict,
    trainer_state: dict,
    stage: str,
    save_language_model: bool,
) -> None:
    """Save a reproducible directory checkpoint for a training stage."""

    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"projector_state_dict": model.multi_modal_projector.state_dict()}, checkpoint_dir / "projector.pt"
    )
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    torch.save(_rng_state(int(training_config.get("seed", 0))), checkpoint_dir / "rng_state.pt")

    if hasattr(model, "config"):
        model.config.save_pretrained(checkpoint_dir / "model_config")
    if processor is not None:
        processor.save_pretrained(checkpoint_dir / "processor")
    if tokenizer is not None:
        tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

    if hasattr(model, "lm_head"):
        torch.save(model.lm_head.state_dict(), checkpoint_dir / "lm_head.pt")
    if save_language_model:
        model.language_model.save_pretrained(checkpoint_dir / "llm", safe_serialization=True)

    full_state = {"stage": stage, **trainer_state}
    _write_json(checkpoint_dir / "trainer_state.json", full_state)
    _write_json(checkpoint_dir / "package_versions.json", _package_versions())
    with (checkpoint_dir / "training_config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(training_config, fh, sort_keys=True, allow_unicode=True)


def load_projector_checkpoint(
    path: str | Path, model, optimizer=None, scheduler=None, restore_rng: bool = False
) -> dict[str, Any]:
    """Load projector checkpoints from new directories or legacy `.pt` files."""

    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return _load_legacy_projector_ckpt(checkpoint_path, model, optimizer, scheduler)

    raw = torch.load(checkpoint_path / "projector.pt", map_location="cpu")
    raw_state = raw["projector_state_dict"] if "projector_state_dict" in raw else raw
    model.multi_modal_projector.load_state_dict(_remap_projector_state(raw_state))
    _maybe_load_optimizer_scheduler(checkpoint_path, optimizer, scheduler)
    if restore_rng:
        _restore_rng_state(checkpoint_path / "rng_state.pt")
    return _load_trainer_state(checkpoint_path)


def load_full_checkpoint(
    path: str | Path, model, optimizer=None, scheduler=None, restore_rng: bool = False
) -> dict[str, Any]:
    """Load trainable full-stage state from a directory checkpoint."""

    checkpoint_dir = Path(path)
    state = load_projector_checkpoint(checkpoint_dir, model, optimizer, scheduler, restore_rng=restore_rng)
    lm_head_path = checkpoint_dir / "lm_head.pt"
    if lm_head_path.exists() and hasattr(model, "lm_head"):
        model.lm_head.load_state_dict(torch.load(lm_head_path, map_location="cpu"))
    return state


def update_checkpoint_pointer(
    output_dir: str | Path,
    name: str,
    checkpoint_path: str | Path,
    *,
    step: int,
    metric_name: str | None = None,
    metric_value: float | None = None,
) -> None:
    """Write a small JSON pointer for best or last checkpoint."""

    payload = {"checkpoint": str(Path(checkpoint_path).resolve()), "step": int(step)}
    if metric_name is not None:
        payload["metric_name"] = metric_name
        payload["metric_value"] = metric_value
    _write_json(Path(output_dir) / f"{name}_checkpoint.json", payload)


def rotate_checkpoints(
    output_dir: str | Path, keep_last_n: int, protected_paths: set[str | Path] | None = None
) -> None:
    """Delete old checkpoints while preserving protected best and last paths."""

    output_path = Path(output_dir)
    protected = {Path(p).resolve() for p in protected_paths or set()}
    for pointer in ("best_checkpoint.json", "last_checkpoint.json"):
        pointer_path = output_path / pointer
        if pointer_path.exists():
            with pointer_path.open("r", encoding="utf-8") as fh:
                protected.add(Path(json.load(fh)["checkpoint"]).resolve())

    checkpoints = []
    for path in output_path.glob("checkpoint-*"):
        step = _checkpoint_step(path)
        if step is not None:
            checkpoints.append((step, path))

    checkpoints.sort(key=lambda item: item[0])
    candidates = [(step, path) for step, path in checkpoints if path.resolve() not in protected]
    to_delete = candidates[:-keep_last_n] if keep_last_n > 0 else candidates
    for _, path in to_delete:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _load_legacy_projector_ckpt(path: Path, model, optimizer=None, scheduler=None) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location="cpu")
    state = _remap_projector_state(ckpt["projector_state_dict"])
    model.multi_modal_projector.load_state_dict(state)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return {"global_step": int(ckpt.get("step", 0)), "epoch": 0, "best_eval_loss": None}


def _load_trainer_state(checkpoint_dir: Path) -> dict[str, Any]:
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        return {"global_step": 0, "epoch": 0, "best_eval_loss": None}
    with trainer_state_path.open("r", encoding="utf-8") as fh:
        state = json.load(fh)
    if "global_step" not in state and "step" in state:
        state["global_step"] = state["step"]
    state.setdefault("global_step", 0)
    state.setdefault("epoch", 0)
    state.setdefault("best_eval_loss", None)
    return state


def _maybe_load_optimizer_scheduler(checkpoint_dir: Path, optimizer=None, scheduler=None) -> None:
    if optimizer is not None and (checkpoint_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu"))
    if scheduler is not None and (checkpoint_dir / "scheduler.pt").exists():
        scheduler.load_state_dict(torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu"))


def _checkpoint_step(path: Path) -> int | None:
    step_text = path.name.replace("checkpoint-", "", 1).removesuffix(".pt")
    return int(step_text) if step_text.isdigit() else None


def _rng_state(seed: int) -> dict[str, Any]:
    state = {"seed": seed, "python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(path: Path) -> None:
    if not path.exists():
        return
    state = torch.load(path, map_location="cpu")
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _package_versions() -> dict[str, Any]:
    packages = {}
    for name in ("accelerate", "torch", "transformers", "pyyaml", "pillow"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "packages": packages,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
