from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch


def _remap_projector_state(state_dict: dict) -> dict:
    """
    Map old nn.Sequential key names → HF LlavaMultiModalProjector key names.

    Old (nn.Sequential): 0.weight, 0.bias, 2.weight, 2.bias
    New (HF):           linear_1.weight, linear_1.bias, linear_2.weight, linear_2.bias

    Keys that do not match the old pattern are passed through unchanged so
    this function is safe to call on checkpoints from either format.
    """
    mapping = {
        "0.weight": "linear_1.weight",
        "0.bias": "linear_1.bias",
        "2.weight": "linear_2.weight",
        "2.bias": "linear_2.bias",
    }
    return {mapping.get(k, k): v for k, v in state_dict.items()}


def save_projector_ckpt(model, optimizer, scheduler, step: int, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "projector_state_dict": model.multi_modal_projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        p,
    )


def load_projector_ckpt(path: str | Path, model, optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(str(path), map_location="cpu")
    state = _remap_projector_state(ckpt["projector_state_dict"])
    model.multi_modal_projector.load_state_dict(state)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt["step"])


def save_full_ckpt(model, tokenizer, optimizer, scheduler, step: int, path: str | Path) -> None:
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"projector_state_dict": model.multi_modal_projector.state_dict()}, d / "projector.pt")
    torch.save(optimizer.state_dict(), d / "optimizer.pt")
    torch.save(scheduler.state_dict(), d / "scheduler.pt")
    model.language_model.save_pretrained(d / "llm", safe_serialization=True)
    tokenizer.save_pretrained(d / "tokenizer")
    with (d / "trainer_state.json").open("w", encoding="utf-8") as fh:
        json.dump({"step": int(step)}, fh, ensure_ascii=False, indent=2)


def load_full_ckpt(path: str | Path, model, optimizer=None, scheduler=None) -> int:
    d = Path(path)
    raw = torch.load(d / "projector.pt", map_location="cpu")
    raw_state = raw["projector_state_dict"] if "projector_state_dict" in raw else raw
    state = _remap_projector_state(raw_state)
    model.multi_modal_projector.load_state_dict(state)
    if optimizer is not None and (d / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(d / "optimizer.pt", map_location="cpu"))
    if scheduler is not None and (d / "scheduler.pt").exists():
        scheduler.load_state_dict(torch.load(d / "scheduler.pt", map_location="cpu"))
    trainer_state_path = d / "trainer_state.json"
    if trainer_state_path.exists():
        with trainer_state_path.open("r", encoding="utf-8") as fh:
            return int(json.load(fh).get("step", 0))
    return 0


def rotate_checkpoints(output_dir: str | Path, keep_last_n: int) -> None:
    d = Path(output_dir)
    checkpoints = []
    for p in d.glob("checkpoint-*"):
        step_str = p.name.replace("checkpoint-", "").rstrip(".pt")
        if step_str.isdigit():
            checkpoints.append((int(step_str), p))
    checkpoints.sort(key=lambda x: x[0])
    for _, p in (checkpoints[:-keep_last_n] if keep_last_n > 0 else checkpoints):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
