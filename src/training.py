from __future__ import annotations

import json
import platform
import random
import shutil
import sys
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Training-loop types
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    global_step: int = 0
    epoch: int = 0
    best_eval_loss: float | None = None


@dataclass(frozen=True)
class StepResult:
    global_step: int
    epoch: int
    train_loss: float
    supervised_tokens: int


ModeCallback = Callable[[], None]
EpochCallback = Callable[[int], None]
ParamsCallback = Callable[[], Iterable[torch.nn.Parameter]]
StepCallback = Callable[[StepResult, TrainingState], None]
SampleLogger = Callable[[object, object], list[str]]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def compute_steps_per_epoch(loader_len: int, grad_accum: int) -> int:
    if grad_accum <= 0:
        raise ValueError("grad_accum must be a positive integer.")
    return (loader_len + grad_accum - 1) // grad_accum


def run_training(
    *,
    model,
    train_loader,
    optimizer,
    scheduler,
    accelerator,
    epochs: int,
    grad_accum: int,
    state: TrainingState,
    set_train_mode: ModeCallback,
    trainable_parameters: ParamsCallback,
    on_step_end: StepCallback | None = None,
    on_epoch_start: EpochCallback | None = None,
    max_grad_norm: float = 1.0,
) -> TrainingState:
    steps_per_epoch = compute_steps_per_epoch(len(train_loader), grad_accum)
    starting_epoch = state.global_step // steps_per_epoch if steps_per_epoch else 0
    batches_to_skip = (state.global_step % steps_per_epoch) * grad_accum if steps_per_epoch else 0

    for epoch in range(starting_epoch, int(epochs)):
        state.epoch = epoch
        if on_epoch_start is not None:
            on_epoch_start(epoch)
        set_train_mode()

        iterator = iter(train_loader)
        for _ in range(batches_to_skip):
            next(iterator, None)
        batches_to_skip = 0

        while True:
            window = _next_window(iterator, grad_accum)
            if not window:
                break
            step_result = _train_window(
                model=model,
                window=window,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=accelerator,
                trainable_parameters=trainable_parameters,
                max_grad_norm=max_grad_norm,
            )
            if step_result is None:
                continue
            state.global_step += 1
            result = StepResult(
                global_step=state.global_step,
                epoch=epoch,
                train_loss=step_result["train_loss"],
                supervised_tokens=step_result["supervised_tokens"],
            )
            if on_step_end is not None:
                on_step_end(result, state)
            set_train_mode()

    return state


def _next_window(iterator, grad_accum: int) -> list[dict[str, Any]]:
    window = []
    for _ in range(grad_accum):
        try:
            window.append(next(iterator))
        except StopIteration:
            break
    return window


def _train_window(
    *,
    model,
    window: list[dict[str, Any]],
    optimizer,
    scheduler,
    accelerator,
    trainable_parameters: ParamsCallback,
    max_grad_norm: float,
) -> dict[str, float | int] | None:
    local_tokens = torch.stack([_supervised_tokens(batch) for batch in window]).sum()
    global_tokens = accelerator.gather(local_tokens).sum().item()
    if global_tokens <= 0:
        return None

    optimizer.zero_grad(set_to_none=True)
    local_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    local_token_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)

    for batch_idx, batch in enumerate(window):
        sync_context = (
            accelerator.no_sync(model)
            if batch_idx < len(window) - 1 and accelerator.num_processes > 1
            else nullcontext()
        )
        with sync_context:
            with accelerator.autocast():
                outputs = model(**batch)
            batch_tokens = _supervised_tokens(batch)
            if batch_tokens.item() <= 0:
                continue
            batch_loss_sum = outputs.loss.float() * batch_tokens.float()
            scaled_loss = batch_loss_sum * accelerator.num_processes / global_tokens
            accelerator.backward(scaled_loss)
            local_loss_sum = local_loss_sum + batch_loss_sum.detach()
            local_token_sum = local_token_sum + batch_tokens.float()

    accelerator.clip_grad_norm_(list(trainable_parameters()), max_grad_norm)
    optimizer.step()
    scheduler.step()

    stats = torch.stack([local_loss_sum, local_token_sum]).unsqueeze(0)
    gathered = accelerator.gather_for_metrics(stats)
    token_count = gathered[:, 1].sum().item()
    train_loss = gathered[:, 0].sum().item() / token_count
    return {"train_loss": float(train_loss), "supervised_tokens": int(token_count)}


def _supervised_tokens(batch: dict[str, Any]) -> torch.Tensor:
    labels = batch["labels"]
    return (labels != -100).sum()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _remap_projector_state(state_dict: dict) -> dict:
    # Legacy checkpoints use sequential keys (0.weight, 2.weight); HF LLaVA expects linear_1/linear_2.
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
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"projector_state_dict": model.multi_modal_projector.state_dict()},
        checkpoint_dir / "projector.pt",
    )
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    torch.save(_rng_state(int(training_config.get("seed", 0))), checkpoint_dir / "rng_state.pt")

    model.config.save_pretrained(checkpoint_dir / "model_config")
    processor.save_pretrained(checkpoint_dir / "processor")
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

    if save_language_model:
        torch.save(model.lm_head.state_dict(), checkpoint_dir / "lm_head.pt")
        model.language_model.save_pretrained(checkpoint_dir / "llm", safe_serialization=True)

    full_state = {"stage": stage, **trainer_state}
    _write_json(checkpoint_dir / "trainer_state.json", full_state)
    _write_json(checkpoint_dir / "package_versions.json", _package_versions())
    with (checkpoint_dir / "training_config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(training_config, fh, sort_keys=True, allow_unicode=True)


def load_projector_checkpoint(
    path: str | Path, model, optimizer=None, scheduler=None, restore_rng: bool = False
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return _load_legacy_projector_ckpt(checkpoint_path, model, optimizer, scheduler)

    raw = torch.load(checkpoint_path / "projector.pt", map_location="cpu", weights_only=True)
    raw_state = raw["projector_state_dict"] if "projector_state_dict" in raw else raw
    model.multi_modal_projector.load_state_dict(_remap_projector_state(raw_state), strict=True)
    _maybe_load_optimizer_scheduler(checkpoint_path, optimizer, scheduler)
    if restore_rng:
        _restore_rng_state(checkpoint_path / "rng_state.pt")
    return _load_trainer_state(checkpoint_path)


def load_full_checkpoint(
    path: str | Path, model, optimizer=None, scheduler=None, restore_rng: bool = False
) -> dict[str, Any]:
    checkpoint_dir = Path(path)
    state = load_projector_checkpoint(checkpoint_dir, model, optimizer, scheduler, restore_rng=restore_rng)
    lm_head_path = checkpoint_dir / "lm_head.pt"
    if lm_head_path.exists():
        model.lm_head.load_state_dict(torch.load(lm_head_path, map_location="cpu", weights_only=True))
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
    payload = {"checkpoint": str(Path(checkpoint_path).expanduser().resolve()), "step": int(step)}
    if metric_name is not None:
        payload["metric_name"] = metric_name
        payload["metric_value"] = metric_value
    _write_json(Path(output_dir) / f"{name}_checkpoint.json", payload)


def rotate_checkpoints(
    output_dir: str | Path, keep_last_n: int, protected_paths: set[str | Path] | None = None
) -> None:
    output_path = Path(output_dir)
    protected = {Path(p).expanduser().resolve() for p in protected_paths or set()}
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
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    state = _remap_projector_state(ckpt["projector_state_dict"])
    model.multi_modal_projector.load_state_dict(state, strict=True)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        _safe_load_optimizer_state(optimizer, ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return {"global_step": int(ckpt.get("step", 0)), "epoch": 0, "best_eval_loss": None}


def _load_trainer_state(checkpoint_dir: Path) -> dict[str, Any]:
    try:
        with (checkpoint_dir / "trainer_state.json").open("r", encoding="utf-8") as fh:
            state = json.load(fh)
    except FileNotFoundError:
        return {"global_step": 0, "epoch": 0, "best_eval_loss": None}
    if "global_step" not in state and "step" in state:
        state["global_step"] = state["step"]
    state.setdefault("global_step", 0)
    state.setdefault("epoch", 0)
    state.setdefault("best_eval_loss", None)
    return state


def _maybe_load_optimizer_scheduler(checkpoint_dir: Path, optimizer=None, scheduler=None) -> None:
    if optimizer is not None and (checkpoint_dir / "optimizer.pt").exists():
        _safe_load_optimizer_state(
            optimizer,
            torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True),
        )
    if scheduler is not None and (checkpoint_dir / "scheduler.pt").exists():
        scheduler.load_state_dict(
            torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu", weights_only=True)
        )


def _safe_load_optimizer_state(optimizer, state_dict) -> None:
    try:
        optimizer.load_state_dict(state_dict)
    except ValueError as error:
        logger.warning("skipped optimizer state because the projector parameter set changed: {}", error)


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
    state = torch.load(path, map_location="cpu", weights_only=False)
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_loss(model, eval_loader, accelerator) -> float:
    total_loss, total_tokens = 0.0, 0.0
    for batch in eval_loader:
        with torch.no_grad(), accelerator.autocast():
            outputs = model(**batch)
        sup_tokens = _supervised_tokens(batch)
        stats = torch.stack(
            [outputs.loss.detach().float() * sup_tokens.float(), sup_tokens.float()]
        )
        gathered = accelerator.gather_for_metrics(stats.unsqueeze(0))
        total_loss += gathered[:, 0].sum().item()
        total_tokens += gathered[:, 1].sum().item()
    return total_loss / total_tokens if total_tokens > 0 else float("nan")


def run_evaluation(
    *,
    model,
    eval_loader,
    accelerator,
    global_step: int,
    logger,
    sample_logger: SampleLogger | None = None,
    restore_train_mode: ModeCallback | None = None,
) -> float:
    accelerator.unwrap_model(model).eval()
    eval_loss = evaluate_loss(model, eval_loader, accelerator)
    logger.info("step {}: eval_loss={:.6f}", global_step, eval_loss)

    if sample_logger is not None and accelerator.is_main_process:
        lines = sample_logger(model, accelerator)
        for line in lines:
            logger.info(line)

    if restore_train_mode is not None:
        restore_train_mode()
    return eval_loss
