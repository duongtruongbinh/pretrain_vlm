from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TrainingState:
    """Mutable progress state shared by training, eval, and checkpoint hooks."""

    global_step: int = 0
    epoch: int = 0
    best_eval_loss: float | None = None


@dataclass(frozen=True)
class StepResult:
    """Token-weighted metrics emitted after one optimizer update."""

    global_step: int
    epoch: int
    train_loss: float
    supervised_tokens: int


StepCallback = Callable[[StepResult, TrainingState], None]
ModeCallback = Callable[[], None]
EpochCallback = Callable[[int], None]
ParamsCallback = Callable[[], Iterable[torch.nn.Parameter]]


def compute_steps_per_epoch(loader_len: int, grad_accum: int) -> int:
    """Return optimizer steps per epoch for a dataloader length."""

    if grad_accum <= 0:
        raise ValueError("grad_accum must be a positive integer.")
    return (loader_len + grad_accum - 1) // grad_accum


def run_training(
    *, model, train_loader, optimizer, scheduler, accelerator, epochs: int, grad_accum: int, state: TrainingState,
    set_train_mode: ModeCallback, trainable_parameters: ParamsCallback, on_step_end: StepCallback | None = None, on_epoch_start: EpochCallback | None = None,
    max_grad_norm: float = 1.0,
) -> TrainingState:
    """Run a token-weighted training loop with explicit accumulation windows."""

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
                model=model, window=window, optimizer=optimizer, scheduler=scheduler, accelerator=accelerator,
                grad_accum=grad_accum, trainable_parameters=trainable_parameters, max_grad_norm=max_grad_norm,
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
    *, model, window: list[dict[str, Any]], optimizer, scheduler, accelerator,
    grad_accum: int, trainable_parameters: ParamsCallback, max_grad_norm: float,
) -> dict[str, float | int] | None:
    local_tokens = sum(_supervised_tokens(batch) for batch in window)
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

            batch_tokens = _supervised_tokens(batch).to(outputs.loss.device)
            if batch_tokens.item() <= 0:
                continue

            batch_loss_sum = outputs.loss.float() * batch_tokens.float()
            scaled_loss = batch_loss_sum * accelerator.num_processes / global_tokens
            accelerator.backward(scaled_loss)
            local_loss_sum = local_loss_sum + batch_loss_sum.detach().to(local_loss_sum.device)
            local_token_sum = local_token_sum + batch_tokens.detach().float().to(local_token_sum.device)

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
