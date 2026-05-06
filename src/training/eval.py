from __future__ import annotations

from collections.abc import Callable

import torch


SampleLogger = Callable[[object, object], list[str]]
ModeCallback = Callable[[], None]


def evaluate_loss(model, eval_loader, accelerator) -> float:
    """Compute token-weighted eval loss over the full eval split."""

    total_loss, total_tokens = 0.0, 0.0
    for batch in eval_loader:
        with torch.no_grad(), accelerator.autocast():
            outputs = model(**batch)
        sup_tokens = (batch["labels"] != -100).sum()
        stats = torch.stack([outputs.loss.detach().float() * sup_tokens.float(), sup_tokens.float()])
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
    """Evaluate token-weighted loss, optionally log generations, then restore modes."""

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
