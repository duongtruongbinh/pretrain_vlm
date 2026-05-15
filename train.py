from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.collators import PROMPT_TEMPLATE, CaptionCollator
from src.inference import eos_token_ids
from src.data import ImageCaptionDataset
from src.modeling import build_model, freeze_components
from src.runtime import (
    append_jsonl,
    build_weighted_sampler,
    current_lr,
    load_config,
    set_seed,
    setup_logger,
)
from src.training import (
    TrainingState,
    compute_steps_per_epoch,
    load_projector_checkpoint,
    rotate_checkpoints,
    run_evaluation,
    run_training,
    save_training_checkpoint,
    update_checkpoint_pointer,
)


def _parse_args() -> argparse.Namespace:
    # runtime-only args that should not live in config.yaml
    parser = argparse.ArgumentParser(description="Caption projector pretraining.")
    parser.add_argument("--config-section", type=str, default="train")
    parser.add_argument("--resume-from", type=str, default=None)
    return parser.parse_args()


def _select_eval_samples(records: list[dict], sample_count: int = 5) -> list[dict]:
    if not records or sample_count <= 0:
        return []
    step = max(1, len(records) // sample_count)
    return [records[i * step] for i in range(min(sample_count, len(records)))]


def _log_eval_samples(model, collator, eval_samples, accelerator, max_new_tokens):
    unwrapped = accelerator.unwrap_model(model)
    tokenizer = collator.tokenizer
    eos_ids = eos_token_ids(tokenizer)
    lines = []
    for idx, sample in enumerate(eval_samples, 1):
        try:
            with Image.open(sample["image"]) as img:
                img = img.convert("RGB")
        except Exception as e:
            line = f"[sample {idx}] failed to load {sample['image']}: {e}"
            lines.append(line)
            continue

        inputs = collator.processor(text=PROMPT_TEMPLATE, images=img, return_tensors="pt")
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        with torch.no_grad(), accelerator.autocast():
            generated_ids = unwrapped.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=5,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_text = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True).strip()
        raw_generated_text = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=False).strip()
        lines.extend([
            f"[sample {idx}] prediction: {generated_text or '<empty>'}",
            f"[sample {idx}] prediction_raw: {raw_generated_text or '<empty>'}",
            f"[sample {idx}] reference: {sample['caption']}",
            f"[sample {idx}] image: {sample['image']}",
        ])
    return lines


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config_section)
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    accelerator = Accelerator()
    logger = setup_logger(output_dir, accelerator)
    set_seed(int(cfg["seed"]))

    collator = CaptionCollator(cfg["vision_model"], cfg["llm_model"])
    train_dataset = ImageCaptionDataset(cfg["train_jsonl"])
    eval_dataset = ImageCaptionDataset(cfg["eval_jsonl"])
    eval_samples = _select_eval_samples(eval_dataset.records, sample_count=5)
    train_sampler = build_weighted_sampler(train_dataset, seed=int(cfg["seed"]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        sampler=train_sampler,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        cfg["vision_model"],
        cfg["llm_model"],
        model_dtype=cfg.get("model_dtype"),
        projector_dtype=cfg.get("projector_dtype", "float32"),
        image_token_id=collator.image_token_id,
        vocab_size=len(collator.tokenizer),
    )
    freeze_components(model, freeze_vision=True, train_projector=True, train_llm=False)

    if bool(cfg.get("gradient_checkpointing", False)):
        model.language_model.config.use_cache = False
        model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    optimizer = AdamW(model.multi_modal_projector.parameters(), lr=float(cfg["lr"]), weight_decay=0.0)
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )

    grad_accum = int(cfg["grad_accum"])
    steps_per_epoch = compute_steps_per_epoch(len(train_loader), grad_accum)
    total_steps = steps_per_epoch * int(cfg["epochs"])
    warmup_steps = int(total_steps * float(cfg["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    state = TrainingState()

    if args.resume_from:
        resume_state = load_projector_checkpoint(
            args.resume_from, accelerator.unwrap_model(model), optimizer, scheduler
        )
        state.global_step = int(resume_state.get("global_step", 0))
        state.best_eval_loss = resume_state.get("best_eval_loss")
        logger.info("Resumed from {} at step {}.", args.resume_from, state.global_step)

    logger.info("Starting training: {} train, {} eval.", len(train_dataset), len(eval_dataset))
    logger.info(
        "Training config: "
        "processes={}, per_device_batch={}, grad_accum={}, effective_batch={}, "
        "steps_per_epoch={}, total_steps={}, warmup_steps={}, output_dir={}",
        accelerator.num_processes,
        int(cfg["batch_size"]),
        grad_accum,
        int(cfg["batch_size"]) * grad_accum * accelerator.num_processes,
        steps_per_epoch,
        total_steps,
        warmup_steps,
        output_dir,
    )
    progress_total = max(total_steps - state.global_step, 0)
    progress = (
        tqdm(total=progress_total, initial=0, desc="stage1", dynamic_ncols=True)
        if accelerator.is_main_process
        else None
    )

    def set_train_mode() -> None:
        accelerator.unwrap_model(model).multi_modal_projector.train()

    def trainable_parameters():
        return accelerator.unwrap_model(model).multi_modal_projector.parameters()

    def save_checkpoint(global_step: int, *, eval_loss: float | None = None) -> Path:
        ckpt_path = output_dir / f"checkpoint-{global_step}"
        is_best = eval_loss is not None and (state.best_eval_loss is None or eval_loss < state.best_eval_loss)
        if is_best:
            state.best_eval_loss = eval_loss
        save_training_checkpoint(
            path=ckpt_path,
            model=accelerator.unwrap_model(model),
            processor=collator.processor,
            tokenizer=collator.tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            training_config=cfg,
            trainer_state={
                "global_step": global_step,
                "epoch": state.epoch,
                "best_eval_loss": state.best_eval_loss,
                "save_best_by": cfg.get("save_best_by", "eval_loss"),
            },
            stage="caption_pretrain",
            save_language_model=False,
        )
        update_checkpoint_pointer(output_dir, "last", ckpt_path, step=global_step)
        if is_best:
            update_checkpoint_pointer(
                output_dir,
                "best",
                ckpt_path,
                step=global_step,
                metric_name="eval_loss",
                metric_value=eval_loss,
            )
        rotate_checkpoints(output_dir, int(cfg["keep_last_n"]))
        return ckpt_path

    def on_step_end(result, _training_state) -> None:
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                train_loss=f"{result.train_loss:.6f}",
                lr=f"{current_lr(scheduler, optimizer):.3e}",
                supervised_tokens=result.supervised_tokens,
            )

        if result.global_step % int(cfg["log_steps"]) == 0:
            logger.info(
                "step {}: train_loss={:.6f}, lr={:.6e}, supervised_tokens={}",
                result.global_step,
                result.train_loss,
                current_lr(scheduler, optimizer),
                result.supervised_tokens,
            )
            if accelerator.is_main_process:
                append_jsonl(metrics_path, {"step": result.global_step, "train_loss": result.train_loss})

        eval_loss = None
        if result.global_step % int(cfg["eval_steps"]) == 0:
            eval_loss = run_evaluation(
                model=model,
                eval_loader=eval_loader,
                accelerator=accelerator,
                global_step=result.global_step,
                logger=logger,
                sample_logger=lambda m, a: _log_eval_samples(m, collator, eval_samples, a, 64),
                restore_train_mode=set_train_mode,
            )
            if accelerator.is_main_process:
                append_jsonl(metrics_path, {"step": result.global_step, "eval_loss": eval_loss})

        if result.global_step % int(cfg["save_steps"]) == 0 and accelerator.is_main_process:
            ckpt_path = save_checkpoint(result.global_step, eval_loss=eval_loss)
            logger.info("Saved checkpoint to {}.", ckpt_path)

    run_training(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        epochs=int(cfg["epochs"]),
        grad_accum=grad_accum,
        state=state,
        set_train_mode=set_train_mode,
        trainable_parameters=trainable_parameters,
        on_step_end=on_step_end,
    )

    if accelerator.is_main_process:
        ckpt_path = save_checkpoint(state.global_step)
        logger.info("Saved final checkpoint to {}.", ckpt_path)
        if progress is not None:
            progress.close()

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
