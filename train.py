from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.collator import PROMPT_TEMPLATE, ImageCaptionCollator
from src.config import load_config
from src.dataset import ImageCaptionDataset
from src.model import build_model, freeze_components
from src.paths import resolve_config_paths
from src.trainer import append_jsonl, build_weighted_sampler, log_message
from src.training.checkpoint import (
    load_projector_checkpoint,
    rotate_checkpoints,
    save_training_checkpoint,
    update_checkpoint_pointer,
)
from src.training.engine import TrainingState, compute_steps_per_epoch, run_training
from src.training.eval import run_evaluation
from src.utils import set_seed


def _parse_args() -> argparse.Namespace:
    """Parse runtime-only arguments that should not live in config.yaml."""

    parser = argparse.ArgumentParser(description="Caption projector pretraining.")
    parser.add_argument("--resume-from", type=str, default=None)
    return parser.parse_args()


def _select_eval_samples(records: list[dict], sample_count: int = 5) -> list[dict]:
    """Select a small deterministic set of unique-image eval samples."""

    if sample_count <= 0 or not records:
        return []
    if len(records) <= sample_count:
        return records

    anchors = [round(i * (len(records) - 1) / (sample_count - 1)) for i in range(sample_count)]
    selected, seen_images = [], set()
    for anchor in anchors:
        for offset in range(len(records)):
            record = records[(anchor + offset) % len(records)]
            image = record.get("image")
            if image not in seen_images:
                selected.append(record)
                seen_images.add(image)
                break

    selected_ids = {id(record) for record in selected}
    for record in records:
        if len(selected) >= sample_count:
            break
        if id(record) not in selected_ids:
            selected.append(record)
    return selected[:sample_count]


def _log_eval_samples(model, collator, eval_samples, accelerator, max_new_tokens):
    unwrapped = accelerator.unwrap_model(model)
    tokenizer = collator.processor.tokenizer
    lines = []
    for idx, sample in enumerate(eval_samples, 1):
        try:
            with Image.open(sample["image"]) as img:
                img = img.convert("RGB")
        except Exception as e:
            line = f"[sample {idx}] failed to load {sample['image']}: {e}"
            accelerator.print(line)
            lines.append(line)
            continue

        inputs = collator.processor(text=PROMPT_TEMPLATE, images=img, return_tensors="pt")
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        eos_ids = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end_of_text|>")})
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
        for line in [
            f"[sample {idx}] prediction: {generated_text or '<empty>'}",
            f"[sample {idx}] prediction_raw: {raw_generated_text or '<empty>'}",
            f"[sample {idx}] reference: {sample['caption']}",
            f"[sample {idx}] image: {sample['image']}",
        ]:
            accelerator.print(line)
            lines.append(line)
    return lines


def main() -> None:
    args = _parse_args()
    cfg = load_config("train")
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    metrics_path = output_dir / "metrics.jsonl"

    accelerator = Accelerator(gradient_accumulation_steps=int(cfg["grad_accum"]))
    set_seed(int(cfg["seed"]))

    collator = ImageCaptionCollator(cfg["vision_model"], cfg["llm_model"])
    train_dataset = ImageCaptionDataset(resolve_config_paths(cfg["train_jsonl"]))
    eval_dataset = ImageCaptionDataset(resolve_config_paths(cfg["eval_jsonl"]))
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
        log_message(f"Resumed from {args.resume_from} at step {state.global_step}.", accelerator, log_path)

    log_message(
        f"Starting training: {len(train_dataset)} train, {len(eval_dataset)} eval.", accelerator, log_path
    )
    log_message(
        "Training config: "
        f"processes={accelerator.num_processes}, "
        f"per_device_batch={int(cfg['batch_size'])}, grad_accum={grad_accum}, "
        f"effective_batch={int(cfg['batch_size']) * grad_accum * accelerator.num_processes}, "
        f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}, output_dir={output_dir}",
        accelerator,
        log_path,
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

    def on_step_end(result, training_state) -> None:
        if result.global_step % int(cfg["log_steps"]) == 0:
            log_message(
                f"step {result.global_step}: train_loss={result.train_loss:.6f}", accelerator, log_path
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
                log_path=log_path,
                sample_logger=lambda m, a: _log_eval_samples(m, collator, eval_samples, a, 64),
                restore_train_mode=set_train_mode,
            )
            if accelerator.is_main_process:
                append_jsonl(metrics_path, {"step": result.global_step, "eval_loss": eval_loss})

        if result.global_step % int(cfg["save_steps"]) == 0 and accelerator.is_main_process:
            ckpt_path = save_checkpoint(result.global_step, eval_loss=eval_loss)
            log_message(f"Saved checkpoint to {ckpt_path}.", accelerator, log_path)

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
        log_message(f"Saved final checkpoint to {ckpt_path}.", accelerator, log_path)

    log_message("Training finished.", accelerator, log_path)


if __name__ == "__main__":
    main()
