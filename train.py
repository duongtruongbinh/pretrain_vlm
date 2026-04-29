from __future__ import annotations

import math
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.checkpoint import load_projector_ckpt, rotate_checkpoints, save_projector_ckpt
from src.collator import PROMPT_TEMPLATE, ImageCaptionCollator
from src.config import load_config
from src.dataset import ImageCaptionDataset
from src.model import build_model, freeze_components
from src.trainer import EpochShuffleSampler, evaluate_loss, log_message
from src.utils import set_seed


def _log_eval_samples(model, collator, eval_samples, accelerator, max_new_tokens, log_path):
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

        with torch.no_grad():
            generated_ids = unwrapped.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

        prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip() or "<empty>"
        for line in [
            f"[sample {idx}] prediction: {prediction}",
            f"[sample {idx}] reference: {sample['caption']}",
            f"[sample {idx}] image: {sample['image']}",
        ]:
            accelerator.print(line)
            lines.append(line)
    return lines


def evaluate(model, eval_loader, accelerator, eval_samples, collator, max_new_tokens, global_step, log_path):
    accelerator.unwrap_model(model).multi_modal_projector.eval()
    eval_loss = evaluate_loss(model, eval_loader, accelerator)
    log_message(f"step {global_step}: eval_loss={eval_loss:.6f}", accelerator, log_path)

    lines = _log_eval_samples(model, collator, eval_samples, accelerator, max_new_tokens, log_path)
    if accelerator.is_main_process:
        with log_path.open("a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")

    accelerator.unwrap_model(model).multi_modal_projector.train()
    return eval_loss


def main() -> None:
    cfg = load_config("train")
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    accelerator = Accelerator(gradient_accumulation_steps=int(cfg["grad_accum"]))
    set_seed(int(cfg["seed"]))

    collator = ImageCaptionCollator(cfg["vision_model"], cfg["llm_model"])
    train_dataset = ImageCaptionDataset(str(Path(cfg["train_jsonl"]).expanduser().resolve()))
    eval_dataset = ImageCaptionDataset(str(Path(cfg["eval_jsonl"]).expanduser().resolve()))
    eval_samples = eval_dataset.records[: min(5, len(eval_dataset.records))]
    train_sampler = EpochShuffleSampler(train_dataset, seed=int(cfg["seed"]))

    train_loader = DataLoader(
        train_dataset, batch_size=int(cfg["batch_size"]), sampler=train_sampler,
        collate_fn=collator, pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=int(cfg["batch_size"]), shuffle=False,
        collate_fn=collator, pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg["vision_model"], cfg["llm_model"], model_dtype=cfg.get("model_dtype"))
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

    steps_per_epoch = math.ceil(len(train_loader) / int(cfg["grad_accum"]))
    total_steps = steps_per_epoch * int(cfg["epochs"])
    warmup_steps = int(total_steps * float(cfg["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    resume_from = cfg.get("resume_from")
    if resume_from:
        global_step = load_projector_ckpt(
            resume_from, accelerator.unwrap_model(model), optimizer, scheduler
        )
        log_message(f"Resumed from {resume_from} at step {global_step}.", accelerator, log_path)

    log_message(
        f"Starting training: {len(train_dataset)} train, {len(eval_dataset)} eval.",
        accelerator, log_path,
    )

    starting_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    batches_to_skip = (global_step % steps_per_epoch) * int(cfg["grad_accum"]) if steps_per_epoch > 0 else 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(starting_epoch, int(cfg["epochs"])):
        train_sampler.set_epoch(epoch)
        accelerator.unwrap_model(model).multi_modal_projector.train()

        for batch_idx, batch in enumerate(train_loader):
            if epoch == starting_epoch and batch_idx < batches_to_skip:
                continue

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        accelerator.unwrap_model(model).multi_modal_projector.parameters(), 1.0
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            mean_loss = accelerator.gather_for_metrics(loss.detach().float().view(1)).mean().item()

            if global_step % int(cfg["log_steps"]) == 0:
                log_message(f"step {global_step}: train_loss={mean_loss:.6f}", accelerator, log_path)

            if global_step % int(cfg["eval_steps"]) == 0:
                evaluate(model, eval_loader, accelerator, eval_samples, collator, 64, global_step, log_path)

            if global_step % int(cfg["save_steps"]) == 0 and accelerator.is_main_process:
                ckpt_path = output_dir / f"checkpoint-{global_step}.pt"
                save_projector_ckpt(accelerator.unwrap_model(model), optimizer, scheduler, global_step, ckpt_path)
                rotate_checkpoints(str(output_dir), int(cfg["keep_last_n"]))
                log_message(f"Saved checkpoint to {ckpt_path}.", accelerator, log_path)

        batches_to_skip = 0

    log_message("Training finished.", accelerator, log_path)


if __name__ == "__main__":
    main()
