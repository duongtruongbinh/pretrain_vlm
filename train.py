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
from src.paths import resolve_config_paths
from src.trainer import append_jsonl, build_weighted_sampler, evaluate_loss, log_message
from src.utils import set_seed


def _select_eval_samples(records: list[dict], sample_count: int = 5) -> list[dict]:
    if sample_count <= 0 or not records:
        return []
    if len(records) <= sample_count:
        return records

    anchors = [
        round(i * (len(records) - 1) / (sample_count - 1)) for i in range(sample_count)
    ]
    selected = []
    seen_images = set()

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
        if id(record) in selected_ids:
            continue
        selected.append(record)

    return selected[:sample_count]


def _log_eval_samples(
    model, collator, eval_samples, accelerator, max_new_tokens, log_path
):
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

        inputs = collator.processor(
            text=PROMPT_TEMPLATE, images=img, return_tensors="pt"
        )
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        eos_ids = sorted(
            {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end_of_text|>")}
        )
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
        generated_text = tokenizer.decode(
            generated_ids[0, input_len:], skip_special_tokens=True
        ).strip()
        raw_generated_text = tokenizer.decode(
            generated_ids[0, input_len:], skip_special_tokens=False
        ).strip()
        prediction = generated_text or "<empty>"
        for line in [
            f"[sample {idx}] prediction: {prediction}",
            f"[sample {idx}] prediction_raw: {raw_generated_text or '<empty>'}",
            f"[sample {idx}] reference: {sample['caption']}",
            f"[sample {idx}] image: {sample['image']}",
        ]:
            accelerator.print(line)
            lines.append(line)
    return lines


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().reshape(-1, values.shape[-1])
    token_norm = flat.norm(dim=-1)
    return {
        "std": float(flat.std().cpu()),
        "token_norm": float(token_norm.mean().cpu()),
    }


def _projector_scale_stats(model, collator, sample: dict, accelerator) -> dict[str, float] | None:
    unwrapped = accelerator.unwrap_model(model)
    tokenizer = collator.processor.tokenizer

    try:
        with Image.open(sample["image"]) as img:
            image = img.convert("RGB")
    except Exception:
        return None

    black_image = Image.new("RGB", image.size, (0, 0, 0))
    vision_dtype = next(unwrapped.vision_tower.parameters()).dtype
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    def encode(image_obj):
        inputs = collator.processor(
            text=PROMPT_TEMPLATE, images=image_obj, return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(accelerator.device)
        pixel_values = inputs["pixel_values"].to(
            device=accelerator.device, dtype=vision_dtype
        )
        vision_outputs = unwrapped.vision_tower(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        feature_layer = unwrapped.config.vision_feature_layer
        select_strategy = unwrapped.config.vision_feature_select_strategy
        if isinstance(feature_layer, int):
            selected_features = vision_outputs.hidden_states[feature_layer]
            if select_strategy == "default":
                selected_features = selected_features[:, 1:]
        else:
            hidden_states = [vision_outputs.hidden_states[idx] for idx in feature_layer]
            if select_strategy == "default":
                hidden_states = [state[:, 1:] for state in hidden_states]
            selected_features = torch.cat(hidden_states, dim=-1)

        image_features = unwrapped.multi_modal_projector(selected_features)
        text_mask = input_ids != image_token_id
        if tokenizer.pad_token_id is not None:
            text_mask = text_mask & (input_ids != tokenizer.pad_token_id)
        text_embeds = unwrapped.get_input_embeddings()(input_ids)[text_mask]
        return image_features, text_embeds

    with torch.no_grad():
        real_features, text_embeds = encode(image)
        black_features, _ = encode(black_image)

    real_stats = _tensor_stats(real_features)
    black_stats = _tensor_stats(black_features)
    text_stats = _tensor_stats(text_embeds)
    real_black_cosine = torch.nn.functional.cosine_similarity(
        real_features.detach().float().flatten(1),
        black_features.detach().float().flatten(1),
        dim=-1,
    ).mean()
    return {
        "real_norm": real_stats["token_norm"],
        "black_norm": black_stats["token_norm"],
        "text_norm": text_stats["token_norm"],
        "real_std": real_stats["std"],
        "black_std": black_stats["std"],
        "text_std": text_stats["std"],
        "real_norm_ratio": real_stats["token_norm"]
        / max(text_stats["token_norm"], 1e-12),
        "black_norm_ratio": black_stats["token_norm"]
        / max(text_stats["token_norm"], 1e-12),
        "real_black_cosine": float(real_black_cosine.cpu()),
    }


def evaluate(
    model,
    eval_loader,
    accelerator,
    eval_samples,
    collator,
    max_new_tokens,
    global_step,
    log_path,
):
    accelerator.unwrap_model(model).multi_modal_projector.eval()
    eval_loss = evaluate_loss(model, eval_loader, accelerator)
    log_message(f"step {global_step}: eval_loss={eval_loss:.6f}", accelerator, log_path)

    lines = _log_eval_samples(
        model, collator, eval_samples, accelerator, max_new_tokens, log_path
    )
    scale_stats = None
    if eval_samples and accelerator.is_main_process:
        scale_stats = _projector_scale_stats(model, collator, eval_samples[0], accelerator)
        if scale_stats is not None:
            log_message(
                "Projector scale: "
                f"real_norm={scale_stats['real_norm']:.3f}, "
                f"black_norm={scale_stats['black_norm']:.3f}, "
                f"text_norm={scale_stats['text_norm']:.3f}, "
                f"real_ratio={scale_stats['real_norm_ratio']:.1f}x, "
                f"black_ratio={scale_stats['black_norm_ratio']:.1f}x, "
                f"real_black_cosine={scale_stats['real_black_cosine']:.4f}",
                accelerator,
                log_path,
            )
    if accelerator.is_main_process:
        with log_path.open("a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")

    accelerator.unwrap_model(model).multi_modal_projector.train()
    return eval_loss, scale_stats


def main() -> None:
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
        projector_norm=cfg.get("projector_norm"),
        projector_norm_target_multiplier=float(
            cfg.get("projector_norm_target_multiplier", 3.0)
        ),
        projector_norm_trainable=bool(cfg.get("projector_norm_trainable", True)),
        projector_norm_min_multiplier=cfg.get("projector_norm_min_multiplier", 1.0),
        projector_norm_max_multiplier=cfg.get("projector_norm_max_multiplier", 10.0),
        projector_norm_eps=float(cfg.get("projector_norm_eps", 1e-6)),
    )
    freeze_components(model, freeze_vision=True, train_projector=True, train_llm=False)

    if bool(cfg.get("gradient_checkpointing", False)):
        model.language_model.config.use_cache = False
        model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    optimizer = AdamW(
        model.multi_modal_projector.parameters(), lr=float(cfg["lr"]), weight_decay=0.0
    )
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
        log_message(
            f"Resumed from {resume_from} at step {global_step}.", accelerator, log_path
        )

    log_message(
        f"Starting training: {len(train_dataset)} train, {len(eval_dataset)} eval.",
        accelerator,
        log_path,
    )
    log_message(
        "Training config: "
        f"processes={accelerator.num_processes}, "
        f"per_device_batch={int(cfg['batch_size'])}, "
        f"grad_accum={int(cfg['grad_accum'])}, "
        f"effective_batch={int(cfg['batch_size']) * int(cfg['grad_accum']) * accelerator.num_processes}, "
        f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}, output_dir={output_dir}",
        accelerator,
        log_path,
    )
    log_message(
        "Runtime dtypes: "
        f"vision={next(accelerator.unwrap_model(model).vision_tower.parameters()).dtype}, "
        f"llm={next(accelerator.unwrap_model(model).language_model.parameters()).dtype}, "
        f"projector={next(accelerator.unwrap_model(model).multi_modal_projector.parameters()).dtype}",
        accelerator,
        log_path,
    )
    log_message(
        "Projector norm: "
        f"type={cfg.get('projector_norm', 'none')}, "
        f"target_multiplier={cfg.get('projector_norm_target_multiplier', 'n/a')}, "
        f"trainable={cfg.get('projector_norm_trainable', False)}, "
        f"clamp=[{cfg.get('projector_norm_min_multiplier', 'none')}, "
        f"{cfg.get('projector_norm_max_multiplier', 'none')}]",
        accelerator,
        log_path,
    )

    starting_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    batches_to_skip = (
        (global_step % steps_per_epoch) * int(cfg["grad_accum"])
        if steps_per_epoch > 0
        else 0
    )
    optimizer.zero_grad(set_to_none=True)
    running_token_count = torch.zeros(1, dtype=torch.long, device=accelerator.device)
    running_sum_loss = torch.zeros(1, dtype=torch.float32, device=accelerator.device)
    projector_params = list(
        accelerator.unwrap_model(model).multi_modal_projector.parameters()
    )
    mean_loss = float("nan")
    mean_loss_tensor = torch.zeros(1, dtype=torch.float32, device=accelerator.device)

    for epoch in range(starting_epoch, int(cfg["epochs"])):
        train_sampler.set_epoch(epoch)
        accelerator.unwrap_model(model).multi_modal_projector.train()

        for batch_idx, batch in enumerate(train_loader):
            if epoch == starting_epoch and batch_idx < batches_to_skip:
                continue

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                n_tokens = (batch["labels"] != -100).sum()
                loss = outputs.loss * n_tokens.float()
                accelerator.backward(loss)
                running_token_count += n_tokens
                running_sum_loss += outputs.loss.detach().float() * n_tokens.float()

                if accelerator.sync_gradients:
                    total_tokens = (
                        accelerator.gather(running_token_count).sum().clamp(min=1).float()
                    )
                    # DDP all-reduce already averages grads across N ranks, so the
                    # effective denominator must be (total_tokens / N) to recover
                    # sum_token_grads / total_tokens.
                    grad_denom = total_tokens / accelerator.num_processes
                    for p in projector_params:
                        if p.grad is not None:
                            p.grad.div_(grad_denom)
                    accelerator.clip_grad_norm_(projector_params, 1.0)
                    mean_loss_tensor = (
                        accelerator.gather(running_sum_loss).sum() / total_tokens
                    ).detach()
                    running_token_count.zero_()
                    running_sum_loss.zero_()

                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            global_step += 1

            if global_step % int(cfg["log_steps"]) == 0:
                mean_loss = mean_loss_tensor.item()
                log_message(
                    f"step {global_step}: train_loss={mean_loss:.6f}",
                    accelerator,
                    log_path,
                )
                if accelerator.is_main_process:
                    append_jsonl(
                        metrics_path, {"step": global_step, "train_loss": mean_loss}
                    )

            if global_step % int(cfg["eval_steps"]) == 0:
                eval_loss, scale_stats = evaluate(
                    model,
                    eval_loader,
                    accelerator,
                    eval_samples,
                    collator,
                    64,
                    global_step,
                    log_path,
                )
                if accelerator.is_main_process:
                    metrics = {"step": global_step, "eval_loss": eval_loss}
                    if scale_stats is not None:
                        metrics.update({f"projector_{k}": v for k, v in scale_stats.items()})
                    append_jsonl(metrics_path, metrics)

            if (
                global_step % int(cfg["save_steps"]) == 0
                and accelerator.is_main_process
            ):
                ckpt_path = output_dir / f"checkpoint-{global_step}.pt"
                save_projector_ckpt(
                    accelerator.unwrap_model(model),
                    optimizer,
                    scheduler,
                    global_step,
                    ckpt_path,
                )
                rotate_checkpoints(str(output_dir), int(cfg["keep_last_n"]))
                log_message(f"Saved checkpoint to {ckpt_path}.", accelerator, log_path)

        batches_to_skip = 0

    if accelerator.is_main_process:
        ckpt_path = output_dir / f"checkpoint-{global_step}.pt"
        save_projector_ckpt(
            accelerator.unwrap_model(model),
            optimizer,
            scheduler,
            global_step,
            ckpt_path,
        )
        rotate_checkpoints(str(output_dir), int(cfg["keep_last_n"]))
        log_message(f"Saved final checkpoint to {ckpt_path}.", accelerator, log_path)

    log_message("Training finished.", accelerator, log_path)


if __name__ == "__main__":
    main()
