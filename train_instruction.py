from __future__ import annotations
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Adafactor, get_cosine_schedule_with_warmup

from src.checkpoint import (
    load_full_ckpt,
    load_projector_ckpt,
    rotate_checkpoints,
    save_full_ckpt,
)
from src.config import load_config
from src.instruction_collator import InstructionCollator
from src.instruction_dataset import ImageInstructionDataset
from src.model import build_model, freeze_components, set_component_modes
from src.paths import resolve_config_path
from src.trainer import EpochShuffleSampler, append_jsonl, evaluate_loss, log_message
from src.utils import set_seed


def _is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    lower = name.lower()
    return param.ndim == 1 or lower.endswith(".bias") or "norm" in lower


def _build_optimizer(model, cfg: dict):
    projector_lr = float(cfg["projector_lr"])
    llm_lr = float(cfg["llm_lr"])
    wd = float(cfg["weight_decay"])
    groups: dict[str, list] = {
        k: [] for k in ("proj_decay", "proj_nodecay", "llm_decay", "llm_nodecay")
    }
    seen_params: set[int] = set()

    def add_params(named_params, decay_key: str, nodecay_key: str) -> None:
        for name, p in named_params:
            if not p.requires_grad or id(p) in seen_params:
                continue
            seen_params.add(id(p))
            groups[nodecay_key if _is_no_decay(name, p) else decay_key].append(p)

    add_params(
        model.multi_modal_projector.named_parameters(), "proj_decay", "proj_nodecay"
    )
    add_params(model.language_model.named_parameters(), "llm_decay", "llm_nodecay")
    add_params(model.lm_head.named_parameters(), "llm_decay", "llm_nodecay")

    param_groups = []
    if groups["proj_decay"]:
        param_groups.append(
            {"params": groups["proj_decay"], "lr": projector_lr, "weight_decay": wd}
        )
    if groups["proj_nodecay"]:
        param_groups.append(
            {"params": groups["proj_nodecay"], "lr": projector_lr, "weight_decay": 0.0}
        )
    if groups["llm_decay"]:
        param_groups.append(
            {"params": groups["llm_decay"], "lr": llm_lr, "weight_decay": wd}
        )
    if groups["llm_nodecay"]:
        param_groups.append(
            {"params": groups["llm_nodecay"], "lr": llm_lr, "weight_decay": 0.0}
        )

    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    opt_type = str(cfg.get("optimizer_type", "adafactor")).strip().lower()
    if opt_type == "adamw":
        return AdamW(param_groups, foreach=bool(cfg.get("adam_foreach", False)))
    if opt_type == "adafactor":
        return Adafactor(
            param_groups, scale_parameter=False, relative_step=False, warmup_init=False
        )
    raise ValueError(f"Unsupported optimizer_type '{opt_type}'.")


def _latest_user_message(messages) -> str:
    for msg in reversed(messages):
        if msg["role"] == "user":
            return str(msg["content"])
    return "<no user message>"


def _select_eval_samples(records, max_samples: int = 5):
    seen, selected = set(), []
    for r in records:
        key = (r.get("sample_type"), r.get("image_id"), r.get("id"))
        if key not in seen:
            seen.add(key)
            selected.append(r)
        if len(selected) >= max_samples:
            break
    return selected


def _resolve_resume_sources(
    resume_dir: Path | None, base_llm_model: str
) -> tuple[str, str]:
    if resume_dir is None:
        return base_llm_model, base_llm_model

    llm_dir = resume_dir / "llm"
    tokenizer_dir = resume_dir / "tokenizer"
    if not llm_dir.exists():
        raise FileNotFoundError(f"Missing resumed LLM weights under {llm_dir}")

    tokenizer_source = (
        str(tokenizer_dir.resolve()) if tokenizer_dir.exists() else base_llm_model
    )
    return str(llm_dir.resolve()), tokenizer_source


def _log_eval_samples(model, collator, eval_samples, accelerator, max_new_tokens):
    unwrapped = accelerator.unwrap_model(model)
    tokenizer = collator.tokenizer
    lines = []
    for idx, sample in enumerate(eval_samples, 1):
        try:
            with Image.open(sample["image"]) as img:
                img = img.convert("RGB")
        except Exception as e:
            line = f"[sample {idx}] failed: {e}"
            accelerator.print(line)
            lines.append(line)
            continue

        prompt_ids, attn_mask = collator.build_prompt_tensors(
            sample["messages"][:-1], device=accelerator.device
        )
        pixel_values = collator.image_processor(images=img, return_tensors="pt")[
            "pixel_values"
        ].to(accelerator.device)

        eos_ids = sorted(
            {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end_of_text|>")}
        )
        with torch.no_grad(), accelerator.autocast():
            generated_ids = unwrapped.generate(
                input_ids=prompt_ids,
                pixel_values=pixel_values,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=5,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

        input_len = prompt_ids.shape[1]
        generated_text = tokenizer.decode(
            generated_ids[0, input_len:], skip_special_tokens=True
        ).strip()
        raw_generated_text = tokenizer.decode(
            generated_ids[0, input_len:], skip_special_tokens=False
        ).strip()
        prediction = generated_text or "<empty>"
        for line in [
            f"[sample {idx}] sample_type: {sample.get('sample_type', 'unknown')}",
            f"[sample {idx}] user: {_latest_user_message(sample['messages'][:-1])}",
            f"[sample {idx}] prediction: {prediction}",
            f"[sample {idx}] prediction_raw: {raw_generated_text or '<empty>'}",
            f"[sample {idx}] reference: {sample['messages'][-1]['content']}",
        ]:
            accelerator.print(line)
            lines.append(line)
    return lines


def evaluate(
    model,
    eval_loader,
    accelerator,
    eval_samples,
    collator,
    max_new_tokens,
    global_step,
    log_path,
    component_modes,
):
    accelerator.unwrap_model(model).eval()
    eval_loss = evaluate_loss(model, eval_loader, accelerator)
    log_message(
        f"[check] step {global_step}: eval_loss={eval_loss:.6f}", accelerator, log_path
    )

    lines = _log_eval_samples(
        model, collator, eval_samples, accelerator, max_new_tokens
    )
    if accelerator.is_main_process:
        with log_path.open("a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")

    set_component_modes(accelerator.unwrap_model(model), **component_modes)
    return eval_loss


def main() -> None:
    cfg = load_config("instruction_train")
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_instruction.log"
    metrics_path = output_dir / "metrics.jsonl"

    resume_dir = (
        Path(cfg["resume_from"]).expanduser().resolve()
        if cfg.get("resume_from")
        else None
    )
    stage1_ckpt = Path(cfg["stage1_projector_ckpt"]).expanduser().resolve()

    mixed_precision = str(cfg.get("mixed_precision", "bf16")).strip().lower()
    if (
        mixed_precision == "bf16"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "bf16 not supported on this device. Set mixed_precision to fp16."
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg["grad_accum"]),
        mixed_precision=mixed_precision,
    )
    set_seed(int(cfg["seed"]))

    llm_source, tokenizer_source = _resolve_resume_sources(resume_dir, cfg["llm_model"])
    collator = InstructionCollator(
        cfg["vision_model"],
        tokenizer_source,
        max_text_tokens=int(cfg["max_text_tokens"]),
    )
    train_dataset = ImageInstructionDataset(resolve_config_path(cfg["train_jsonl"]))
    eval_dataset = ImageInstructionDataset(resolve_config_path(cfg["eval_jsonl"]))
    eval_samples = _select_eval_samples(eval_dataset.records)
    train_sampler = EpochShuffleSampler(train_dataset, seed=int(cfg["seed"]))

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
        llm_source,
        tokenizer_name_or_path=tokenizer_source,
        model_dtype=cfg.get("model_dtype"),
        projector_dtype=cfg.get("projector_dtype", "float32"),
    )
    freeze_components(
        model,
        freeze_vision=bool(cfg.get("freeze_vision", True)),
        train_projector=bool(cfg.get("train_projector", True)),
        train_llm=bool(cfg.get("train_llm", True)),
    )
    component_modes = {
        "freeze_vision": bool(cfg.get("freeze_vision", True)),
        "train_projector": bool(cfg.get("train_projector", True)),
        "train_llm": bool(cfg.get("train_llm", True)),
    }

    if bool(cfg.get("gradient_checkpointing", False)):
        model.language_model.config.use_cache = False
        model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    stage1_step = None
    if resume_dir is None:
        if not stage1_ckpt.exists():
            raise FileNotFoundError(f"Missing stage-1 checkpoint: {stage1_ckpt}")
        stage1_step = load_projector_ckpt(str(stage1_ckpt), model)

    optimizer = _build_optimizer(model, cfg)
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )

    steps_per_epoch = math.ceil(len(train_loader) / int(cfg["grad_accum"]))
    total_steps = steps_per_epoch * int(cfg["epochs"])
    warmup_steps = int(total_steps * float(cfg["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    if resume_dir:
        global_step = load_full_ckpt(
            str(resume_dir), accelerator.unwrap_model(model), optimizer, scheduler
        )

    log_message(
        f"[check] train={len(train_dataset)} eval={len(eval_dataset)}",
        accelerator,
        log_path,
    )
    if stage1_step is not None:
        log_message(
            f"[check] warm-started projector from stage-1 step {stage1_step}",
            accelerator,
            log_path,
        )
    if resume_dir:
        log_message(
            f"[check] resumed from {resume_dir} at step {global_step}",
            accelerator,
            log_path,
        )

    optimizer.zero_grad(set_to_none=True)
    set_component_modes(accelerator.unwrap_model(model), **component_modes)
    running_token_count = torch.zeros(1, dtype=torch.long, device=accelerator.device)
    running_sum_loss = torch.zeros(1, dtype=torch.float32, device=accelerator.device)
    trainable_params = [
        p
        for p in accelerator.unwrap_model(model).parameters()
        if p.requires_grad
    ]
    mean_loss = float("nan")
    mean_loss_tensor = torch.zeros(1, dtype=torch.float32, device=accelerator.device)

    starting_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    batches_to_skip = (
        (global_step % steps_per_epoch) * int(cfg["grad_accum"])
        if steps_per_epoch > 0
        else 0
    )

    for epoch in range(starting_epoch, int(cfg["epochs"])):
        train_sampler.set_epoch(epoch)
        set_component_modes(accelerator.unwrap_model(model), **component_modes)

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
                    for p in trainable_params:
                        if p.grad is not None:
                            p.grad.div_(grad_denom)
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
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
                eval_loss = evaluate(
                    model,
                    eval_loader,
                    accelerator,
                    eval_samples,
                    collator,
                    int(cfg["max_new_tokens"]),
                    global_step,
                    log_path,
                    component_modes,
                )
                if accelerator.is_main_process:
                    append_jsonl(
                        metrics_path, {"step": global_step, "eval_loss": eval_loss}
                    )

            if (
                global_step % int(cfg["save_steps"]) == 0
                and accelerator.is_main_process
            ):
                ckpt_path = output_dir / f"checkpoint-{global_step}"
                save_full_ckpt(
                    accelerator.unwrap_model(model),
                    collator.tokenizer,
                    optimizer,
                    scheduler,
                    global_step,
                    ckpt_path,
                )
                rotate_checkpoints(str(output_dir), int(cfg["keep_last_n"]))
                log_message(
                    f"[check] saved checkpoint to {ckpt_path}", accelerator, log_path
                )

        batches_to_skip = 0

    if accelerator.is_main_process:
        ckpt_path = output_dir / f"checkpoint-{global_step}"
        save_full_ckpt(
            accelerator.unwrap_model(model),
            collator.tokenizer,
            optimizer,
            scheduler,
            global_step,
            ckpt_path,
        )
        rotate_checkpoints(str(output_dir), int(cfg["keep_last_n"]))
        log_message(
            f"[check] saved final instruction checkpoint to {ckpt_path}",
            accelerator,
            log_path,
        )

    log_message("Instruction finetuning finished.", accelerator, log_path)


if __name__ == "__main__":
    main()
