from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Adafactor, get_cosine_schedule_with_warmup

from src.collators import InstructionCollator
from src.data import ImageInstructionDataset
from src.modeling import build_model, freeze_components, set_component_modes
from src.runtime import (
    EpochShuffleSampler,
    append_jsonl,
    load_config,
    resolve_config_path,
    set_seed,
    setup_logger,
)
from src.training.checkpoint import (
    load_full_checkpoint,
    load_projector_checkpoint,
    rotate_checkpoints,
    save_training_checkpoint,
    update_checkpoint_pointer,
)
from src.training.engine import TrainingState, compute_steps_per_epoch, run_training
from src.training.eval import run_evaluation


def _parse_args() -> argparse.Namespace:
    """Parse runtime-only arguments that should not live in config.yaml."""

    parser = argparse.ArgumentParser(description="Instruction tuning.")
    parser.add_argument("--resume-from", type=str, default=None)
    return parser.parse_args()


def _is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    lower = name.lower()
    return param.ndim == 1 or lower.endswith(".bias") or "norm" in lower


def _build_optimizer(model, cfg: dict):
    projector_lr = float(cfg["projector_lr"])
    llm_lr = float(cfg["llm_lr"])
    wd = float(cfg["weight_decay"])
    groups: dict[str, list] = {k: [] for k in ("proj_decay", "proj_nodecay", "llm_decay", "llm_nodecay")}
    seen_params: set[int] = set()

    def add_params(named_params, decay_key: str, nodecay_key: str) -> None:
        for name, param in named_params:
            if not param.requires_grad or id(param) in seen_params:
                continue
            seen_params.add(id(param))
            groups[nodecay_key if _is_no_decay(name, param) else decay_key].append(param)

    add_params(model.multi_modal_projector.named_parameters(), "proj_decay", "proj_nodecay")
    add_params(model.language_model.named_parameters(), "llm_decay", "llm_nodecay")
    add_params(model.lm_head.named_parameters(), "llm_decay", "llm_nodecay")

    param_groups = []
    if groups["proj_decay"]:
        param_groups.append({"params": groups["proj_decay"], "lr": projector_lr, "weight_decay": wd})
    if groups["proj_nodecay"]:
        param_groups.append({"params": groups["proj_nodecay"], "lr": projector_lr, "weight_decay": 0.0})
    if groups["llm_decay"]:
        param_groups.append({"params": groups["llm_decay"], "lr": llm_lr, "weight_decay": wd})
    if groups["llm_nodecay"]:
        param_groups.append({"params": groups["llm_nodecay"], "lr": llm_lr, "weight_decay": 0.0})

    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    opt_type = str(cfg.get("optimizer_type", "adafactor")).strip().lower()
    if opt_type == "adamw":
        return AdamW(param_groups, foreach=bool(cfg.get("adam_foreach", False)))
    if opt_type == "adafactor":
        return Adafactor(param_groups, scale_parameter=False, relative_step=False, warmup_init=False)
    raise ValueError(f"Unsupported optimizer_type '{opt_type}'.")


def _latest_user_message(messages) -> str:
    for msg in reversed(messages):
        if msg["role"] == "user":
            return str(msg["content"])
    return "<no user message>"


def _select_eval_samples(records, max_samples: int = 5):
    seen, selected = set(), []
    for record in records:
        key = (record.get("sample_type"), record.get("image_id"), record.get("id"))
        if key not in seen:
            seen.add(key)
            selected.append(record)
        if len(selected) >= max_samples:
            break
    return selected


def _resolve_resume_sources(resume_dir: Path | None, base_llm_model: str) -> tuple[str, str]:
    if resume_dir is None:
        return base_llm_model, base_llm_model

    llm_dir = resume_dir / "llm"
    tokenizer_dir = resume_dir / "tokenizer"
    if not llm_dir.exists():
        raise FileNotFoundError(f"Missing resumed LLM weights under {llm_dir}")

    tokenizer_source = str(tokenizer_dir.resolve()) if tokenizer_dir.exists() else base_llm_model
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
            lines.append(line)
            continue

        prompt_ids, attn_mask, pixel_values = collator.build_prompt_tensors(
            sample["messages"][:-1], img, device=accelerator.device
        )
        eos_ids = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end_of_text|>")})
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
        generated_text = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True).strip()
        raw_generated_text = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=False).strip()
        for line in [
            f"[sample {idx}] sample_type: {sample.get('sample_type', 'unknown')}",
            f"[sample {idx}] user: {_latest_user_message(sample['messages'][:-1])}",
            f"[sample {idx}] prediction: {generated_text or '<empty>'}",
            f"[sample {idx}] prediction_raw: {raw_generated_text or '<empty>'}",
            f"[sample {idx}] reference: {sample['messages'][-1]['content']}",
        ]:
            lines.append(line)
    return lines


def _current_lr(scheduler, optimizer) -> float:
    try:
        return float(scheduler.get_last_lr()[0])
    except Exception:
        return float(optimizer.param_groups[0]["lr"])


def main() -> None:
    args = _parse_args()
    cfg = load_config("instruction_train")
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    resume_dir = Path(args.resume_from).expanduser().resolve() if args.resume_from else None
    stage1_ckpt = Path(cfg["stage1_projector_ckpt"]).expanduser().resolve()
    mixed_precision = str(cfg.get("mixed_precision", "bf16")).strip().lower()
    if mixed_precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 not supported on this device. Set mixed_precision to fp16.")

    accelerator = Accelerator(mixed_precision=mixed_precision)
    logger = setup_logger(output_dir, accelerator)
    set_seed(int(cfg["seed"]))

    llm_source, tokenizer_source = _resolve_resume_sources(resume_dir, cfg["llm_model"])
    collator = InstructionCollator(
        cfg["vision_model"], tokenizer_source, max_text_tokens=int(cfg["max_text_tokens"])
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
    component_modes = {
        "freeze_vision": bool(cfg.get("freeze_vision", True)),
        "train_projector": bool(cfg.get("train_projector", True)),
        "train_llm": bool(cfg.get("train_llm", True)),
    }
    freeze_components(model, **component_modes)

    if bool(cfg.get("gradient_checkpointing", False)):
        model.language_model.config.use_cache = False
        model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    stage1_step = None
    if resume_dir is None:
        if not stage1_ckpt.exists():
            raise FileNotFoundError(f"Missing stage-1 checkpoint: {stage1_ckpt}")
        stage1_state = load_projector_checkpoint(stage1_ckpt, model)
        stage1_step = int(stage1_state.get("global_step", 0))

    optimizer = _build_optimizer(model, cfg)
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )

    grad_accum = int(cfg["grad_accum"])
    steps_per_epoch = compute_steps_per_epoch(len(train_loader), grad_accum)
    total_steps = steps_per_epoch * int(cfg["epochs"])
    warmup_steps = int(total_steps * float(cfg["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    state = TrainingState()

    if resume_dir:
        resume_state = load_full_checkpoint(resume_dir, accelerator.unwrap_model(model), optimizer, scheduler)
        state.global_step = int(resume_state.get("global_step", 0))
        state.best_eval_loss = resume_state.get("best_eval_loss")

    logger.info("[check] train={} eval={}", len(train_dataset), len(eval_dataset))
    if stage1_step is not None:
        logger.info("[check] warm-started projector from stage-1 step {}", stage1_step)
    if resume_dir:
        logger.info("[check] resumed from {} at step {}", resume_dir, state.global_step)

    progress_total = max(total_steps - state.global_step, 0)
    progress = (
        tqdm(total=progress_total, initial=0, desc="instruction", dynamic_ncols=True)
        if accelerator.is_main_process
        else None
    )

    def set_train_mode() -> None:
        set_component_modes(accelerator.unwrap_model(model), **component_modes)

    def trainable_parameters():
        return (p for p in accelerator.unwrap_model(model).parameters() if p.requires_grad)

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
            stage="instruction_tuning",
            save_language_model=True,
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

    def on_epoch_start(epoch: int) -> None:
        train_sampler.set_epoch(epoch)

    def on_step_end(result, training_state) -> None:
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                train_loss=f"{result.train_loss:.6f}",
                lr=f"{_current_lr(scheduler, optimizer):.3e}",
                supervised_tokens=result.supervised_tokens,
            )

        if result.global_step % int(cfg["log_steps"]) == 0:
            logger.info(
                "step {}: train_loss={:.6f}, lr={:.6e}, supervised_tokens={}",
                result.global_step,
                result.train_loss,
                _current_lr(scheduler, optimizer),
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
                sample_logger=lambda m, a: _log_eval_samples(
                    m, collator, eval_samples, a, int(cfg["max_new_tokens"])
                ),
                restore_train_mode=set_train_mode,
            )
            if accelerator.is_main_process:
                append_jsonl(metrics_path, {"step": result.global_step, "eval_loss": eval_loss})

        if result.global_step % int(cfg["save_steps"]) == 0 and accelerator.is_main_process:
            ckpt_path = save_checkpoint(result.global_step, eval_loss=eval_loss)
            logger.info("[check] saved checkpoint to {}", ckpt_path)

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
        on_epoch_start=on_epoch_start,
    )

    if accelerator.is_main_process:
        ckpt_path = save_checkpoint(state.global_step)
        logger.info("[check] saved final instruction checkpoint to {}", ckpt_path)
        if progress is not None:
            progress.close()

    logger.info("Instruction finetuning finished.")


if __name__ == "__main__":
    main()
