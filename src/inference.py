"""Shared model loading, generation, and IO helpers for evaluation scripts and demos."""

from __future__ import annotations

import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.prompts import render


DEFAULT_SYSTEM_PROMPT = render("vqa_system.j2")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_json_or_jsonl(path: str | Path) -> Any:
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Device / token helpers
# ---------------------------------------------------------------------------

def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def eos_token_ids(tokenizer) -> list[int]:
    ids = {tokenizer.eos_token_id}
    for token in ("<|eot_id|>", "<|end_of_text|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    return sorted(i for i in ids if i is not None)


# ---------------------------------------------------------------------------
# Checkpoint source resolution
# ---------------------------------------------------------------------------

def resolve_stage1_tokenizer(checkpoint_path: str | Path, fallback_llm_model: str = "") -> str:
    """Return tokenizer source for a Stage-1 checkpoint directory."""
    tokenizer_dir = Path(checkpoint_path).expanduser().resolve() / "tokenizer"
    if not tokenizer_dir.exists():
        if fallback_llm_model:
            return fallback_llm_model
        raise FileNotFoundError(f"Missing Stage-1 tokenizer: {tokenizer_dir}")
    return str(tokenizer_dir)


def resolve_stage2_sources(checkpoint_path: str | Path) -> tuple[str, str]:
    """Return (llm_source, tokenizer_source) for a Stage-2 checkpoint directory."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    llm_dir = checkpoint / "llm"
    tokenizer_dir = checkpoint / "tokenizer"
    if not llm_dir.exists():
        raise FileNotFoundError(f"Missing Stage-2 LLM weights: {llm_dir}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Missing Stage-2 tokenizer: {tokenizer_dir}")
    return str(llm_dir), str(tokenizer_dir)


# ---------------------------------------------------------------------------
# Stage-1 model loading and caption generation
# ---------------------------------------------------------------------------

def load_stage1_model(checkpoint_path: str | Path, config_section: str, device_name: str):
    """Load Stage-1 (projector-only) model for caption evaluation."""
    from src.modeling import build_model, build_processor
    from src.runtime import load_config
    from src.training import load_projector_checkpoint

    cfg = load_config(config_section)
    device = resolve_device(device_name)
    tokenizer_source = resolve_stage1_tokenizer(checkpoint_path)
    processor = build_processor(cfg["vision_model"], tokenizer_source)
    model = build_model(
        cfg["vision_model"],
        cfg["llm_model"],
        tokenizer_name_or_path=tokenizer_source,
        model_dtype=cfg.get("model_dtype"),
        projector_dtype=cfg.get("projector_dtype", "float32"),
        image_token_id=processor.tokenizer.convert_tokens_to_ids("<image>"),
        vocab_size=len(processor.tokenizer),
    )
    state = load_projector_checkpoint(checkpoint_path, model)
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, processor, state


def generate_caption(model, processor, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    vision_dtype = next(model.vision_tower.parameters()).dtype
    tokenizer = processor.tokenizer
    inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif key == "pixel_values":
            moved[key] = value.to(device=device, dtype=vision_dtype)
        else:
            moved[key] = value.to(device=device)

    autocast_context = nullcontext()
    if device.type == "cuda" and vision_dtype in (torch.float16, torch.bfloat16):
        autocast_context = torch.autocast(device_type="cuda", dtype=vision_dtype)

    with torch.inference_mode(), autocast_context:
        generated_ids = model.generate(
            **moved,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            eos_token_id=eos_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = moved["input_ids"].shape[1]
    return tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Stage-2 model loading and answer generation
# ---------------------------------------------------------------------------

def load_stage2_model(
    checkpoint_path: str | Path, config_section: str, device_name: str, max_text_tokens: int | None
):
    """Load Stage-2 (instruction-tuned) model for VQA evaluation."""
    from src.collators import InstructionCollator
    from src.modeling import build_model
    from src.runtime import load_config
    from src.training import load_full_checkpoint

    cfg = load_config(config_section)
    device = resolve_device(device_name)
    llm_source, tokenizer_source = resolve_stage2_sources(checkpoint_path)
    collator = InstructionCollator(
        cfg["vision_model"],
        tokenizer_source,
        max_text_tokens=max_text_tokens or int(cfg["max_text_tokens"]),
    )
    model = build_model(
        cfg["vision_model"],
        llm_source,
        tokenizer_name_or_path=tokenizer_source,
        model_dtype=cfg.get("model_dtype"),
        projector_dtype=cfg.get("projector_dtype", "float32"),
        image_token_id=collator.image_token_id,
        vocab_size=len(collator.tokenizer),
    )
    state = load_full_checkpoint(checkpoint_path, model)
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, collator, state


def extract_short_answer(text: str) -> str:
    first_line = str(text).strip().splitlines()[0] if str(text).strip() else ""
    answer = re.sub(
        r"^(?:câu\s+)?(?:trả\s+lời|đáp\s+án|answer)\s*(?:là|[:：-])\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    ).strip()
    return re.sub(r"[.。!！?？]+$", "", answer).strip()


def generate_answer(
    model,
    collator,
    image: Image.Image,
    question: str,
    *,
    system_prompt: str,
    max_new_tokens: int,
) -> tuple[str, str]:
    device = next(model.parameters()).device
    vision_dtype = next(model.vision_tower.parameters()).dtype
    tokenizer = collator.tokenizer
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": render("vqa_question.j2", question=question)},
    ]
    input_ids, attention_mask, pixel_values = collator.build_prompt_tensors(
        messages, image.convert("RGB"), device=device
    )
    pixel_values = pixel_values.to(dtype=vision_dtype)

    autocast_context = nullcontext()
    if device.type == "cuda" and vision_dtype in (torch.float16, torch.bfloat16):
        autocast_context = torch.autocast(device_type="cuda", dtype=vision_dtype)

    with torch.inference_mode(), autocast_context:
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=eos_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = input_ids.shape[1]
    raw_text = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True).strip()
    return extract_short_answer(raw_text), raw_text
