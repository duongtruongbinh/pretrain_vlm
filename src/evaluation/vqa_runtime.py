"""Runtime helpers for Stage-2 VQA benchmark evaluation."""

from __future__ import annotations

import re
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image

from src.collators import InstructionCollator
from src.modeling import build_model
from src.prompts import render
from src.runtime import load_config
from src.training.checkpoint import load_full_checkpoint

DEFAULT_SYSTEM_PROMPT = render("vqa_system.j2")


def resolve_device(device_name: str):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_checkpoint_sources(checkpoint_path: str | Path) -> tuple[str, str]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    llm_dir = checkpoint / "llm"
    tokenizer_dir = checkpoint / "tokenizer"
    if not llm_dir.exists():
        raise FileNotFoundError(f"Missing Stage-2 LLM weights: {llm_dir}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Missing Stage-2 tokenizer: {tokenizer_dir}")
    return str(llm_dir), str(tokenizer_dir)


def load_stage2_model(checkpoint_path: str | Path, config_section: str, device_name: str, max_text_tokens: int | None):
    cfg = load_config(config_section)
    device = resolve_device(device_name)
    llm_source, tokenizer_source = resolve_checkpoint_sources(checkpoint_path)
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
    torch.cuda.empty_cache() if device.type == "cuda" else None
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
    input_ids, attention_mask, pixel_values = collator.build_prompt_tensors(messages, image.convert("RGB"), device=device)
    pixel_values = pixel_values.to(dtype=vision_dtype)

    eos_ids = {tokenizer.eos_token_id}
    for token in ("<|eot_id|>", "<|end_of_text|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            eos_ids.add(token_id)

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
            eos_token_id=sorted(i for i in eos_ids if i is not None),
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = input_ids.shape[1]
    raw_text = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True).strip()
    return extract_short_answer(raw_text), raw_text
