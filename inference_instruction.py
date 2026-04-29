from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.checkpoint import load_full_ckpt
from src.config import load_config
from src.instruction_collator import InstructionCollator
from src.model import build_model, freeze_components


def main() -> None:
    cfg = load_config("instruction_inference")
    ckpt_dir = Path(cfg["ckpt_dir"]).expanduser().resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing instruction checkpoint directory: {ckpt_dir}")

    llm_source = str((ckpt_dir / "llm").resolve()) if (ckpt_dir / "llm").exists() else cfg["llm_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    collator = InstructionCollator(
        cfg["vision_model"], llm_source, max_text_tokens=int(cfg.get("max_text_tokens", 1024))
    )
    model = build_model(cfg["vision_model"], llm_source, model_dtype=cfg.get("model_dtype"))
    load_full_ckpt(ckpt_dir, model)
    freeze_components(model, freeze_vision=True, train_projector=False, train_llm=False)
    model.eval().to(device)

    system_prompt = str(cfg.get("system_prompt", "")).strip()
    question = str(cfg["question"]).strip()
    if not question:
        raise ValueError("instruction_infer.question must be a non-empty string.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    prompt_ids, attention_mask = collator.build_prompt_tensors(messages, device=device)

    with Image.open(Path(cfg["image"]).expanduser().resolve()) as img:
        img = img.convert("RGB")
    pixel_values = collator.image_processor(images=img, return_tensors="pt")["pixel_values"].to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=prompt_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=int(cfg["max_new_tokens"]),
            eos_token_id=collator.tokenizer.eos_token_id,
            pad_token_id=collator.tokenizer.pad_token_id,
            do_sample=False,
        )

    print(collator.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
