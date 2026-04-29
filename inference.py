from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.checkpoint import load_projector_ckpt
from src.collator import PROMPT_TEMPLATE, ImageCaptionCollator
from src.config import load_config
from src.model import build_model, freeze_components


def main() -> None:
    cfg = load_config("inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    collator = ImageCaptionCollator(cfg["vision_model"], cfg["llm_model"])
    model = build_model(cfg["vision_model"], cfg["llm_model"], model_dtype=cfg.get("model_dtype"))
    load_projector_ckpt(cfg["ckpt"], model)
    freeze_components(model, freeze_vision=True, train_projector=False, train_llm=False)
    model.eval().to(device)

    with Image.open(Path(cfg["image"]).expanduser().resolve()) as img:
        img = img.convert("RGB")

    prompt_text = cfg.get("prompt") or PROMPT_TEMPLATE
    inputs = collator.processor(text=prompt_text, images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=int(cfg["max_new_tokens"]),
            eos_token_id=collator.processor.tokenizer.eos_token_id,
            pad_token_id=collator.processor.tokenizer.pad_token_id,
            do_sample=False,
        )

    print(collator.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
