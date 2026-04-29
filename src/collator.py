from __future__ import annotations

import torch

from src.model import IMAGE_TOKEN, build_processor

_CAPTION_PROMPT = "Mo ta hinh anh nay: <"
PROMPT_TEMPLATE = f"{IMAGE_TOKEN}\n{_CAPTION_PROMPT}"


class ImageCaptionCollator:
    def __init__(self, vision_model_name: str, llm_model_name: str):
        self.processor = build_processor(vision_model_name, llm_model_name)

    def __call__(self, batch):
        samples = [s for s in batch if s is not None]
        if not samples:
            raise RuntimeError("Received an empty batch after filtering invalid samples.")

        eos = self.processor.tokenizer.eos_token
        # Append EOS explicitly so it always falls in the supervised span.
        # Relying on the processor's default add_special_tokens behaviour is
        # fragile — the old collator always added <|end_of_text|> explicitly.
        full_texts = [PROMPT_TEMPLATE + s["caption"].strip() + eos for s in samples]
        images = [s["image"] for s in samples]

        inputs = self.processor(
            text=full_texts, images=images, return_tensors="pt", padding=True,
        )

        labels = inputs["input_ids"].clone()
        for i, s in enumerate(samples):
            prompt_inputs = self.processor(
                text=PROMPT_TEMPLATE, images=s["image"], return_tensors="pt",
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels[i, :prompt_len] = -100
        labels[inputs["attention_mask"] == 0] = -100

        return {
            "pixel_values": inputs["pixel_values"],
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }
