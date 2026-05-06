from __future__ import annotations

from src.model import IMAGE_TOKEN, build_processor

_CAPTION_PROMPT = "Mô tả hình ảnh này: "
PROMPT_TEMPLATE = f"{IMAGE_TOKEN}\n{_CAPTION_PROMPT}"


class ImageCaptionCollator:
    def __init__(self, vision_model_name: str, llm_model_name: str):
        self.processor = build_processor(vision_model_name, llm_model_name)
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.eos_token_id = self.tokenizer.eos_token_id

        if self.image_token_id is None or self.image_token_id < 0:
            raise ValueError(f"Tokenizer does not expose the {IMAGE_TOKEN} token.")
        if self.eos_token_id is None or self.tokenizer.eos_token is None:
            raise ValueError("Tokenizer must expose an EOS token.")

    def __call__(self, batch):
        samples = [s for s in batch if s is not None]
        if not samples:
            raise RuntimeError("Received an empty batch after filtering invalid samples.")

        images, texts, prompt_lengths = [], [], []
        for sample in samples:
            caption = sample["caption"].strip()
            if not caption:
                raise ValueError("Caption tokenization produced an empty caption sequence.")
            image = sample["image"]
            full_text = f"{PROMPT_TEMPLATE}{caption}{self.tokenizer.eos_token}"
            prompt_inputs = self.processor(text=PROMPT_TEMPLATE, images=image, return_tensors="pt")
            images.append(image)
            texts.append(full_text)
            prompt_lengths.append(int(prompt_inputs["input_ids"].shape[1]))

        encoded = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        for row, prompt_len in enumerate(prompt_lengths):
            labels[row, :prompt_len] = -100

        return {
            "pixel_values": encoded["pixel_values"],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }
