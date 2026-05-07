"""Batch collation for caption pretraining and instruction tuning."""

from __future__ import annotations

import warnings

import torch

from src.modeling import IMAGE_TOKEN, build_processor

_CAPTION_PROMPT = "Mô tả hình ảnh này: "
PROMPT_TEMPLATE = f"{IMAGE_TOKEN}\n{_CAPTION_PROMPT}"


def _image_seq_length(processor) -> int:
    """Number of image feature tokens produced by the processor for one image."""
    h = processor.image_processor.size["height"]
    return (h // processor.patch_size) ** 2


class CaptionCollator:
    """Collate image-caption samples into LlavaProcessor tensors with prompt masking."""

    def __init__(self, vision_model_name: str, llm_model_name: str):
        self.processor = build_processor(vision_model_name, llm_model_name)
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        if self.image_token_id is None or self.image_token_id < 0:
            raise ValueError(f"Tokenizer does not expose the {IMAGE_TOKEN} token.")
        if self.tokenizer.eos_token is None:
            raise ValueError("Tokenizer must expose an EOS token.")

        # Prompt is fixed → compute its expanded token length once.
        _n = _image_seq_length(self.processor)
        _expanded = PROMPT_TEMPLATE.replace(IMAGE_TOKEN, IMAGE_TOKEN * _n)
        self._prompt_len = len(self.tokenizer(_expanded)["input_ids"])

    def __call__(self, batch):
        samples = [s for s in batch if s is not None]
        if not samples:
            raise RuntimeError("Received an empty batch after filtering invalid samples.")

        for s in samples:
            if not s["caption"].strip():
                raise ValueError("Received an empty caption.")

        images = [s["image"] for s in samples]
        texts = [
            f"{PROMPT_TEMPLATE}{s['caption'].strip()}{self.tokenizer.eos_token}"
            for s in samples
        ]

        encoded = self.processor(text=texts, images=images, padding=True, return_tensors="pt")

        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        labels[:, : self._prompt_len] = -100

        if torch.any((labels != -100).sum(dim=1) == 0):
            raise RuntimeError("At least one caption sample produced zero supervised tokens.")

        return {
            "pixel_values": encoded["pixel_values"],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


class InstructionCollator:
    """Collate instruction samples into tensors with chat-template prompt masking."""

    def __init__(self, vision_model_name: str, llm_model_name: str, max_text_tokens: int):
        self.processor = build_processor(vision_model_name, llm_model_name)
        self.tokenizer = self.processor.tokenizer
        self.max_text_tokens = int(max_text_tokens)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        if self.image_token_id is None or self.image_token_id < 0:
            raise ValueError(f"Tokenizer does not expose the {IMAGE_TOKEN} token.")
        self._image_seq_length = _image_seq_length(self.processor)

    def _inject_image_token(self, messages: list[dict]) -> list[dict]:
        result, injected = [], False
        for msg in messages:
            if msg["role"] == "user" and not injected:
                result.append({**msg, "content": f"{IMAGE_TOKEN}\n{msg['content']}"})
                injected = True
            else:
                result.append(msg)
        return result

    def _build_training_texts(self, messages, *, sample_id: str) -> tuple[str, str]:
        messages = self._inject_image_token(messages)

        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if not full_text:
            raise ValueError(f"Sample '{sample_id}' produced an empty token sequence.")
        if not full_text.startswith(prompt_text):
            raise ValueError(f"Sample '{sample_id}' failed the prompt-prefix masking check.")
        return prompt_text, full_text

    def build_prompt_tensors(self, messages, image, device=None):
        messages = self._inject_image_token(messages)
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.processor(
            text=prompt_text, images=image, return_tensors="pt",
            truncation=True, max_length=self.max_text_tokens,
        )
        return (
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
            encoded["pixel_values"].to(device),
        )

    def __call__(self, batch):
        valid = []
        for sample in batch:
            try:
                prompt_text, full_text = self._build_training_texts(
                    sample["messages"], sample_id=sample["id"]
                )
            except Exception as e:
                warnings.warn(f"Skipping sample '{sample['id']}': {e}")
                continue
            valid.append(
                {
                    "pixel_image": sample["image"],
                    "prompt_text": prompt_text,
                    "full_text": full_text,
                    "sample_id": sample["id"],
                }
            )

        if not valid:
            raise RuntimeError("Received an empty batch after filtering invalid instruction samples.")

        images = [s["pixel_image"] for s in valid]
        full_texts = [s["full_text"] for s in valid]
        encoded = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100

        for row, sample in enumerate(valid):
            # Expand image token in prompt text, then tokenize text-only to get prompt length.
            # Avoids re-encoding the image; token count is identical to the batch-encoded prefix.
            prompt_expanded = sample["prompt_text"].replace(
                IMAGE_TOKEN, IMAGE_TOKEN * self._image_seq_length
            )
            prompt_len = len(self.tokenizer(prompt_expanded)["input_ids"])
            labels[row, :prompt_len] = -100

            if not torch.any(labels[row] != -100):
                warnings.warn(
                    f"Sample '{sample['sample_id']}' produced no supervised tokens after truncation."
                )

        return {
            "pixel_values": encoded["pixel_values"],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

