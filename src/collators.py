"""Batch collation for caption pretraining and instruction tuning."""

from __future__ import annotations

import warnings

import torch

from src.modeling import IMAGE_TOKEN, build_processor
from src.prompts import render

PROMPT_TEMPLATE = render("caption_prompt.j2")


def _image_seq_length(processor) -> int:
    h = processor.image_processor.size["height"]
    n = (h // processor.patch_size) ** 2
    if processor.vision_feature_select_strategy == "default":
        n -= 1
    return n + processor.num_additional_image_tokens


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
        for s in batch:
            if not s["caption"].strip():
                raise ValueError("Received an empty caption.")

        images = [s["image"] for s in batch]
        texts = [f"{PROMPT_TEMPLATE}{s['caption'].strip()}{self.tokenizer.eos_token}" for s in batch]
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

    def _build_training_texts(self, messages, *, sample_id: str) -> tuple[list[dict], str]:
        messages = self._inject_image_token(messages)
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if not full_text:
            raise ValueError(f"Sample '{sample_id}' produced empty full text.")
        return messages, full_text

    def _assistant_token_spans(self, messages: list[dict]) -> list[tuple[int, int]]:
        """Return (start, end) token index for every assistant turn in the conversation."""
        spans = []
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            prefix = self.tokenizer.apply_chat_template(
                messages[:i], tokenize=False, add_generation_prompt=True
            ).replace(IMAGE_TOKEN, IMAGE_TOKEN * self._image_seq_length)
            full = self.tokenizer.apply_chat_template(
                messages[:i + 1], tokenize=False, add_generation_prompt=False
            ).replace(IMAGE_TOKEN, IMAGE_TOKEN * self._image_seq_length)
            start = len(self.tokenizer(prefix)["input_ids"])
            end = len(self.tokenizer(full)["input_ids"])
            spans.append((start, end))
        return spans

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
                injected_messages, full_text = self._build_training_texts(
                    sample["messages"], sample_id=sample["id"]
                )
            except Exception as e:
                warnings.warn(f"Skipping sample '{sample['id']}': {e}")
                continue
            valid.append(
                {
                    "pixel_image": sample["image"],
                    "messages": injected_messages,
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
        seq_len = encoded["input_ids"].shape[1]
        labels = torch.full_like(encoded["input_ids"], -100)

        for row, sample in enumerate(valid):
            # Supervise all assistant turns; mask system/user turns.
            for start, end in self._assistant_token_spans(sample["messages"]):
                end = min(end, seq_len)
                if start < seq_len:
                    labels[row, start:end] = encoded["input_ids"][row, start:end]

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

