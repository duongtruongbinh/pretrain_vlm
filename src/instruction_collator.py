from __future__ import annotations

import warnings

import torch

from src.model import IMAGE_TOKEN, build_processor


class InstructionCollator:
    def __init__(self, vision_model_name: str, llm_model_name: str, max_text_tokens: int):
        self.processor = build_processor(vision_model_name, llm_model_name)
        self.max_text_tokens = int(max_text_tokens)
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        if self.image_token_id is None or self.image_token_id < 0:
            raise ValueError(f"Tokenizer does not expose the {IMAGE_TOKEN} token.")

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def image_processor(self):
        return self.processor.image_processor

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
            text=prompt_text,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_tokens,
        )
        input_ids = encoded["input_ids"].to(device)
        return (input_ids, encoded["attention_mask"].to(device), encoded["pixel_values"].to(device))

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
            try:
                prompt_inputs = self.processor(
                    text=sample["prompt_text"],
                    images=sample["pixel_image"],
                    truncation=True,
                    max_length=self.max_text_tokens,
                    return_tensors="pt",
                )
                prompt_len = int(prompt_inputs["input_ids"].shape[1])
                labels[row, :prompt_len] = -100
            except Exception as e:
                warnings.warn(f"Could not compute prompt mask for sample '{sample['sample_id']}': {e}")
                labels[row, :] = -100

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
