from __future__ import annotations

import warnings

import torch

from src.model import IMAGE_TOKEN, build_processor


class InstructionCollator:
    def __init__(self, vision_model_name: str, llm_model_name: str, max_text_tokens: int):
        self.processor = build_processor(vision_model_name, llm_model_name)
        self.max_text_tokens = int(max_text_tokens)
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

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

    def _build_training_tokens(self, messages, *, sample_id: str) -> tuple[list[int], list[int]]:
        messages = self._inject_image_token(messages)

        prompt_ids = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True,
        )
        full_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
        )

        if not full_ids:
            raise ValueError(f"Sample '{sample_id}' produced an empty token sequence.")
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Sample '{sample_id}' failed the prompt-prefix masking check.")

        assistant_ids = full_ids[len(prompt_ids):]
        if not assistant_ids:
            raise ValueError(f"Sample '{sample_id}' produced no assistant supervision tokens.")

        max_prompt_len = self.max_text_tokens - len(assistant_ids)
        if max_prompt_len < 2:
            warnings.warn(
                f"Sample '{sample_id}': assistant target ({len(assistant_ids)} tokens) exceeds "
                f"max_text_tokens={self.max_text_tokens}. Truncating assistant response."
            )
            assistant_ids = assistant_ids[: self.max_text_tokens - 2]
            max_prompt_len = self.max_text_tokens - len(assistant_ids)

        if len(prompt_ids) > max_prompt_len:
            prompt_ids = [prompt_ids[0]] + prompt_ids[-(max_prompt_len - 1):]

        input_ids = prompt_ids + assistant_ids
        labels = ([-100] * len(prompt_ids)) + assistant_ids
        return input_ids, labels

    def build_prompt_tensors(self, messages, device=None):
        messages = self._inject_image_token(messages)
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
        if len(prompt_ids) > self.max_text_tokens:
            prompt_ids = [prompt_ids[0]] + prompt_ids[-(self.max_text_tokens - 1):]
        t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        return t, torch.ones_like(t)

    def __call__(self, batch):
        valid = []
        for sample in batch:
            try:
                input_ids, labels = self._build_training_tokens(
                    sample["messages"], sample_id=sample["id"]
                )
            except Exception as e:
                warnings.warn(f"Skipping sample '{sample['id']}': {e}")
                continue
            valid.append({"pixel_image": sample["image"], "input_ids": input_ids, "labels": labels})

        if not valid:
            raise RuntimeError("Received an empty batch after filtering invalid instruction samples.")

        pixel_values = self.image_processor(
            images=[s["pixel_image"] for s in valid], return_tensors="pt"
        )["pixel_values"]

        max_len = max(len(s["input_ids"]) for s in valid)
        pad_id = self.tokenizer.pad_token_id
        padded_input_ids, padded_labels, padded_masks = [], [], []

        for s in valid:
            ids = torch.tensor(s["input_ids"], dtype=torch.long)
            lbl = torch.tensor(s["labels"], dtype=torch.long)
            mask = torch.ones_like(ids)
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([ids, ids.new_full((pad_len,), pad_id)])
                lbl = torch.cat([lbl, lbl.new_full((pad_len,), -100)])
                mask = torch.cat([mask, mask.new_zeros((pad_len,))])
            padded_input_ids.append(ids)
            padded_labels.append(lbl)
            padded_masks.append(mask)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_masks),
            "labels": torch.stack(padded_labels),
        }
