from __future__ import annotations

import torch

from src.model import IMAGE_TOKEN, build_processor

_CAPTION_PROMPT = "Mô tả hình ảnh này: "
PROMPT_TEMPLATE = f"{IMAGE_TOKEN}\n{_CAPTION_PROMPT}"


class ImageCaptionCollator:
    def __init__(self, vision_model_name: str, llm_model_name: str):
        self.processor = build_processor(vision_model_name, llm_model_name)
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.prompt_ids = self.tokenizer(
            f"\n{_CAPTION_PROMPT}", add_special_tokens=False
        ).input_ids

        if self.image_token_id is None or self.image_token_id < 0:
            raise ValueError(f"Tokenizer does not expose the {IMAGE_TOKEN} token.")
        if self.bos_token_id is None or self.eos_token_id is None:
            raise ValueError("Tokenizer must expose both BOS and EOS tokens.")

    def _num_image_tokens(self, pixel_values: torch.Tensor) -> int:
        height, width = pixel_values.shape[-2:]
        count = (height // self.processor.patch_size) * (
            width // self.processor.patch_size
        )
        count += int(getattr(self.processor, "num_additional_image_tokens", 0) or 0)
        if self.processor.vision_feature_select_strategy == "default":
            count -= 1
        return int(count)

    def __call__(self, batch):
        samples = [s for s in batch if s is not None]
        if not samples:
            raise RuntimeError(
                "Received an empty batch after filtering invalid samples."
            )

        images = [s["image"] for s in samples]
        pixel_values = self.processor.image_processor(
            images=images, return_tensors="pt"
        )["pixel_values"]
        image_token_count = self._num_image_tokens(pixel_values)
        prompt_prefix = (
            [self.bos_token_id]
            + ([self.image_token_id] * image_token_count)
            + self.prompt_ids
        )

        input_id_tensors, label_tensors, attention_tensors = [], [], []
        for sample in samples:
            caption_ids = self.tokenizer(
                sample["caption"].strip(), add_special_tokens=False
            ).input_ids
            if not caption_ids:
                raise ValueError(
                    "Caption tokenization produced an empty caption sequence."
                )

            input_ids = prompt_prefix + caption_ids + [self.eos_token_id]
            labels = ([-100] * len(prompt_prefix)) + caption_ids + [self.eos_token_id]
            attention_mask = [1] * len(input_ids)

            input_id_tensors.append(torch.tensor(input_ids, dtype=torch.long))
            label_tensors.append(torch.tensor(labels, dtype=torch.long))
            attention_tensors.append(torch.tensor(attention_mask, dtype=torch.long))

        max_len = max(t.size(0) for t in input_id_tensors)
        padded_input_ids, padded_labels, padded_attention_masks = [], [], []
        for input_ids, labels, attention_mask in zip(
            input_id_tensors, label_tensors, attention_tensors
        ):
            pad_len = max_len - input_ids.size(0)
            if pad_len > 0:
                input_ids = torch.cat(
                    [input_ids, input_ids.new_full((pad_len,), self.pad_token_id)]
                )
                labels = torch.cat([labels, labels.new_full((pad_len,), -100)])
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros((pad_len,))]
                )
            padded_input_ids.append(input_ids)
            padded_labels.append(labels)
            padded_attention_masks.append(attention_mask)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "labels": torch.stack(padded_labels),
        }
