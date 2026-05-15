"""Model/processor construction utilities for LLaVA-style VLM training."""

from __future__ import annotations

from types import MethodType

import torch
from transformers import (
    AutoConfig, AutoImageProcessor, AutoModel, AutoModelForCausalLM,
    AutoTokenizer, LlavaConfig, LlavaForConditionalGeneration, LlavaProcessor,
)

IMAGE_TOKEN = "<image>"
# vision_feature_select_strategy is set here and propagated to both
# LlavaConfig and LlavaProcessor so they never drift apart.
_VISION_FEATURE_SELECT_STRATEGY = "full"
_NUM_ADDITIONAL_IMAGE_TOKENS = 0
DEFAULT_PROJECTOR_DTYPE = "float32"


def build_processor(vision_model_name: str, llm_model_name: str) -> LlavaProcessor:
    # patch_size must come from the vision model config so the processor inserts
    # the correct number of <image> tokens; otherwise shape mismatch at forward.
    vision_root_config = AutoConfig.from_pretrained(vision_model_name)
    vision_config = getattr(vision_root_config, "vision_config", vision_root_config)
    patch_size = vision_config.patch_size  # 16 for siglip2-so400m-patch16-384

    image_processor = AutoImageProcessor.from_pretrained(vision_model_name, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return LlavaProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=patch_size,
        vision_feature_select_strategy=_VISION_FEATURE_SELECT_STRATEGY,
        num_additional_image_tokens=_NUM_ADDITIONAL_IMAGE_TOKENS,
    )


def build_model(
    vision_model_name: str,
    llm_model_name: str,
    tokenizer_name_or_path: str | None = None,
    model_dtype: str | None = None,
    projector_dtype: str | None = DEFAULT_PROJECTOR_DTYPE,
    projector_state: dict | None = None,
    image_token_id: int | None = None,
    vocab_size: int | None = None,
) -> LlavaForConditionalGeneration:
    # image_token_id + vocab_size from caller's processor avoids a redundant build.
    dtype = _resolve_dtype(model_dtype)
    resolved_projector_dtype = _resolve_dtype(projector_dtype) or torch.float32

    if image_token_id is not None:
        if vocab_size is None:
            raise ValueError("vocab_size is required when image_token_id is provided")
        image_token_index = image_token_id
        _vocab_size = vocab_size
    else:
        _proc = build_processor(vision_model_name, tokenizer_name_or_path or llm_model_name)
        image_token_index = _proc.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        _vocab_size = len(_proc.tokenizer)

    vision_root_config = AutoConfig.from_pretrained(vision_model_name)
    vision_config = getattr(vision_root_config, "vision_config", vision_root_config)
    text_config = AutoConfig.from_pretrained(llm_model_name)

    llava_config = LlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        projector_hidden_act="gelu",
        vision_feature_select_strategy=_VISION_FEATURE_SELECT_STRATEGY,
        vision_feature_layer=-1,
        image_token_index=image_token_index,
    )

    model = LlavaForConditionalGeneration(llava_config)

    _load_vision_weights(model, vision_model_name, dtype)
    _load_llm_weights(model, llm_model_name, dtype)

    # Resize after loading the pretrained LLM so existing token embeddings and
    # lm_head rows are preserved. The new <image> token is only a placeholder
    # that HF LLaVA replaces with projected image features during forward.
    if _vocab_size != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(_vocab_size)

    _cast_runtime_dtypes(model, dtype, resolved_projector_dtype)
    if projector_state is not None:
        model.multi_modal_projector.load_state_dict(projector_state, strict=True)
    _patch_projector_input_dtype(model.multi_modal_projector)
    _patch_last_hidden_state_image_features(model)

    return model


def freeze_components(
    model: LlavaForConditionalGeneration, freeze_vision: bool, train_projector: bool, train_llm: bool
) -> None:
    model.vision_tower.requires_grad_(not freeze_vision)
    model.multi_modal_projector.requires_grad_(train_projector)
    model.language_model.requires_grad_(train_llm)
    model.lm_head.requires_grad_(train_llm)

    set_component_modes(model, freeze_vision, train_projector, train_llm)


def set_component_modes(
    model: LlavaForConditionalGeneration, freeze_vision: bool, train_projector: bool, train_llm: bool
) -> None:
    model.vision_tower.train(not freeze_vision)
    model.multi_modal_projector.train(train_projector)
    model.language_model.train(train_llm)
    model.lm_head.train(train_llm)


def _resolve_dtype(dtype_name: str | None) -> torch.dtype | None:
    if not dtype_name or str(dtype_name).strip().lower() in {"", "auto", "none", "null"}:
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(dtype_name).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported model_dtype '{dtype_name}'.")
    return mapping[key]


def _cast_runtime_dtypes(
    model: LlavaForConditionalGeneration, model_dtype: torch.dtype | None, projector_dtype: torch.dtype
) -> None:
    if model_dtype is not None:
        model.vision_tower.to(model_dtype)
        model.language_model.to(model_dtype)
        model.lm_head.to(model_dtype)
    model.multi_modal_projector.to(projector_dtype)


def _patch_projector_input_dtype(projector) -> None:
    # Vision tower (bf16) feeds directly into projector (fp32 during stage-1).
    # Cast input to projector weight dtype to avoid Linear dtype mismatch.
    def forward(self, image_features):
        projector_dtype = self.linear_1.weight.dtype
        if image_features.dtype != projector_dtype:
            image_features = image_features.to(projector_dtype)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        return self.linear_2(hidden_states)

    projector.forward = MethodType(forward, projector)


def _patch_last_hidden_state_image_features(model: LlavaForConditionalGeneration) -> None:
    # Stock HF LLaVA passes output_hidden_states=True, retaining all 48 intermediate
    # activations (~hundreds MB VRAM). For vision_feature_layer=-1, hidden_states[-1]
    # equals last_hidden_state, so use output_hidden_states=False instead.
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer=None,
        vision_feature_select_strategy: str | None = None,
        **kwargs,
    ):
        del vision_feature_layer  # always use last_hidden_state regardless of layer index
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        if vision_feature_select_strategy not in {"default", "full"}:
            raise ValueError(
                f"Unexpected select feature strategy: {vision_feature_select_strategy}"
            )

        image_sizes = kwargs.pop("image_sizes", None)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        image_outputs = self.vision_tower(
            pixel_values,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )
        selected_image_feature = image_outputs.last_hidden_state
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]

        image_features = self.multi_modal_projector(selected_image_feature)
        if image_sizes is not None:
            patch_size = getattr(
                self.vision_tower,
                "patch_size",
                self.config.vision_config.patch_size,
            )
            if isinstance(image_sizes, torch.Tensor):
                image_sizes = image_sizes.tolist()
            split_sizes = [
                (int(height) // patch_size) * (int(width) // patch_size)
                for height, width in image_sizes
            ]
            image_features = torch.split(image_features.squeeze(0), split_sizes)
        else:
            image_features = list(image_features)
        return image_features

    model.model.get_image_features = MethodType(get_image_features, model.model)


def _load_vision_weights(
    model: LlavaForConditionalGeneration, vision_model_name: str, dtype: torch.dtype | None
) -> None:
    full_siglip = AutoModel.from_pretrained(
        vision_model_name, dtype=dtype, low_cpu_mem_usage=True
    ).vision_model
    target = model.vision_tower
    if hasattr(target, "vision_model"):
        target = target.vision_model
    target.load_state_dict(full_siglip.state_dict(), strict=True)
    del full_siglip


def _load_llm_weights(
    model: LlavaForConditionalGeneration, llm_model_name: str, dtype: torch.dtype | None
) -> None:
    llm = AutoModelForCausalLM.from_pretrained(
        llm_model_name, dtype=dtype, low_cpu_mem_usage=True
    )
    # LlavaForConditionalGeneration.language_model is the bare LlamaModel.
    # AutoModelForCausalLM.state_dict() keys are prefixed with "model.", so
    # loading that dict directly into language_model silently leaves the LLM
    # randomly initialized when strict=False.
    model.language_model.load_state_dict(llm.model.state_dict(), strict=True)
    model.lm_head.load_state_dict(llm.lm_head.state_dict(), strict=True)
    del llm
