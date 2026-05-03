from __future__ import annotations

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)

IMAGE_TOKEN = "<image>"
# vision_feature_select_strategy is set here and propagated to both
# LlavaConfig and LlavaProcessor so they never drift apart.
_VISION_FEATURE_SELECT_STRATEGY = "full"
DEFAULT_PROJECTOR_DTYPE = "float32"


def build_processor(vision_model_name: str, llm_model_name: str) -> LlavaProcessor:
    """
    Build a fully configured LlavaProcessor.

    patch_size and vision_feature_select_strategy are read from the vision
    model config so the processor inserts the correct number of <image> tokens.
    Without these, processor(text="<image>...", images=img) would insert only
    one token while the model expects num_patches tokens — causing a runtime
    shape mismatch in _merge_input_ids_with_image_features.
    """
    vision_root_config = AutoConfig.from_pretrained(vision_model_name)
    vision_config = getattr(vision_root_config, "vision_config", vision_root_config)
    patch_size = vision_config.patch_size  # 16 for siglip2-so400m-patch16-384

    image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
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
    )


def build_model(
    vision_model_name: str,
    llm_model_name: str,
    tokenizer_name_or_path: str | None = None,
    model_dtype: str | None = None,
    projector_dtype: str | None = DEFAULT_PROJECTOR_DTYPE,
    projector_state: dict | None = None,
) -> LlavaForConditionalGeneration:
    """
    Build LlavaForConditionalGeneration from pretrained SigLIP2 + Llama weights.

    projector_state: if provided, load into multi_modal_projector (stage-1 warm-start).
    The projector is otherwise randomly initialised.

    Note: build_processor() must be called with the same vision_model_name and
    llm_model_name so the processor's patch_size matches the model's vision config.
    """
    torch_dtype = _resolve_dtype(model_dtype)
    projector_torch_dtype = _resolve_dtype(projector_dtype) or torch.float32

    # Use build_processor to resolve patch_size and token IDs consistently.
    processor = build_processor(
        vision_model_name, tokenizer_name_or_path or llm_model_name
    )
    tokenizer = processor.tokenizer
    image_token_index = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

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

    _load_vision_weights(model, vision_model_name, torch_dtype)
    _load_llm_weights(model, llm_model_name, torch_dtype)

    # Resize after loading the pretrained LLM so existing token embeddings and
    # lm_head rows are preserved. The new <image> token is only a placeholder
    # that HF LLaVA replaces with projected image features during forward.
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    if projector_state is not None:
        model.multi_modal_projector.load_state_dict(projector_state)

    _cast_runtime_dtypes(model, torch_dtype, projector_torch_dtype)

    return model


def freeze_components(
    model: LlavaForConditionalGeneration,
    freeze_vision: bool,
    train_projector: bool,
    train_llm: bool,
) -> None:
    model.vision_tower.requires_grad_(not freeze_vision)
    model.multi_modal_projector.requires_grad_(train_projector)
    model.language_model.requires_grad_(train_llm)
    model.lm_head.requires_grad_(train_llm)

    set_component_modes(model, freeze_vision, train_projector, train_llm)


def set_component_modes(
    model: LlavaForConditionalGeneration,
    freeze_vision: bool,
    train_projector: bool,
    train_llm: bool,
) -> None:
    model.vision_tower.train(not freeze_vision)
    model.multi_modal_projector.train(train_projector)
    model.language_model.train(train_llm)
    model.lm_head.train(train_llm)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_dtype(dtype_name: str | None) -> torch.dtype | None:
    if not dtype_name or str(dtype_name).strip().lower() in {
        "",
        "auto",
        "none",
        "null",
    }:
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
    model: LlavaForConditionalGeneration,
    model_dtype: torch.dtype | None,
    projector_dtype: torch.dtype,
) -> None:
    if model_dtype is not None:
        model.vision_tower.to(model_dtype)
        model.language_model.to(model_dtype)
        model.lm_head.to(model_dtype)
    model.multi_modal_projector.to(projector_dtype)


def _load_vision_weights(
    model: LlavaForConditionalGeneration,
    vision_model_name: str,
    torch_dtype: torch.dtype | None,
) -> None:
    full_siglip = AutoModel.from_pretrained(
        vision_model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    ).vision_model
    model.vision_tower.load_state_dict(full_siglip.state_dict(), strict=False)
    del full_siglip


def _load_llm_weights(
    model: LlavaForConditionalGeneration,
    llm_model_name: str,
    torch_dtype: torch.dtype | None,
) -> None:
    llm = AutoModelForCausalLM.from_pretrained(
        llm_model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    # LlavaForConditionalGeneration.language_model is the bare LlamaModel.
    # AutoModelForCausalLM.state_dict() keys are prefixed with "model.", so
    # loading that dict directly into language_model silently leaves the LLM
    # randomly initialized when strict=False.
    model.language_model.load_state_dict(llm.model.state_dict(), strict=True)
    model.lm_head.load_state_dict(llm.lm_head.state_dict(), strict=True)
    del llm
