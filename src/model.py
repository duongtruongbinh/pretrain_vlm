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


def build_processor(vision_model_name: str, llm_model_name: str) -> LlavaProcessor:
    """
    Build a fully configured LlavaProcessor.

    patch_size and vision_feature_select_strategy are read from the vision
    model config so the processor inserts the correct number of <image> tokens.
    Without these, processor(text="<image>...", images=img) would insert only
    one token while the model expects num_patches tokens — causing a runtime
    shape mismatch in _merge_input_ids_with_image_features.
    """
    siglip_config = AutoConfig.from_pretrained(vision_model_name)
    patch_size = siglip_config.vision_config.patch_size  # 16 for siglip2-so400m-patch16-384

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
    model_dtype: str | None = None,
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

    # Use build_processor to resolve patch_size and token IDs consistently.
    processor = build_processor(vision_model_name, llm_model_name)
    tokenizer = processor.tokenizer
    image_token_index = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    siglip_config = AutoConfig.from_pretrained(vision_model_name)
    vision_config = siglip_config.vision_config
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
    # Resize to accommodate the added <image> token.
    model.resize_token_embeddings(len(tokenizer))

    _load_vision_weights(model, vision_model_name, torch_dtype)
    _load_llm_weights(model, llm_model_name, torch_dtype)

    if projector_state is not None:
        model.multi_modal_projector.load_state_dict(projector_state)

    if torch_dtype is not None:
        model = model.to(torch_dtype)

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

    if freeze_vision:
        model.vision_tower.eval()
    if not train_projector:
        model.multi_modal_projector.eval()
    if not train_llm:
        model.language_model.eval()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_dtype(dtype_name: str | None) -> torch.dtype | None:
    if not dtype_name or str(dtype_name).strip().lower() in {"", "auto", "none", "null"}:
        return None
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    key = str(dtype_name).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported model_dtype '{dtype_name}'.")
    return mapping[key]


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
    model.language_model.load_state_dict(llm.state_dict(), strict=False)
    del llm
