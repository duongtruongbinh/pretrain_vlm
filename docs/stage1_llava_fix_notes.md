# Stage 1 LLAVA Fix Notes

## Summary

The main issue was not the sampler, normalization, or learning rate. The issue was in the pretrained SigLIP loading path for the vision tower.

The old code loaded the SigLIP state dict into `model.vision_tower` with `strict=False`. That allowed PyTorch to ignore mismatched keys. The source SigLIP checkpoint uses keys like `embeddings...` and `encoder...`, while the HF LLaVA vision tower keeps those weights under the nested `vision_model` module. As a result, the pretrained weights were not copied into the right module, but training still continued without an error.

The vision tower was then frozen, so the projector learned from fixed random image features. The model could still produce different outputs for a real image and a black image, but that difference did not carry image semantics. This explains why predictions often repeated generic captions about people, bicycles, trucks, or stores instead of matching the actual object in the image.

## How the issue was verified

The old loading path produced `448` missing keys, `448` unexpected keys, and `0` matching keys. That means the pretrained vision weights were effectively not loaded into the active vision tower.

After the fix, tensors from `model.vision_tower.vision_model` were compared directly with the pretrained SigLIP source. Key tensors such as `embeddings.patch_embedding.weight`, `encoder.layers.0.self_attn.k_proj.weight`, `encoder.layers.0.layer_norm1.weight`, and `post_layernorm.weight` all had `max_abs_diff` equal to `0.0`. This confirms that the current vision tower uses the pretrained SigLIP weights.

## Code changes

`_load_vision_weights()` now loads into `model.vision_tower.vision_model` when that nested module exists, then calls `load_state_dict()` with `strict=True`. If a key or shape mismatch happens, the program stops immediately instead of training with the wrong backbone.

HF LLaVA normally selects features from the vision tower hidden states list. The local `vlm_pretrain` baseline uses `last_hidden_state`. To keep both implementations on the same feature type, the code now patches `get_image_features()` so it passes `image_outputs.last_hidden_state` into the projector.

The projector is kept in `float32` because it is the trainable module in stage 1. The vision tower and LLM stay in `bfloat16` because both are frozen. Full model `float32` is not needed for this stage, costs more VRAM, and does not address the root cause.

## New training config

The full COCO plus UIT run writes to `outputs/stage_1_projector_coco_uit_balanced_fixed_last_hidden_bf16_projfp32_lr1e3`.

The UIT only run writes to `outputs/stage_1_projector_uit_only_fixed_last_hidden_bf16_projfp32_lr1e3`.

Both baselines use learning rate `0.001`, `projector_norm=none`, `model_dtype=bfloat16`, and `projector_dtype=float32`. The goal of this baseline is to confirm that the model learns from real image features after the vision tower loading fix. RMSNorm or LayerNorm should be tested later as separate ablations once this clean baseline is working.

## Old checkpoints

Old LLAVA stage 1 checkpoints should not be resumed. Those checkpoints trained the projector on features from a vision tower that had not loaded the pretrained weights correctly. The projector learned against the wrong feature distribution, so continuing from those checkpoints would make the result hard to interpret and hard to compare.

New runs should start from scratch with the fixed code and new output directories.
