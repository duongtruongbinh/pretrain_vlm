# 🦙🔭 Vietnamese VLM Pretraining

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.55%2B-yellow)](https://huggingface.co/docs/transformers)
[![Accelerate](https://img.shields.io/badge/🤗_Accelerate-1.12%2B-orange)](https://huggingface.co/docs/accelerate)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A lightweight, two-stage LLaVA-style training pipeline for Vietnamese Vision-Language Models.

**Architecture:** [SigLIP2-so400m-patch16-384](https://huggingface.co/google/siglip2-so400m-patch16-384) → MLP projector → [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), assembled via HuggingFace [`LlavaForConditionalGeneration`](https://huggingface.co/docs/transformers/model_doc/llava).

---

## Table of Contents

- [Overview](#overview)
- [Training Stages](#training-stages)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Configuration](#configuration)
- [Checkpoints](#checkpoints)

---

## Overview

| Component | Details |
|---|---|
| Vision encoder | `google/siglip2-so400m-patch16-384` · 576 tokens/image · 48 layers |
| Language model | `meta-llama/Llama-3.2-1B-Instruct` |
| Stage 1 data | COCO 2017 (Vietnamese captions) + UIT-OpenViIC |
| Stage 2 data | Viet-ShareGPT-4o-Text-VQA; optional 5CD-AI/Viet-Localization-VQA |
| Precision | BF16 (vision + LLM) · FP32 projector (Stage 1) · BF16 all (Stage 2) |
| Target hardware | Single/multi GPU · 24 GB VRAM per device |

**Key design choices:**

- Token-weighted cross-entropy loss — correct gradient accumulation across variable-length sequences and multi-GPU setups
- `output_hidden_states=False` on the vision tower — avoids storing all 48 intermediate hidden states (~hundreds of MB peak VRAM savings for SigLIP2-so400m-384)
- FP32 projector during Stage 1 — stable alignment when vision tower runs BF16 and is frozen; BF16 from Stage 2 onward
- `WeightedRandomSampler` in Stage 1 — equal per-source contribution when mixing multiple caption datasets
- HuggingFace `LlavaProcessor` handles `<image>` → 576-token expansion; no manual token counting

---

## Training Stages

### Stage 1 — Projector Alignment

Train only the MLP projector. Vision tower and LLM are frozen.

| Setting | Value |
|---|---|
| Trainable | Projector only |
| Optimizer | AdamW, lr=1e-3, no weight decay |
| Batch | 1 per GPU · grad_accum=32 · effective=64 (2 GPUs) |
| Schedule | Cosine with 3% warmup |
| Epochs | 2 |

### Stage 2 — Instruction Tuning

Fine-tune projector + LLM jointly. Vision tower remains frozen.

| Setting | Value |
|---|---|
| Trainable | Projector + LLM + lm_head |
| Optimizer | Adafactor · projector_lr=1e-4 · llm_lr=2e-5 |
| Batch | 1 per GPU · grad_accum=16 |
| Max text tokens | 1024 |
| Gradient checkpointing | Enabled |
| Schedule | Cosine with 3% warmup |
| Epochs | 1 |

---

## Project Structure

```
pretrain_vlm/
├── train.py                    # Stage 1 entrypoint
├── train_instruction.py        # Stage 2 entrypoint
├── config.yaml                 # All training + data configs
├── scripts/
│   ├── prepare_data.py         # Download & process UIT-OpenViIC
│   ├── prepare_coco_data.py    # Process COCO 2017 (Vietnamese captions)
│   ├── prepare_instruction_viet_sharegpt.py  # Process Viet-ShareGPT-4o-Text-VQA
│   ├── prepare_instruction_5cd_localization.py  # Process 5CD Localization VQA
│   └── train_stage1.sh         # Multi-GPU Stage 1 launcher
└── src/
    ├── modeling.py             # Model & processor construction
    ├── collators.py            # CaptionCollator / InstructionCollator
    ├── data.py                 # ImageCaptionDataset / ImageInstructionDataset
    ├── runtime.py              # Seed, logging, samplers, config loading
    └── training/
        ├── engine.py           # Token-weighted training loop
        ├── checkpoint.py       # Save/load/rotate checkpoints
        └── eval.py             # Evaluation loop
```

---

## Setup

<details>
<summary><strong>Option A — uv (recommended)</strong></summary>

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create env and install dependencies
uv sync

# Activate (optional — uv run works without activation)
source .venv/bin/activate
```

Run any command with `uv run python ...` or activate the venv first.

</details>

<details>
<summary><strong>Option B — pip + requirements.txt</strong></summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

</details>

**HuggingFace access** — both models require accepting their license on HuggingFace Hub and authenticating:

```bash
huggingface-cli login
```

---

## Data Preparation

Run scripts in order. All paths and dataset names are read from `config.yaml`.

```bash
# 1. UIT-OpenViIC (Vietnamese image captioning — downloads from Google Drive)
python scripts/prepare_data.py

# 2. COCO 2017 with Vietnamese captions (streams from HuggingFace Hub)
python scripts/prepare_coco_data.py

# 3a. Viet-ShareGPT-4o-Text-VQA (instruction tuning — from HuggingFace Hub)
python scripts/prepare_instruction_viet_sharegpt.py

# 3b. 5CD-AI/Viet-Localization-VQA (optional instruction data; requires HF access approval)
python scripts/prepare_instruction_5cd_localization.py
```

Each script produces JSONL files under `data/`. Stage 1 expects `image` + `caption` fields; Stage 2 expects `image` + `messages` (OpenAI chat format, must end with an assistant turn).

---

## Training

### Stage 1

**Single GPU:**

```bash
accelerate launch --num_processes 1 train.py --config-section train
```

**Multi-GPU (via script):**

```bash
# Defaults: 2 GPUs, BF16, config-section=train
bash scripts/train_stage1.sh

# Override GPU count or config section
NUM_PROCESSES=4 CONFIG_SECTION=train_uit_only bash scripts/train_stage1.sh
```

**Resume from checkpoint:**

```bash
accelerate launch train.py --config-section train --resume-from outputs/run1/checkpoint-500
```

### Stage 2

```bash
# Fresh run (warm-starts projector from stage1_projector_ckpt in config.yaml)
accelerate launch train_instruction.py

# Resume
accelerate launch train_instruction.py --resume-from outputs/instruction_run1/checkpoint-1000
```

---

## Configuration

All configuration lives in `config.yaml`. Relevant sections:

| Section | Used by |
|---|---|
| `train` | Stage 1 (COCO + UIT-OpenViIC) |
| `train_uit_only` | Stage 1 (UIT-OpenViIC only) |
| `instruction_train` | Stage 2 |
| `prepare_data` | `scripts/prepare_data.py` |
| `prepare_coco` | `scripts/prepare_coco_data.py` |
| `instruction_data_gpt` | `scripts/prepare_instruction_viet_sharegpt.py` |
| `instruction_data_5cd` | `scripts/prepare_instruction_5cd_localization.py` |

<details>
<summary><strong>Key Stage 1 fields</strong></summary>

```yaml
train:
  vision_model: google/siglip2-so400m-patch16-384
  llm_model: meta-llama/Llama-3.2-1B-Instruct
  model_dtype: bfloat16      # vision tower + LLM precision
  projector_dtype: float32   # projector precision (stable alignment)
  lr: 1.0e-3
  epochs: 2
  batch_size: 1
  grad_accum: 32
  warmup_ratio: 0.03
  output_dir: outputs/stage_1_...
```

</details>

<details>
<summary><strong>Key Stage 2 fields</strong></summary>

```yaml
instruction_train:
  stage1_projector_ckpt: outputs/run1/checkpoint-2500  # warm-start source
  freeze_vision: true
  train_projector: true
  train_llm: true
  optimizer_type: adafactor
  projector_lr: 1.0e-4
  llm_lr: 2.0e-5
  max_text_tokens: 1024
  gradient_checkpointing: true
  mixed_precision: bf16
```

</details>

---

## Checkpoints

Each checkpoint is a directory saved under `output_dir/checkpoint-{step}/`:

```
checkpoint-500/
├── projector.pt          # MLP projector weights
├── optimizer.pt          # Optimizer state
├── scheduler.pt          # LR scheduler state
├── rng_state.pt          # RNG state for exact resume
├── trainer_state.json    # step, epoch, best_eval_loss
├── training_config.yaml  # frozen copy of config at save time
├── model_config/         # LlavaConfig (HF format)
├── processor/            # LlavaProcessor
├── tokenizer/            # tokenizer files
└── llm/                  # LLM weights (Stage 2 only, safetensors)
```

`best_checkpoint.json` and `last_checkpoint.json` are written to `output_dir/` as pointers. Old checkpoints are automatically rotated — `keep_last_n` controls how many are retained (best and last are always protected).
