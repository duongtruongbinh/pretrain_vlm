# Vietnamese VLM Pretraining

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.55%2B-yellow)](https://huggingface.co/docs/transformers)
[![Accelerate](https://img.shields.io/badge/🤗_Accelerate-1.12%2B-orange)](https://huggingface.co/docs/accelerate)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A two-stage LLaVA-style pretraining pipeline for Vietnamese Vision-Language Models.

**Architecture:** [SigLIP2-so400m-patch16-384](https://huggingface.co/google/siglip2-so400m-patch16-384) → MLP projector → [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), assembled via HuggingFace [`LlavaForConditionalGeneration`](https://huggingface.co/docs/transformers/model_doc/llava).

---

## Table of Contents

- [Overview](#overview)
- [Training Stages](#training-stages)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Streamlit Demos](#streamlit-demos)
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
├── pyproject.toml
├── demos/
│   ├── stage1.py               # Streamlit demo: Stage 1 captioning
│   ├── instruction.py          # Streamlit demo: Stage 2 instruction chat
│   └── _utils.py               # Shared demo utilities
├── scripts/
│   ├── prepare_data.py         # Data prep entrypoint (subcommands per dataset)
│   ├── evaluate.py             # Benchmark eval entrypoint (subcommands per task)
│   ├── download_benchmarks.py  # Download KTVIC + Vista conversation datasets
│   ├── prepare_uit_openviic.py
│   ├── prepare_coco_data.py
│   ├── prepare_instruction_common.py
│   ├── prepare_vietnamtourism_data.py
│   ├── generate_qa_vietnamtourism.py
│   ├── crawl_vietnamtourism.py
│   ├── train_stage1.sh
│   └── train_instruction.sh
└── src/
    ├── modeling.py             # Model & processor construction
    ├── collators.py            # CaptionCollator / InstructionCollator
    ├── data.py                 # ImageCaptionDataset / ImageInstructionDataset
    ├── runtime.py              # Seed, logging, samplers, config loading, Jinja2 render
    ├── training.py             # Token-weighted training loop, checkpointing, eval
    ├── inference.py            # Model loading, generation, IO helpers
    ├── metrics.py              # Corpus-level caption and VQA metrics
    └── prompts/                # Jinja2 prompt templates
        ├── caption_prompt.j2
        ├── vqa_system.j2
        ├── vqa_question.j2
        ├── qa_gen_system.j2
        └── qa_gen_instruction.j2
```

---

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

`uv sync` installs all dependencies and the `src` package in editable mode. All commands are run via `uv run` — no manual venv activation needed.

**HuggingFace access** — both models require accepting their license and authenticating:

```bash
uv run huggingface-cli login
```

---

## Data Preparation

Run scripts in order. All paths are read from `config.yaml`.

```bash
# Stage 1 data
uv run python scripts/prepare_data.py uit
uv run python scripts/prepare_data.py coco

# Stage 2 data
uv run python scripts/prepare_data.py sharegpt
uv run python scripts/prepare_data.py 5cd       # optional; requires HF access approval

# VietnamTourism data (optional)
uv run python scripts/prepare_data.py vietnamtourism-crawl
uv run python scripts/prepare_data.py vietnamtourism-qa      # uses OpenAI Batch API
uv run python scripts/prepare_data.py vietnamtourism-prepare
```

Stage 1 expects `image` + `caption` fields. Stage 2 expects `image` + `messages` (OpenAI chat format, must end with an assistant turn).

---

## Training

### Stage 1

```bash
# Single GPU
uv run accelerate launch --num_processes 1 train.py --config-section train

# Multi-GPU (edit CUDA_VISIBLE_DEVICES and NUM_PROCESSES in the script as needed)
bash scripts/train_stage1.sh

# Single GPU via shell script
NUM_PROCESSES=1 CUDA_VISIBLE_DEVICES=0 bash scripts/train_stage1.sh

# Resume
uv run accelerate launch train.py --config-section train --resume-from outputs/run1/checkpoint-500
```

### Stage 2

```bash
# Fresh run (warm-starts projector from stage1_projector_ckpt in config.yaml)
uv run accelerate launch train_instruction.py

# Resume
uv run accelerate launch train_instruction.py --resume-from outputs/instruction_run1/checkpoint-1000
```

---

## Evaluation

```bash
# Download benchmarks first
uv run python scripts/download_benchmarks.py --output-root data/benchmarks

# Stage 1 — KTVIC captioning benchmark
uv run python scripts/evaluate.py ktvic \
  --annotations data/benchmarks/ktvic/raw/ktvic_dataset/test_data.json \
  --image-root data/benchmarks/ktvic/raw/ktvic_dataset/public-test-images \
  --checkpoint outputs/stage1/checkpoint-2500

# Stage 2 — Vista conversation
uv run python scripts/evaluate.py vista-conversation \
  --annotations data/benchmarks/vista/data/vi_llava_conversation/validation-00000-of-00001.parquet \
  --image-root data/benchmarks/vista/images/coco2017/val2017 \
  --checkpoint outputs/instruction_run1/checkpoint-1000
```

---

## Streamlit Demos

Run from the project root:

```bash
# Stage 1 projector demo (captioning + projector scale diagnostics)
uv run streamlit run demos/stage1.py

# Stage 2 instruction demo (multi-turn chat)
uv run streamlit run demos/instruction.py
```

Select checkpoint, device, and generation parameters in the sidebar. Both demos load eval samples automatically if the configured JSONL paths exist.

---

## Configuration

All configuration lives in `config.yaml`.

| Section | Used by |
|---|---|
| `train` | Stage 1 (COCO + UIT-OpenViIC) |
| `train_uit_only` | Stage 1 (UIT-OpenViIC only) |
| `instruction_train` | Stage 2 |
| `prepare_data` | `scripts/prepare_uit_openviic.py` |
| `prepare_coco` | `scripts/prepare_coco_data.py` |
| `instruction_data_gpt` | `scripts/prepare_instruction_viet_sharegpt.py` |
| `instruction_data_5cd` | `scripts/prepare_instruction_5cd_localization.py` |
| `prepare_vietnamtourism` | `scripts/prepare_vietnamtourism_data.py` |
| `generate_qa_vietnamtourism` | `scripts/generate_qa_vietnamtourism.py` |

<details>
<summary><strong>Key Stage 1 fields</strong></summary>

```yaml
train:
  vision_model: google/siglip2-so400m-patch16-384
  llm_model: meta-llama/Llama-3.2-1B-Instruct
  model_dtype: bfloat16
  projector_dtype: float32
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
  stage1_projector_ckpt: outputs/run1/checkpoint-2500
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

Each checkpoint is saved under `output_dir/checkpoint-{step}/`:

```
checkpoint-500/
├── projector.pt          # MLP projector weights
├── optimizer.pt
├── scheduler.pt
├── rng_state.pt
├── trainer_state.json    # step, epoch, best_eval_loss
├── training_config.yaml  # frozen copy of config at save time
├── model_config/         # LlavaConfig (HF format)
├── processor/
├── tokenizer/
└── llm/                  # LLM weights (Stage 2 only, safetensors)
```

`best_checkpoint.json` and `last_checkpoint.json` are written to `output_dir/` as pointers. Old checkpoints are rotated — `keep_last_n` controls how many are retained (best and last are always kept).
