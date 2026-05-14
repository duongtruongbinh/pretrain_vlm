# LLAVA - Vietnamese Vision-Language Model Training

Dự án này xây dựng pipeline huấn luyện mô hình Vision-Language tiếng Việt theo phong cách LLaVA. Mục tiêu là kết hợp mô hình nhìn ảnh, projector đa phương thức và mô hình ngôn ngữ để tạo một trợ lý có thể hiểu ảnh, mô tả ảnh và trả lời câu hỏi bằng tiếng Việt.

## Tổng Quan

Pipeline sử dụng kiến trúc LLaVA gồm ba phần chính:

| Thành phần | Mô tả |
| --- | --- |
| Vision encoder | `google/siglip2-so400m-patch16-384`, dùng để trích xuất đặc trưng ảnh |
| Multimodal projector | MLP projector, nối đặc trưng ảnh sang không gian embedding của LLM |
| Language model | `meta-llama/Llama-3.2-1B-Instruct`, sinh câu trả lời tiếng Việt |
| Framework | PyTorch, Hugging Face Transformers, Accelerate |

Mô hình được tạo bằng `LlavaForConditionalGeneration`. `LlavaProcessor` xử lý ảnh, text, token `<image>` và tokenizer để đưa dữ liệu vào model đúng format.

## Mục Tiêu Dự Án

- Huấn luyện một mô hình VLM tiếng Việt theo pipeline hai giai đoạn.
- Tận dụng pretrained SigLIP2 và Llama 3.2 thay vì train từ đầu.
- Chuẩn hóa nhiều bộ dữ liệu ảnh - văn bản tiếng Việt về cùng format JSONL.
- Hỗ trợ captioning, OCR/VQA, mô tả ảnh, hỏi đáp theo ảnh và hội thoại đa lượt.
- Có script test checkpoint bằng Streamlit để kiểm tra nhanh chất lượng mô hình.

## Pipeline Huấn Luyện

### Stage 1 - Projector Alignment

Stage 1 chỉ train `multi_modal_projector`. Vision encoder và LLM được đóng băng.

Mục đích của giai đoạn này là giúp projector học cách biến đặc trưng ảnh thành embedding mà Llama có thể hiểu. Dữ liệu chính là các cặp ảnh + caption tiếng Việt.

Thiết lập chính:

- Entry point: `train.py`
- Dataset: COCO 2017 caption tiếng Việt, UIT-OpenViIC
- Trainable: projector
- Frozen: vision encoder, LLM, `lm_head`
- Optimizer: AdamW
- Loss: token-weighted cross entropy
- Output: checkpoint projector cho Stage 2

### Stage 2 - Instruction Tuning

Stage 2 dùng projector từ Stage 1 làm warm-start, sau đó fine-tune model trên dữ liệu instruction/hội thoại có ảnh.

Mục đích của giai đoạn này là giúp model trả lời câu hỏi, mô tả chi tiết, đọc thông tin trong ảnh và hội thoại theo ngữ cảnh ảnh.

Thiết lập chính:

- Entry point: `train_instruction.py`
- Dataset: Viet-ShareGPT-4o-Text-VQA, Viet-Localization-VQA, VietnamTourism
- Trainable: projector, LLM, `lm_head`
- Frozen: vision encoder
- Optimizer: Adafactor hoặc AdamW
- Loss: chỉ supervise token thuộc câu trả lời của assistant
- Output: checkpoint đầy đủ gồm projector và LLM đã fine-tune

## Dữ Liệu

Dữ liệu được lưu trong `data/` và được chuẩn hóa về JSONL.

### Dữ liệu Stage 1

Stage 1 dùng record dạng:

```json
{"image": "path/to/image.jpg", "caption": "Mô tả ảnh bằng tiếng Việt"}
```

Nguồn dữ liệu:

- `data/coco2017/`: COCO 2017 có caption tiếng Việt.
- `data/uit-openviic/`: UIT-OpenViIC cho image captioning tiếng Việt.

### Dữ liệu Stage 2

Stage 2 dùng record dạng:

```json
{
  "id": "sample-id",
  "image": "path/to/image.jpg",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Nguồn dữ liệu:

- `data/viet-sharegpt-4o-text-vqa/`: dữ liệu OCR/VQA và mô tả ảnh.
- `data/viet-localization-vqa/`: dữ liệu hỏi đáp/định vị nội dung trong ảnh.
- `data/vietnamtourism/`: dữ liệu du lịch Việt Nam, tạo từ ảnh và metadata crawl từ `vietnamtourism.gov.vn`.
- `data/vietnamtourism-raw/`: dữ liệu crawl thô, batch input/output và kết quả sinh hội thoại.

## Cấu Trúc Dự Án

```text
LLAVA/
├── README.md                         # Tài liệu mô tả dự án
├── config.yaml                       # Cấu hình model, data, training
├── pyproject.toml                    # Khai báo package và dependencies cho uv
├── requirements.txt                  # Dependencies dạng pip
├── train.py                          # Stage 1 - projector alignment
├── train_instruction.py              # Stage 2 - instruction tuning
├── streamlit_stage1_test.py          # Giao diện test checkpoint Stage 1
├── streamlit_instruction_test.py     # Giao diện test checkpoint Stage 2
├── scripts/                          # Script chuẩn bị dữ liệu và chạy training
├── src/                              # Source code lõi
├── data/                             # Dữ liệu đã chuẩn bị
├── outputs/                          # Checkpoint, logs, metrics
├── docs/                             # Ghi chú kỹ thuật
└── tests/                            # Unit tests
```

## Các File Chính

### `config.yaml`

File cấu hình trung tâm của dự án. Các section quan trọng:

- `prepare_data`: tải và xử lý UIT-OpenViIC.
- `prepare_coco`: xử lý COCO 2017.
- `train`: cấu hình Stage 1 với COCO + UIT.
- `train_uit_only`: cấu hình Stage 1 chỉ dùng UIT.
- `instruction_data_gpt`: chuẩn bị Viet-ShareGPT-4o-Text-VQA.
- `instruction_data_5cd`: chuẩn bị Viet-Localization-VQA.
- `instruction_train`: cấu hình Stage 2.
- `crawl_vietnamtourism`: crawl ảnh và metadata từ VietnamTourism.
- `generate_qa_vietnamtourism`: sinh hội thoại bằng OpenAI Batch API.
- `prepare_vietnamtourism`: chuyển dữ liệu raw sang JSONL train/val/test.

### `train.py`

Entry point cho Stage 1.

Luồng xử lý chính:

1. Đọc config từ `config.yaml`.
2. Load dataset caption bằng `ImageCaptionDataset`.
3. Tạo batch bằng `CaptionCollator`.
4. Build model LLaVA bằng `build_model()`.
5. Đóng băng vision tower và LLM.
6. Chỉ train `multi_modal_projector`.
7. Chạy training loop bằng `run_training()`.
8. Eval định kỳ bằng `run_evaluation()`.
9. Lưu checkpoint projector, processor, tokenizer, optimizer, scheduler và trainer state.

Chạy trực tiếp:

```bash
accelerate launch train.py --config-section train
```

Chạy với cấu hình UIT-only:

```bash
accelerate launch train.py --config-section train_uit_only
```

Resume:

```bash
accelerate launch train.py --config-section train --resume-from outputs/.../checkpoint-500
```

### `train_instruction.py`

Entry point cho Stage 2.

Luồng xử lý chính:

1. Đọc config từ section `instruction_train`.
2. Load dataset instruction bằng `ImageInstructionDataset`.
3. Tạo batch bằng `InstructionCollator`.
4. Build model LLaVA.
5. Load projector từ `stage1_projector_ckpt` nếu train mới.
6. Nếu resume, load lại LLM/tokenizer từ checkpoint Stage 2.
7. Đóng băng vision tower, train projector + LLM + `lm_head`.
8. Dùng weighted sampler nếu có `sample_weights`.
9. Eval định kỳ và log sample generation.
10. Lưu checkpoint đầy đủ gồm projector, LLM, tokenizer, processor và trạng thái train.

Chạy trực tiếp:

```bash
accelerate launch train_instruction.py
```

Resume:

```bash
accelerate launch train_instruction.py --resume-from outputs/stage_2_instruction_final/checkpoint-1000
```

### `streamlit_stage1_test.py`

Giao diện Streamlit để test checkpoint Stage 1. App cho phép chọn checkpoint projector, chọn ảnh, nhập prompt caption và sinh mô tả ảnh.

Chạy:

```bash
streamlit run streamlit_stage1_test.py
```

### `streamlit_instruction_test.py`

Giao diện Streamlit để test checkpoint Stage 2. App hỗ trợ hội thoại theo ảnh, chọn checkpoint instruction và sinh câu trả lời từ model đã fine-tune.

Chạy:

```bash
streamlit run streamlit_instruction_test.py
```

## Thư Mục `src/`

`src/` chứa phần code lõi dùng chung cho training, eval và app test.

### `src/modeling.py`

Tạo processor và model LLaVA.

Nhiệm vụ chính:

- Tạo `LlavaProcessor` từ image processor của SigLIP2 và tokenizer của Llama.
- Thêm token đặc biệt `<image>`.
- Tạo `LlavaForConditionalGeneration`.
- Load pretrained weights cho vision tower và LLM.
- Resize token embeddings sau khi thêm `<image>`.
- Patch projector để xử lý khác dtype giữa vision features và projector.
- Patch cách lấy image features để dùng `last_hidden_state` và giảm VRAM.

### `src/data.py`

Định nghĩa dataset.

- `ImageCaptionDataset`: đọc JSONL dạng ảnh + caption cho Stage 1.
- `ImageInstructionDataset`: đọc JSONL dạng ảnh + messages cho Stage 2.
- Có kiểm tra format messages và bỏ qua ảnh lỗi/corrupt khi load.

### `src/collators.py`

Tạo batch tensor cho model.

- `CaptionCollator`: tạo prompt caption, encode ảnh/text và mask phần prompt để chỉ train caption.
- `InstructionCollator`: áp chat template, chèn `<image>`, truncate text và chỉ supervise token thuộc assistant.

### `src/runtime.py`

Các tiện ích dùng chung:

- Đọc config.
- Resolve đường dẫn dữ liệu.
- Set seed.
- Setup logger.
- Tạo sampler.
- Ghi JSONL metrics.

### `src/training/engine.py`

Training loop chung cho cả hai stage.

Điểm quan trọng là loss được tính theo số supervised token, giúp ổn định khi batch có sequence dài/ngắn khác nhau và khi chạy multi-GPU.

### `src/training/eval.py`

Tính eval loss theo supervised token và log sample generation nếu được truyền callback.

### `src/training/checkpoint.py`

Quản lý checkpoint.

Chức năng chính:

- Save/load projector checkpoint.
- Save/load full checkpoint cho Stage 2.
- Lưu tokenizer, processor, model config, optimizer, scheduler, RNG state.
- Ghi `best_checkpoint.json` và `last_checkpoint.json`.
- Xoay vòng checkpoint cũ bằng `keep_last_n`.

## Thư Mục `scripts/`

### Script chuẩn bị dữ liệu

- `prepare_uit_openviic.py`: tải và chuyển UIT-OpenViIC sang JSONL.
- `prepare_coco_data.py`: xử lý COCO 2017 từ Hugging Face Hub.
- `prepare_instruction_common.py`: logic chung để convert dataset instruction.
- `prepare_instruction_viet_sharegpt.py`: xử lý Viet-ShareGPT-4o-Text-VQA.
- `prepare_instruction_5cd_localization.py`: xử lý Viet-Localization-VQA.
- `prepare_instruction_data.py`: entrypoint tương thích cũ cho instruction data.
- `crawl_vietnamtourism.py`: crawl ảnh và metadata từ VietnamTourism.
- `generate_qa_vietnamtourism.py`: sinh mô tả/hội thoại từ ảnh bằng OpenAI Batch API.
- `prepare_vietnamtourism_data.py`: chuyển dữ liệu VietnamTourism raw sang train/val/test JSONL.

### Script chạy training

- `train_stage1.sh`: launcher multi-GPU cho Stage 1, mặc định dùng `train`.
- `train_stage1_uit_card2.sh`: launcher một GPU cho Stage 1, mặc định dùng `train_uit_only`.
- `train_instruction.sh`: launcher multi-GPU cho Stage 2.

Các launcher thiết lập GPU, NCCL env, mixed precision và gọi `accelerate launch`.

## Luồng Chạy Đề Xuất

### 1. Cài môi trường

```bash
uv sync
source .venv/bin/activate
```

Hoặc dùng pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu dùng model từ Hugging Face cần login trước:

```bash
huggingface-cli login
```

### 2. Chuẩn bị dữ liệu Stage 1

```bash
python scripts/prepare_uit_openviic.py
python scripts/prepare_coco_data.py
```

### 3. Train Stage 1

```bash
bash scripts/train_stage1.sh
```

Hoặc chỉ train UIT:

```bash
CONFIG_SECTION=train_uit_only bash scripts/train_stage1.sh
```

### 4. Chuẩn bị dữ liệu Stage 2

```bash
python scripts/prepare_instruction_viet_sharegpt.py
python scripts/prepare_instruction_5cd_localization.py
python scripts/prepare_vietnamtourism_data.py
```

Nếu cần tạo lại dữ liệu VietnamTourism từ đầu:

```bash
python scripts/crawl_vietnamtourism.py
python scripts/generate_qa_vietnamtourism.py
python scripts/prepare_vietnamtourism_data.py
```

### 5. Train Stage 2

Kiểm tra `instruction_train.stage1_projector_ckpt` trong `config.yaml`, sau đó chạy:

```bash
bash scripts/train_instruction.sh
```

### 6. Test checkpoint

```bash
streamlit run streamlit_stage1_test.py
streamlit run streamlit_instruction_test.py
```

### 7. Chạy unit test

```bash
pytest tests
```

## Checkpoint Và Output

Output được lưu trong `outputs/`.

Mỗi checkpoint thường có cấu trúc:

```text
checkpoint-500/
├── projector.pt
├── optimizer.pt
├── scheduler.pt
├── rng_state.pt
├── trainer_state.json
├── training_config.yaml
├── model_config/
├── processor/
├── tokenizer/
└── llm/                 # Chỉ có ở checkpoint Stage 2
```

Các file quan trọng trong output dir:

- `metrics.jsonl`: lưu `train_loss` và `eval_loss` theo step.
- `train.log`: log chi tiết quá trình train/eval.
- `best_checkpoint.json`: trỏ tới checkpoint tốt nhất theo `eval_loss`.
- `last_checkpoint.json`: trỏ tới checkpoint mới nhất.

## Ghi Chú Kỹ Thuật

- Stage 1 giữ projector ở `float32` để alignment ổn định hơn.
- Vision tower và LLM thường dùng `bfloat16` để tiết kiệm VRAM.
- Vision tower được freeze trong cả hai stage.
- Processor tự mở rộng `<image>` thành số token ảnh phù hợp, không cần đếm token thủ công.
- Training loop dùng token-weighted loss để tránh lệch loss khi sequence dài/ngắn khác nhau.
- Code đã sửa đường load SigLIP để pretrained vision weights được nạp đúng vào `model.vision_tower.vision_model`.
- Không nên resume các checkpoint Stage 1 cũ được train trước khi fix lỗi load vision tower, vì projector khi đó học trên image features sai.

## Test Và Kiểm Tra

Unit tests hiện kiểm tra các phần quan trọng:

- Entry point chuẩn bị instruction data.
- Lưu ảnh streamed về RGB JPEG.
- Ưu tiên tokenizer/LLM từ checkpoint khi test bằng Streamlit.
- Validate lỗi ảnh không đọc được.
- Parse và chuẩn hóa hội thoại VietnamTourism.

Chạy:

```bash
pytest tests
```

## Kết Quả Mong Muốn

Sau khi hoàn tất Stage 1 và Stage 2, model có thể:

- Mô tả ảnh bằng tiếng Việt.
- Trả lời câu hỏi dựa trên nội dung ảnh.
- Đọc và giải thích văn bản trong ảnh ở mức VQA/OCR.
- Hội thoại đa lượt với ngữ cảnh ảnh.
- Xử lý tốt hơn các ảnh, địa danh và ngữ cảnh liên quan đến du lịch Việt Nam.

## Phụ Lục: Full Code

Phần này tổng hợp code nguồn chính của project để tiện đọc trong một file. Không bao gồm `.env`, dữ liệu trong `data/`, checkpoint/log trong `outputs/`, cache hoặc file nhị phân.

### `config.yaml`

````yaml
prepare_data:
  raw_dir: data/uit-openviic-raw
  output_dir: data/uit-openviic
  download:
    enabled: true
    images:
      file_id: 1eRWL751fGv4WPTg0dhBGMRbcUgh9xKA5
      filename: images.zip
    annotations:
      train:
        file_id: 1kOgUE3duQjxJ6aaXrEHN8YlPQxF08JxW
        filename: uit-openviic-annotation-train.json
      val:
        file_id: 1eQbmeU5x3GL0_JO_mZgYKYNAFLvpCdL5
        filename: uit-openviic-annotation-dev.json
      test:
        file_id: 126gNnTrh13AZ2mc51bEMvB29huwuSB98
        filename: uit-openviic-annotation-test.json

prepare_coco:
  dataset_name: DavidPhilips/coco2017
  dataset_config: default
  output_dir: data/coco2017
  split_map:
    train: train
    validation: val
  image_field: image
  image_id_field: image_id
  caption_id_field: caption_id
  caption_field: caption_vi
  streaming: true
  overwrite: false
  max_rows_per_split: null

train:
  train_jsonl:
    - data/coco2017/train.jsonl
    - data/uit-openviic/train.jsonl
  eval_jsonl:
    - data/coco2017/val.jsonl
    - data/uit-openviic/val.jsonl
  output_dir: outputs/stage_1_projector_coco_uit_test
  vision_model: google/siglip2-so400m-patch16-384
  llm_model: meta-llama/Llama-3.2-1B-Instruct
  epochs: 2
  batch_size: 1
  grad_accum: 32
  gradient_checkpointing: false
  lr: 1.0e-3
  warmup_ratio: 0.03
  log_steps: 10
  eval_steps: 1000
  save_steps: 500
  keep_last_n: 3
  save_best_by: eval_loss
  seed: 42
  model_dtype: bfloat16
  projector_dtype: float32

train_uit_only:
  train_jsonl:
    - data/uit-openviic/train.jsonl
  eval_jsonl:
    - data/uit-openviic/val.jsonl
  output_dir: outputs/stage_1_projector_uit_only_fixed_last_hidden_bf16_projfp32_lr1e3
  vision_model: google/siglip2-so400m-patch16-384
  llm_model: meta-llama/Llama-3.2-1B-Instruct
  epochs: 2
  batch_size: 1
  grad_accum: 32
  gradient_checkpointing: false
  lr: 1.0e-3
  warmup_ratio: 0.03
  log_steps: 10
  eval_steps: 500
  save_steps: 500
  keep_last_n: 3
  save_best_by: eval_loss
  seed: 42
  model_dtype: bfloat16
  projector_dtype: float32

instruction_data_gpt:
  dataset_name: 5CD-AI/Viet-ShareGPT-4o-Text-VQA
  output_dir: data/viet-sharegpt-4o-text-vqa
  system_prompt: Bạn là một trợ lý thị giác tiếng Việt, trả lời trung thực và chỉ dựa trên nội dung nhìn thấy trong ảnh.
  description_user_prompt: Hãy mô tả thật chi tiết hình ảnh này, bao gồm văn bản, bố cục, đối tượng và vị trí của chúng.
  use_description_samples: true
  use_qna_samples: true
  split_mode: auto
  split_seed: 42
  val_ratio: 0.01
  test_ratio: 0.01
  image_field: null
  description_field: null
  qna_field: null
  max_rows: null
  inspect_only: false

instruction_data_5cd:
  dataset_name: 5CD-AI/Viet-Localization-VQA
  output_dir: data/viet-localization-vqa
  system_prompt: Bạn là một trợ lý thị giác tiếng Việt, trả lời trung thực và chỉ dựa trên nội dung nhìn thấy trong ảnh.
  description_user_prompt: Hãy mô tả thật chi tiết hình ảnh này, bao gồm bối cảnh, đối tượng, hoạt động, màu sắc và các chi tiết văn hóa nếu có.
  use_description_samples: true
  use_qna_samples: true
  split_mode: image_level
  split_seed: 42
  val_ratio: 0.01
  test_ratio: 0.01
  image_field: image
  description_field: description
  qna_field: conversations
  streaming: true
  max_rows: null
  inspect_only: false

instruction_train:
  train_jsonl:
    - data/viet-sharegpt-4o-text-vqa/train.jsonl
    - data/viet-localization-vqa/train.jsonl
    - data/vietnamtourism/train.jsonl
  eval_jsonl:
    - data/viet-sharegpt-4o-text-vqa/val.jsonl
    - data/viet-localization-vqa/val.jsonl
    - data/vietnamtourism/val.jsonl
  sample_weights: [40, 55, 5]
  output_dir: outputs/stage_2_instruction_final
  stage1_projector_ckpt: outputs/stage_1_projector_coco_uit_test/checkpoint-19782
  vision_model: google/siglip2-so400m-patch16-384
  llm_model: meta-llama/Llama-3.2-1B-Instruct
  freeze_vision: true
  train_projector: true
  train_llm: true
  mixed_precision: bf16
  model_dtype: bfloat16
  projector_dtype: float32
  optimizer_type: adafactor
  adam_foreach: false
  max_text_tokens: 2048
  epochs: 1
  batch_size: 1
  grad_accum: 16
  projector_lr: 1.0e-4
  llm_lr: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.03
  gradient_checkpointing: false
  log_steps: 10
  eval_steps: 1000
  save_steps: 500
  keep_last_n: 3
  save_best_by: eval_loss
  max_new_tokens: 768
  seed: 42

crawl_vietnamtourism:
  output_dir: data/vietnamtourism-raw
  category_id: 55
  max_pages: null
  max_images: 10000
  delay_seconds: 1.0
  min_image_width: 200

generate_qa_vietnamtourism:
  raw_dir: data/vietnamtourism-raw
  model: gpt-5-nano
  max_tokens: 16000
  batch_chunk_mb: 150
  # Keep only a few OpenAI batches enqueued at once to avoid org token-limit failures.
  max_active_batches: 3
  # Number of new images to generate in this run. Existing batch_results.jsonl
  # image_ids are skipped, so 2000 here means "generate about 2k more".
  max_images: 2000

prepare_vietnamtourism:
  raw_dir: data/vietnamtourism-raw
  output_dir: data/vietnamtourism
  system_prompt: "Bạn là một trợ lý thị giác tiếng Việt, trả lời trung thực dựa trên ảnh và ngữ cảnh bài viết được cung cấp."
  val_ratio: 0.01
  test_ratio: 0.01
  seed: 42
````

### `pyproject.toml`

````toml
[project]
name = "pretrain-vlm"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "accelerate>=1.12.0,<2",
    "beautifulsoup4>=4.12.0",
    "datasets>=3.6.0,<5",
    "gdown>=6.0.0,<7",
    "loguru>=0.7.3",
    "openai>=2.0.0",
    "python-dotenv>=1.0.0",
    "pillow>=12.2.0,<13",
    "pyyaml>=6.0.3,<7",
    "requests>=2.32.0",
    "torch>=2.10.0,<3",
    "transformers>=4.55.4,<5",
]

[tool.ruff]
target-version = "py311"
line-length = 110

[tool.ruff.format]
skip-magic-trailing-comma = true
````

### `requirements.txt`

````text
accelerate>=1.12.0,<2
beautifulsoup4>=4.12.0
datasets>=3.6.0,<5
gdown>=6.0.0,<7
loguru>=0.7.3
openai>=2.0.0
python-dotenv>=1.0.0
pillow>=12.2.0,<13
pyyaml>=6.0.3,<7
requests>=2.32.0
torch>=2.10.0,<3
transformers>=4.55.4,<5
streamlit
````

### `train.py`

````python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.collators import PROMPT_TEMPLATE, CaptionCollator
from src.data import ImageCaptionDataset
from src.modeling import build_model, freeze_components
from src.runtime import (
    append_jsonl,
    build_weighted_sampler,
    current_lr,
    load_config,
    resolve_config_paths,
    set_seed,
    setup_logger,
)
from src.training.checkpoint import (
    load_projector_checkpoint,
    rotate_checkpoints,
    save_training_checkpoint,
    update_checkpoint_pointer,
)
from src.training.engine import TrainingState, compute_steps_per_epoch, run_training
from src.training.eval import run_evaluation


def _parse_args() -> argparse.Namespace:
    """Parse runtime-only arguments that should not live in config.yaml."""

    parser = argparse.ArgumentParser(description="Caption projector pretraining.")
    parser.add_argument("--config-section", type=str, default="train")
    parser.add_argument("--resume-from", type=str, default=None)
    return parser.parse_args()


def _select_eval_samples(records: list[dict], sample_count: int = 5) -> list[dict]:
    if not records or sample_count <= 0:
        return []
    step = max(1, len(records) // sample_count)
    return [records[i * step] for i in range(min(sample_count, len(records)))]


def _log_eval_samples(model, collator, eval_samples, accelerator, max_new_tokens):
    unwrapped = accelerator.unwrap_model(model)
    tokenizer = collator.tokenizer
    eos_ids = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end_of_text|>")})
    lines = []
    for idx, sample in enumerate(eval_samples, 1):
        try:
            with Image.open(sample["image"]) as img:
                img = img.convert("RGB")
        except Exception as e:
            line = f"[sample {idx}] failed to load {sample['image']}: {e}"
            lines.append(line)
            continue

        inputs = collator.processor(text=PROMPT_TEMPLATE, images=img, return_tensors="pt")
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        with torch.no_grad(), accelerator.autocast():
            generated_ids = unwrapped.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=5,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_text = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True).strip()
        raw_generated_text = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=False).strip()
        for line in [
            f"[sample {idx}] prediction: {generated_text or '<empty>'}",
            f"[sample {idx}] prediction_raw: {raw_generated_text or '<empty>'}",
            f"[sample {idx}] reference: {sample['caption']}",
            f"[sample {idx}] image: {sample['image']}",
        ]:
            lines.append(line)
    return lines


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config_section)
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    accelerator = Accelerator()
    logger = setup_logger(output_dir, accelerator)
    set_seed(int(cfg["seed"]))

    collator = CaptionCollator(cfg["vision_model"], cfg["llm_model"])
    train_dataset = ImageCaptionDataset(resolve_config_paths(cfg["train_jsonl"]))
    eval_dataset = ImageCaptionDataset(resolve_config_paths(cfg["eval_jsonl"]))
    eval_samples = _select_eval_samples(eval_dataset.records, sample_count=5)
    train_sampler = build_weighted_sampler(train_dataset, seed=int(cfg["seed"]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        sampler=train_sampler,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        cfg["vision_model"],
        cfg["llm_model"],
        model_dtype=cfg.get("model_dtype"),
        projector_dtype=cfg.get("projector_dtype", "float32"),
        image_token_id=collator.image_token_id,
        vocab_size=len(collator.tokenizer),
    )
    freeze_components(model, freeze_vision=True, train_projector=True, train_llm=False)

    if bool(cfg.get("gradient_checkpointing", False)):
        model.language_model.config.use_cache = False
        model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    optimizer = AdamW(model.multi_modal_projector.parameters(), lr=float(cfg["lr"]), weight_decay=0.0)
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )

    grad_accum = int(cfg["grad_accum"])
    steps_per_epoch = compute_steps_per_epoch(len(train_loader), grad_accum)
    total_steps = steps_per_epoch * int(cfg["epochs"])
    warmup_steps = int(total_steps * float(cfg["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    state = TrainingState()

    if args.resume_from:
        resume_state = load_projector_checkpoint(
            args.resume_from, accelerator.unwrap_model(model), optimizer, scheduler
        )
        state.global_step = int(resume_state.get("global_step", 0))
        state.best_eval_loss = resume_state.get("best_eval_loss")
        logger.info("Resumed from {} at step {}.", args.resume_from, state.global_step)

    logger.info("Starting training: {} train, {} eval.", len(train_dataset), len(eval_dataset))
    logger.info(
        "Training config: "
        "processes={}, per_device_batch={}, grad_accum={}, effective_batch={}, "
        "steps_per_epoch={}, total_steps={}, warmup_steps={}, output_dir={}",
        accelerator.num_processes,
        int(cfg["batch_size"]),
        grad_accum,
        int(cfg["batch_size"]) * grad_accum * accelerator.num_processes,
        steps_per_epoch,
        total_steps,
        warmup_steps,
        output_dir,
    )
    progress_total = max(total_steps - state.global_step, 0)
    progress = (
        tqdm(total=progress_total, initial=0, desc="stage1", dynamic_ncols=True)
        if accelerator.is_main_process
        else None
    )

    def set_train_mode() -> None:
        accelerator.unwrap_model(model).multi_modal_projector.train()

    def trainable_parameters():
        return accelerator.unwrap_model(model).multi_modal_projector.parameters()

    def save_checkpoint(global_step: int, *, eval_loss: float | None = None) -> Path:
        ckpt_path = output_dir / f"checkpoint-{global_step}"
        is_best = eval_loss is not None and (state.best_eval_loss is None or eval_loss < state.best_eval_loss)
        if is_best:
            state.best_eval_loss = eval_loss
        save_training_checkpoint(
            path=ckpt_path,
            model=accelerator.unwrap_model(model),
            processor=collator.processor,
            tokenizer=collator.tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            training_config=cfg,
            trainer_state={
                "global_step": global_step,
                "epoch": state.epoch,
                "best_eval_loss": state.best_eval_loss,
                "save_best_by": cfg.get("save_best_by", "eval_loss"),
            },
            stage="caption_pretrain",
            save_language_model=False,
        )
        update_checkpoint_pointer(output_dir, "last", ckpt_path, step=global_step)
        if is_best:
            update_checkpoint_pointer(
                output_dir,
                "best",
                ckpt_path,
                step=global_step,
                metric_name="eval_loss",
                metric_value=eval_loss,
            )
        rotate_checkpoints(output_dir, int(cfg["keep_last_n"]))
        return ckpt_path

    def on_step_end(result, _training_state) -> None:
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                train_loss=f"{result.train_loss:.6f}",
                lr=f"{current_lr(scheduler, optimizer):.3e}",
                supervised_tokens=result.supervised_tokens,
            )

        if result.global_step % int(cfg["log_steps"]) == 0:
            logger.info(
                "step {}: train_loss={:.6f}, lr={:.6e}, supervised_tokens={}",
                result.global_step,
                result.train_loss,
                current_lr(scheduler, optimizer),
                result.supervised_tokens,
            )
            if accelerator.is_main_process:
                append_jsonl(metrics_path, {"step": result.global_step, "train_loss": result.train_loss})

        eval_loss = None
        if result.global_step % int(cfg["eval_steps"]) == 0:
            eval_loss = run_evaluation(
                model=model,
                eval_loader=eval_loader,
                accelerator=accelerator,
                global_step=result.global_step,
                logger=logger,
                sample_logger=lambda m, a: _log_eval_samples(m, collator, eval_samples, a, 64),
                restore_train_mode=set_train_mode,
            )
            if accelerator.is_main_process:
                append_jsonl(metrics_path, {"step": result.global_step, "eval_loss": eval_loss})

        if result.global_step % int(cfg["save_steps"]) == 0 and accelerator.is_main_process:
            ckpt_path = save_checkpoint(result.global_step, eval_loss=eval_loss)
            logger.info("Saved checkpoint to {}.", ckpt_path)

    run_training(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        epochs=int(cfg["epochs"]),
        grad_accum=grad_accum,
        state=state,
        set_train_mode=set_train_mode,
        trainable_parameters=trainable_parameters,
        on_step_end=on_step_end,
    )

    if accelerator.is_main_process:
        ckpt_path = save_checkpoint(state.global_step)
        logger.info("Saved final checkpoint to {}.", ckpt_path)
        if progress is not None:
            progress.close()

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
````

### `train_instruction.py`

````python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Adafactor, get_cosine_schedule_with_warmup

from src.collators import InstructionCollator
from src.data import ImageInstructionDataset
from src.modeling import build_model, freeze_components, set_component_modes
from src.runtime import (
    EpochShuffleSampler,
    append_jsonl,
    build_weighted_sampler,
    current_lr,
    load_config,
    resolve_config_path,
    set_seed,
    setup_logger,
)
from src.training.checkpoint import (
    load_full_checkpoint,
    load_projector_checkpoint,
    rotate_checkpoints,
    save_training_checkpoint,
    update_checkpoint_pointer,
)
from src.training.engine import TrainingState, compute_steps_per_epoch, run_training
from src.training.eval import run_evaluation


def _parse_args() -> argparse.Namespace:
    """Parse runtime-only arguments that should not live in config.yaml."""

    parser = argparse.ArgumentParser(description="Instruction tuning.")
    parser.add_argument("--resume-from", type=str, default=None)
    return parser.parse_args()


def _is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    lower = name.lower()
    return param.ndim == 1 or lower.endswith(".bias") or "norm" in lower


def _build_optimizer(model, cfg: dict):
    projector_lr = float(cfg["projector_lr"])
    llm_lr = float(cfg["llm_lr"])
    wd = float(cfg["weight_decay"])
    groups: dict[str, list] = {k: [] for k in ("proj_decay", "proj_nodecay", "llm_decay", "llm_nodecay")}
    seen_params: set[int] = set()

    def add_params(named_params, decay_key: str, nodecay_key: str) -> None:
        for name, param in named_params:
            if not param.requires_grad or id(param) in seen_params:
                continue
            seen_params.add(id(param))
            groups[nodecay_key if _is_no_decay(name, param) else decay_key].append(param)

    add_params(model.multi_modal_projector.named_parameters(), "proj_decay", "proj_nodecay")
    add_params(model.language_model.named_parameters(), "llm_decay", "llm_nodecay")
    add_params(model.lm_head.named_parameters(), "llm_decay", "llm_nodecay")

    param_groups = []
    if groups["proj_decay"]:
        param_groups.append({"params": groups["proj_decay"], "lr": projector_lr, "weight_decay": wd})
    if groups["proj_nodecay"]:
        param_groups.append({"params": groups["proj_nodecay"], "lr": projector_lr, "weight_decay": 0.0})
    if groups["llm_decay"]:
        param_groups.append({"params": groups["llm_decay"], "lr": llm_lr, "weight_decay": wd})
    if groups["llm_nodecay"]:
        param_groups.append({"params": groups["llm_nodecay"], "lr": llm_lr, "weight_decay": 0.0})

    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    opt_type = str(cfg.get("optimizer_type", "adafactor")).strip().lower()
    if opt_type == "adamw":
        return AdamW(param_groups, foreach=bool(cfg.get("adam_foreach", False)))
    if opt_type == "adafactor":
        return Adafactor(param_groups, scale_parameter=False, relative_step=False, warmup_init=False)
    raise ValueError(f"Unsupported optimizer_type '{opt_type}'.")


def _latest_user_message(messages) -> str:
    for msg in reversed(messages):
        if msg["role"] == "user":
            return str(msg["content"])
    return "<no user message>"


def _select_eval_samples(records, max_samples: int = 5):
    seen, selected = set(), []
    for record in records:
        key = (record.get("sample_type"), record.get("image_id"), record.get("id"))
        if key not in seen:
            seen.add(key)
            selected.append(record)
        if len(selected) >= max_samples:
            break
    return selected


def _resolve_resume_sources(resume_dir: Path | None, base_llm_model: str) -> tuple[str, str]:
    if resume_dir is None:
        return base_llm_model, base_llm_model

    llm_dir = resume_dir / "llm"
    tokenizer_dir = resume_dir / "tokenizer"
    if not llm_dir.exists():
        raise FileNotFoundError(f"Missing resumed LLM weights under {llm_dir}")

    tokenizer_source = str(tokenizer_dir.resolve()) if tokenizer_dir.exists() else base_llm_model
    return str(llm_dir.resolve()), tokenizer_source


def _log_eval_samples(model, collator, eval_samples, accelerator, max_new_tokens):
    unwrapped = accelerator.unwrap_model(model)
    tokenizer = collator.tokenizer
    eos_ids = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end_of_text|>")})
    lines = []
    for idx, sample in enumerate(eval_samples, 1):
        try:
            with Image.open(sample["image"]) as img:
                img = img.convert("RGB")
        except Exception as e:
            line = f"[sample {idx}] failed: {e}"
            lines.append(line)
            continue

        prompt_ids, attn_mask, pixel_values = collator.build_prompt_tensors(
            sample["messages"][:-1], img, device=accelerator.device
        )
        with torch.no_grad(), accelerator.autocast():
            generated_ids = unwrapped.generate(
                input_ids=prompt_ids,
                pixel_values=pixel_values,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=5,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

        input_len = prompt_ids.shape[1]
        generated_text = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True).strip()
        raw_generated_text = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=False).strip()
        for line in [
            f"[sample {idx}] sample_type: {sample.get('sample_type', 'unknown')}",
            f"[sample {idx}] user: {_latest_user_message(sample['messages'][:-1])}",
            f"[sample {idx}] prediction: {generated_text or '<empty>'}",
            f"[sample {idx}] prediction_raw: {raw_generated_text or '<empty>'}",
            f"[sample {idx}] reference: {sample['messages'][-1]['content']}",
        ]:
            lines.append(line)
    return lines


def main() -> None:
    args = _parse_args()
    cfg = load_config("instruction_train")
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    resume_dir = Path(args.resume_from).expanduser().resolve() if args.resume_from else None
    stage1_ckpt = Path(cfg["stage1_projector_ckpt"]).expanduser().resolve()
    mixed_precision = str(cfg.get("mixed_precision", "bf16")).strip().lower()
    if mixed_precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 not supported on this device. Set mixed_precision to fp16.")

    accelerator = Accelerator(mixed_precision=mixed_precision)
    logger = setup_logger(output_dir, accelerator)
    set_seed(int(cfg["seed"]))

    llm_source, tokenizer_source = _resolve_resume_sources(resume_dir, cfg["llm_model"])
    collator = InstructionCollator(
        cfg["vision_model"], tokenizer_source, max_text_tokens=int(cfg["max_text_tokens"])
    )
    train_jsonl = cfg["train_jsonl"]
    eval_jsonl = cfg["eval_jsonl"]
    train_dataset = ImageInstructionDataset(
        [resolve_config_path(p) for p in train_jsonl] if isinstance(train_jsonl, list) else resolve_config_path(train_jsonl)
    )
    eval_dataset = ImageInstructionDataset(
        [resolve_config_path(p) for p in eval_jsonl] if isinstance(eval_jsonl, list) else resolve_config_path(eval_jsonl)
    )
    eval_samples = _select_eval_samples(eval_dataset.records)
    sample_weights = cfg.get("sample_weights")
    if sample_weights is not None:
        train_sampler = build_weighted_sampler(
            train_dataset, seed=int(cfg["seed"]), source_weights=[float(w) for w in sample_weights]
        )
    else:
        train_sampler = EpochShuffleSampler(train_dataset, seed=int(cfg["seed"]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        sampler=train_sampler,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        cfg["vision_model"],
        llm_source,
        tokenizer_name_or_path=tokenizer_source,
        model_dtype=cfg.get("model_dtype"),
        projector_dtype=cfg.get("projector_dtype", "float32"),
        image_token_id=collator.image_token_id,
        vocab_size=len(collator.tokenizer),
    )
    component_modes = {
        "freeze_vision": bool(cfg.get("freeze_vision", True)),
        "train_projector": bool(cfg.get("train_projector", True)),
        "train_llm": bool(cfg.get("train_llm", True)),
    }
    freeze_components(model, **component_modes)

    if bool(cfg.get("gradient_checkpointing", False)):
        model.language_model.config.use_cache = False
        model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    stage1_step = None
    if resume_dir is None:
        if not stage1_ckpt.exists():
            raise FileNotFoundError(f"Missing stage-1 checkpoint: {stage1_ckpt}")
        stage1_state = load_projector_checkpoint(stage1_ckpt, model)
        stage1_step = int(stage1_state.get("global_step", 0))

    optimizer = _build_optimizer(model, cfg)
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )

    grad_accum = int(cfg["grad_accum"])
    steps_per_epoch = compute_steps_per_epoch(len(train_loader), grad_accum)
    total_steps = steps_per_epoch * int(cfg["epochs"])
    warmup_steps = int(total_steps * float(cfg["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    state = TrainingState()

    if resume_dir:
        resume_state = load_full_checkpoint(resume_dir, accelerator.unwrap_model(model), optimizer, scheduler)
        state.global_step = int(resume_state.get("global_step", 0))
        state.best_eval_loss = resume_state.get("best_eval_loss")

    logger.info("[check] train={} eval={}", len(train_dataset), len(eval_dataset))
    if stage1_step is not None:
        logger.info("[check] warm-started projector from stage-1 step {}", stage1_step)
    if resume_dir:
        logger.info("[check] resumed from {} at step {}", resume_dir, state.global_step)

    progress_total = max(total_steps - state.global_step, 0)
    progress = (
        tqdm(total=progress_total, initial=0, desc="instruction", dynamic_ncols=True)
        if accelerator.is_main_process
        else None
    )

    def set_train_mode() -> None:
        set_component_modes(accelerator.unwrap_model(model), **component_modes)

    def trainable_parameters():
        return (p for p in accelerator.unwrap_model(model).parameters() if p.requires_grad)

    def save_checkpoint(global_step: int, *, eval_loss: float | None = None) -> Path:
        ckpt_path = output_dir / f"checkpoint-{global_step}"
        is_best = eval_loss is not None and (state.best_eval_loss is None or eval_loss < state.best_eval_loss)
        if is_best:
            state.best_eval_loss = eval_loss
        save_training_checkpoint(
            path=ckpt_path,
            model=accelerator.unwrap_model(model),
            processor=collator.processor,
            tokenizer=collator.tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            training_config=cfg,
            trainer_state={
                "global_step": global_step,
                "epoch": state.epoch,
                "best_eval_loss": state.best_eval_loss,
                "save_best_by": cfg.get("save_best_by", "eval_loss"),
            },
            stage="instruction_tuning",
            save_language_model=True,
        )
        update_checkpoint_pointer(output_dir, "last", ckpt_path, step=global_step)
        if is_best:
            update_checkpoint_pointer(
                output_dir,
                "best",
                ckpt_path,
                step=global_step,
                metric_name="eval_loss",
                metric_value=eval_loss,
            )
        rotate_checkpoints(output_dir, int(cfg["keep_last_n"]))
        return ckpt_path

    def on_epoch_start(epoch: int) -> None:
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

    def on_step_end(result, _training_state) -> None:
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                train_loss=f"{result.train_loss:.6f}",
                lr=f"{current_lr(scheduler, optimizer):.3e}",
                supervised_tokens=result.supervised_tokens,
            )

        if result.global_step % int(cfg["log_steps"]) == 0:
            logger.info(
                "step {}: train_loss={:.6f}, lr={:.6e}, supervised_tokens={}",
                result.global_step,
                result.train_loss,
                current_lr(scheduler, optimizer),
                result.supervised_tokens,
            )
            if accelerator.is_main_process:
                append_jsonl(metrics_path, {"step": result.global_step, "train_loss": result.train_loss})

        eval_loss = None
        if result.global_step % int(cfg["eval_steps"]) == 0:
            eval_loss = run_evaluation(
                model=model,
                eval_loader=eval_loader,
                accelerator=accelerator,
                global_step=result.global_step,
                logger=logger,
                sample_logger=lambda m, a: _log_eval_samples(
                    m, collator, eval_samples, a, int(cfg["max_new_tokens"])
                ),
                restore_train_mode=set_train_mode,
            )
            if accelerator.is_main_process:
                append_jsonl(metrics_path, {"step": result.global_step, "eval_loss": eval_loss})

        if result.global_step % int(cfg["save_steps"]) == 0 and accelerator.is_main_process:
            ckpt_path = save_checkpoint(result.global_step, eval_loss=eval_loss)
            logger.info("[check] saved checkpoint to {}", ckpt_path)

    run_training(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        epochs=int(cfg["epochs"]),
        grad_accum=grad_accum,
        state=state,
        set_train_mode=set_train_mode,
        trainable_parameters=trainable_parameters,
        on_step_end=on_step_end,
        on_epoch_start=on_epoch_start,
    )

    if accelerator.is_main_process:
        ckpt_path = save_checkpoint(state.global_step)
        logger.info("[check] saved final instruction checkpoint to {}", ckpt_path)
        if progress is not None:
            progress.close()

    logger.info("Instruction finetuning finished.")


if __name__ == "__main__":
    main()
````

### `streamlit_stage1_test.py`

````python
from __future__ import annotations

import json
import os
import re
from contextlib import nullcontext
from pathlib import Path

from PIL import Image
import streamlit as st
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage_1_projector_coco_uit_final"
DEFAULT_PROMPT = "<image>\nMô tả hình ảnh này: "


st.set_page_config(page_title="Stage 1 Projector Test", layout="wide")


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)(?:\.pt)?$", path.name)
    return int(match.group(1)) if match else -1


def _is_checkpoint(path: Path) -> bool:
    if not path.name.startswith("checkpoint-"):
        return False
    if path.is_file():
        return path.suffix == ".pt"
    return (path / "projector.pt").exists()


def find_checkpoints(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    checkpoints = [p for p in output_dir.iterdir() if _is_checkpoint(p)]
    return sorted(checkpoints, key=checkpoint_step, reverse=True)


def read_checkpoint_pointer(output_dir: Path, name: str) -> Path | None:
    pointer_path = output_dir / f"{name}_checkpoint.json"
    if not pointer_path.exists():
        return None
    try:
        with pointer_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        checkpoint = Path(str(payload["checkpoint"])).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = output_dir / checkpoint
        checkpoint = checkpoint.resolve()
        return checkpoint if _is_checkpoint(checkpoint) else None
    except Exception:
        return None


def default_checkpoint_index(output_dir: Path, checkpoints: list[Path]) -> int:
    resolved = [path.resolve() for path in checkpoints]
    for pointer_name in ("best", "last"):
        pointer = read_checkpoint_pointer(output_dir, pointer_name)
        if pointer and pointer.resolve() in resolved:
            return resolved.index(pointer.resolve())
    return 0


def load_checkpoint_training_config(checkpoint_path: Path) -> dict:
    config_path = checkpoint_path / "training_config.yaml" if checkpoint_path.is_dir() else None
    if not config_path or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"{config_path} must contain a mapping.")
    return config


def merge_checkpoint_config(base_config: dict, checkpoint_path: Path) -> dict:
    merged = dict(base_config)
    merged.update(load_checkpoint_training_config(checkpoint_path))
    return merged


def resolve_tokenizer_source(checkpoint_path: str | Path, fallback_llm_model: str) -> str:
    tokenizer_dir = Path(checkpoint_path).expanduser().resolve() / "tokenizer"
    return str(tokenizer_dir) if tokenizer_dir.exists() else fallback_llm_model


@st.cache_data(show_spinner=False)
def load_train_config() -> dict:
    from src.runtime import load_config

    return load_config("train")


def as_list(value) -> list:
    return value if isinstance(value, list) else [value]


@st.cache_data(show_spinner=False)
def load_eval_samples(eval_jsonl, limit: int = 200) -> list[dict]:
    from src.runtime import resolve_record_image_path

    samples: list[dict] = []
    jsonl_paths = as_list(eval_jsonl)
    per_path_limit = max(1, limit // max(len(jsonl_paths), 1))
    for jsonl_path in jsonl_paths:
        p = Path(jsonl_path).expanduser().resolve()
        if not p.exists():
            continue
        path_count = 0
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                if path_count >= per_path_limit or len(samples) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                image = resolve_record_image_path(record.get("image", ""), jsonl_path=p)
                caption = str(record.get("caption", "")).strip()
                if image and Path(image).exists():
                    samples.append({"image": image, "caption": caption, "source": p.parent.name})
                    path_count += 1
    return samples


def detect_devices() -> list[str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return ["cpu"]
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())] + ["cpu"]
    except Exception:
        return ["cpu"]


def default_device_index(devices: list[str]) -> int:
    requested = os.environ.get("STREAMLIT_DEVICE", "").strip()
    if requested in devices:
        return devices.index(requested)
    if "cuda:2" in devices:
        return devices.index("cuda:2")
    if "cuda:0" in devices:
        return devices.index("cuda:0")
    return 0


def device_label(device_name: str) -> str:
    try:
        import torch

        if not device_name.startswith("cuda") or not torch.cuda.is_available():
            return device_name
        device = torch.device(device_name)
        props = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        used_gb = (total_bytes - free_bytes) / 1024**3
        total_gb = total_bytes / 1024**3
        return f"{device_name} | {props.name} | {used_gb:.1f}/{total_gb:.1f} GB"
    except Exception:
        return device_name


@st.cache_resource(show_spinner="Đang load model và checkpoint...")
def load_model_resource(
    checkpoint_path: str,
    vision_model: str,
    llm_model: str,
    model_dtype: str,
    projector_dtype: str,
    device_name: str,
):
    import torch

    from src.modeling import build_model, build_processor
    from src.training.checkpoint import load_projector_checkpoint

    device = torch.device(device_name)
    tokenizer_source = resolve_tokenizer_source(checkpoint_path, llm_model)
    processor = build_processor(vision_model, tokenizer_source)
    model = build_model(
        vision_model,
        llm_model,
        tokenizer_name_or_path=tokenizer_source,
        model_dtype=model_dtype,
        projector_dtype=projector_dtype,
        image_token_id=processor.tokenizer.convert_tokens_to_ids("<image>"),
        vocab_size=len(processor.tokenizer),
    )
    state = load_projector_checkpoint(checkpoint_path, model)
    step = int(state.get("global_step") or checkpoint_step(Path(checkpoint_path)))
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return model, processor, step


def move_inputs_to_device(inputs: dict, model):
    import torch

    device = next(model.parameters()).device
    vision_dtype = next(model.vision_tower.parameters()).dtype
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif key == "pixel_values":
            moved[key] = value.to(device=device, dtype=vision_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def eos_token_ids(tokenizer) -> list[int]:
    ids = {tokenizer.eos_token_id}
    for token in ("<|eot_id|>", "<|end_of_text|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    return sorted(i for i in ids if i is not None)


def generate_caption(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, str]:
    import torch

    image = image.convert("RGB")
    tokenizer = processor.tokenizer
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = move_inputs_to_device(inputs, model)
    device = next(model.parameters()).device
    vision_dtype = next(model.vision_tower.parameters()).dtype
    use_sampling = temperature > 0.0

    autocast_context = nullcontext()
    if device.type == "cuda" and vision_dtype in (torch.float16, torch.bfloat16):
        autocast_context = torch.autocast(device_type="cuda", dtype=vision_dtype)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "eos_token_id": eos_token_ids(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": use_sampling,
    }
    if use_sampling:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.inference_mode(), autocast_context:
        generated_ids = model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[0, input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    return text, raw_text


def _tensor_stats(values) -> dict[str, float]:
    flat = values.detach().float().reshape(-1, values.shape[-1])
    token_norm = flat.norm(dim=-1)
    return {
        "std": float(flat.std().cpu()),
        "rms": float(flat.pow(2).mean().sqrt().cpu()),
        "token_norm": float(token_norm.mean().cpu()),
    }


def projector_scale_diagnostics(model, processor, image: Image.Image, prompt: str) -> dict[str, float]:
    import torch

    image = image.convert("RGB")
    tokenizer = processor.tokenizer
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = move_inputs_to_device(inputs, model)
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    device = next(model.parameters()).device
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    text_mask = input_ids != image_token_id
    if tokenizer.pad_token_id is not None:
        text_mask = text_mask & (input_ids != tokenizer.pad_token_id)

    with torch.inference_mode():
        vision_outputs = model.vision_tower(
            pixel_values=pixel_values, output_hidden_states=True, return_dict=True
        )
        feature_layer = model.config.vision_feature_layer
        select_strategy = model.config.vision_feature_select_strategy
        if isinstance(feature_layer, int):
            selected_features = vision_outputs.hidden_states[feature_layer]
            if select_strategy == "default":
                selected_features = selected_features[:, 1:]
        else:
            hidden_states = [vision_outputs.hidden_states[idx] for idx in feature_layer]
            if select_strategy == "default":
                hidden_states = [state[:, 1:] for state in hidden_states]
            selected_features = torch.cat(hidden_states, dim=-1)

        image_features = model.multi_modal_projector(selected_features)
        text_embeds = model.get_input_embeddings()(input_ids.to(device))[text_mask]

    image_stats = _tensor_stats(image_features)
    text_stats = _tensor_stats(text_embeds)
    return {
        "image_tokens": int(image_features.shape[1]),
        "projector_std": image_stats["std"],
        "projector_rms": image_stats["rms"],
        "projector_token_norm": image_stats["token_norm"],
        "text_std": text_stats["std"],
        "text_rms": text_stats["rms"],
        "text_token_norm": text_stats["token_norm"],
        "std_ratio": image_stats["std"] / max(text_stats["std"], 1e-12),
        "norm_ratio": image_stats["token_norm"] / max(text_stats["token_norm"], 1e-12),
    }


def format_diag_row(name: str, diag: dict[str, float]) -> dict[str, str]:
    return {
        "input": name,
        "image_tokens": str(diag["image_tokens"]),
        "projector_norm": f"{diag['projector_token_norm']:.3f}",
        "text_norm": f"{diag['text_token_norm']:.3f}",
        "norm_ratio": f"{diag['norm_ratio']:.1f}x",
        "projector_std": f"{diag['projector_std']:.4f}",
        "text_std": f"{diag['text_std']:.4f}",
        "std_ratio": f"{diag['std_ratio']:.1f}x",
    }


def open_image_from_upload(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def open_image_from_path(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def main() -> None:
    st.title("Test Stage 1 Projector")

    try:
        cfg = load_train_config()
    except Exception as error:
        st.error(f"Không đọc được config.yaml: {error}")
        st.stop()

    with st.sidebar:
        st.header("Checkpoint")
        output_dir_text = st.text_input("Output dir", value=str(DEFAULT_OUTPUT_DIR))
        output_dir = Path(output_dir_text).expanduser().resolve()
        checkpoints = find_checkpoints(output_dir)
        if not checkpoints:
            st.error(f"Không tìm thấy checkpoint có projector.pt hoặc checkpoint-*.pt trong {output_dir}")
            st.stop()

        checkpoint = st.selectbox(
            "Checkpoint",
            checkpoints,
            index=default_checkpoint_index(output_dir, checkpoints),
            format_func=lambda p: f"{p.name} (step {checkpoint_step(p)})",
        )

        effective_cfg = merge_checkpoint_config(cfg, checkpoint)

        st.header("Model")
        vision_model = st.text_input("Vision model", value=str(effective_cfg["vision_model"]))
        llm_model = st.text_input("LLM model", value=str(effective_cfg["llm_model"]))
        model_dtype = st.selectbox(
            "Model dtype",
            ["bfloat16", "float16", "float32", "auto"],
            index=["bfloat16", "float16", "float32", "auto"].index(
                str(effective_cfg.get("model_dtype", "bfloat16"))
            )
            if str(effective_cfg.get("model_dtype", "bfloat16")) in ["bfloat16", "float16", "float32", "auto"]
            else 0,
        )
        projector_dtype = st.selectbox(
            "Projector dtype",
            ["bfloat16", "float16", "float32"],
            index=["bfloat16", "float16", "float32"].index(
                str(effective_cfg.get("projector_dtype", "float32"))
            )
            if str(effective_cfg.get("projector_dtype", "float32")) in ["bfloat16", "float16", "float32"]
            else 0,
        )
        devices = detect_devices()
        device_name = st.selectbox(
            "Device", devices, index=default_device_index(devices), format_func=device_label
        )
        if device_name == "cuda:2":
            st.caption("Đang chọn card vật lý cuda:2. Nếu chạy bằng CUDA_VISIBLE_DEVICES=2 thì chọn cuda:0.")

        st.header("Generate")
        max_new_tokens = st.slider("Max new tokens", 8, 256, 64, step=8)
        min_new_tokens = st.slider("Min new tokens", 0, 64, 5, step=1)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.0, step=0.05)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
        st.header("Diagnostics")
        run_black_baseline = st.checkbox("Black-image baseline", value=True)
        run_scale_diagnostics = st.checkbox("Projector scale", value=True)

        if st.button("Clear cache"):
            load_model_resource.clear()
            st.rerun()

    samples = load_eval_samples(cfg.get("eval_jsonl", []), limit=200)
    uploaded_file = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png", "webp"])

    left, right = st.columns([0.45, 0.55], gap="large")
    with left:
        sample = None
        if uploaded_file is None and samples:
            sample = st.selectbox(
                "Ảnh mẫu từ eval",
                samples,
                format_func=lambda x: (
                    f"{x.get('source', 'eval')} | {Path(x['image']).name} | {x['caption'][:80]}"
                ),
            )

        try:
            if uploaded_file is not None:
                image = open_image_from_upload(uploaded_file)
                reference = ""
                image_path = uploaded_file.name
            elif sample is not None:
                image = open_image_from_path(sample["image"])
                reference = sample["caption"]
                image_path = sample["image"]
            else:
                st.info("Upload một ảnh để test.")
                st.stop()
        except Exception as error:
            st.error(f"Không đọc được ảnh: {error}")
            st.stop()

        st.image(image, caption=str(image_path), use_container_width=True)
        if reference:
            st.caption(f"Reference: {reference}")

    with right:
        prompt_text = st.text_area("Prompt", value=DEFAULT_PROMPT, height=90)
        run = st.button("Sinh mô tả", type="primary", use_container_width=True)

        if run:
            try:
                model, processor, step = load_model_resource(
                    str(checkpoint), vision_model, llm_model, model_dtype, projector_dtype, device_name
                )
                text, raw_text = generate_caption(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except Exception as error:
                st.error(f"Không chạy được inference: {error}")
                st.exception(error)
                st.stop()

            st.subheader(f"Prediction - step {step}")
            st.write(text or "<empty>")
            with st.expander("Raw decode"):
                st.code(raw_text or "<empty>")

            if run_black_baseline:
                black_image = Image.new("RGB", image.size, (0, 0, 0))
                black_text, black_raw_text = generate_caption(
                    model=model,
                    processor=processor,
                    image=black_image,
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                st.subheader("Black-image baseline")
                st.write(black_text or "<empty>")
                with st.expander("Black raw decode"):
                    st.code(black_raw_text or "<empty>")

            if run_scale_diagnostics:
                rows = [
                    format_diag_row(
                        "real",
                        projector_scale_diagnostics(
                            model=model, processor=processor, image=image, prompt=prompt_text
                        ),
                    )
                ]
                if run_black_baseline:
                    rows.append(
                        format_diag_row(
                            "black",
                            projector_scale_diagnostics(
                                model=model, processor=processor, image=black_image, prompt=prompt_text
                            ),
                        )
                    )
                st.subheader("Projector Scale")
                st.table(rows)
                max_ratio = max(float(row["norm_ratio"].rstrip("x")) for row in rows)
                if max_ratio > 50:
                    st.warning(
                        "Projector norm đang lớn hơn text embedding rất nhiều. "
                        "Checkpoint này có nguy cơ bị visual soft-prompt collapse."
                    )


if __name__ == "__main__":
    main()
````

### `streamlit_instruction_test.py`

````python
from __future__ import annotations

import json
import os
import re
from contextlib import nullcontext
from pathlib import Path

from PIL import Image
import streamlit as st
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "instruction_run2"
DEFAULT_SYSTEM_PROMPT = (
    "Bạn là một trợ lý thị giác tiếng Việt, trả lời trung thực và chỉ dựa trên nội dung nhìn thấy trong ảnh."
)

st.set_page_config(page_title="Instruction Model Test", layout="wide")


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _is_checkpoint(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("checkpoint-") and (path / "projector.pt").exists()


def find_checkpoints(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted([p for p in output_dir.iterdir() if _is_checkpoint(p)], key=checkpoint_step, reverse=True)


def read_checkpoint_pointer(output_dir: Path, name: str) -> Path | None:
    pointer_path = output_dir / f"{name}_checkpoint.json"
    if not pointer_path.exists():
        return None
    try:
        with pointer_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        ckpt = Path(str(payload["checkpoint"])).expanduser().resolve()
        return ckpt if _is_checkpoint(ckpt) else None
    except Exception:
        return None


def default_checkpoint_index(output_dir: Path, checkpoints: list[Path]) -> int:
    resolved = [p.resolve() for p in checkpoints]
    for name in ("best", "last"):
        ptr = read_checkpoint_pointer(output_dir, name)
        if ptr and ptr.resolve() in resolved:
            return resolved.index(ptr.resolve())
    return 0


def load_checkpoint_config(checkpoint_path: Path) -> dict:
    cfg_path = checkpoint_path / "training_config.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg if isinstance(cfg, dict) else {}


def merge_config(base: dict, checkpoint_path: Path) -> dict:
    merged = dict(base)
    merged.update(load_checkpoint_config(checkpoint_path))
    return merged


def resolve_checkpoint_sources(checkpoint_path: str | Path, fallback_llm_model: str) -> tuple[str, str]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    llm_dir = checkpoint / "llm"
    tokenizer_dir = checkpoint / "tokenizer"
    llm_source = str(llm_dir) if llm_dir.exists() else fallback_llm_model
    tokenizer_source = str(tokenizer_dir) if tokenizer_dir.exists() else fallback_llm_model
    return llm_source, tokenizer_source


@st.cache_data(show_spinner=False)
def load_instruction_config() -> dict:
    from src.runtime import load_config

    return load_config("instruction_train")


def as_list(value) -> list:
    return value if isinstance(value, list) else [value]


@st.cache_data(show_spinner=False)
def load_eval_samples(eval_jsonl, limit: int = 200) -> list[dict]:
    from src.runtime import resolve_record_image_path

    samples: list[dict] = []
    paths = as_list(eval_jsonl)
    per_path = max(1, limit // max(len(paths), 1))
    for jsonl_path in paths:
        p = Path(jsonl_path).expanduser().resolve()
        if not p.exists():
            continue
        count = 0
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if count >= per_path or len(samples) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                image_val = record.get("image", "")
                if not image_val:
                    continue
                image = resolve_record_image_path(image_val, jsonl_path=p)
                messages = record.get("messages", [])
                if not messages or not Path(image).exists():
                    continue
                first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
                samples.append(
                    {
                        "image": image,
                        "messages": messages,
                        "first_user": str(first_user)[:100],
                        "source": p.parent.name,
                        "sample_type": record.get("sample_type", ""),
                    }
                )
                count += 1
    return samples


def detect_devices() -> list[str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return ["cpu"]
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"]
    except Exception:
        return ["cpu"]


def default_device_index(devices: list[str]) -> int:
    requested = os.environ.get("STREAMLIT_DEVICE", "").strip()
    if requested in devices:
        return devices.index(requested)
    for preferred in ("cuda:2", "cuda:0"):
        if preferred in devices:
            return devices.index(preferred)
    return 0


def device_label(name: str) -> str:
    try:
        import torch

        if not name.startswith("cuda") or not torch.cuda.is_available():
            return name
        props = torch.cuda.get_device_properties(torch.device(name))
        free, total = torch.cuda.mem_get_info(torch.device(name))
        return f"{name} | {props.name} | {(total - free) / 1024**3:.1f}/{total / 1024**3:.1f} GB"
    except Exception:
        return name


@st.cache_resource(show_spinner="Đang load model...")
def load_model_resource(
    checkpoint_path: str,
    vision_model: str,
    llm_model: str,
    model_dtype: str,
    projector_dtype: str,
    max_text_tokens: int,
    device_name: str,
):
    import torch
    from src.collators import InstructionCollator
    from src.modeling import build_model
    from src.training.checkpoint import load_full_checkpoint

    device = torch.device(device_name)
    llm_source, tokenizer_source = resolve_checkpoint_sources(checkpoint_path, llm_model)
    collator = InstructionCollator(vision_model, tokenizer_source, max_text_tokens=max_text_tokens)
    model = build_model(
        vision_model,
        llm_source,
        tokenizer_name_or_path=tokenizer_source,
        model_dtype=model_dtype,
        projector_dtype=projector_dtype,
        image_token_id=collator.image_token_id,
        vocab_size=len(collator.tokenizer),
    )
    state = load_full_checkpoint(checkpoint_path, model)
    step = int(state.get("global_step") or checkpoint_step(Path(checkpoint_path)))
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return model, collator, step


def eos_token_ids(tokenizer) -> list[int]:
    ids = {tokenizer.eos_token_id}
    for token in ("<|eot_id|>", "<|end_of_text|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0:
            ids.add(tid)
    return sorted(i for i in ids if i is not None)


def generate_reply(
    model,
    collator,
    image: Image.Image,
    messages: list[dict],
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.1,
) -> tuple[str, str]:
    import torch

    device = next(model.parameters()).device
    vision_dtype = next(model.vision_tower.parameters()).dtype
    tokenizer = collator.tokenizer

    prompt_ids, attn_mask, pixel_values = collator.build_prompt_tensors(messages, image, device=device)
    pixel_values = pixel_values.to(dtype=vision_dtype)

    use_sampling = temperature > 0.0
    gen_kwargs: dict = {
        "input_ids": prompt_ids,
        "pixel_values": pixel_values,
        "attention_mask": attn_mask,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "eos_token_id": eos_token_ids(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": use_sampling,
        "repetition_penalty": repetition_penalty,
    }
    if use_sampling:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})

    autocast_ctx = nullcontext()
    if device.type == "cuda" and vision_dtype in (torch.float16, torch.bfloat16):
        autocast_ctx = torch.autocast(device_type="cuda", dtype=vision_dtype)

    with torch.inference_mode(), autocast_ctx:
        generated_ids = model.generate(**gen_kwargs)

    input_len = prompt_ids.shape[1]
    new_tokens = generated_ids[0, input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    return text, raw_text


def main() -> None:
    st.title("Test Instruction Model")

    try:
        cfg = load_instruction_config()
    except Exception as error:
        st.error(f"Không đọc được config.yaml: {error}")
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Checkpoint")
        output_dir_text = st.text_input("Output dir", value=str(DEFAULT_OUTPUT_DIR))
        output_dir = Path(output_dir_text).expanduser().resolve()
        checkpoints = find_checkpoints(output_dir)
        if not checkpoints:
            st.error(f"Không tìm thấy checkpoint trong {output_dir}")
            st.stop()

        checkpoint = st.selectbox(
            "Checkpoint",
            checkpoints,
            index=default_checkpoint_index(output_dir, checkpoints),
            format_func=lambda p: f"{p.name} (step {checkpoint_step(p)})",
        )
        effective_cfg = merge_config(cfg, checkpoint)

        st.header("Model")
        vision_model = st.text_input("Vision model", value=str(effective_cfg.get("vision_model", "")))
        llm_model = st.text_input("LLM model", value=str(effective_cfg.get("llm_model", "")))

        dtype_opts = ["bfloat16", "float16", "float32", "auto"]
        mdtype_val = str(effective_cfg.get("model_dtype", "bfloat16"))
        model_dtype = st.selectbox(
            "Model dtype", dtype_opts, index=dtype_opts.index(mdtype_val) if mdtype_val in dtype_opts else 0
        )

        pdtype_opts = ["float32", "bfloat16", "float16"]
        pdtype_val = str(effective_cfg.get("projector_dtype", "float32"))
        projector_dtype = st.selectbox(
            "Projector dtype",
            pdtype_opts,
            index=pdtype_opts.index(pdtype_val) if pdtype_val in pdtype_opts else 0,
        )

        max_text_tokens = st.number_input(
            "Max text tokens",
            min_value=256,
            max_value=4096,
            value=int(effective_cfg.get("max_text_tokens", 2048)),
            step=256,
        )

        devices = detect_devices()
        device_name = st.selectbox(
            "Device", devices, index=default_device_index(devices), format_func=device_label
        )

        st.header("System Prompt")
        system_prompt = st.text_area(
            "system_prompt", label_visibility="collapsed", value=DEFAULT_SYSTEM_PROMPT, height=110
        )

        st.header("Generate")
        max_new_tokens = st.slider(
            "Max new tokens", 16, 768, int(effective_cfg.get("max_new_tokens", 256)), step=16
        )
        min_new_tokens = st.slider("Min new tokens", 0, 64, 5, step=1)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.0, step=0.05)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
        repetition_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.1, step=0.05)

        if st.button("Clear model cache"):
            load_model_resource.clear()
            st.rerun()

    # --- Image selection ---
    samples = load_eval_samples(cfg.get("eval_jsonl", []), limit=200)
    uploaded_file = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png", "webp"])

    left, right = st.columns([0.4, 0.6], gap="large")

    with left:
        selected_sample = None
        if uploaded_file is None and samples:
            selected_sample = st.selectbox(
                "Ảnh mẫu từ eval",
                samples,
                format_func=lambda x: (
                    f"[{x.get('source', '')}] {Path(x['image']).name} | {x['first_user'][:55]}"
                ),
            )

        try:
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                image_key = uploaded_file.name
            elif selected_sample is not None:
                with Image.open(selected_sample["image"]) as img:
                    image = img.convert("RGB")
                image_key = selected_sample["image"]
            else:
                st.info("Upload một ảnh hoặc chọn từ eval để bắt đầu.")
                st.stop()
        except Exception as error:
            st.error(f"Không đọc được ảnh: {error}")
            st.stop()

        st.image(image, caption=Path(image_key).name, use_container_width=True)

        if selected_sample and uploaded_file is None:
            with st.expander("Reference conversation"):
                for msg in selected_sample["messages"]:
                    role = msg["role"]
                    content = str(msg["content"])
                    if role == "system":
                        st.caption(f"**[system]** {content}")
                    elif role == "user":
                        st.markdown(f"**User:** {content}")
                    else:
                        st.markdown(f"**Assistant:** {content}")

    # --- Chat interface ---
    with right:
        session_key = f"chat_{image_key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = []

        chat_history: list[dict] = st.session_state[session_key]

        if st.button("Xóa conversation"):
            st.session_state[session_key] = []
            st.rerun()

        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Nhập câu hỏi...")
        if user_input:
            chat_history.append({"role": "user", "content": user_input})

            model_messages: list[dict] = []
            if system_prompt.strip():
                model_messages.append({"role": "system", "content": system_prompt.strip()})
            for msg in chat_history:
                model_messages.append({"role": msg["role"], "content": msg["content"]})

            with st.spinner("Đang sinh câu trả lời..."):
                try:
                    model, collator, step = load_model_resource(
                        str(checkpoint),
                        vision_model,
                        llm_model,
                        model_dtype,
                        projector_dtype,
                        int(max_text_tokens),
                        device_name,
                    )
                    text, raw_text = generate_reply(
                        model=model,
                        collator=collator,
                        image=image,
                        messages=model_messages,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                except Exception as error:
                    st.error(f"Lỗi inference: {error}")
                    st.exception(error)
                    chat_history.pop()
                    st.stop()

            chat_history.append({"role": "assistant", "content": text or "<empty>"})
            st.rerun()


if __name__ == "__main__":
    main()
````

### `src/modeling.py`

````python
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
    """
    Build LlavaForConditionalGeneration from pretrained SigLIP2 + Llama weights.

    Pass image_token_id and vocab_size from an already-built processor to avoid
    a redundant processor build. If None, build_processor() resolves them.
    projector_state: if provided, load into multi_modal_projector (stage-1 warm-start).
    """

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
    """Enable/disable gradients for each major model component."""

    model.vision_tower.requires_grad_(not freeze_vision)
    model.multi_modal_projector.requires_grad_(train_projector)
    model.language_model.requires_grad_(train_llm)
    model.lm_head.requires_grad_(train_llm)

    set_component_modes(model, freeze_vision, train_projector, train_llm)


def set_component_modes(
    model: LlavaForConditionalGeneration, freeze_vision: bool, train_projector: bool, train_llm: bool
) -> None:
    """Set `.train()` modes for each major model component."""

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
    """
    HF LLaVA passes vision-tower hidden states directly into the projector.

    When the frozen vision tower runs in bf16 but the projector is kept in fp32
    for stable stage-1 alignment, the stock projector forward raises a dtype
    mismatch in the first Linear. Keep the original module and state_dict keys,
    but cast inputs to the projector weight dtype before the linear layers.
    """

    def forward(self, image_features):
        projector_dtype = self.linear_1.weight.dtype
        if image_features.dtype != projector_dtype:
            image_features = image_features.to(projector_dtype)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        return self.linear_2(hidden_states)

    projector.forward = MethodType(forward, projector)


def _patch_last_hidden_state_image_features(model: LlavaForConditionalGeneration) -> None:
    """
    Stock HF LLaVA calls vision_tower with output_hidden_states=True, which retains
    every layer's activations in memory. For SigLIP2-so400m-384px (576 tokens × 48 layers)
    this is hundreds of MB of VRAM with no benefit when vision_feature_layer=-1, because
    hidden_states[-1] == last_hidden_state. This patch uses output_hidden_states=False and
    reads last_hidden_state directly, giving identical results at much lower peak VRAM.
    """

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
````

### `src/data.py`

````python
"""Dataset definitions for caption pretraining and instruction tuning."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.runtime import resolve_record_image_path


class ImageCaptionDataset(Dataset):
    """Load image-caption pairs from one or more JSONL files."""

    def __init__(self, jsonl_path: str | Path | list[str | Path]):
        paths = [jsonl_path] if isinstance(jsonl_path, (str, Path)) else jsonl_path
        self.jsonl_paths = [Path(path).expanduser().resolve() for path in paths]
        self.records = []
        self.bad_indices = set()

        self.source_indices: list[int] = []

        for source_idx, p in enumerate(self.jsonl_paths):
            with p.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    record = json.loads(line)
                    if "image" not in record or "caption" not in record:
                        raise ValueError(
                            f"Each JSONL line must contain 'image' and 'caption'. Bad line {line_number} in {p}."
                        )
                    record["image"] = resolve_record_image_path(record["image"], jsonl_path=p)
                    self.records.append(record)
                    self.source_indices.append(source_idx)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        n = len(self.records)
        for offset in range(n):
            current_index = (index + offset) % n
            if current_index in self.bad_indices:
                continue

            record = self.records[current_index]
            image_path = record["image"]

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
            except Exception as error:
                warnings.warn(f"Skipping corrupt image at {image_path}: {error}")
                self.bad_indices.add(current_index)
                continue

            return {"image": image, "caption": record["caption"]}

        raise RuntimeError(f"All images at index {index} are invalid or unreadable.")


def _validate_messages(messages, *, sample_id: str) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Sample '{sample_id}' must contain a non-empty 'messages' list.")

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Sample '{sample_id}' message #{message_index} must be a mapping.")

        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Sample '{sample_id}' message #{message_index} has unsupported role '{role}'.")
        if not content:
            raise ValueError(f"Sample '{sample_id}' message #{message_index} has empty content.")

    if messages[-1]["role"] != "assistant":
        raise ValueError(f"Sample '{sample_id}' must end with an assistant message.")


class ImageInstructionDataset(Dataset):
    """Load image-chat examples for instruction tuning from one or more JSONL files."""

    def __init__(self, jsonl_path: str | Path | list[str | Path]):
        paths = [jsonl_path] if isinstance(jsonl_path, (str, Path)) else jsonl_path
        self.jsonl_paths = [Path(p).expanduser().resolve() for p in paths]
        self.records = []
        self.source_indices: list[int] = []
        self.bad_indices = set()

        for source_idx, p in enumerate(self.jsonl_paths):
            with p.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    record = json.loads(line)
                    sample_id = str(record.get("id", f"line_{line_number}"))
                    image_path = str(record.get("image", "")).strip()
                    messages = record.get("messages")

                    if not image_path:
                        raise ValueError(f"Sample '{sample_id}' on line {line_number} is missing 'image'.")
                    _validate_messages(messages, sample_id=sample_id)
                    image_path = resolve_record_image_path(image_path, jsonl_path=p)

                    self.records.append(
                        {
                            "id": sample_id,
                            "image": image_path,
                            "messages": messages,
                            "sample_type": str(record.get("sample_type", "unknown")),
                            "image_id": str(record.get("image_id", "")),
                            "source_dataset": str(record.get("source_dataset", "")),
                        }
                    )
                    self.source_indices.append(source_idx)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        n = len(self.records)
        for offset in range(n):
            current_index = (index + offset) % n
            if current_index in self.bad_indices:
                continue

            record = self.records[current_index]
            image_path = record["image"]

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
            except Exception as error:
                warnings.warn(f"Skipping corrupt instruction image at {image_path}: {error}")
                self.bad_indices.add(current_index)
                continue

            return {
                "id": record["id"],
                "image": image,
                "image_path": image_path,
                "messages": record["messages"],
                "sample_type": record["sample_type"],
                "image_id": record["image_id"],
                "source_dataset": record["source_dataset"],
            }

        paths = ", ".join(str(path) for path in self.jsonl_paths)
        raise RuntimeError(f"All images in {paths} are invalid or unreadable.")
````

### `src/collators.py`

````python
"""Batch collation for caption pretraining and instruction tuning."""

from __future__ import annotations

import warnings

import torch

from src.modeling import IMAGE_TOKEN, build_processor

_CAPTION_PROMPT = "Mô tả hình ảnh này: "
PROMPT_TEMPLATE = f"{IMAGE_TOKEN}\n{_CAPTION_PROMPT}"


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
````

### `src/runtime.py`

````python
"""Cross-cutting runtime utilities used by scripts and training entrypoints."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
import yaml
from loguru import logger
from torch.utils.data import Sampler, WeightedRandomSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(section_name: str) -> dict:
    """Load a named config section from `config.yaml`."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if section_name not in config:
        raise KeyError(f"Missing '{section_name}' section in {CONFIG_PATH}")

    section = config[section_name]
    if not isinstance(section, dict):
        raise TypeError(f"Config section '{section_name}' must be a mapping.")
    return section


def resolve_config_path(value: str | Path) -> str:
    """Resolve a single path-like config value to an absolute string path."""

    return str(Path(value).expanduser().resolve())


def resolve_config_paths(value: str | Path | list[str | Path]) -> str | list[str]:
    """Resolve one or many path-like config values to absolute string paths."""

    if isinstance(value, list):
        return [resolve_config_path(item) for item in value]
    return resolve_config_path(value)


def resolve_record_image_path(image_value: str | Path, *, jsonl_path: Path) -> str:
    """Resolve an image path stored in a JSONL record."""

    raw_path = Path(str(image_value).strip()).expanduser()
    if raw_path.is_absolute():
        return str(raw_path)

    candidates = (PROJECT_ROOT / raw_path, Path.cwd() / raw_path, jsonl_path.parent / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str((PROJECT_ROOT / raw_path).resolve())


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs for reproducible runs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: str | Path, accelerator):
    """Configure Loguru sinks for the main process only."""

    logger.remove()
    if not accelerator.is_main_process:
        return logger

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.add(_tqdm_sink, level="INFO", 
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <6}</level> | {message}",
        colorize=True)
    logger.add(
        output_path / "train.log", level="INFO",
        encoding="utf-8", rotation="50 MB",
        retention=5, enqueue=True,
    )
    return logger


def _tqdm_sink(message) -> None:
    tqdm.write(str(message), end="")


class EpochShuffleSampler(Sampler[int]):
    """Shuffle indices deterministically per epoch."""

    def __init__(self, dataset, seed: int):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


def build_weighted_sampler(
    dataset, seed: int, source_weights: list[float] | None = None
) -> WeightedRandomSampler:
    """Weighted sampler across multiple source jsonl files.

    source_weights: desired sampling proportion per source (will be normalised).
    If None, each source contributes equally regardless of size.
    """
    n_sources = len(dataset.jsonl_paths)
    counts = [0] * n_sources
    for src_idx in dataset.source_indices:
        counts[src_idx] += 1

    if source_weights is None:
        target = [1.0] * n_sources
    else:
        if len(source_weights) != n_sources:
            raise ValueError(
                f"sample_weights has {len(source_weights)} entries but dataset has {n_sources} sources."
            )
        target = list(source_weights)

    total = sum(target)
    per_sample = [t / (total * c) if c > 0 else 0.0 for t, c in zip(target, counts)]
    weights = [per_sample[src_idx] for src_idx in dataset.source_indices]

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=weights, num_samples=len(dataset), replacement=True, generator=generator
    )


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record to a JSONL file."""

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def current_lr(scheduler, optimizer) -> float:
    """Return current learning rate from scheduler, falling back to optimizer."""
    try:
        return float(scheduler.get_last_lr()[0])
    except Exception:
        return float(optimizer.param_groups[0]["lr"])
````

### `src/training/__init__.py`

````python
"""Shared training utilities for caption pretraining and instruction tuning."""
````

### `src/training/checkpoint.py`

````python
from __future__ import annotations

import json
import platform
import random
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import yaml
from loguru import logger


def _remap_projector_state(state_dict: dict) -> dict:
    """Map legacy sequential projector keys to HF LLaVA projector keys."""

    mapping = {
        "0.weight": "linear_1.weight",
        "0.bias": "linear_1.bias",
        "2.weight": "linear_2.weight",
        "2.bias": "linear_2.bias",
    }
    return {mapping.get(k, k): v for k, v in state_dict.items()}


def save_training_checkpoint(
    *,
    path: str | Path,
    model,
    processor,
    tokenizer,
    optimizer,
    scheduler,
    training_config: dict,
    trainer_state: dict,
    stage: str,
    save_language_model: bool,
) -> None:
    """Save a reproducible directory checkpoint for a training stage."""

    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"projector_state_dict": model.multi_modal_projector.state_dict()}, checkpoint_dir / "projector.pt"
    )
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    torch.save(_rng_state(int(training_config.get("seed", 0))), checkpoint_dir / "rng_state.pt")

    model.config.save_pretrained(checkpoint_dir / "model_config")
    processor.save_pretrained(checkpoint_dir / "processor")
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

    if save_language_model:
        torch.save(model.lm_head.state_dict(), checkpoint_dir / "lm_head.pt")
        model.language_model.save_pretrained(checkpoint_dir / "llm", safe_serialization=True)

    full_state = {"stage": stage, **trainer_state}
    _write_json(checkpoint_dir / "trainer_state.json", full_state)
    _write_json(checkpoint_dir / "package_versions.json", _package_versions())
    with (checkpoint_dir / "training_config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(training_config, fh, sort_keys=True, allow_unicode=True)


def load_projector_checkpoint(
    path: str | Path, model, optimizer=None, scheduler=None, restore_rng: bool = False
) -> dict[str, Any]:
    """Load projector checkpoints from new directories or legacy `.pt` files."""

    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return _load_legacy_projector_ckpt(checkpoint_path, model, optimizer, scheduler)

    raw = torch.load(checkpoint_path / "projector.pt", map_location="cpu", weights_only=True)
    raw_state = raw["projector_state_dict"] if "projector_state_dict" in raw else raw
    model.multi_modal_projector.load_state_dict(_remap_projector_state(raw_state), strict=True)
    _maybe_load_optimizer_scheduler(checkpoint_path, optimizer, scheduler)
    if restore_rng:
        _restore_rng_state(checkpoint_path / "rng_state.pt")
    return _load_trainer_state(checkpoint_path)


def load_full_checkpoint(
    path: str | Path, model, optimizer=None, scheduler=None, restore_rng: bool = False
) -> dict[str, Any]:
    """Load trainable full-stage state from a directory checkpoint."""

    checkpoint_dir = Path(path)
    state = load_projector_checkpoint(checkpoint_dir, model, optimizer, scheduler, restore_rng=restore_rng)
    lm_head_path = checkpoint_dir / "lm_head.pt"
    if lm_head_path.exists():
        model.lm_head.load_state_dict(torch.load(lm_head_path, map_location="cpu", weights_only=True))
    return state


def update_checkpoint_pointer(
    output_dir: str | Path,
    name: str,
    checkpoint_path: str | Path,
    *,
    step: int,
    metric_name: str | None = None,
    metric_value: float | None = None,
) -> None:
    """Write a small JSON pointer for best or last checkpoint."""

    payload = {"checkpoint": str(Path(checkpoint_path).resolve()), "step": int(step)}
    if metric_name is not None:
        payload["metric_name"] = metric_name
        payload["metric_value"] = metric_value
    _write_json(Path(output_dir) / f"{name}_checkpoint.json", payload)


def rotate_checkpoints(
    output_dir: str | Path, keep_last_n: int, protected_paths: set[str | Path] | None = None
) -> None:
    """Delete old checkpoints while preserving protected best and last paths."""

    output_path = Path(output_dir)
    protected = {Path(p).resolve() for p in protected_paths or set()}
    for pointer in ("best_checkpoint.json", "last_checkpoint.json"):
        pointer_path = output_path / pointer
        if pointer_path.exists():
            with pointer_path.open("r", encoding="utf-8") as fh:
                protected.add(Path(json.load(fh)["checkpoint"]).resolve())

    checkpoints = []
    for path in output_path.glob("checkpoint-*"):
        step = _checkpoint_step(path)
        if step is not None:
            checkpoints.append((step, path))

    checkpoints.sort(key=lambda item: item[0])
    candidates = [(step, path) for step, path in checkpoints if path.resolve() not in protected]
    to_delete = candidates[:-keep_last_n] if keep_last_n > 0 else candidates
    for _, path in to_delete:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _load_legacy_projector_ckpt(path: Path, model, optimizer=None, scheduler=None) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    state = _remap_projector_state(ckpt["projector_state_dict"])
    model.multi_modal_projector.load_state_dict(state, strict=True)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        _safe_load_optimizer_state(optimizer, ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return {"global_step": int(ckpt.get("step", 0)), "epoch": 0, "best_eval_loss": None}


def _load_trainer_state(checkpoint_dir: Path) -> dict[str, Any]:
    try:
        with (checkpoint_dir / "trainer_state.json").open("r", encoding="utf-8") as fh:
            state = json.load(fh)
    except FileNotFoundError:
        return {"global_step": 0, "epoch": 0, "best_eval_loss": None}
    if "global_step" not in state and "step" in state:
        state["global_step"] = state["step"]
    state.setdefault("global_step", 0)
    state.setdefault("epoch", 0)
    state.setdefault("best_eval_loss", None)
    return state


def _maybe_load_optimizer_scheduler(checkpoint_dir: Path, optimizer=None, scheduler=None) -> None:
    if optimizer is not None and (checkpoint_dir / "optimizer.pt").exists():
        _safe_load_optimizer_state(
            optimizer, torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True)
        )
    if scheduler is not None and (checkpoint_dir / "scheduler.pt").exists():
        scheduler.load_state_dict(torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu", weights_only=True))


def _safe_load_optimizer_state(optimizer, state_dict) -> None:
    try:
        optimizer.load_state_dict(state_dict)
    except ValueError as error:
        logger.warning(
            "skipped optimizer state because the "
            f"projector parameter set changed: {error}"
        )


def _checkpoint_step(path: Path) -> int | None:
    step_text = path.name.replace("checkpoint-", "", 1).removesuffix(".pt")
    return int(step_text) if step_text.isdigit() else None


def _rng_state(seed: int) -> dict[str, Any]:
    state = {"seed": seed, "python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(path: Path) -> None:
    if not path.exists():
        return
    state = torch.load(path, map_location="cpu", weights_only=False)  # contains Python random state
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _package_versions() -> dict[str, Any]:
    packages = {}
    for name in ("accelerate", "torch", "transformers", "pyyaml", "pillow"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "packages": packages,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
````

### `src/training/engine.py`

````python
from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TrainingState:
    """Mutable progress state shared by training, eval, and checkpoint hooks."""

    global_step: int = 0
    epoch: int = 0
    best_eval_loss: float | None = None


@dataclass(frozen=True)
class StepResult:
    """Token-weighted metrics emitted after one optimizer update."""

    global_step: int
    epoch: int
    train_loss: float
    supervised_tokens: int


StepCallback = Callable[[StepResult, TrainingState], None]
ModeCallback = Callable[[], None]
EpochCallback = Callable[[int], None]
ParamsCallback = Callable[[], Iterable[torch.nn.Parameter]]


def compute_steps_per_epoch(loader_len: int, grad_accum: int) -> int:
    """Return optimizer steps per epoch for a dataloader length."""

    if grad_accum <= 0:
        raise ValueError("grad_accum must be a positive integer.")
    return (loader_len + grad_accum - 1) // grad_accum


def run_training(
    *, model, train_loader, optimizer, scheduler, accelerator, epochs: int, grad_accum: int, state: TrainingState,
    set_train_mode: ModeCallback, trainable_parameters: ParamsCallback, on_step_end: StepCallback | None = None, on_epoch_start: EpochCallback | None = None,
    max_grad_norm: float = 1.0,
) -> TrainingState:
    """Run a token-weighted training loop with explicit accumulation windows."""

    steps_per_epoch = compute_steps_per_epoch(len(train_loader), grad_accum)
    starting_epoch = state.global_step // steps_per_epoch if steps_per_epoch else 0
    batches_to_skip = (state.global_step % steps_per_epoch) * grad_accum if steps_per_epoch else 0

    for epoch in range(starting_epoch, int(epochs)):
        state.epoch = epoch
        if on_epoch_start is not None:
            on_epoch_start(epoch)
        set_train_mode()

        iterator = iter(train_loader)
        for _ in range(batches_to_skip):
            next(iterator, None)
        batches_to_skip = 0

        while True:
            window = _next_window(iterator, grad_accum)
            if not window:
                break

            step_result = _train_window(
                model=model, window=window, optimizer=optimizer, scheduler=scheduler, accelerator=accelerator,
                trainable_parameters=trainable_parameters, max_grad_norm=max_grad_norm,
            )
            if step_result is None:
                continue

            state.global_step += 1
            result = StepResult(
                global_step=state.global_step,
                epoch=epoch,
                train_loss=step_result["train_loss"],
                supervised_tokens=step_result["supervised_tokens"],
            )
            if on_step_end is not None:
                on_step_end(result, state)
            set_train_mode()

    return state


def _next_window(iterator, grad_accum: int) -> list[dict[str, Any]]:
    window = []
    for _ in range(grad_accum):
        try:
            window.append(next(iterator))
        except StopIteration:
            break
    return window


def _train_window(
    *, model, window: list[dict[str, Any]], optimizer, scheduler, accelerator,
    trainable_parameters: ParamsCallback, max_grad_norm: float,
) -> dict[str, float | int] | None:
    local_tokens = torch.stack([_supervised_tokens(batch) for batch in window]).sum()
    global_tokens = accelerator.gather(local_tokens).sum().item()
    if global_tokens <= 0:
        return None

    optimizer.zero_grad(set_to_none=True)
    local_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    local_token_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)

    for batch_idx, batch in enumerate(window):
        sync_context = (
            accelerator.no_sync(model)
            if batch_idx < len(window) - 1 and accelerator.num_processes > 1
            else nullcontext()
        )
        with sync_context:
            with accelerator.autocast():
                outputs = model(**batch)

            batch_tokens = _supervised_tokens(batch)
            if batch_tokens.item() <= 0:
                continue

            batch_loss_sum = outputs.loss.float() * batch_tokens.float()
            scaled_loss = batch_loss_sum * accelerator.num_processes / global_tokens
            accelerator.backward(scaled_loss)
            local_loss_sum = local_loss_sum + batch_loss_sum.detach()
            local_token_sum = local_token_sum + batch_tokens.float()

    accelerator.clip_grad_norm_(list(trainable_parameters()), max_grad_norm)
    optimizer.step()
    scheduler.step()

    stats = torch.stack([local_loss_sum, local_token_sum]).unsqueeze(0)
    gathered = accelerator.gather_for_metrics(stats)
    token_count = gathered[:, 1].sum().item()
    train_loss = gathered[:, 0].sum().item() / token_count
    return {"train_loss": float(train_loss), "supervised_tokens": int(token_count)}


def _supervised_tokens(batch: dict[str, Any]) -> torch.Tensor:
    labels = batch["labels"]
    return (labels != -100).sum()
````

### `src/training/eval.py`

````python
from __future__ import annotations

from collections.abc import Callable

import torch


SampleLogger = Callable[[object, object], list[str]]
ModeCallback = Callable[[], None]


def evaluate_loss(model, eval_loader, accelerator) -> float:
    """Compute token-weighted eval loss over the full eval split."""

    total_loss, total_tokens = 0.0, 0.0
    for batch in eval_loader:
        with torch.no_grad(), accelerator.autocast():
            outputs = model(**batch)
        sup_tokens = (batch["labels"] != -100).sum()
        stats = torch.stack([outputs.loss.detach().float() * sup_tokens.float(), sup_tokens.float()])
        gathered = accelerator.gather_for_metrics(stats.unsqueeze(0))
        total_loss += gathered[:, 0].sum().item()
        total_tokens += gathered[:, 1].sum().item()
    return total_loss / total_tokens if total_tokens > 0 else float("nan")


def run_evaluation(
    *,
    model,
    eval_loader,
    accelerator,
    global_step: int,
    logger,
    sample_logger: SampleLogger | None = None,
    restore_train_mode: ModeCallback | None = None,
) -> float:
    """Evaluate token-weighted loss, optionally log generations, then restore modes."""

    accelerator.unwrap_model(model).eval()
    eval_loss = evaluate_loss(model, eval_loader, accelerator)
    logger.info("step {}: eval_loss={:.6f}", global_step, eval_loss)

    if sample_logger is not None and accelerator.is_main_process:
        lines = sample_logger(model, accelerator)
        for line in lines:
            logger.info(line)

    if restore_train_mode is not None:
        restore_train_mode()
    return eval_loss
````

### `scripts/prepare_uit_openviic.py`

````python
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import gdown

from src.runtime import load_config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_valid_download(output_path: Path) -> bool:
    if not output_path.exists():
        return False

    if output_path.suffix.lower() == ".json":
        try:
            with output_path.open("r", encoding="utf-8") as handle:
                json.load(handle)
            return True
        except Exception:
            return False

    if output_path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(output_path, "r") as archive:
                return archive.testzip() is None
        except zipfile.BadZipFile:
            return False

    return output_path.stat().st_size > 0


def download_file(file_id: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_valid_download(output_path):
        print(f"Skip existing download: {output_path}")
        return
    if output_path.exists():
        output_path.unlink()

    gdown.download(id=file_id, output=str(output_path), quiet=False, resume=True)
    if not output_path.exists():
        raise FileNotFoundError(f"Download did not create expected file: {output_path}")


def extract_images_zip(images_zip_path: Path, images_root: Path) -> None:
    marker_path = images_root / ".extracted"
    if marker_path.exists():
        print(f"Skip extraction, marker found: {marker_path}")
        return

    temp_extract_dir = images_root.parent / "_images_extract_tmp"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(images_zip_path, "r") as archive:
        archive.extractall(temp_extract_dir)

    extracted_entries = list(temp_extract_dir.iterdir())
    source_root = (
        extracted_entries[0]
        if len(extracted_entries) == 1 and extracted_entries[0].is_dir()
        else temp_extract_dir
    )

    images_root.mkdir(parents=True, exist_ok=True)
    for child in source_root.iterdir():
        destination = images_root / child.name
        if destination.exists():
            continue
        shutil.move(str(child), str(destination))

    shutil.rmtree(temp_extract_dir, ignore_errors=True)
    marker_path.write_text("ok\n", encoding="utf-8")


def ensure_symlink(source_path: Path, output_path: Path) -> None:
    source_path = source_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink():
        if output_path.resolve() == source_path:
            return
        raise ValueError(f"{output_path} already exists and points somewhere else.")

    if output_path.exists():
        raise ValueError(f"{output_path} already exists and is not a symlink.")

    os.symlink(source_path, output_path, target_is_directory=source_path.is_dir())


def build_image_index(images_root: Path) -> dict[str, Path]:
    image_index = {}
    for image_path in images_root.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if image_path.name in image_index:
            raise ValueError(f"Duplicate image filename found in extracted images: {image_path.name}")
        image_index[image_path.name] = image_path

    if not image_index:
        raise RuntimeError(f"No images found under {images_root}")
    return image_index


def iter_annotation_records(annotation_path: Path):
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotation_data = json.load(handle)

    if isinstance(annotation_data, dict) and "images" in annotation_data and "annotations" in annotation_data:
        image_id_to_file_name = {
            image_info["id"]: image_info["file_name"] for image_info in annotation_data["images"]
        }
        for annotation in annotation_data["annotations"]:
            image_name = image_id_to_file_name[annotation["image_id"]]
            captions = [annotation["caption"]]
            yield image_name, captions
        return

    if isinstance(annotation_data, dict):
        for image_name, metadata in annotation_data.items():
            if not isinstance(metadata, dict) or "captions" not in metadata:
                raise ValueError(
                    f"Unsupported annotation entry for image '{image_name}' in {annotation_path}"
                )
            yield image_name, metadata["captions"]
        return

    raise ValueError(f"Unsupported annotation format in {annotation_path}")


def maybe_download_dataset(config: dict, raw_dir: Path) -> None:
    download_config = config.get("download", {})
    if not download_config.get("enabled", False):
        return

    downloads_dir = raw_dir / "downloads"
    annotations_dir = raw_dir / "annotations"
    images_root = raw_dir / "images"

    images_download = download_config["images"]
    images_zip_path = downloads_dir / images_download["filename"]
    download_file(images_download["file_id"], images_zip_path)
    extract_images_zip(images_zip_path, images_root)

    annotation_name_map = {"train": "train.json", "val": "val.json", "test": "test.json"}
    for split, target_name in annotation_name_map.items():
        annotation_download = download_config["annotations"][split]
        downloaded_path = downloads_dir / annotation_download["filename"]
        target_path = annotations_dir / target_name
        download_file(annotation_download["file_id"], downloaded_path)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded_path, target_path)


def main() -> None:
    config = load_config("prepare_data")
    raw_dir = Path(config["raw_dir"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    maybe_download_dataset(config, raw_dir)

    source_annotations_dir = raw_dir / "annotations"
    source_images_root = raw_dir / "images"
    if not source_annotations_dir.exists():
        raise FileNotFoundError(f"Missing annotations directory: {source_annotations_dir}")
    if not source_images_root.exists():
        raise FileNotFoundError(f"Missing images directory: {source_images_root}")

    image_index = build_image_index(source_images_root)
    output_images_root = output_dir / "images"
    ensure_symlink(source_images_root, output_images_root)

    for split in ("train", "val", "test"):
        annotation_path = source_annotations_dir / f"{split}.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
        output_jsonl_path = output_dir / f"{split}.jsonl"
        skipped_empty_captions = 0
        written_rows = 0
        with output_jsonl_path.open("w", encoding="utf-8") as handle:
            for image_name, captions in iter_annotation_records(annotation_path):
                if image_name not in image_index:
                    raise FileNotFoundError(f"Missing image referenced by annotations: {image_name}")

                source_image_path = image_index[image_name]
                relative_image_path = source_image_path.relative_to(source_images_root)
                output_image_path = output_images_root / relative_image_path

                for caption in captions:
                    caption = str(caption).strip()
                    if not caption:
                        skipped_empty_captions += 1
                        continue

                    json.dump(
                        {"image": str(output_image_path.resolve()), "caption": caption},
                        handle,
                        ensure_ascii=False,
                    )
                    handle.write("\n")
                    written_rows += 1

        print(
            f"Wrote {output_jsonl_path} (rows={written_rows}, skipped_empty_captions={skipped_empty_captions})"
        )


if __name__ == "__main__":
    main()
````

### `scripts/prepare_coco_data.py`

````python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare DavidPhilips/coco2017 as local image-caption JSONL."
    )
    parser.add_argument("--config-section", default="prepare_coco")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def sanitize_id(value: str, fallback: str) -> str:
    text = str(value or "").strip() or fallback
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or fallback


def split_mapping(config: dict) -> dict[str, str]:
    raw = config.get("split_map") or {"train": "train", "validation": "val"}
    if not isinstance(raw, dict) or not raw:
        raise ValueError("prepare_coco.split_map must be a non-empty mapping.")
    return {str(hf_split): str(local_split) for hf_split, local_split in raw.items()}


def choose_caption(row: dict, caption_field: str) -> str:
    return str(row.get(caption_field, "") or "").strip()


def save_image(image_value, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    should_close = False
    if isinstance(image_value, Image.Image):
        image = image_value
    elif isinstance(image_value, dict) and image_value.get("bytes") is not None:
        from io import BytesIO

        image = Image.open(BytesIO(image_value["bytes"]))
        should_close = True
    elif isinstance(image_value, dict) and image_value.get("path"):
        image = Image.open(image_value["path"])
        should_close = True
    else:
        raise TypeError(f"Unsupported image payload type: {type(image_value)!r}")

    try:
        image.convert("RGB").save(destination, format="JPEG", quality=95)
    finally:
        if should_close:
            image.close()


def prepare_split(config: dict, hf_split: str, local_split: str, output_dir: Path) -> None:
    dataset_name = str(config["dataset_name"]).strip()
    dataset_config = config.get("dataset_config")
    caption_field = str(config.get("caption_field", "caption_vi")).strip()
    image_field = str(config.get("image_field", "image")).strip()
    image_id_field = str(config.get("image_id_field", "image_id")).strip()
    caption_id_field = str(config.get("caption_id_field", "caption_id")).strip()
    streaming = bool(config.get("streaming", True))
    overwrite = bool(config.get("overwrite", False))
    max_rows = config.get("max_rows_per_split")
    max_rows = int(max_rows) if max_rows is not None else None

    output_jsonl = output_dir / f"{local_split}.jsonl"
    if output_jsonl.exists() and not overwrite:
        print(f"Skip existing {output_jsonl}. Set overwrite=true to rebuild.")
        return

    print(
        f"[coco] loading {dataset_name} config={dataset_config or 'default'} split={hf_split} streaming={streaming}"
    )
    dataset = load_dataset(dataset_name, dataset_config, split=hf_split, streaming=streaming)

    images_dir = output_dir / "images" / local_split
    stats = Counter()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(dataset):
            if max_rows is not None and row_index >= max_rows:
                break

            image_id = sanitize_id(row.get(image_id_field), f"{local_split}_{row_index:08d}")
            caption_id = sanitize_id(row.get(caption_id_field), f"{row_index:08d}")
            caption = choose_caption(row, caption_field)
            if not caption:
                stats["skipped_empty_caption"] += 1
                continue

            image_path = images_dir / f"{image_id}.jpg"
            try:
                save_image(row[image_field], image_path)
            except Exception as error:
                stats["skipped_bad_image"] += 1
                print(f"[coco][warn] skip image_id={image_id} row={row_index}: {error}")
                continue

            json.dump(
                {
                    "image": str(image_path.resolve()),
                    "caption": caption,
                    "source_dataset": dataset_name,
                    "source_split": hf_split,
                    "image_id": image_id,
                    "caption_id": caption_id,
                },
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")
            stats["rows_written"] += 1

    print(
        f"[coco] wrote {output_jsonl} | "
        + " | ".join(f"{key}={value}" for key, value in sorted(stats.items()))
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config_section)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_rows_per_split is not None:
        config["max_rows_per_split"] = args.max_rows_per_split
    if args.overwrite:
        config["overwrite"] = True
    if args.inspect_only:
        config["max_rows_per_split"] = 3
        config["overwrite"] = True

    output_dir = Path(config["output_dir"]).expanduser().resolve()
    for hf_split, local_split in split_mapping(config).items():
        prepare_split(config, hf_split, local_split, output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    sys.stdout.flush()
    sys.stderr.flush()
    # Streaming image datasets can abort during native teardown in this env after
    # all Python file handles are closed. Exit directly once preparation succeeds.
    os._exit(0)
````

### `scripts/prepare_instruction_common.py`

````python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Image as HfImage
from datasets import load_dataset
from PIL import Image

from src.runtime import load_config


IMAGE_FIELD_CANDIDATES = ("image", "Image", "img")
DESCRIPTION_FIELD_CANDIDATES = ("Description", "description", "caption", "Caption")
QNA_FIELD_CANDIDATES = ("QnA", "qna", "messages", "conversations", "conversation")
DEFAULT_CONFIG_SECTION = "instruction_data_gpt"
SPLIT_NAME_MAP = {
    "train": "train",
    "training": "train",
    "validation": "val",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def log(message: str) -> None:
    print(message)


def log_check(name: str, condition: bool, detail: str) -> None:
    prefix = "[check]" if condition else "[error]"
    log(f"{prefix} {name}: {detail}")
    if not condition:
        raise RuntimeError(detail)


def sanitize_image_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "image"


def select_field(field_names, *, override, candidates, field_label: str) -> str:
    if override is not None:
        override = str(override).strip()
        log_check(
            f"{field_label}_override",
            override in field_names,
            f"Using configured {field_label} field '{override}'.",
        )
        return override

    for candidate in candidates:
        if candidate in field_names:
            log_check(
                f"{field_label}_auto_select", True, f"Selected field '{candidate}' from {list(field_names)}."
            )
            return candidate

    raise KeyError(f"Could not infer {field_label} field from available columns: {list(field_names)}")


def summarize_value(value):
    if isinstance(value, dict):
        return {key: summarize_value(sub_value) for key, sub_value in list(value.items())[:4]}
    if isinstance(value, list):
        return [summarize_value(item) for item in value[:2]]
    if isinstance(value, str):
        return value[:160]
    return str(type(value).__name__)


def normalize_qna_messages(raw_messages) -> list[dict[str, str]]:
    if raw_messages is None:
        return []
    if isinstance(raw_messages, str):
        raw_messages = json.loads(raw_messages)
    if not isinstance(raw_messages, list):
        raise ValueError("QnA field must be a list or a JSON-encoded list.")
    if len(raw_messages) % 2 != 0:
        raw_messages = raw_messages[:-1]  # drop trailing unpaired user message

    normalized_messages = []
    expected_role = "user"
    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("Each QnA message must be a mapping.")

        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role != expected_role:
            raise ValueError(f"Expected QnA role '{expected_role}' but received '{role}'.")
        if not content:
            raise ValueError(f"Encountered an empty '{role}' message.")

        normalized_messages.append({"role": role, "content": content})
        expected_role = "assistant" if expected_role == "user" else "user"

    return normalized_messages


def stable_split_for_image(image_key: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.sha1(f"{seed}:{image_key}".encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    if score < float(test_ratio):
        return "test"
    if score < float(test_ratio) + float(val_ratio):
        return "val"
    return "train"


def determine_output_split(
    raw_split: str, *, split_mode: str, image_key: str, seed: int, val_ratio: float, test_ratio: float
) -> str:
    normalized_mode = str(split_mode).strip().lower()
    normalized_raw_split = SPLIT_NAME_MAP.get(str(raw_split).strip().lower())

    if normalized_mode == "source":
        if normalized_raw_split is None:
            raise ValueError(f"Source split '{raw_split}' cannot be mapped to train/val/test.")
        return normalized_raw_split

    if normalized_mode == "image_level":
        return stable_split_for_image(image_key, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)

    if normalized_mode != "auto":
        raise ValueError(f"Unsupported split_mode '{split_mode}'.")

    return stable_split_for_image(image_key, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)


def infer_image_key(row: dict, image_field: str, row_index: int, raw_split: str) -> str:
    for candidate in ("image_id", "id", "file_name", "filename", "name"):
        candidate_value = row.get(candidate)
        if candidate_value is None:
            continue
        candidate_text = str(candidate_value).strip()
        if candidate_text:
            return sanitize_image_key(Path(candidate_text).stem or candidate_text)

    image_value = row[image_field]
    if isinstance(image_value, dict):
        image_path = image_value.get("path")
        if image_path:
            return sanitize_image_key(Path(str(image_path)).stem)

        image_bytes = image_value.get("bytes")
        if image_bytes:
            return hashlib.sha1(image_bytes).hexdigest()[:16]

    return f"{raw_split}_{row_index:07d}"


def infer_image_extension(image_value, image_key: str) -> str:
    if isinstance(image_value, dict):
        image_path = image_value.get("path")
        if image_path:
            suffix = Path(str(image_path)).suffix.lower()
            if suffix:
                return suffix
    if isinstance(image_value, Image.Image):
        return ".jpg"
    return ".png"


def save_image_asset(image_value, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image_value, dict):
        image_path = image_value.get("path")
        image_bytes = image_value.get("bytes")
        if image_path and Path(str(image_path)).exists():
            shutil.copy2(str(image_path), destination)
        elif image_bytes is not None:
            destination.write_bytes(image_bytes)
        else:
            raise ValueError("Image payload does not provide a valid path or bytes.")
    elif isinstance(image_value, Image.Image):
        image = image_value
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(destination, format="JPEG", quality=95)
    else:
        raise TypeError(f"Unsupported image value type: {type(image_value)!r}")

    with Image.open(destination) as image:
        if image.mode != "RGB":
            image.convert("RGB").save(destination)


def build_description_sample(
    *,
    sample_id: str,
    image_path: Path,
    image_key: str,
    system_prompt: str,
    user_prompt: str,
    assistant_text: str,
    source_dataset: str,
) -> dict:
    return {
        "id": sample_id,
        "image": str(image_path.resolve()),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ],
        "sample_type": "description",
        "source_dataset": source_dataset,
        "image_id": image_key,
    }


def build_qna_samples(
    *,
    image_path: Path,
    image_key: str,
    qna_messages: list[dict[str, str]],
    system_prompt: str,
    sample_id_prefix: str,
    source_dataset: str,
) -> list[dict]:
    return [
        {
            "id": f"{sample_id_prefix}_qa",
            "image": str(image_path.resolve()),
            "messages": [{"role": "system", "content": system_prompt}, *qna_messages],
            "sample_type": "qa",
            "source_dataset": source_dataset,
            "image_id": image_key,
        }
    ]


def parse_args(argv: list[str] | None = None, *, default_config_section: str = DEFAULT_CONFIG_SECTION):
    parser = argparse.ArgumentParser(
        description="Prepare image instruction-tuning JSONL from a Hugging Face dataset."
    )
    parser.add_argument("--config-section", default=default_config_section)
    return parser.parse_args(argv)


def load_dataset_from_config(config: dict):
    dataset_name = str(config["dataset_name"]).strip()
    dataset_config = config.get("dataset_config")
    kwargs = {"streaming": bool(config.get("streaming", False))}
    if dataset_config is not None:
        kwargs["name"] = str(dataset_config).strip()
    return load_dataset(dataset_name, **kwargs)


def get_first_row(dataset_dict, first_split: str, *, streaming: bool):
    first_dataset = dataset_dict[first_split]
    if streaming:
        return next(iter(first_dataset))
    return first_dataset[0]


def maybe_cast_images(dataset_dict, image_field: str, *, streaming: bool):
    if streaming:
        return dataset_dict
    return {
        split_name: split_dataset.cast_column(image_field, HfImage(decode=False))
        for split_name, split_dataset in dataset_dict.items()
    }


def dataset_len_text(dataset) -> str:
    try:
        return str(len(dataset))
    except TypeError:
        return "unknown (streaming)"


def run(config_section: str) -> None:
    config = load_config(config_section)
    dataset_name = str(config["dataset_name"]).strip()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    images_root = output_dir / "images"
    streaming = bool(config.get("streaming", False))

    log(f"[instruction-data] loading dataset '{dataset_name}' from config section '{config_section}'...")
    dataset_dict = load_dataset_from_config(config)
    raw_splits = list(dataset_dict.keys())
    log_check("dataset_load", bool(raw_splits), f"Loaded raw splits: {raw_splits}")

    first_split = raw_splits[0]
    first_features = list(dataset_dict[first_split].features.keys())
    image_field = select_field(
        first_features,
        override=config.get("image_field"),
        candidates=IMAGE_FIELD_CANDIDATES,
        field_label="image",
    )
    description_field = select_field(
        first_features,
        override=config.get("description_field"),
        candidates=DESCRIPTION_FIELD_CANDIDATES,
        field_label="description",
    )
    qna_field = select_field(
        first_features, override=config.get("qna_field"), candidates=QNA_FIELD_CANDIDATES, field_label="qna"
    )

    log("[instruction-data] schema summary:")
    for raw_split in raw_splits:
        log(f"  - {raw_split}: columns={list(dataset_dict[raw_split].features.keys())}")

    preview_row = get_first_row(dataset_dict, first_split, streaming=streaming)
    preview_summary = {
        image_field: summarize_value(preview_row[image_field]),
        description_field: summarize_value(preview_row.get(description_field)),
        qna_field: summarize_value(preview_row.get(qna_field)),
    }
    log(f"[instruction-data] first-row preview: {json.dumps(preview_summary, ensure_ascii=False)}")

    dataset_dict = maybe_cast_images(dataset_dict, image_field, streaming=streaming)

    configured_split_mode = str(config.get("split_mode", "auto")).strip().lower()
    mapped_raw_splits = {SPLIT_NAME_MAP.get(str(split_name).strip().lower()) for split_name in raw_splits}
    if configured_split_mode == "auto":
        effective_split_mode = (
            "source" if {"train", "val", "test"}.issubset(mapped_raw_splits) else "image_level"
        )
    else:
        effective_split_mode = configured_split_mode
    log_check(
        "split_mode",
        True,
        f"Resolved split_mode='{effective_split_mode}' from configured value '{configured_split_mode}'.",
    )

    if bool(config.get("inspect_only", False)):
        log("[instruction-data] inspect_only=true, stopping after schema checks.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    output_handles = {
        "train": (output_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (output_dir / "val.jsonl").open("w", encoding="utf-8"),
        "test": (output_dir / "test.jsonl").open("w", encoding="utf-8"),
    }

    counters = Counter()
    split_to_image_ids = defaultdict(set)
    image_id_to_split = {}
    written_images = set()
    approx_token_lengths = []
    max_rows = config.get("max_rows")
    if max_rows is not None:
        max_rows = int(max_rows)

    try:
        processed_rows = 0
        for raw_split in raw_splits:
            raw_dataset = dataset_dict[raw_split]
            log(
                f"[instruction-data] processing raw split '{raw_split}' "
                f"with {dataset_len_text(raw_dataset)} rows..."
            )

            for row_index, row in enumerate(raw_dataset):
                if max_rows is not None and processed_rows >= max_rows:
                    log(f"[instruction-data] reached max_rows={max_rows}, stopping early.")
                    break

                processed_rows += 1
                counters["rows_total"] += 1
                try:
                    image_key = infer_image_key(
                        row, image_field=image_field, row_index=row_index, raw_split=raw_split
                    )
                    output_split = determine_output_split(
                        raw_split,
                        split_mode=effective_split_mode,
                        image_key=image_key,
                        seed=int(config.get("split_seed", 42)),
                        val_ratio=float(config.get("val_ratio", 0.01)),
                        test_ratio=float(config.get("test_ratio", 0.01)),
                    )

                    previous_split = image_id_to_split.setdefault(image_key, output_split)
                    if previous_split != output_split:
                        raise RuntimeError(
                            f"Image '{image_key}' was assigned to both "
                            f"'{previous_split}' and '{output_split}'."
                        )

                    image_value = row[image_field]
                    image_extension = infer_image_extension(image_value, image_key=image_key)
                    image_output_path = images_root / output_split / f"{image_key}{image_extension}"
                    if image_output_path not in written_images:
                        save_image_asset(image_value, image_output_path)
                        written_images.add(image_output_path)
                        counters[f"{output_split}_images_written"] += 1

                    built_samples = []

                    description_text = str(row.get(description_field, "") or "").strip()
                    if bool(config.get("use_description_samples", True)):
                        if description_text:
                            built_samples.append(
                                build_description_sample(
                                    sample_id=f"{output_split}_{image_key}_desc",
                                    image_path=image_output_path,
                                    image_key=image_key,
                                    system_prompt=str(config["system_prompt"]).strip(),
                                    user_prompt=str(config["description_user_prompt"]).strip(),
                                    assistant_text=description_text,
                                    source_dataset=dataset_name,
                                )
                            )
                        else:
                            counters["rows_empty_description"] += 1

                    if bool(config.get("use_qna_samples", True)):
                        raw_qna = row.get(qna_field)
                        if raw_qna is not None and raw_qna != "":
                            qna_messages = normalize_qna_messages(raw_qna)
                            if qna_messages:
                                built_samples.extend(
                                    build_qna_samples(
                                        image_path=image_output_path,
                                        image_key=image_key,
                                        qna_messages=qna_messages,
                                        system_prompt=str(config["system_prompt"]).strip(),
                                        sample_id_prefix=f"{output_split}_{image_key}",
                                        source_dataset=dataset_name,
                                    )
                                )
                            else:
                                counters["rows_empty_qna"] += 1
                    if not built_samples:
                        counters["rows_skipped_no_samples"] += 1
                        continue

                    split_to_image_ids[output_split].add(image_key)
                    for sample in built_samples:
                        json.dump(sample, output_handles[output_split], ensure_ascii=False)
                        output_handles[output_split].write("\n")
                        counters[f"{output_split}_samples_written"] += 1
                        counters["samples_written"] += 1
                        approx_token_lengths.append(
                            sum(len(str(message["content"]).split()) for message in sample["messages"])
                        )
                except Exception as error:
                    error_text = str(error).lower()
                    if (
                        "image payload" in error_text
                        or "unsupported image" in error_text
                        or "cannot identify image file" in error_text
                    ):
                        counters["rows_missing_or_bad_image"] += 1
                    elif "qna" in error_text or "assistant" in error_text or "user" in error_text:
                        counters["rows_malformed_qna"] += 1
                    else:
                        counters["rows_other_errors"] += 1
                    counters["rows_skipped"] += 1
                    log(f"[instruction-data][warn] skipped row split={raw_split} index={row_index}: {error}")

            if max_rows is not None and processed_rows >= max_rows:
                break
    finally:
        for handle in output_handles.values():
            handle.close()

    log_check(
        "samples_written",
        counters["samples_written"] > 0,
        f"Wrote {counters['samples_written']} instruction samples.",
    )
    leakage_free = (
        split_to_image_ids["train"].isdisjoint(split_to_image_ids["val"])
        and split_to_image_ids["train"].isdisjoint(split_to_image_ids["test"])
        and split_to_image_ids["val"].isdisjoint(split_to_image_ids["test"])
    )
    log_check("image_level_split", leakage_free, "Verified that train/val/test image sets are disjoint.")

    report = {
        "dataset_name": dataset_name,
        "raw_splits": raw_splits,
        "selected_fields": {"image": image_field, "description": description_field, "qna": qna_field},
        "config": {
            "split_mode": effective_split_mode,
            "split_seed": int(config.get("split_seed", 42)),
            "val_ratio": float(config.get("val_ratio", 0.01)),
            "test_ratio": float(config.get("test_ratio", 0.01)),
            "use_description_samples": bool(config.get("use_description_samples", True)),
            "use_qna_samples": bool(config.get("use_qna_samples", True)),
        },
        "counts": {
            "rows_total": counters["rows_total"],
            "rows_skipped": counters["rows_skipped"],
            "rows_skipped_no_samples": counters["rows_skipped_no_samples"],
            "rows_empty_description": counters["rows_empty_description"],
            "rows_empty_qna": counters["rows_empty_qna"],
            "rows_missing_or_bad_image": counters["rows_missing_or_bad_image"],
            "rows_malformed_qna": counters["rows_malformed_qna"],
            "rows_other_errors": counters["rows_other_errors"],
            "samples_written": counters["samples_written"],
            "train_samples_written": counters["train_samples_written"],
            "val_samples_written": counters["val_samples_written"],
            "test_samples_written": counters["test_samples_written"],
            "train_images_written": counters["train_images_written"],
            "val_images_written": counters["val_images_written"],
            "test_images_written": counters["test_images_written"],
            "train_unique_images": len(split_to_image_ids["train"]),
            "val_unique_images": len(split_to_image_ids["val"]),
            "test_unique_images": len(split_to_image_ids["test"]),
        },
        "approx_message_word_count": {
            "min": min(approx_token_lengths) if approx_token_lengths else 0,
            "max": max(approx_token_lengths) if approx_token_lengths else 0,
            "mean": round(sum(approx_token_lengths) / len(approx_token_lengths), 2)
            if approx_token_lengths
            else 0.0,
        },
    }
    report_path = output_dir / "prepare_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    log("[instruction-data] done")
    log(f"[instruction-data] report written to {report_path}")


def main(argv: list[str] | None = None, *, default_config_section: str = DEFAULT_CONFIG_SECTION) -> None:
    args = parse_args(argv, default_config_section=default_config_section)
    run(args.config_section)


if __name__ == "__main__":
    main()
````

### `scripts/prepare_instruction_viet_sharegpt.py`

````python
"""Prepare Viet-ShareGPT-4o-Text-VQA for instruction tuning."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_instruction_common import main  # noqa: E402


DEFAULT_CONFIG_SECTION = "instruction_data_gpt"


if __name__ == "__main__":
    main(default_config_section=DEFAULT_CONFIG_SECTION)
````

### `scripts/prepare_instruction_5cd_localization.py`

````python
"""Prepare 5CD-AI/Viet-Localization-VQA for instruction tuning."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_instruction_common import main  # noqa: E402


DEFAULT_CONFIG_SECTION = "instruction_data_5cd"


if __name__ == "__main__":
    main(default_config_section=DEFAULT_CONFIG_SECTION)
````

### `scripts/prepare_instruction_data.py`

````python
"""Backward-compatible instruction-data entrypoint.

Prefer dataset-specific scripts:
  - scripts/prepare_instruction_viet_sharegpt.py
  - scripts/prepare_instruction_5cd_localization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_instruction_common import DEFAULT_CONFIG_SECTION, main, parse_args, run  # noqa: E402


if __name__ == "__main__":
    main(default_config_section=DEFAULT_CONFIG_SECTION)
````

### `scripts/crawl_vietnamtourism.py`

````python
"""Crawl images and metadata from vietnamtourism.gov.vn/cat/55 via public JSON API."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from bs4 import BeautifulSoup

from src.runtime import load_config

API_BASE = "https://public.vietnamtourism.gov.vn"
# Images served from CDN: relative paths (/images/...) from API resolve under /vn/
IMG_CDN = "https://images.vietnamtourism.gov.vn/vn"
_VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# vietnamtourism.gov.vn uses <em> and <i> interchangeably for captions
_ITALIC = ["em", "i"]


def build_api_url(cat_id: int, page: int, lang: str = "vi") -> tuple[str, dict]:
    url = f"{API_BASE}/cat/{cat_id}"
    param = json.dumps({"offset": page, "callType": 1, "lang": lang})
    return url, {"type": "1", "param": param}


def make_image_id(post_id: str, img_src: str) -> str:
    return hashlib.sha1(f"{post_id}:{img_src}".encode()).hexdigest()[:16]


def _resolve_img_src(src: str, cdn_base: str) -> str:
    if src.startswith("http://") or src.startswith("https://"):
        return src
    # Relative path like /images/2026/... → cdn_base + /images/2026/...
    return cdn_base + (src if src.startswith("/") else "/" + src)


def _clean_caption(caption: str) -> str:
    caption = caption.replace("\xa0", " ").strip()
    # Parenthesized notes first — prevents bare-credit regex eating inside "(Ảnh: ...)"
    caption = re.sub(r"\s*\([Ảả]nh[^)]*\)\s*$", "", caption).strip()
    # Credit with colon: "Ảnh: ..." (uppercase only — "Trong ảnh:" uses lowercase)
    caption = re.sub(r"[,\s]*[-–]?\s*Ảnh\s*:\s*[^\n]*$", "", caption).strip()
    # Credit without colon: ". Ảnh Name" (lookbehind keeps the period) / ", ảnh Name"
    # Safe: "cảnh", "chụp ảnh", "hình ảnh" are preceded by letters/spaces, not "." or ","
    caption = re.sub(r"(?<=\.)\s*[Ảả]nh\s+\S[^\n]*$", "", caption).strip()
    caption = re.sub(r",\s*[Ảả]nh\s+\S[^\n]*$", "", caption).strip()
    # Standalone credit-only caption: "Ảnh TITC", "Ảnh minh họa/Nguồn: ..." etc.
    if re.fullmatch(r"[Ảả]nh(\s+\S[^\n]*)?", caption):
        return ""
    return caption


def _parse_width_from_style(style: str) -> int:
    # Match standalone `width:` but not `border-width:` or `max-width:`
    m = re.search(r"(?<![a-z-])width\s*:\s*(\d+)", style or "")
    return int(m.group(1)) if m else 0


def extract_images_from_html(
    html: str,
    min_width: int = 200,
    cdn_base: str = IMG_CDN,
) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []

    for img in soup.find_all("img"):
        src: str = img.get("src") or img.get("data-src") or ""
        if not src or src.endswith(".svg"):
            continue

        width = _parse_width_from_style(img.get("style", ""))
        if 0 < width < min_width:
            continue

        src = _resolve_img_src(src, cdn_base)

        # Three caption patterns (site uses <em> and <i> interchangeably):
        # 1. italic tag is ancestor of img:  <p><em><img><br>caption</em></p>
        # 2. caption in next sibling <p>:    <p><img></p><p><em|i>caption</em|i></p>
        # 3. italic sibling in same <p>:     <p><a><img></a><br><em|i>caption</em|i></p>
        caption = ""

        # Pattern 1
        italic_anc = img.find_parent(_ITALIC)
        if italic_anc:
            caption = italic_anc.get_text(separator=" ", strip=True)

        if not caption:
            parent_p = img.find_parent("p")
            if parent_p:
                # Pattern 3: italic sibling in same <p> (iterate all to skip any that wrap img)
                for el in parent_p.find_all(_ITALIC):
                    if not el.find("img"):
                        caption = el.get_text(separator=" ", strip=True)
                        break

                # Pattern 2: italic in next sibling <p>
                if not caption:
                    next_p = parent_p.find_next_sibling("p")
                    if next_p and not next_p.find("img"):
                        el = next_p.find(_ITALIC)
                        if el:
                            caption = el.get_text(separator=" ", strip=True)

        if not caption:
            caption = img.get("alt", "").strip()

        results.append({"src": src, "caption": caption})

    return results


def download_image(src: str, dest: Path, session: requests.Session, timeout: int = 15) -> bool:
    try:
        resp = session.get(src, timeout=timeout, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            print(f"[warn] unexpected content-type {content_type!r} for {src}")
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return True
    except Exception as exc:
        print(f"[warn] download failed {src}: {exc}")
        return False


def fetch_page(session: requests.Session, cat_id: int, page: int) -> list[dict]:
    url, params = build_api_url(cat_id=cat_id, page=page)
    resp = session.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("child") or []


def main() -> None:
    cfg = load_config("crawl_vietnamtourism")
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "raw_crawl.jsonl"

    cat_id = int(cfg.get("category_id", 55))
    max_pages = cfg.get("max_pages")
    max_images = int(cfg.get("max_images", 5000))
    delay = float(cfg.get("delay_seconds", 1.0))
    min_width = int(cfg.get("min_image_width", 200))

    # Resume: skip already-crawled image IDs
    crawled_ids: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    crawled_ids.add(json.loads(line)["image_id"])
    already_crawled = len(crawled_ids)
    print(f"[crawl] resuming — {already_crawled} images already crawled")

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research crawler)"

    total_new = 0
    page = 1

    with jsonl_path.open("a", encoding="utf-8") as out:
        while True:
            if max_pages is not None and page > int(max_pages):
                break
            if total_new + already_crawled >= max_images:
                print(f"[crawl] reached max_images={max_images}, stopping")
                break

            print(f"[crawl] page {page} ...", flush=True)
            try:
                posts = fetch_page(session, cat_id=cat_id, page=page)
            except Exception as exc:
                print(f"[warn] page {page} failed: {exc}")
                break

            if not posts:
                print(f"[crawl] empty page {page}, done")
                break

            for post in posts:
                if total_new + already_crawled >= max_images:
                    break

                post_id = str(post["id"])
                title = post.get("title", "").strip()
                date = (post.get("dateedit") or "")[:10]
                article_url = f"https://vietnamtourism.gov.vn/post/{post_id}"

                for img_info in extract_images_from_html(
                    post.get("content", ""), min_width=min_width
                ):
                    if total_new + already_crawled >= max_images:
                        break

                    image_id = make_image_id(post_id, img_info["src"])
                    if image_id in crawled_ids:
                        continue

                    ext = Path(urlparse(img_info["src"]).path).suffix.lower() or ".jpg"
                    if ext not in _VALID_EXTS:
                        continue

                    dest = images_dir / f"{image_id}{ext}"
                    if not download_image(img_info["src"], dest, session):
                        continue

                    record = {
                        "image_id": image_id,
                        "image_path": str(dest),
                        "title": title,
                        "caption": _clean_caption(img_info["caption"]),
                        "article_url": article_url,
                        "date": date,
                        "post_id": post_id,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    crawled_ids.add(image_id)
                    total_new += 1

            page += 1
            time.sleep(delay)

    print(f"[crawl] done — {total_new} new images, {len(crawled_ids)} total in {jsonl_path}")


if __name__ == "__main__":
    main()
````

### `scripts/generate_qa_vietnamtourism.py`

````python
"""Generate coherent multi-turn conversations from crawled images via OpenAI Batch API.

Prompt design references:
  - LLaVA (Liu et al., 2023): description + complex-reasoning instruction types
  - ShareGPT4V (Chen et al., 2023): rich spatial descriptions with color/layout
  - InstructBLIP (Dai et al., 2023): diverse short-answer factual templates
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from PIL import Image

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import append_jsonl, load_config  # noqa: E402

_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

_OPENAI_SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}
_FINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


def _to_supported_image(image_path: Path) -> tuple[bytes, str]:
    """Return (bytes, media_type), converting to JPEG if the format is unsupported by OpenAI."""
    raw = image_path.read_bytes()
    img = Image.open(io.BytesIO(raw))
    if img.format in _OPENAI_SUPPORTED_FORMATS:
        return raw, _MEDIA_TYPES.get(image_path.suffix.lower(), "image/jpeg")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.getchannel("A"))
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


_SYSTEM_PROMPT = (
    "Bạn là trợ lý thị giác tiếng Việt đang nhìn thấy một bức ảnh. "
    "Trả lời mọi câu hỏi bằng cách bạn đang thực sự quan sát ảnh đó."
)

_INSTRUCTION = """\
Nhìn vào bức ảnh và ngữ cảnh bên dưới, thiết kế một hội thoại 3 lượt tự nhiên giữa
user và assistant về bức ảnh này.

Mỗi câu hỏi của user PHẢI đáp ứng đồng thời ba điều kiện:
(1) Là câu hỏi thực sự (kết thúc bằng "?"), không phải lệnh như "Hãy...", "Đề xuất...".
(2) Bắt buộc phải nhìn thấy ảnh mới trả lời được, nếu chỉ đọc tiêu đề/chú thích mà
    không thấy ảnh thì không thể trả lời chính xác.
(3) Câu hỏi không được chứa tên địa danh, tên sự kiện, tên người lấy từ tiêu đề/chú thích
    trừ khi thông tin đó nhìn thấy được trong ảnh (ví dụ: đọc được trên biển tên, băng rôn).
    Tuyệt đối không hỏi dạng "Theo chú thích...", "Dựa vào tiêu đề...", hay dùng tên địa
    danh/sự kiện mà người xem ảnh không thể biết nếu không đọc caption.

Trước khi viết mỗi câu hỏi, tự kiểm tra: "Nếu tôi che ảnh đi và chỉ đọc tiêu đề/chú thích
hoặc dựa vào kiến thức chung, tôi có thể trả lời câu này không?" — Nếu có, bỏ câu đó và
đặt câu hỏi khác gắn với chi tiết nhìn thấy trong ảnh.

Câu hỏi có thể về: số lượng đối tượng, màu sắc, vị trí không gian, hành động, trang phục,
biểu cảm, hoặc ý nghĩa văn hóa/du lịch gắn với chi tiết nhìn thấy trong ảnh.
Nếu ảnh có chi tiết văn hóa hoặc du lịch nhận ra được (trang phục truyền thống, địa danh,
hoạt động đặc trưng...), hãy khai thác ý nghĩa của chi tiết đó trong một lượt.
Câu hỏi từ lượt 2 trở đi phải nối tiếp nội dung đã nói ở lượt trước.
Câu trả lời phức tạp nên có giải thích cụ thể. Nếu hỏi về thứ không có trong ảnh,
assistant trả lời phủ định thay vì bịa đặt.

Trả về JSON hợp lệ, không thêm nội dung nào khác:
{
  "description": "mô tả chi tiết bức ảnh kết hợp tiêu đề/chú thích",
  "conversation": [
    {"role": "user", "content": "...?"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "...?"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "...?"},
    {"role": "assistant", "content": "..."}
  ]
}\
"""


def build_user_message(title: str, caption: str, image_path: Path) -> list[dict]:
    context: list[str] = []
    if title:
        context.append(f"Tiêu đề bài viết: {title}")
    if caption:
        context.append(f"Chú thích ảnh: {caption}")

    text = "\n".join(context + ["", _INSTRUCTION]) if context else _INSTRUCTION

    img_bytes, media_type = _to_supported_image(image_path)
    b64 = base64.standard_b64encode(img_bytes).decode()

    return [
        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
        {"type": "text", "text": text},
    ]


def _strip_json_fence(content: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE).strip()


def _repair_unescaped_quotes(text: str) -> str:
    """Escape unescaped double-quotes inside JSON string values.

    Models sometimes output Vietnamese text with unescaped " characters inside
    string values, e.g. "description": "...có tiêu đề "Lễ hội đền Hà" và...".
    Uses a state machine: a closing " is one whose next non-whitespace char is
    , } ] " or : (structural JSON); otherwise the " is inside a value and must
    be escaped.
    """
    in_string = False
    result = []
    i = 0
    while i < len(text):
        c = text[i]
        if not in_string:
            result.append(c)
            if c == '"':
                in_string = True
        elif c == "\\":
            result.append(c)
            i += 1
            if i < len(text):
                result.append(text[i])
        elif c == '"':
            j = i + 1
            while j < len(text) and text[j] in " \t\r\n":
                j += 1
            if j >= len(text) or text[j] in ',}]":':
                in_string = False
                result.append(c)
            else:
                result.append('\\"')
        else:
            result.append(c)
        i += 1
    return "".join(result)


def _normalize_conversation(raw_messages) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        raise ValueError(f"Expected 'conversation' to be a list, got {type(raw_messages).__name__}")
    if len(raw_messages) % 2 != 0:
        raw_messages = raw_messages[:-1]

    normalized_messages: list[dict[str, str]] = []
    expected_role = "user"
    for index, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            raise ValueError(f"Conversation message #{index} must be a dict.")

        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role != expected_role:
            raise ValueError(
                f"Expected conversation role '{expected_role}' at message #{index}, got '{role}'."
            )
        if not content:
            raise ValueError(f"Conversation message #{index} has empty content.")

        normalized_messages.append({"role": role, "content": content})
        expected_role = "assistant" if expected_role == "user" else "user"

    if len(normalized_messages) < 2:
        raise ValueError("Conversation must contain at least one user-assistant pair.")
    return normalized_messages


def _qa_pairs_to_conversation(qa_list) -> list[dict[str, str]]:
    if not isinstance(qa_list, list):
        raise ValueError(f"Expected 'qa_pairs' to be a list, got {type(qa_list).__name__}")

    messages: list[dict[str, str]] = []
    for item in qa_list:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
    return _normalize_conversation(messages)


def parse_qa_response(content: str) -> tuple[str, list[dict[str, str]]]:
    cleaned = _strip_json_fence(content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(_repair_unescaped_quotes(cleaned))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse QA response: {exc}\nContent: {content[:300]}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected a dict with 'description' and 'conversation', got {type(parsed).__name__}"
        )
    description = str(parsed.get("description", "")).strip()
    if not description:
        raise ValueError("Expected a non-empty 'description'.")

    if "conversation" in parsed:
        return description, _normalize_conversation(parsed["conversation"])

    # Backward compatibility for older batch outputs created before the conversation schema.
    if "qa_pairs" in parsed:
        return description, _qa_pairs_to_conversation(parsed["qa_pairs"])

    raise ValueError("Expected either 'conversation' or legacy 'qa_pairs' in model response.")


def read_done_image_ids(batch_results: Path) -> set[str]:
    done_ids: set[str] = set()
    if not batch_results.exists():
        return done_ids
    with batch_results.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                done_ids.add(str(json.loads(line)["image_id"]))
    return done_ids


def save_batch_result_text(
    result_text: str, id_to_record: dict[str, dict], batch_results: Path, done_ids: set[str]
) -> int:
    saved = 0
    for line in result_text.splitlines():
        line = line.strip()
        if not line:
            continue
        result = json.loads(line)
        custom_id = result.get("custom_id") or result.get("id", "")
        rec = id_to_record.get(custom_id)
        if rec is None:
            continue
        image_id = str(rec["image_id"])
        if image_id in done_ids:
            continue
        try:
            resp_body = result["response"]["body"]
            if "error" in resp_body:
                raise ValueError(resp_body["error"].get("message", "api error"))
            choice = resp_body["choices"][0]
            content = choice["message"]["content"] or ""
            if not content:
                raise ValueError(f"empty content (finish_reason={choice.get('finish_reason')})")
            description, conversation = parse_qa_response(content)
        except Exception as exc:
            print(f"[warn] skipping {custom_id}: {exc}")
            continue
        append_jsonl(
            batch_results,
            {
                "image_id": rec["image_id"],
                "image_path": rec["image_path"],
                "title": rec["title"],
                "caption": rec["caption"],
                "article_url": rec["article_url"],
                "date": rec["date"],
                "description": description,
                "conversation": conversation,
            },
        )
        done_ids.add(image_id)
        saved += 1
    return saved


def print_batch_errors(client, batch) -> None:
    errors = getattr(getattr(batch, "errors", None), "data", None) or []
    for err in errors[:5]:
        print(f"  error sample: {getattr(err, 'message', err)}")
    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        err_text = client.files.content(error_file_id).text
        for err_line in err_text.splitlines()[:5]:
            print(f"  error sample: {err_line.strip()}")


def save_completed_batch(
    client, batch, id_to_record: dict[str, dict], batch_results: Path, done_ids: set[str]
) -> int:
    if getattr(batch, "status", None) != "completed":
        print(f"[warn] batch {batch.id} ended with status={batch.status}, skipping")
        print_batch_errors(client, batch)
        return 0
    if not getattr(batch, "output_file_id", None):
        print(f"[warn] batch {batch.id} completed but output_file_id is None (all requests may have failed)")
        print_batch_errors(client, batch)
        return 0
    result_text = client.files.content(batch.output_file_id).text
    return save_batch_result_text(result_text, id_to_record, batch_results, done_ids)


_REASONING_MODELS = {"o1", "o3", "o4"}


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _REASONING_MODELS)


def build_batch_request(record: dict, *, model: str, max_tokens: int) -> dict:
    image_path = Path(record["image_path"])
    body: dict = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "messages": [
            {"role": "developer", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_message(
                    title=record.get("title", ""), caption=record.get("caption", ""), image_path=image_path
                ),
            },
        ],
    }
    if _is_reasoning_model(model):
        body["reasoning_effort"] = "medium"
    return {
        "custom_id": f"img-{record['image_id']}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def submit_batch_chunk(
    client, raw_dir: Path, chunk_index: int, total_chunks: int, chunk: list[tuple[dict, dict]]
):
    chunk_path = raw_dir / f"batch_input_{chunk_index:03d}.jsonl"
    chunk_path.write_text(
        "\n".join(json.dumps(req, ensure_ascii=False) for _, req in chunk), encoding="utf-8"
    )
    with chunk_path.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h"
    )
    print(f"[qa-gen] chunk {chunk_index + 1}/{total_chunks} ({len(chunk)} reqs) → {batch.id}")
    return batch


def main() -> None:
    import openai

    cfg = load_config("generate_qa_vietnamtourism")
    raw_dir = Path(cfg["raw_dir"]).expanduser().resolve()
    model = str(cfg.get("model", "gpt-4o"))
    max_tokens = int(cfg.get("max_tokens", 1500))
    max_active_batches = max(1, int(cfg.get("max_active_batches", 3)))
    max_images = cfg.get("max_images")
    if max_images is not None:
        max_images = int(max_images)

    raw_jsonl = raw_dir / "raw_crawl.jsonl"
    batch_results = raw_dir / "batch_results.jsonl"

    if not raw_jsonl.exists():
        raise FileNotFoundError(f"Run crawl script first: {raw_jsonl}")

    done_ids = read_done_image_ids(batch_results)
    print(f"[qa-gen] {len(done_ids)} already generated, skipping")

    records: list[dict] = []
    with raw_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if max_images is not None and len(records) >= max_images:
                break
            rec = json.loads(line)
            if (
                str(rec["image_id"]) not in done_ids
                and Path(rec["image_path"]).exists()
                and rec.get("caption", "").strip()
                and rec.get("title", "").strip()
            ):
                records.append(rec)

    if not records:
        print("[qa-gen] nothing to generate, done")
        return

    print(f"[qa-gen] building {len(records)} batch requests ...")
    valid: list[tuple[dict, dict]] = []
    for rec in records:
        try:
            valid.append((rec, build_batch_request(rec, model=model, max_tokens=max_tokens)))
        except Exception as exc:
            print(f"[warn] skipping {rec['image_id']}: {exc}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set")
    client = openai.OpenAI(api_key=api_key)

    # Chunk by serialised byte size to stay under OpenAI's 200 MB batch file limit.
    max_chunk_bytes = int(cfg.get("batch_chunk_mb", 150)) * 1024 * 1024
    chunks: list[list[tuple[dict, dict]]] = [[]]
    chunk_bytes = 0
    for pair in valid:
        line_bytes = len(json.dumps(pair[1], ensure_ascii=False).encode())
        if chunk_bytes + line_bytes > max_chunk_bytes and chunks[-1]:
            chunks.append([])
            chunk_bytes = 0
        chunks[-1].append(pair)
        chunk_bytes += line_bytes
    print(
        f"[qa-gen] splitting into {len(chunks)} batch(es)"
        f" (≤{max_chunk_bytes // 1024 // 1024} MB each, max_active={max_active_batches}) ..."
    )

    id_to_record = {f"img-{rec['image_id']}": rec for rec in records}
    saved = 0
    active_batches: dict[str, object] = {}
    next_chunk_index = 0
    print("[qa-gen] polling (may take minutes to hours) ...")
    try:
        while next_chunk_index < len(chunks) or active_batches:
            while next_chunk_index < len(chunks) and len(active_batches) < max_active_batches:
                batch = submit_batch_chunk(
                    client, raw_dir, next_chunk_index, len(chunks), chunks[next_chunk_index]
                )
                active_batches[batch.id] = batch
                next_chunk_index += 1

            still_active: dict[str, object] = {}
            for bid in active_batches:
                b = client.batches.retrieve(bid)
                counts = b.request_counts
                print(
                    f"  [{bid[-12:]}] status={b.status}"
                    f" completed={getattr(counts, 'completed', '?')}/{getattr(counts, 'total', '?')}",
                    flush=True,
                )
                if b.status == "completed":
                    batch_saved = save_completed_batch(client, b, id_to_record, batch_results, done_ids)
                    saved += batch_saved
                    print(f"  [{bid[-12:]}] saved={batch_saved}", flush=True)
                elif b.status in _FINAL_BATCH_STATUSES:
                    save_completed_batch(client, b, id_to_record, batch_results, done_ids)
                else:
                    still_active[bid] = b
            active_batches = still_active

            if next_chunk_index < len(chunks) or active_batches:
                time.sleep(60)
    except KeyboardInterrupt:
        if active_batches:
            print("\n[qa-gen] interrupted; these submitted batches may still finish on OpenAI:")
            for bid in active_batches:
                print(f"  {bid}")
            print("[qa-gen] completed batches already seen by this process were saved before exit.")
        raise

    print(f"[qa-gen] done — {saved} records saved to {batch_results}")


if __name__ == "__main__":
    main()
````

### `scripts/prepare_vietnamtourism_data.py`

````python
"""Convert raw crawl + generated conversations to project instruction-tuning JSONL."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime import load_config


def assign_split(image_id: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.sha1(f"{seed}:{image_id}".encode()).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    if score < test_ratio:
        return "test"
    if score < test_ratio + val_ratio:
        return "val"
    return "train"


def normalize_conversation_messages(raw_messages) -> list[dict[str, str]]:
    if raw_messages is None:
        return []
    if not isinstance(raw_messages, list):
        raise ValueError("conversation must be a list.")
    if len(raw_messages) % 2 != 0:
        raw_messages = raw_messages[:-1]

    normalized_messages: list[dict[str, str]] = []
    expected_role = "user"
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise ValueError(f"conversation message #{index} must be a dict.")

        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role != expected_role:
            raise ValueError(
                f"expected role '{expected_role}' at conversation message #{index}, got '{role}'."
            )
        if not content:
            raise ValueError(f"conversation message #{index} has empty content.")

        normalized_messages.append({"role": role, "content": content})
        expected_role = "assistant" if expected_role == "user" else "user"

    return normalized_messages


def qa_pairs_to_conversation(qa_pairs: list[dict]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for qa in qa_pairs:
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()
        if not question or not answer:
            continue
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    return normalize_conversation_messages(messages)


def conversation_from_record(record: dict) -> list[dict[str, str]]:
    conversation = normalize_conversation_messages(record.get("conversation"))
    if conversation:
        return conversation
    return qa_pairs_to_conversation(record.get("qa_pairs", []))


def build_instruction_entry(
    *,
    image_path: Path,
    conversation: list[dict[str, str]],
    system_prompt: str,
    image_id: str,
    source: str,
    title: str = "",
    caption: str = "",
    description: str = "",
    article_url: str = "",
    date: str = "",
    post_id: str = "",
) -> dict:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation)
    return {
        "id": f"vntour_{image_id}",
        "image": str(image_path),
        "messages": messages,
        "sample_type": "conversation",
        "source_dataset": source,
        "image_id": image_id,
        "title": title,
        "caption": caption,
        "description": description,
        "article_url": article_url,
        "date": date,
        "post_id": post_id,
    }


def main() -> None:
    cfg = load_config("prepare_vietnamtourism")
    raw_dir = Path(cfg["raw_dir"]).expanduser().resolve()
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    system_prompt = str(cfg["system_prompt"])
    seed = int(cfg.get("seed", 42))
    val_ratio = float(cfg.get("val_ratio", 0.01))
    test_ratio = float(cfg.get("test_ratio", 0.01))
    source = "vietnamtourism"

    batch_results = raw_dir / "batch_results.jsonl"
    if not batch_results.exists():
        raise FileNotFoundError(f"Run generate script first: {batch_results}")

    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    counters: Counter = Counter()
    split_image_ids: dict[str, set] = defaultdict(set)

    try:
        with batch_results.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                image_id = rec["image_id"]
                image_path = Path(rec["image_path"])

                if not image_path.exists():
                    counters["skipped_missing_image"] += 1
                    continue

                split = assign_split(image_id, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
                split_image_ids[split].add(image_id)

                try:
                    conversation = conversation_from_record(rec)
                except ValueError as exc:
                    counters["skipped_invalid_conversation"] += 1
                    print(f"[warn] skipping {image_id}: {exc}")
                    continue
                if not conversation:
                    counters["skipped_empty_conversation"] += 1
                    continue

                entry = build_instruction_entry(
                    image_path=image_path,
                    conversation=conversation,
                    system_prompt=system_prompt,
                    image_id=image_id,
                    source=source,
                    title=rec.get("title", ""),
                    caption=rec.get("caption", ""),
                    description=rec.get("description", ""),
                    article_url=rec.get("article_url", ""),
                    date=rec.get("date", ""),
                    post_id=rec.get("post_id", ""),
                )
                handles[split].write(json.dumps(entry, ensure_ascii=False) + "\n")
                counters[f"{split}_images"] += 1
                counters["total_images"] += 1
    finally:
        for h in handles.values():
            h.close()

    print("[prepare] done")
    for split in ("train", "val", "test"):
        print(f"  {split}: {counters[f'{split}_images']} images → {split}.jsonl")
    print(f"  skipped missing images: {counters['skipped_missing_image']}")
    print(f"  skipped empty conv:     {counters['skipped_empty_conversation']}")
    print(f"  skipped invalid conv:   {counters['skipped_invalid_conversation']}")

    (output_dir / "prepare_report.json").write_text(
        json.dumps(
            {
                "source": source,
                "counts": dict(counters),
                "split_images": {k: len(v) for k, v in split_image_ids.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
````

### `scripts/train_stage1.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

/home/shared/miniconda3/envs/nhantd_env/bin/accelerate launch \
  --num_processes "${NUM_PROCESSES:-2}" \
  --num_machines 1 \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --main_process_port "${MAIN_PROCESS_PORT:-0}" \
  --dynamo_backend no \
  --multi_gpu \
  train.py --config-section "${CONFIG_SECTION:-train}" "$@"
````

### `scripts/train_stage1_uit_card2.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

/home/shared/miniconda3/envs/nhantd_env/bin/accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --main_process_port "${MAIN_PROCESS_PORT:-0}" \
  --dynamo_backend no \
  train.py --config-section "${CONFIG_SECTION:-train_uit_only}" "$@"
````

### `scripts/train_instruction.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

/home/shared/miniconda3/envs/nhantd_env/bin/accelerate launch \
  --num_processes "${NUM_PROCESSES:-2}" \
  --num_machines 1 \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --main_process_port "${MAIN_PROCESS_PORT:-0}" \
  --dynamo_backend no \
  --multi_gpu \
  train_instruction.py "$@"
````

### `tests/test_instruction_entrypoints.py`

````python
import importlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class InstructionEntrypointTest(unittest.TestCase):
    def test_dataset_specific_entrypoints_have_separate_config_sections(self):
        gpt = importlib.import_module("scripts.prepare_instruction_viet_sharegpt")
        five_cd = importlib.import_module("scripts.prepare_instruction_5cd_localization")

        self.assertEqual(gpt.DEFAULT_CONFIG_SECTION, "instruction_data_gpt")
        self.assertEqual(five_cd.DEFAULT_CONFIG_SECTION, "instruction_data_5cd")

    def test_generic_entrypoint_keeps_config_section_override(self):
        generic = importlib.import_module("scripts.prepare_instruction_data")

        args = generic.parse_args(["--config-section", "instruction_data_5cd"])
        self.assertEqual(args.config_section, "instruction_data_5cd")

    def test_streamed_cmyk_images_are_saved_as_rgb_jpeg(self):
        common = importlib.import_module("scripts.prepare_instruction_common")
        image = Image.new("CMYK", (8, 8), color=(0, 128, 128, 0))

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "sample.jpg"
            common.save_image_asset(image, destination)

            with Image.open(destination) as saved:
                self.assertEqual(saved.mode, "RGB")
                self.assertEqual(saved.format, "JPEG")

    def test_streamed_pil_images_use_jpeg_extension(self):
        common = importlib.import_module("scripts.prepare_instruction_common")
        image = Image.new("CMYK", (8, 8), color=(0, 128, 128, 0))

        self.assertEqual(common.infer_image_extension(image, image_key="sample"), ".jpg")

    def test_instruction_streamlit_prefers_checkpoint_llm_and_tokenizer_sources(self):
        app = importlib.import_module("streamlit_instruction_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint-10"
            (checkpoint / "llm").mkdir(parents=True)
            (checkpoint / "tokenizer").mkdir()

            llm_source, tokenizer_source = app.resolve_checkpoint_sources(checkpoint, "base-llm")

            self.assertEqual(llm_source, str((checkpoint / "llm").resolve()))
            self.assertEqual(tokenizer_source, str((checkpoint / "tokenizer").resolve()))

    def test_stage1_streamlit_uses_checkpoint_tokenizer_when_available(self):
        app = importlib.import_module("streamlit_stage1_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint-10"
            (checkpoint / "tokenizer").mkdir(parents=True)

            tokenizer_source = app.resolve_tokenizer_source(checkpoint, "base-llm")

            self.assertEqual(tokenizer_source, str((checkpoint / "tokenizer").resolve()))

    def test_instruction_dataset_reports_jsonl_paths_when_all_images_are_unreadable(self):
        data = importlib.import_module("src.data")

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "samples.jsonl"
            record = {
                "id": "bad-image",
                "image": "missing.jpg",
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Ảnh có gì?"},
                    {"role": "assistant", "content": "Không đọc được ảnh."},
                ],
            }
            jsonl_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = data.ImageInstructionDataset(jsonl_path)

            with self.assertRaisesRegex(RuntimeError, "samples.jsonl"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
````

### `tests/test_vietnamtourism_conversations.py`

````python
import json
import tempfile
import unittest
from pathlib import Path


class VietnamTourismConversationTest(unittest.TestCase):
    def test_parse_new_conversation_schema(self):
        generator = __import__("scripts.generate_qa_vietnamtourism", fromlist=["parse_qa_response"])
        payload = {
            "description": "Một hội nghị trong hội trường.",
            "conversation": [
                {"role": "user", "content": "Ảnh này có gì nổi bật?"},
                {"role": "assistant", "content": "Ảnh cho thấy một hội nghị trong hội trường lớn."},
                {"role": "user", "content": "Chi tiết hội trường trông ra sao?"},
                {"role": "assistant", "content": "Hội trường có sân khấu, màn hình lớn và nhiều hàng ghế."},
            ],
        }

        description, conversation = generator.parse_qa_response(json.dumps(payload, ensure_ascii=False))

        self.assertEqual(description, payload["description"])
        self.assertEqual(conversation, payload["conversation"])

    def test_parse_legacy_qa_pairs_schema(self):
        generator = __import__("scripts.generate_qa_vietnamtourism", fromlist=["parse_qa_response"])
        payload = {
            "description": "Một đoàn du khách trên tàu.",
            "qa_pairs": [
                {
                    "type": "description",
                    "question": "Trong ảnh có hoạt động gì?",
                    "answer": "Ảnh cho thấy du khách đang tham dự một hoạt động trên tàu.",
                }
            ],
        }

        _, conversation = generator.parse_qa_response(json.dumps(payload, ensure_ascii=False))

        self.assertEqual(
            conversation,
            [
                {"role": "user", "content": "Trong ảnh có hoạt động gì?"},
                {
                    "role": "assistant",
                    "content": "Ảnh cho thấy du khách đang tham dự một hoạt động trên tàu.",
                },
            ],
        )

    def test_prepare_entry_keeps_article_context_out_of_training_messages(self):
        prepare = __import__("scripts.prepare_vietnamtourism_data", fromlist=["build_instruction_entry"])
        conversation = [
            {"role": "user", "content": "Ảnh này mô tả điều gì?"},
            {"role": "assistant", "content": "Ảnh mô tả một sự kiện du lịch."},
            {"role": "user", "content": "Sự kiện đó có ý nghĩa gì?"},
            {"role": "assistant", "content": "Nó góp phần quảng bá điểm đến địa phương."},
        ]

        entry = prepare.build_instruction_entry(
            image_path=Path("/tmp/sample.jpg"),
            conversation=conversation,
            system_prompt="System",
            image_id="abc",
            source="vietnamtourism",
            title="Quảng bá du lịch Việt Nam",
            caption="Du khách tham gia sự kiện.",
            description="Một nhóm du khách đang tham gia sự kiện ngoài trời.",
        )

        self.assertEqual(entry["messages"][1]["content"], "Ảnh này mô tả điều gì?")
        self.assertEqual(entry["messages"][3]["content"], "Sự kiện đó có ý nghĩa gì?")
        self.assertEqual(entry["title"], "Quảng bá du lịch Việt Nam")
        self.assertEqual(entry["caption"], "Du khách tham gia sự kiện.")
        self.assertEqual(entry["description"], "Một nhóm du khách đang tham gia sự kiện ngoài trời.")
        self.assertEqual(entry["sample_type"], "conversation")

    def test_save_batch_result_text_appends_only_new_records(self):
        generator = __import__("scripts.generate_qa_vietnamtourism", fromlist=["save_batch_result_text"])
        response_payload = {
            "description": "Một nhóm du khách đang đứng trước cổng.",
            "conversation": [
                {"role": "user", "content": "Trong ảnh có mấy người nổi bật?"},
                {"role": "assistant", "content": "Có ba người nổi bật ở phía trước ảnh."},
            ],
        }
        result_text = "\n".join(
            [
                json.dumps(
                    {
                        "custom_id": "img-existing",
                        "response": {
                            "body": {
                                "choices": [
                                    {"message": {"content": json.dumps(response_payload, ensure_ascii=False)}}
                                ]
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "custom_id": "img-new",
                        "response": {
                            "body": {
                                "choices": [
                                    {"message": {"content": json.dumps(response_payload, ensure_ascii=False)}}
                                ]
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        records = {
            "img-existing": {
                "image_id": "existing",
                "image_path": "/tmp/existing.jpg",
                "title": "Tựa đề cũ",
                "caption": "Caption cũ",
                "article_url": "https://example.com/old",
                "date": "2026-05-12",
            },
            "img-new": {
                "image_id": "new",
                "image_path": "/tmp/new.jpg",
                "title": "Tựa đề mới",
                "caption": "Caption mới",
                "article_url": "https://example.com/new",
                "date": "2026-05-12",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "batch_results.jsonl"
            done_ids = {"existing"}

            saved = generator.save_batch_result_text(result_text, records, output_path, done_ids)

            self.assertEqual(saved, 1)
            self.assertEqual(done_ids, {"existing", "new"})
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            saved_record = json.loads(lines[0])
            self.assertEqual(saved_record["image_id"], "new")
            self.assertEqual(saved_record["description"], response_payload["description"])


if __name__ == "__main__":
    unittest.main()
````
