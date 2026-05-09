# Stage 2 Data Pipeline — vietnamtourism.gov.vn

**Goal:** Crawl ~1,000–5,000 tourism images from vietnamtourism.gov.vn, generate 4 QA pairs per image
via GPT-4o Vision, and produce train/val/test JSONL files compatible with the existing Stage 2
instruction-tuning format.

---

## Data Sources

| Source | Type | Est. size |
|--------|------|-----------|
| vietnamtourism.gov.vn/cat/55 | Crawled images + metadata | 1,000–5,000 images |
| 5CD-AI/Viet-Localization-VQA | HuggingFace open-source | Existing pipeline |
| 5CD-AI/Viet-ShareGPT-4o-Text-VQA | HuggingFace open-source | Already in `prepare_instruction_data.py` |

This spec covers only the crawled data pipeline (3 scripts). Open-source datasets are already handled
by `scripts/prepare_instruction_data.py`.

---

## Architecture (Approach B — Two-Stage)

```
public.vietnamtourism.gov.vn JSON API
        │  GET /cat/55?type=1&param={"offset":N,...}
        │  15 posts/page · 43,110 total · content has <img> in HTML
        ▼
scripts/crawl_vietnamtourism.py       (requests + BeautifulSoup)
        │  → data/vietnamtourism-raw/raw_crawl.jsonl
        │  → data/vietnamtourism-raw/images/{image_id}.jpg
        ▼
scripts/generate_qa_vietnamtourism.py  (openai Batch API)
        │  → data/vietnamtourism-raw/batch_input.jsonl  (uploaded to OpenAI)
        │  → data/vietnamtourism-raw/batch_results.jsonl
        ▼
scripts/prepare_vietnamtourism_data.py (format conversion + split)
        │  → data/vietnamtourism/train.jsonl
        │  → data/vietnamtourism/val.jsonl
        │  → data/vietnamtourism/test.jsonl
```

Output JSONL format matches `ImageInstructionDataset` in `src/data.py`:
```json
{
  "id": "vntour_<image_id>_qa_001",
  "image": "/abs/path/to/image.jpg",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "sample_type": "qa",
  "source_dataset": "vietnamtourism",
  "image_id": "<image_id>"
}
```

---

## API Discovery

- **Base:** `https://public.vietnamtourism.gov.vn`
- **Listing:** `GET /cat/55?type=1&param={"offset":<page>,"callType":1,"lang":"vi"}`
  - Returns 15 posts per page; `offset` is 1-indexed page number
  - `info.total` gives total post count
- **Post fields used:** `id`, `title`, `content` (HTML), `dateedit`, `source`
- **Images:** Embedded in `content` HTML as `<img src="/images/...">`, relative to
  `https://vietnamtourism.gov.vn`
- **Captions:** In `<em>` tag wrapping `<img>` + caption text

---

## Prompt Template (GPT-4o)

```
SYSTEM:
Bạn là chuyên gia văn hóa và du lịch Việt Nam. Nhiệm vụ của bạn là tạo
các cặp câu hỏi-trả lời chất lượng cao bằng tiếng Việt về hình ảnh du
lịch Việt Nam, dựa trên nội dung thực sự quan sát được trong ảnh.

USER: [IMAGE]
Tiêu đề bài viết: {title}
Mô tả ảnh: {caption}   ← bỏ dòng này nếu caption rỗng

Hãy tạo đúng 4 cặp câu hỏi-trả lời về hình ảnh bằng tiếng Việt, trả về
JSON hợp lệ theo định dạng sau (không thêm gì ngoài JSON):
[
  {"type": "description", "question": "...", "answer": "..."},
  {"type": "location",    "question": "...", "answer": "..."},
  {"type": "cultural",    "question": "...", "answer": "..."},
  {"type": "reasoning",   "question": "...", "answer": "..."}
]
Yêu cầu: câu hỏi đa dạng và tự nhiên, câu trả lời đầy đủ 2–4 câu, không
bịa thông tin không có trong ảnh hoặc tiêu đề.
```

QA types:
- `description` — mô tả cảnh vật/không gian/con người trong ảnh
- `location` — tỉnh/thành/địa danh du lịch cụ thể
- `cultural` — lễ hội/ẩm thực/trang phục/phong tục
- `reasoning` — ý nghĩa/liên hệ giữa ảnh và tiêu đề bài viết

---

## Config (config.yaml additions)

```yaml
crawl_vietnamtourism:
  output_dir: data/vietnamtourism-raw
  category_id: 55
  max_pages: null        # null = crawl all (offset until empty)
  max_images: 5000       # hard cap on total images downloaded
  delay_seconds: 1.0
  min_image_width: 200   # skip thumbnails

generate_qa_vietnamtourism:
  raw_dir: data/vietnamtourism-raw
  model: gpt-4o
  max_tokens: 1024
  batch_api: true

prepare_vietnamtourism:
  raw_dir: data/vietnamtourism-raw
  output_dir: data/vietnamtourism
  system_prompt: "Bạn là một trợ lý thị giác tiếng Việt, trả lời trung thực và chỉ dựa trên nội dung nhìn thấy trong ảnh."
  val_ratio: 0.01
  test_ratio: 0.01
  seed: 42
```

---

## Files Created / Modified

| File | Action |
|------|--------|
| `config.yaml` | Add 3 config sections |
| `scripts/crawl_vietnamtourism.py` | Create |
| `scripts/generate_qa_vietnamtourism.py` | Create |
| `scripts/prepare_vietnamtourism_data.py` | Create |
| `tests/test_crawl_vietnamtourism.py` | Create |
| `tests/test_generate_qa_vietnamtourism.py` | Create |
| `tests/test_prepare_vietnamtourism_data.py` | Create |
