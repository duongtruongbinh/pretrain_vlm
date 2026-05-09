# vietnamtourism Stage 2 Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three scripts that crawl ~1,000–5,000 images from vietnamtourism.gov.vn, generate 4 QA
pairs per image via GPT-4o Batch API, and produce train/val/test JSONL files for Stage 2 training.

**Architecture:** Pure JSON API crawl (no Playwright) → OpenAI Batch API for async QA generation →
format-conversion script aligned with existing `prepare_instruction_data.py` patterns.

**Tech Stack:** Python 3.11, `requests`, `beautifulsoup4`, `openai` (Batch API), `pillow`,
`pyyaml`, `pytest`. Run with `/home/shared/miniconda3/envs/nhantd_env/bin/python`.

---

## File Map

| File | Responsibility |
|------|---------------|
| `config.yaml` | Add `crawl_vietnamtourism`, `generate_qa_vietnamtourism`, `prepare_vietnamtourism` sections |
| `scripts/crawl_vietnamtourism.py` | API pagination, HTML image extraction, image download, `raw_crawl.jsonl` |
| `scripts/generate_qa_vietnamtourism.py` | Build batch requests, submit to OpenAI Batch API, poll, parse results |
| `scripts/prepare_vietnamtourism_data.py` | Join raw + QA results, convert to project JSONL, split train/val/test |
| `tests/test_crawl_vietnamtourism.py` | Unit tests for pure functions in crawler |
| `tests/test_generate_qa_vietnamtourism.py` | Unit tests for batch request building and response parsing |
| `tests/test_prepare_vietnamtourism_data.py` | Unit tests for split assignment and sample building |

---

## Task 1: Config sections

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Add three config sections to config.yaml**

Append after the last section:

```yaml
crawl_vietnamtourism:
  output_dir: data/vietnamtourism-raw
  category_id: 55
  max_pages: null
  max_images: 5000
  delay_seconds: 1.0
  min_image_width: 200

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

- [ ] **Step 2: Verify config loads**

```bash
/home/shared/miniconda3/envs/nhantd_env/bin/python -c "
import sys; sys.path.insert(0, '.')
from src.runtime import load_config
for s in ['crawl_vietnamtourism','generate_qa_vietnamtourism','prepare_vietnamtourism']:
    cfg = load_config(s); print(s, list(cfg.keys()))
"
```

Expected: prints the three sections with their keys.

---

## Task 2: Crawler — pure functions + tests

**Files:**
- Create: `scripts/crawl_vietnamtourism.py`
- Create: `tests/test_crawl_vietnamtourism.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_crawl_vietnamtourism.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.crawl_vietnamtourism import (
    build_api_url,
    make_image_id,
    extract_images_from_html,
)


def test_build_api_url_page1():
    url, params = build_api_url(cat_id=55, page=1, lang="vi")
    assert url == "https://public.vietnamtourism.gov.vn/cat/55"
    assert params["type"] == "1"
    import json
    p = json.loads(params["param"])
    assert p["offset"] == 1
    assert p["lang"] == "vi"


def test_build_api_url_page5():
    url, params = build_api_url(cat_id=55, page=5, lang="vi")
    import json
    p = json.loads(params["param"])
    assert p["offset"] == 5


def test_make_image_id_stable():
    a = make_image_id("123", "/images/2026/foo.jpg")
    b = make_image_id("123", "/images/2026/foo.jpg")
    assert a == b
    assert len(a) == 16
    assert a.isalnum()


def test_make_image_id_unique():
    a = make_image_id("123", "/images/a.jpg")
    b = make_image_id("123", "/images/b.jpg")
    assert a != b


def test_extract_images_from_html_basic():
    html = """
    <html><body>
    <p><em><a href="/images/foo.jpg"><img src="/images/foo.jpg" style="width:700px"></a><br>
    Cảnh đẹp Hội An</em></p>
    </body></html>
    """
    imgs = extract_images_from_html(html, base_url="https://vietnamtourism.gov.vn")
    assert len(imgs) == 1
    assert imgs[0]["src"] == "https://vietnamtourism.gov.vn/images/foo.jpg"
    assert "Hội An" in imgs[0]["caption"]


def test_extract_images_from_html_skips_tiny():
    html = """
    <html><body>
    <img src="/images/icon.png" style="width:50px">
    <img src="/images/photo.jpg" style="width:700px">
    </body></html>
    """
    imgs = extract_images_from_html(html, base_url="https://vietnamtourism.gov.vn", min_width=200)
    assert len(imgs) == 1
    assert "photo.jpg" in imgs[0]["src"]


def test_extract_images_from_html_no_images():
    imgs = extract_images_from_html("<html><body><p>No images</p></body></html>",
                                    base_url="https://vietnamtourism.gov.vn")
    assert imgs == []


def test_extract_images_from_html_absolute_src():
    html = '<html><body><img src="https://cdn.example.com/photo.jpg" style="width:700px"></body></html>'
    imgs = extract_images_from_html(html, base_url="https://vietnamtourism.gov.vn")
    assert imgs[0]["src"] == "https://cdn.example.com/photo.jpg"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/docs/nhantd/LLAVA && \
/home/shared/miniconda3/envs/nhantd_env/bin/python -m pytest tests/test_crawl_vietnamtourism.py -v 2>&1 | head -30
```

Expected: ImportError or ModuleNotFoundError for `scripts.crawl_vietnamtourism`.

- [ ] **Step 3: Implement crawler pure functions**

Create `scripts/crawl_vietnamtourism.py` (pure functions section only):

```python
"""Crawl images and metadata from vietnamtourism.gov.vn/cat/55 via public JSON API."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.runtime import load_config

API_BASE = "https://public.vietnamtourism.gov.vn"
IMG_BASE = "https://vietnamtourism.gov.vn"
PAGE_SIZE = 15


def build_api_url(cat_id: int, page: int, lang: str = "vi") -> tuple[str, dict]:
    url = f"{API_BASE}/cat/{cat_id}"
    param = json.dumps({"offset": page, "callType": 1, "lang": lang})
    return url, {"type": "1", "param": param}


def make_image_id(post_id: str, img_src: str) -> str:
    raw = f"{post_id}:{img_src}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _parse_width_from_style(style: str) -> int:
    m = re.search(r"width\s*:\s*(\d+)", style or "")
    return int(m.group(1)) if m else 0


def extract_images_from_html(html: str, base_url: str, min_width: int = 200) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src or src.endswith(".svg"):
            continue

        # Skip thumbnails by inline style width
        style = img.get("style", "")
        width = _parse_width_from_style(style)
        if 0 < width < min_width:
            continue

        src = urljoin(base_url, src)

        # Caption: text of the enclosing <em> tag, minus the img itself
        caption = ""
        em = img.find_parent("em")
        if em:
            caption = em.get_text(separator=" ", strip=True)
        if not caption:
            alt = img.get("alt", "").strip()
            caption = alt

        results.append({"src": src, "caption": caption})

    return results
```

- [ ] **Step 4: Run tests — should pass**

```bash
cd /home/docs/nhantd/LLAVA && \
/home/shared/miniconda3/envs/nhantd_env/bin/python -m pytest tests/test_crawl_vietnamtourism.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/crawl_vietnamtourism.py tests/test_crawl_vietnamtourism.py
git commit -m "feat: add crawler pure functions with tests"
```

---

## Task 3: Crawler — main orchestration

**Files:**
- Modify: `scripts/crawl_vietnamtourism.py` (add `download_image` and `main`)

- [ ] **Step 1: Add download_image and main to crawl_vietnamtourism.py**

Append to `scripts/crawl_vietnamtourism.py`:

```python
def download_image(src: str, dest: Path, session: requests.Session, timeout: int = 15) -> bool:
    try:
        resp = session.get(src, timeout=timeout, stream=True)
        resp.raise_for_status()
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
    data = resp.json()
    return data.get("child") or []


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

    # Load already-crawled image IDs to allow resume
    crawled_ids: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    crawled_ids.add(rec["image_id"])
    print(f"[crawl] resuming — {len(crawled_ids)} images already crawled")

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research crawler; contact: research)"

    total_images = 0
    page = 1

    with jsonl_path.open("a", encoding="utf-8") as out:
        while True:
            if max_pages is not None and page > int(max_pages):
                break
            if total_images >= max_images:
                print(f"[crawl] reached max_images={max_images}, stopping")
                break

            print(f"[crawl] page {page} ...", flush=True)
            try:
                posts = fetch_page(session, cat_id=cat_id, page=page)
            except Exception as exc:
                print(f"[warn] page {page} failed: {exc}")
                break

            if not posts:
                print(f"[crawl] no posts on page {page}, done")
                break

            for post in posts:
                if total_images >= max_images:
                    break

                post_id = str(post["id"])
                title = post.get("title", "").strip()
                date = post.get("dateedit", "")[:10]
                article_url = f"https://vietnamtourism.gov.vn/post/{post_id}"
                content_html = post.get("content", "")

                imgs = extract_images_from_html(content_html, base_url=IMG_BASE, min_width=min_width)
                for img_info in imgs:
                    if total_images >= max_images:
                        break

                    image_id = make_image_id(post_id, img_info["src"])
                    if image_id in crawled_ids:
                        continue

                    ext = Path(urlparse(img_info["src"]).path).suffix.lower() or ".jpg"
                    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                        continue

                    dest = images_dir / f"{image_id}{ext}"
                    if not download_image(img_info["src"], dest, session):
                        continue

                    record = {
                        "image_id": image_id,
                        "image_path": str(dest),
                        "title": title,
                        "caption": img_info["caption"],
                        "article_url": article_url,
                        "date": date,
                        "post_id": post_id,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    crawled_ids.add(image_id)
                    total_images += 1

            page += 1
            time.sleep(delay)

    print(f"[crawl] done — {total_images} new images, {len(crawled_ids)} total in {jsonl_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (5 images, 1 page)**

```bash
cd /home/docs/nhantd/LLAVA && \
/home/shared/miniconda3/envs/nhantd_env/bin/python -c "
import sys; sys.path.insert(0, '.')
import yaml
# Temporarily override max_images for smoke test
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('config.yaml').read_text())
cfg['crawl_vietnamtourism']['max_images'] = 5
cfg['crawl_vietnamtourism']['max_pages'] = 1
Path('config.yaml').write_text(yaml.dump(cfg, allow_unicode=True))
" && \
/home/shared/miniconda3/envs/nhantd_env/bin/python scripts/crawl_vietnamtourism.py && \
cat data/vietnamtourism-raw/raw_crawl.jsonl | head -2 | python3 -m json.tool
```

Expected: 2–5 records in `raw_crawl.jsonl`, images downloaded.

- [ ] **Step 3: Restore config and commit**

```bash
# Restore max_images/max_pages to null/5000 in config.yaml, then:
git add scripts/crawl_vietnamtourism.py config.yaml data/vietnamtourism-raw/.gitkeep
git commit -m "feat: add crawl_vietnamtourism main orchestration"
```

---

## Task 4: QA Generator — pure functions + tests

**Files:**
- Create: `scripts/generate_qa_vietnamtourism.py`
- Create: `tests/test_generate_qa_vietnamtourism.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_generate_qa_vietnamtourism.py`:

```python
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from scripts.generate_qa_vietnamtourism import (
    build_system_prompt,
    build_user_text,
    parse_qa_response,
    build_batch_request,
)

SAMPLE_RECORD = {
    "image_id": "abc123",
    "image_path": "",  # will use a tmp file in fixture tests
    "title": "Lễ hội Hoa Đà Lạt 2024",
    "caption": "Du khách tham quan vườn hoa",
    "article_url": "https://vietnamtourism.gov.vn/post/123",
    "date": "2024-05-15",
}


def test_build_user_text_with_caption():
    text = build_user_text(title="Vịnh Hạ Long", caption="Du thuyền trên biển")
    assert "Vịnh Hạ Long" in text
    assert "Du thuyền trên biển" in text


def test_build_user_text_no_caption():
    text = build_user_text(title="Phố cổ Hội An", caption="")
    assert "Phố cổ Hội An" in text
    assert "Mô tả ảnh" not in text


def test_parse_qa_response_valid():
    raw = json.dumps([
        {"type": "description", "question": "Ảnh mô tả gì?", "answer": "Ảnh chụp..."},
        {"type": "location",    "question": "Đây là đâu?",  "answer": "Đây là..."},
        {"type": "cultural",    "question": "Lễ hội gì?",   "answer": "Đây là lễ..."},
        {"type": "reasoning",   "question": "Tại sao?",     "answer": "Vì..."},
    ])
    pairs = parse_qa_response(raw)
    assert len(pairs) == 4
    assert pairs[0]["type"] == "description"
    assert pairs[0]["question"]
    assert pairs[0]["answer"]


def test_parse_qa_response_wrapped_in_markdown():
    raw = '```json\n[{"type":"description","question":"Q","answer":"A"}]\n```'
    pairs = parse_qa_response(raw)
    assert len(pairs) == 1


def test_parse_qa_response_invalid_json_raises():
    with pytest.raises(ValueError, match="parse"):
        parse_qa_response("not json at all !!!")


def test_parse_qa_response_wrong_shape_raises():
    with pytest.raises(ValueError, match="list"):
        parse_qa_response(json.dumps({"type": "description"}))


def test_build_batch_request_structure(tmp_path):
    img = tmp_path / "img.jpg"
    from PIL import Image
    Image.new("RGB", (100, 100), color=(255, 0, 0)).save(img)

    record = {**SAMPLE_RECORD, "image_path": str(img)}
    req = build_batch_request(record, model="gpt-4o", max_tokens=512)

    assert req["custom_id"] == f"img-{record['image_id']}"
    assert req["method"] == "POST"
    assert req["url"] == "/v1/chat/completions"
    body = req["body"]
    assert body["model"] == "gpt-4o"
    assert body["max_tokens"] == 512
    msgs = body["messages"]
    assert msgs[0]["role"] == "system"
    user_content = msgs[1]["content"]
    assert isinstance(user_content, list)
    types = [c["type"] for c in user_content]
    assert "image_url" in types
    assert "text" in types
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/docs/nhantd/LLAVA && \
/home/shared/miniconda3/envs/nhantd_env/bin/python -m pytest tests/test_generate_qa_vietnamtourism.py -v 2>&1 | head -15
```

Expected: ImportError.

- [ ] **Step 3: Implement pure functions**

Create `scripts/generate_qa_vietnamtourism.py`:

```python
"""Generate QA pairs from crawled images via OpenAI Batch API."""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

from src.runtime import load_config

SYSTEM_PROMPT = (
    "Bạn là chuyên gia văn hóa và du lịch Việt Nam. Nhiệm vụ của bạn là tạo "
    "các cặp câu hỏi-trả lời chất lượng cao bằng tiếng Việt về hình ảnh du lịch "
    "Việt Nam, dựa trên nội dung thực sự quan sát được trong ảnh."
)

USER_INSTRUCTION = (
    "Hãy tạo đúng 4 cặp câu hỏi-trả lời về hình ảnh bằng tiếng Việt, trả về JSON "
    "hợp lệ theo định dạng sau (không thêm gì ngoài JSON):\n"
    "[\n"
    '  {"type": "description", "question": "...", "answer": "..."},\n'
    '  {"type": "location",    "question": "...", "answer": "..."},\n'
    '  {"type": "cultural",    "question": "...", "answer": "..."},\n'
    '  {"type": "reasoning",   "question": "...", "answer": "..."}\n'
    "]\n"
    "Yêu cầu: câu hỏi đa dạng và tự nhiên, câu trả lời đầy đủ 2–4 câu, "
    "không bịa thông tin không có trong ảnh hoặc tiêu đề."
)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_text(title: str, caption: str) -> str:
    lines = []
    if title:
        lines.append(f"Tiêu đề bài viết: {title}")
    if caption:
        lines.append(f"Mô tả ảnh: {caption}")
    lines.append("")
    lines.append(USER_INSTRUCTION)
    return "\n".join(lines)


def _encode_image(image_path: Path) -> str:
    return base64.standard_b64encode(image_path.read_bytes()).decode()


def _image_media_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")


def parse_qa_response(content: str) -> list[dict]:
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse QA response as JSON: {exc}\nContent: {content[:200]}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list of QA pairs, got {type(parsed).__name__}")
    return [
        {
            "type": str(item.get("type", "")),
            "question": str(item.get("question", "")).strip(),
            "answer": str(item.get("answer", "")).strip(),
        }
        for item in parsed
        if isinstance(item, dict)
    ]


def build_batch_request(record: dict, *, model: str, max_tokens: int) -> dict:
    image_path = Path(record["image_path"])
    b64 = _encode_image(image_path)
    media_type = _image_media_type(image_path)
    user_text = build_user_text(title=record.get("title", ""), caption=record.get("caption", ""))

    return {
        "custom_id": f"img-{record['image_id']}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": build_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
        },
    }
```

- [ ] **Step 4: Run tests — should pass**

```bash
cd /home/docs/nhantd/LLAVA && \
/home/shared/miniconda3/envs/nhantd_env/bin/python -m pytest tests/test_generate_qa_vietnamtourism.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_qa_vietnamtourism.py tests/test_generate_qa_vietnamtourism.py
git commit -m "feat: add QA generator pure functions with tests"
```

---

## Task 5: QA Generator — Batch API orchestration

**Files:**
- Modify: `scripts/generate_qa_vietnamtourism.py` (add `main`)

- [ ] **Step 1: Append main() to generate_qa_vietnamtourism.py**

```python
def main() -> None:
    import openai

    cfg = load_config("generate_qa_vietnamtourism")
    raw_dir = Path(cfg["raw_dir"]).expanduser().resolve()
    model = str(cfg.get("model", "gpt-4o"))
    max_tokens = int(cfg.get("max_tokens", 1024))

    raw_jsonl = raw_dir / "raw_crawl.jsonl"
    batch_input = raw_dir / "batch_input.jsonl"
    batch_results = raw_dir / "batch_results.jsonl"

    if not raw_jsonl.exists():
        raise FileNotFoundError(f"Run crawl script first: {raw_jsonl}")

    # Load already-generated image IDs to allow resume
    done_ids: set[str] = set()
    if batch_results.exists():
        with batch_results.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done_ids.add(rec["image_id"])
    print(f"[qa-gen] {len(done_ids)} already generated, skipping")

    records = []
    with raw_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["image_id"] not in done_ids and Path(rec["image_path"]).exists():
                records.append(rec)

    if not records:
        print("[qa-gen] nothing to generate, done")
        return

    print(f"[qa-gen] building {len(records)} batch requests ...")
    requests_list = []
    for rec in records:
        try:
            req = build_batch_request(rec, model=model, max_tokens=max_tokens)
            requests_list.append((rec, req))
        except Exception as exc:
            print(f"[warn] skipping {rec['image_id']}: {exc}")

    batch_input.write_text(
        "\n".join(json.dumps(req, ensure_ascii=False) for _, req in requests_list),
        encoding="utf-8",
    )
    print(f"[qa-gen] wrote {len(requests_list)} requests to {batch_input}")

    client = openai.OpenAI()

    print("[qa-gen] uploading batch input file ...")
    with batch_input.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    print(f"[qa-gen] file uploaded: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"[qa-gen] batch created: {batch.id}")

    print("[qa-gen] polling for completion (this may take minutes to hours) ...")
    while True:
        batch = client.batches.retrieve(batch.id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else "?"
        total = batch.request_counts.total if batch.request_counts else "?"
        print(f"  status={status} completed={completed}/{total}", flush=True)
        if status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(60)

    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status={batch.status}")

    print(f"[qa-gen] downloading results from file {batch.output_file_id} ...")
    result_content = client.files.content(batch.output_file_id).text

    # Map custom_id → record for lookup
    id_to_record = {f"img-{rec['image_id']}": rec for rec in records}

    with batch_results.open("a", encoding="utf-8") as out:
        for line in result_content.splitlines():
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            custom_id = result.get("id") or result.get("custom_id", "")
            rec = id_to_record.get(custom_id)
            if rec is None:
                continue

            try:
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                qa_pairs = parse_qa_response(content)
            except Exception as exc:
                print(f"[warn] parse failed for {custom_id}: {exc}")
                continue

            out_record = {
                "image_id": rec["image_id"],
                "image_path": rec["image_path"],
                "title": rec["title"],
                "caption": rec["caption"],
                "article_url": rec["article_url"],
                "date": rec["date"],
                "qa_pairs": qa_pairs,
            }
            out.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"[qa-gen] done — results in {batch_results}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script is importable and main is defined**

```bash
cd /home/docs/nhantd/LLAVA && \
/home/shared/miniconda3/envs/nhantd_env/bin/python -c "
import sys; sys.path.insert(0, '.')
from scripts.generate_qa_vietnamtourism import main, build_batch_request
print('ok')
"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_qa_vietnamtourism.py
git commit -m "feat: add QA generator Batch API orchestration"
```

---

## Task 6: Prepare script — pure functions + tests

**Files:**
- Create: `scripts/prepare_vietnamtourism_data.py`
- Create: `tests/test_prepare_vietnamtourism_data.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_prepare_vietnamtourism_data.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare_vietnamtourism_data import assign_split, build_instruction_sample


def test_assign_split_deterministic():
    a = assign_split("abc123", seed=42, val_ratio=0.1, test_ratio=0.1)
    b = assign_split("abc123", seed=42, val_ratio=0.1, test_ratio=0.1)
    assert a == b
    assert a in {"train", "val", "test"}


def test_assign_split_distribution():
    # With 1000 IDs, roughly 10% val, 10% test
    ids = [f"img{i:04d}" for i in range(1000)]
    splits = [assign_split(i, seed=42, val_ratio=0.1, test_ratio=0.1) for i in ids]
    from collections import Counter
    counts = Counter(splits)
    assert 50 < counts["val"] < 200
    assert 50 < counts["test"] < 200
    assert counts["train"] > 600


def test_build_instruction_sample_structure():
    qa = {"type": "description", "question": "Ảnh mô tả gì?", "answer": "Đây là..."}
    sample = build_instruction_sample(
        image_path=Path("/data/images/abc.jpg"),
        qa_pair=qa,
        system_prompt="Bạn là trợ lý.",
        image_id="abc123",
        source="vietnamtourism",
        sample_index=0,
    )
    assert sample["image"] == "/data/images/abc.jpg"
    assert sample["image_id"] == "abc123"
    assert sample["source_dataset"] == "vietnamtourism"
    assert sample["sample_type"] == "qa"
    msgs = sample["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert "Ảnh mô tả gì?" in msgs[1]["content"]
    assert "Đây là..." in msgs[2]["content"]


def test_build_instruction_sample_id_format():
    qa = {"type": "location", "question": "Q", "answer": "A"}
    sample = build_instruction_sample(
        image_path=Path("/img.jpg"),
        qa_pair=qa,
        system_prompt="sys",
        image_id="xyz",
        source="src",
        sample_index=2,
    )
    assert sample["id"] == "vntour_xyz_qa_003"
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/docs/nhantd/LLAVA && \
/home/shared/miniconda3/envs/nhantd_env/bin/python -m pytest tests/test_prepare_vietnamtourism_data.py -v 2>&1 | head -15
```

Expected: ImportError.

- [ ] **Step 3: Implement pure functions**

Create `scripts/prepare_vietnamtourism_data.py`:

```python
"""Convert raw_crawl.jsonl + batch_results.jsonl to project instruction-tuning JSONL."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.runtime import load_config


def assign_split(image_id: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.sha1(f"{seed}:{image_id}".encode()).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    if score < test_ratio:
        return "test"
    if score < test_ratio + val_ratio:
        return "val"
    return "train"


def build_instruction_sample(
    *,
    image_path: Path,
    qa_pair: dict,
    system_prompt: str,
    image_id: str,
    source: str,
    sample_index: int,
) -> dict:
    return {
        "id": f"vntour_{image_id}_qa_{sample_index + 1:03d}",
        "image": str(image_path),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": qa_pair["question"]},
            {"role": "assistant", "content": qa_pair["answer"]},
        ],
        "sample_type": "qa",
        "source_dataset": source,
        "image_id": image_id,
    }
```

- [ ] **Step 4: Run tests — should pass**

```bash
cd /home/docs/nhantd/LLAVA && \
/home/shared/miniconda3/envs/nhantd_env/bin/python -m pytest tests/test_prepare_vietnamtourism_data.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_vietnamtourism_data.py tests/test_prepare_vietnamtourism_data.py
git commit -m "feat: add prepare_vietnamtourism pure functions with tests"
```

---

## Task 7: Prepare script — main orchestration

**Files:**
- Modify: `scripts/prepare_vietnamtourism_data.py` (add `main`)

- [ ] **Step 1: Append main() to prepare_vietnamtourism_data.py**

```python
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

                for idx, qa_pair in enumerate(rec.get("qa_pairs", [])):
                    if not qa_pair.get("question") or not qa_pair.get("answer"):
                        counters["skipped_empty_qa"] += 1
                        continue
                    sample = build_instruction_sample(
                        image_path=image_path,
                        qa_pair=qa_pair,
                        system_prompt=system_prompt,
                        image_id=image_id,
                        source=source,
                        sample_index=idx,
                    )
                    handles[split].write(json.dumps(sample, ensure_ascii=False) + "\n")
                    counters[f"{split}_samples"] += 1
                    counters["total_samples"] += 1
    finally:
        for h in handles.values():
            h.close()

    print(f"[prepare] done")
    print(f"  train: {counters['train_samples']} samples from {len(split_image_ids['train'])} images")
    print(f"  val:   {counters['val_samples']} samples from {len(split_image_ids['val'])} images")
    print(f"  test:  {counters['test_samples']} samples from {len(split_image_ids['test'])} images")
    print(f"  skipped missing images: {counters['skipped_missing_image']}")

    report = {
        "source": source,
        "counts": dict(counters),
        "split_unique_images": {k: len(v) for k, v in split_image_ids.items()},
    }
    (output_dir / "prepare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify all tests still pass**

```bash
cd /home/docs/nhantd/LLAVA && \
/home/shared/miniconda3/envs/nhantd_env/bin/python -m pytest \
  tests/test_crawl_vietnamtourism.py \
  tests/test_generate_qa_vietnamtourism.py \
  tests/test_prepare_vietnamtourism_data.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Final commit**

```bash
git add scripts/prepare_vietnamtourism_data.py
git commit -m "feat: add prepare_vietnamtourism main orchestration"
```
