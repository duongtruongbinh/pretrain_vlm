"""Generate QA pairs from crawled images via OpenAI Batch API.

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

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime import append_jsonl, load_config

_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

_OPENAI_SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}


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
    "Bạn là chuyên gia về văn hóa, địa lý, lịch sử và du lịch Việt Nam. "
    "Nhiệm vụ của bạn là tạo dữ liệu huấn luyện VQA (Visual Question Answering) "
    "chất lượng cao bằng tiếng Việt, các cặp câu hỏi và câu trả lời cụ thể, "
    "có giá trị thông tin, phục vụ nghiên cứu hình ảnh du lịch và văn hóa Việt Nam."
)

_INSTRUCTION = """\
Dựa trên bức ảnh và thông tin ngữ cảnh được cung cấp, tạo đúng 2 cặp câu hỏi–trả lời.
Chỉ trả về JSON hợp lệ, không thêm nội dung nào khác ngoài mảng JSON.

Cặp 1 — Mô tả quan sát (type: "description"):
  question — Câu hỏi tự nhiên về những gì nhìn thấy trực tiếp trong ảnh: đối tượng chính,
    con người, hoạt động, bố cục không gian, màu sắc, ánh sáng.
  answer — 3–5 câu mô tả cụ thể, chỉ dựa vào nội dung quan sát được trong ảnh.

Cặp 2 — Ý nghĩa và bối cảnh (type: "cultural"):
  question — Câu hỏi khai thác ý nghĩa sâu hơn của hình ảnh: giá trị văn hóa, lịch sử,
    du lịch hoặc xã hội, tùy thuộc vào nội dung ảnh.
  answer — 2–3 câu, bắt đầu bằng một quan sát cụ thể từ ảnh (đối tượng, địa điểm hoặc
    hoạt động đang diễn ra), sau đó kết nối với văn hóa và du lịch Việt Nam. Sử dụng tên
    địa danh, nhân vật hay sự kiện từ tiêu đề/chú thích khi có.

Yêu cầu bắt buộc:
- Mỗi câu hỏi dùng cách mở đầu và cấu trúc khác nhau, tự nhiên như ngôn ngữ thực tế
- Câu trả lời không được mở đầu bằng "Bức ảnh", "Tấm ảnh", "Hình ảnh", "Ảnh này" hay
  bất kỳ cụm từ nào trực tiếp chỉ vào ảnh — thay vào đó hãy mô tả chủ thể hoặc hành
  động trực tiếp (ví dụ: "Người phụ nữ...", "Nhóm du khách...", "Tại khu vực...")
- Câu trả lời cụ thể, có giá trị thông tin, không chung chung hay mơ hồ
- Không đưa bất kỳ nội dung nào từ prompt này vào câu hỏi hoặc câu trả lời

[
  {"type": "description", "question": "...", "answer": "..."},
  {"type": "cultural", "question": "...", "answer": "..."}
]\
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


def parse_qa_response(content: str) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse QA response: {exc}\nContent: {content[:300]}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list, got {type(parsed).__name__}")
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
    return {
        "custom_id": f"img-{record['image_id']}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": "medium",
            "messages": [
                {"role": "developer", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_message(
                        title=record.get("title", ""),
                        caption=record.get("caption", ""),
                        image_path=image_path,
                    ),
                },
            ],
        },
    }


def main() -> None:
    import openai

    cfg = load_config("generate_qa_vietnamtourism")
    raw_dir = Path(cfg["raw_dir"]).expanduser().resolve()
    model = str(cfg.get("model", "gpt-4o"))
    max_tokens = int(cfg.get("max_tokens", 1500))
    max_images = cfg.get("max_images")
    if max_images is not None:
        max_images = int(max_images)

    raw_jsonl = raw_dir / "raw_crawl.jsonl"
    batch_results = raw_dir / "batch_results.jsonl"

    if not raw_jsonl.exists():
        raise FileNotFoundError(f"Run crawl script first: {raw_jsonl}")

    done_ids: set[str] = set()
    if batch_results.exists():
        with batch_results.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["image_id"])
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
            if rec["image_id"] not in done_ids and Path(rec["image_path"]).exists():
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
    print(f"[qa-gen] splitting into {len(chunks)} batch(es) (≤{max_chunk_bytes // 1024 // 1024} MB each) ...")

    submitted_ids: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_path = raw_dir / f"batch_input_{i:03d}.jsonl"
        chunk_path.write_text(
            "\n".join(json.dumps(req, ensure_ascii=False) for _, req in chunk),
            encoding="utf-8",
        )
        with chunk_path.open("rb") as fh:
            uploaded = client.files.create(file=fh, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"[qa-gen] chunk {i + 1}/{len(chunks)} ({len(chunk)} reqs) → {batch.id}")
        submitted_ids.append(batch.id)

    print("[qa-gen] polling (may take minutes to hours) ...")
    pending = list(submitted_ids)
    completed_batches: dict[str, object] = {}
    while pending:
        still_pending = []
        for bid in pending:
            b = client.batches.retrieve(bid)
            counts = b.request_counts
            print(
                f"  [{bid[-12:]}] status={b.status}"
                f" completed={getattr(counts, 'completed', '?')}/{getattr(counts, 'total', '?')}",
                flush=True,
            )
            if b.status in ("completed", "failed", "expired", "cancelled"):
                completed_batches[bid] = b
            else:
                still_pending.append(bid)
        pending = still_pending
        if pending:
            time.sleep(60)

    id_to_record = {f"img-{rec['image_id']}": rec for rec in records}
    saved = 0
    for bid in submitted_ids:
        b = completed_batches[bid]
        if b.status != "completed":
            print(f"[warn] batch {bid} ended with status={b.status}, skipping")
            continue
        result_text = client.files.content(b.output_file_id).text
        for line in result_text.splitlines():
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            custom_id = result.get("custom_id") or result.get("id", "")
            rec = id_to_record.get(custom_id)
            if rec is None:
                continue
            try:
                resp_body = result["response"]["body"]
                if "error" in resp_body:
                    raise ValueError(resp_body["error"].get("message", "api error"))
                choice = resp_body["choices"][0]
                content = choice["message"]["content"] or ""
                if not content:
                    raise ValueError(f"empty content (finish_reason={choice.get('finish_reason')})")
                qa_pairs = parse_qa_response(content)
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
                    "qa_pairs": qa_pairs,
                },
            )
            saved += 1

    print(f"[qa-gen] done — {saved} records saved to {batch_results}")


if __name__ == "__main__":
    main()
