"""Generate QA pairs from crawled images via OpenAI Batch API.

Prompt design references:
  - LLaVA (Liu et al., 2023): description + complex-reasoning instruction types
  - ShareGPT4V (Chen et al., 2023): rich spatial descriptions with color/layout
  - InstructBLIP (Dai et al., 2023): diverse short-answer factual templates
"""
from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime import load_config

_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

_SYSTEM_PROMPT = (
    "Bạn là chuyên gia văn hóa, địa lý và du lịch Việt Nam với kiến thức sâu rộng "
    "về cảnh quan, phong tục, lịch sử và ẩm thực Việt Nam. Nhiệm vụ của bạn là tạo "
    "các cặp câu hỏi-trả lời chất lượng cao bằng tiếng Việt về hình ảnh du lịch, "
    "chỉ dựa trên những gì quan sát được trong ảnh và thông tin ngữ cảnh được cung cấp."
)

_INSTRUCTION = """\
Tạo đúng 4 cặp câu hỏi-trả lời về bức ảnh theo 4 loại sau.
Trả về JSON hợp lệ (không thêm bất kỳ nội dung nào ngoài JSON):

[
  {"type": "description",
   "question": "<hỏi mô tả tổng thể: cảnh vật, con người, màu sắc, bố cục không gian>",
   "answer":   "<3–5 câu: vật thể chính, màu sắc, vị trí tương đối, hoạt động, ánh sáng/thời tiết>"},
  {"type": "factual",
   "question": "<hỏi ngắn về 1 yếu tố cụ thể quan sát được: đối tượng, hành động, số lượng, màu sắc>",
   "answer":   "<trả lời trực tiếp 1–2 câu>"},
  {"type": "cultural",
   "question": "<hỏi về ý nghĩa văn hóa, lễ hội, phong tục, ẩm thực hoặc kiến trúc đặc trưng Việt Nam>",
   "answer":   "<2–3 câu kết nối nội dung ảnh với bối cảnh văn hóa-du lịch Việt Nam>"},
  {"type": "reasoning",
   "question": "<hỏi phân tích suy luận: tại sao, như thế nào, kết nối nhiều yếu tố trong ảnh>",
   "answer":   "<2–3 câu lập luận từ bằng chứng quan sát, trình bày theo logic>"}
]

Yêu cầu bắt buộc:
- Câu hỏi đa dạng cách hỏi, tự nhiên (không lặp cùng công thức như "Trong ảnh có gì?")
- Chỉ đề cập những gì thực sự thấy được trong ảnh hoặc có trong tiêu đề/chú thích
- Câu trả lời cụ thể và thông tin, tránh chung chung và mơ hồ\
"""


def build_user_message(title: str, caption: str, image_path: Path) -> list[dict]:
    context: list[str] = []
    if title:
        context.append(f"Tiêu đề bài viết: {title}")
    if caption:
        context.append(f"Chú thích ảnh: {caption}")

    text = "\n".join(context + ["", _INSTRUCTION]) if context else _INSTRUCTION

    b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
    media_type = _MEDIA_TYPES.get(image_path.suffix.lower(), "image/jpeg")

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
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
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

    raw_jsonl = raw_dir / "raw_crawl.jsonl"
    batch_input = raw_dir / "batch_input.jsonl"
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

    batch_input.write_text(
        "\n".join(json.dumps(req, ensure_ascii=False) for _, req in valid),
        encoding="utf-8",
    )
    print(f"[qa-gen] {len(valid)} requests → {batch_input}")

    client = openai.OpenAI()

    print("[qa-gen] uploading batch input ...")
    with batch_input.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    print(f"[qa-gen] uploaded file: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"[qa-gen] batch created: {batch.id}")

    print("[qa-gen] polling (may take minutes to hours) ...")
    while True:
        batch = client.batches.retrieve(batch.id)
        counts = batch.request_counts
        completed = getattr(counts, "completed", "?")
        total = getattr(counts, "total", "?")
        print(f"  status={batch.status} completed={completed}/{total}", flush=True)
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(60)

    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status={batch.status}")

    result_text = client.files.content(batch.output_file_id).text
    id_to_record = {f"img-{rec['image_id']}": rec for rec in records}

    saved = 0
    with batch_results.open("a", encoding="utf-8") as out:
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
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                qa_pairs = parse_qa_response(content)
            except Exception as exc:
                print(f"[warn] parse failed for {custom_id}: {exc}")
                continue
            out.write(
                json.dumps(
                    {
                        "image_id": rec["image_id"],
                        "image_path": rec["image_path"],
                        "title": rec["title"],
                        "caption": rec["caption"],
                        "article_url": rec["article_url"],
                        "date": rec["date"],
                        "qa_pairs": qa_pairs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            saved += 1

    print(f"[qa-gen] done — {saved} records saved to {batch_results}")


if __name__ == "__main__":
    main()
