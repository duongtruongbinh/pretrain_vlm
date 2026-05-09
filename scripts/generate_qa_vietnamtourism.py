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

_USER_INSTRUCTION = (
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

_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_text(title: str, caption: str) -> str:
    lines: list[str] = []
    if title:
        lines.append(f"Tiêu đề bài viết: {title}")
    if caption:
        lines.append(f"Mô tả ảnh: {caption}")
    lines.append("")
    lines.append(_USER_INSTRUCTION)
    return "\n".join(lines)


def _encode_image(image_path: Path) -> str:
    return base64.standard_b64encode(image_path.read_bytes()).decode()


def _media_type(image_path: Path) -> str:
    return _MEDIA_TYPES.get(image_path.suffix.lower(), "image/jpeg")


def parse_qa_response(content: str) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse QA response: {exc}\nContent: {content[:300]}") from exc
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
    user_text = build_user_text(
        title=record.get("title", ""),
        caption=record.get("caption", ""),
    )
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
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{_media_type(image_path)};base64,{b64}"},
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
        },
    }


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

    # Resume: skip already-generated image IDs
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
    valid_requests: list[tuple[dict, dict]] = []
    for rec in records:
        try:
            req = build_batch_request(rec, model=model, max_tokens=max_tokens)
            valid_requests.append((rec, req))
        except Exception as exc:
            print(f"[warn] skipping {rec['image_id']}: {exc}")

    batch_input.write_text(
        "\n".join(json.dumps(req, ensure_ascii=False) for _, req in valid_requests),
        encoding="utf-8",
    )
    print(f"[qa-gen] wrote {len(valid_requests)} requests to {batch_input}")

    client = openai.OpenAI()

    print("[qa-gen] uploading batch input ...")
    with batch_input.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    print(f"[qa-gen] uploaded: {uploaded.id}")

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
        completed = counts.completed if counts else "?"
        total = counts.total if counts else "?"
        print(f"  status={batch.status} completed={completed}/{total}", flush=True)
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(60)

    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status={batch.status}")

    print(f"[qa-gen] downloading results ...")
    result_text = client.files.content(batch.output_file_id).text

    id_to_record = {f"img-{rec['image_id']}": rec for rec in records}

    with batch_results.open("a", encoding="utf-8") as out:
        for line in result_text.splitlines():
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

    print(f"[qa-gen] done — results in {batch_results}")


if __name__ == "__main__":
    main()
