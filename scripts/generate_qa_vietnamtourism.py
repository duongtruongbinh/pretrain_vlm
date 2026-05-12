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
