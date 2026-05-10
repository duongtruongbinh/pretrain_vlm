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
    # Bare credit: uppercase "Ảnh:" only — "Trong ảnh:" uses lowercase and is kept
    caption = re.sub(r"[,\s]*[-–]?\s*Ảnh\s*:\s*[^\n]*$", "", caption).strip()
    # Standalone illustrative-photo markers
    if re.fullmatch(r"[Ảả]nh\s+minh\s+\S+", caption):
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
