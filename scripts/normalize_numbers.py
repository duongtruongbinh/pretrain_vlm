from __future__ import annotations

import json
import re
from pathlib import Path

_NUMBER_WORDS: dict[int, str] = {
    1: "một", 2: "hai", 3: "ba", 4: "bốn", 5: "năm",
    6: "sáu", 7: "bảy", 8: "tám", 9: "chín", 10: "mười",
    11: "mười một", 12: "mười hai", 13: "mười ba", 14: "mười bốn",
    15: "mười lăm", 16: "mười sáu", 17: "mười bảy", 18: "mười tám",
    19: "mười chín", 20: "hai mươi",
}

_CLASSIFIERS = [
    "người", "cái", "con", "chiếc", "bức", "tờ", "cốc", "cây", "cuốn", "quyển",
    "cặp", "đôi", "đàn", "nhóm", "tốp", "bộ", "túi", "gói", "chai", "lon", "hộp",
    "tầng", "tòa", "căn", "phòng", "màn",
    # thêm vòng 1
    "cuộn", "lối", "phần", "tấm", "miếng", "mảnh", "hàng", "bậc", "dải", "cọng",
    "bát", "đĩa", "mâm", "que", "thanh", "sợi", "vòng",
    # thêm vòng 2
    "bánh", "cửa", "cánh", "cột", "lá", "kệ", "ngôi", "quả", "chậu", "tô",
    "quầy", "dãy", "đứa", "vùng", "góc", "cô", "chân", "hướng", "loại", "máy",
    "xe", "bên", "tay", "dòng", "hình",
    # thêm vòng 3
    "biển", "khu", "bảng", "đường", "chén", "cổng", "cầu", "ly", "sạp", "ngọn",
    "tủ", "bãi", "dĩa", "chỗ", "chú", "gian", "đoàn", "đèn", "màu",
    # thêm vòng 4
    "bàn", "thùng", "món", "vòi", "khung", "rổ", "bó", "giỏ", "hồ", "ghế",
    "trái", "bông", "bồn", "nhà", "lư", "tán", "ngã", "phía",
    # thêm vòng 5
    "làn", "nơi", "bóng", "lầu", "vật", "mái", "bịch", "lỗ", "dây", "lò",
    "đoạn", "tàu", "cụm", "tốp", "ổ", "bụi", "khối", "mảng", "súng",
    # thêm vòng 6
    "mũ", "nhân", "bé", "sân", "chàng", "mặt", "lớp", "bạn", "vạch", "tư",
    "ông", "anh", "em", "chị", "cậu", "đội", "ban", "tổ", "toán", "nhóm",
    # thêm vòng 7
    "băng", "lát", "cạnh", "bình", "ô", "manơcanh", "ngón", "nửa", "ảnh",
    "quán", "bục", "phông", "cành", "vị",
    # thêm vòng 8 (long tail)
    "sấp", "lan", "đài", "biểu", "toa", "bước",
    # thêm vòng 9 - full long tail classifiers
    "ngựa", "viên", "nến", "phím", "bếp", "chuột", "lưỡi", "móc", "nón", "trụ",
    "thảm", "luồng", "lưới", "nhánh", "chữ", "lồng", "bữa", "cục", "rãnh", "ngõ",
    "thỏi", "chùm", "thúng", "mẩu", "hang", "lề", "khúc", "lùm", "bản", "kim",
    "nền", "loạt", "cảnh", "trần", "lẳng", "nam", "gái", "thìa", "điểm", "lốc",
    "pho", "hương", "cung", "buổi", "thời", "bưc", "bờ", "cặp",
    # thêm vòng 10 - từ phát hiện qua scan thực tế
    "rừng", "bầu", "mô", "hành", "từ", "ánh", "dinh",
]

_PATTERN = re.compile(
    r'\b(\d{1,2})\s+(' + "|".join(re.escape(c) for c in _CLASSIFIERS) + r')\b'
)


def number_to_words(n: int) -> str | None:
    return _NUMBER_WORDS.get(n)


def normalize_caption(text: str) -> str:
    def _replace(m: re.Match) -> str:
        word = number_to_words(int(m.group(1)))
        if word is None:
            return m.group(0)
        return f"{word} {m.group(2)}"

    return _PATTERN.sub(_replace, text)


def process_file(path: Path) -> int:
    tmp = path.with_suffix(".tmp")
    total = 0
    with path.open(encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            original = obj.get("caption", "")
            obj["caption"] = normalize_caption(original)
            if obj["caption"] != original:
                total += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return total


def main() -> None:
    targets = [
        Path("data/coco2017/train.jsonl"),
        Path("data/uit-openviic/train.jsonl"),
    ]
    for path in targets:
        if not path.exists():
            print(f"SKIP (not found): {path}")
            continue
        count = process_file(path)
        print(f"{path}: {count} replacements")


if __name__ == "__main__":
    main()
