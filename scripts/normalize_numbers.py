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


def number_to_words(n: int) -> str | None:
    return _NUMBER_WORDS.get(n)
