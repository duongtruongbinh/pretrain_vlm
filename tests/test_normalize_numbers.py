from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from normalize_numbers import number_to_words


def test_basic_numbers():
    assert number_to_words(1) == "một"
    assert number_to_words(2) == "hai"
    assert number_to_words(5) == "năm"
    assert number_to_words(10) == "mười"


def test_irregular_forms():
    assert number_to_words(15) == "mười lăm"   # not "mười năm"
    assert number_to_words(20) == "hai mươi"


def test_teens():
    assert number_to_words(11) == "mười một"
    assert number_to_words(14) == "mười bốn"
    assert number_to_words(19) == "mười chín"


def test_out_of_range_returns_none():
    assert number_to_words(0) is None
    assert number_to_words(21) is None
    assert number_to_words(99) is None
