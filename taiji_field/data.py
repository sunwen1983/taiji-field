"""Hexagram data: loading and identity.

Bit convention (critical, see README):
- yang = 1, yin = 0; bits ordered bottom-to-top, L1..L6.
- A hexagram's canonical identity is its 6-bit string, e.g. "111111" (Qian).
- This matches the ``zhouyi-6bit`` package. The Go library ``godcong/yi``
  uses the OPPOSITE convention (yang = 0) — do not mix the two.

Data source: ``zhouyi-6bit`` (PyPI, MIT license, https://hexsage.ai/hexagrams).
``load_hexagrams`` reads ``hexagrams_64_binary.json`` from the installed
package when available and falls back to the vendored copy in
``taiji_field/data/`` (same file, same license, kept only as a fallback).

The Chinese names table (``CHINESE_NAMES``) is the standard King Wen
sequence of the 64 hexagram names — canonical public-domain knowledge,
not research data. It is cross-checked against the dataset's pinyin in
the test suite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Standard King Wen order, hexagrams 1-64.
CHINESE_NAMES = {
    1: "乾", 2: "坤", 3: "屯", 4: "蒙", 5: "需", 6: "讼", 7: "师", 8: "比",
    9: "小畜", 10: "履", 11: "泰", 12: "否", 13: "同人", 14: "大有",
    15: "谦", 16: "豫", 17: "随", 18: "蛊", 19: "临", 20: "观",
    21: "噬嗑", 22: "贲", 23: "剥", 24: "复", 25: "无妄", 26: "大畜",
    27: "颐", 28: "大过", 29: "坎", 30: "离", 31: "咸", 32: "恒",
    33: "遁", 34: "大壮", 35: "晋", 36: "明夷", 37: "家人", 38: "睽",
    39: "蹇", 40: "解", 41: "损", 42: "益", 43: "夬", 44: "姤",
    45: "萃", 46: "升", 47: "困", 48: "井", 49: "革", 50: "鼎",
    51: "震", 52: "艮", 53: "渐", 54: "归妹", 55: "丰", 56: "旅",
    57: "巽", 58: "兑", 59: "涣", 60: "节", 61: "中孚", 62: "小过",
    63: "既济", 64: "未济",
}


@dataclass(frozen=True)
class Hexagram:
    """One hexagram. ``bits`` is the L1..L6 string, e.g. "111111"."""

    king_wen: int
    unicode: str
    name_zh: str
    pinyin: str
    english: str
    upper_trigram: str
    lower_trigram: str
    bits: str
    shao_yong: int
    # Relations as recorded by the dataset (used to cross-check our code).
    nuclear_bits: str
    opposite_bits: str
    inverted_bits: str


def _load_from_installed_package() -> dict:
    import importlib.resources as resources

    import zhouyi_6bit

    path = resources.files(zhouyi_6bit) / "hexagrams_64_binary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_vendored() -> dict:
    path = Path(__file__).parent / "data" / "hexagrams_64_binary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_hexagrams() -> list[Hexagram]:
    """Load all 64 hexagrams, King Wen order.

    Prefers the installed ``zhouyi-6bit`` package; falls back to the
    vendored copy shipped in this repo (same data, same MIT license).
    """
    try:
        raw = _load_from_installed_package()
    except Exception:
        raw = _load_vendored()
    records = raw["hexagrams"]
    if len(records) != 64:
        raise ValueError(f"expected 64 hexagrams, got {len(records)}")
    out = []
    for r in records:
        kw = int(r["king_wen"])
        out.append(
            Hexagram(
                king_wen=kw,
                unicode=r["unicode"],
                name_zh=CHINESE_NAMES[kw],
                pinyin=r["pinyin"],
                english=r["english"],
                upper_trigram=r["upper_trigram"],
                lower_trigram=r["lower_trigram"],
                bits=r["bits_L1_to_L6"],
                shao_yong=int(r["shao_yong_decimal"]),
                nuclear_bits=r["nuclear_bits_L1_to_L6"],
                opposite_bits=r["opposite_bits_L1_to_L6"],
                inverted_bits=r["inverted_bits_L1_to_L6"],
            )
        )
    if len({h.bits for h in out}) != 64:
        raise ValueError("hexagram bit strings are not unique")
    return sorted(out, key=lambda h: h.king_wen)
