"""Tests for the Pass C adjudicated label set (100 scenarios).

Pass C is a careful Muse adjudication (see docs/label-review-pass-c.md),
NOT independent human review. These tests check structural validity only:
100 entries, consistent IDs, and every end hexagram equal to start + lines.
"""

import json
from pathlib import Path

from taiji_field.data import load_hexagrams

PASS_C_PATH = Path(__file__).resolve().parent.parent / "taiji_field" / "data" / "exam_100_pass_c.json"


def load_pass_c():
    return json.loads(PASS_C_PATH.read_text(encoding="utf-8"))


def bits_maps():
    hexagrams = load_hexagrams()
    bits_by_number = {h.king_wen: h.bits for h in hexagrams}
    return bits_by_number, {bits: num for num, bits in bits_by_number.items()}


def flip_bits(bits: str, lines) -> str:
    chars = list(bits)
    for line in lines:
        chars[line - 1] = "0" if chars[line - 1] == "1" else "1"
    return "".join(chars)


def test_pass_c_has_100_entries_with_expected_ids():
    entries = load_pass_c()
    assert len(entries) == 100
    assert [e["id"] for e in sorted(entries, key=lambda e: int(e["id"][1:]))] == [
        f"F{i:02d}" for i in range(1, 101)
    ]


def test_pass_c_entries_are_structurally_valid():
    for entry in load_pass_c():
        start = entry["reviewed_start_king_wen"]
        end = entry["reviewed_end_king_wen"]
        lines = entry["reviewed_moving_lines"]
        assert 1 <= start <= 64
        assert 1 <= end <= 64
        assert 1 <= len(lines) <= 3
        assert all(isinstance(line, int) and 1 <= line <= 6 for line in lines)
        assert entry["source"] in ("A", "B", "system", "baseline", "custom")
        assert isinstance(entry["note"], str) and entry["note"]


def test_pass_c_end_equals_start_plus_lines():
    bits_by_number, number_by_bits = bits_maps()
    for entry in load_pass_c():
        start_bits = bits_by_number[entry["reviewed_start_king_wen"]]
        expected_end = number_by_bits[flip_bits(start_bits, entry["reviewed_moving_lines"])]
        assert entry["reviewed_end_king_wen"] == expected_end, entry["id"]
