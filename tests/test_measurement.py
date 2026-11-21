"""Tests for the measurement step: richer hexagram documents, the
100-scenario exam, the baseline scorer, and the harness on exam 100.

The embedding-model probes reuse the module-scoped real encoder from
the Phase 1 suite pattern: the 2-scenario harness test must run the
real pipeline, not the TF-IDF fallback.
"""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from taiji_field import grounding
from taiji_field.data import load_hexagrams
from taiji_field.documents import (
    PARAGRAPHS,
    READINGS,
    TRIGRAMS,
    hexagram_documents,
    trigram_info,
)
from taiji_field.dynamics import DEFAULT_WEIGHTS
from taiji_field.exam import load_exam, run_exam
from taiji_field.graph import flip_bit

ROOT = Path(__file__).resolve().parent.parent
EXAM_100_PATH = ROOT / "taiji_field" / "data" / "exam_100.json"
REPORT_V2_PATH = ROOT / "taiji_field" / "data" / "exam_report_v2.json"


def _load_baseline_module():
    spec = importlib.util.spec_from_file_location(
        "run_baseline", ROOT / "scripts" / "run_baseline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def encoder():
    """Real encoder (sentence-transformers). Fails loudly if the model
    cannot load — a silent TF-IDF fallback must never fake a pass here."""
    enc = grounding.GroundingEncoder()
    assert enc.backend.startswith("sentence-transformers"), (
        f"probe tests need the embedding model, got backend {enc.backend!r}"
    )
    return enc


# --- richer documents -------------------------------------------------------

def test_rich_docs_cover_all_64():
    docs = hexagram_documents()
    assert len(docs) == 64
    assert {d["king_wen"] for d in docs} == set(range(1, 65))
    assert len(READINGS) == 64 and len(PARAGRAPHS) == 64
    # One-line readings are unchanged (the baseline prompt used them).
    assert READINGS[1].startswith("创造与主导")
    assert READINGS[64].startswith("未成定局")


def test_every_hexagram_has_paragraph():
    for kw in range(1, 65):
        p = PARAGRAPHS[kw]
        assert isinstance(p, str) and len(p) >= 30, f"hexagram {kw} paragraph too short"
        assert "动向" in p, f"hexagram {kw} paragraph must state what it moves toward"


def test_trigrams_cover_all_eight():
    assert set(TRIGRAMS) == {"Qian", "Kun", "Zhen", "Kan", "Gen", "Xun", "Li", "Dui"}
    for name, (zh, meaning) in TRIGRAMS.items():
        assert len(zh) == 1 and len(meaning) >= 10, name
    assert trigram_info("Qian") == ("乾", "天，创造与主导：主动开创的力量。")
    assert trigram_info("Kun")[0] == "坤"


def test_every_hexagram_has_trigram_info():
    by_kw = {h.king_wen: h for h in load_hexagrams()}
    for doc in hexagram_documents():
        h = by_kw[doc["king_wen"]]
        assert doc["upper_trigram"] == h.upper_trigram
        assert doc["lower_trigram"] == h.lower_trigram
        for key in ("upper_trigram_zh", "lower_trigram_zh",
                    "upper_trigram_meaning", "lower_trigram_meaning"):
            assert doc[key] and isinstance(doc[key], str)
        # The embedded text carries the paragraph and both trigrams.
        assert doc["paragraph"] in doc["text"]
        assert doc["upper_trigram_zh"] in doc["text"]
        assert doc["lower_trigram_zh"] in doc["text"]


# --- exam 100 consistency ---------------------------------------------------

def test_exam_100_structure():
    data = load_exam(EXAM_100_PATH)
    scenarios = data["scenarios"]
    assert len(scenarios) == 100
    counts = Counter(s["category"] for s in scenarios)
    assert counts == {c: 20 for c in ["工作决策", "谈判", "关系", "创意", "日常"]}
    assert len({s["id"] for s in scenarios}) == 100


def test_exam_100_end_hexagram_computed_from_start_and_lines():
    data = load_exam(EXAM_100_PATH)
    by_bits = {h.bits: h for h in load_hexagrams()}
    for s in data["scenarios"]:
        p = s["proposed"]
        bits = p["start_bits"]
        for line in p["moving_lines"]:
            assert 1 <= line <= 6
            bits = flip_bit(bits, line)
        assert by_bits[bits].king_wen == p["end_king_wen"], s["id"]
        assert by_bits[bits].bits == p["end_bits"]
        assert 1 <= len(p["moving_lines"]) <= 3
        assert len(set(p["moving_lines"])) == len(p["moving_lines"])


def test_exam_100_passes_and_agreement_consistent():
    data = load_exam(EXAM_100_PATH)
    assert "NOT 3-human" in data["note"]
    assert data["annotation"]["pass_C"].startswith("pending-human-review")
    n_start = n_full = 0
    for s in data["scenarios"]:
        a, b = s["pass_a"], s["pass_b"]
        assert s["agreement"]["start_agree"] == (a["start_king_wen"] == b["start_king_wen"])
        assert s["agreement"]["full_agree"] == (
            a["start_king_wen"] == b["start_king_wen"]
            and a["moving_lines"] == b["moving_lines"]
        )
        assert s["flagged"] == (not s["agreement"]["full_agree"])
        # With pass C pending, a disagreement has no majority: the
        # tie-break takes pass A, so the final label always equals A.
        assert s["proposed"]["start_king_wen"] == a["start_king_wen"]
        assert s["proposed"]["moving_lines"] == a["moving_lines"]
        n_start += s["agreement"]["start_agree"]
        n_full += s["agreement"]["full_agree"]
    assert data["agreement_summary"]["start_agree"] == n_start
    assert data["agreement_summary"]["full_agree"] == n_full
    assert data["agreement_summary"]["flagged"] == 100 - n_full


# --- baseline scorer: same definitions as the harness (regression) ----------

def test_baseline_scorer_matches_harness_definitions():
    baseline = _load_baseline_module()
    proposed = {"start_king_wen": 12, "moving_lines": [5]}
    # Start + lines both right.
    assert baseline.score_prediction(
        {"start_king_wen": 12, "moving_lines": [5]}, proposed
    ) == {"top1_match": True, "path_match": True}
    # Right start, wrong lines: top-1 only.
    assert baseline.score_prediction(
        {"start_king_wen": 12, "moving_lines": [1]}, proposed
    ) == {"top1_match": True, "path_match": False}
    # REGRESSION (Phase 2 bug, see docs/phase2-notes.md): right lines on
    # a WRONG start must NOT count as a path match. An earlier fit
    # counted lines only and inflated the path count by 3/30.
    assert baseline.score_prediction(
        {"start_king_wen": 35, "moving_lines": [5]}, proposed
    ) == {"top1_match": False, "path_match": False}
    # Unparsed prediction scores zero on both.
    assert baseline.score_prediction(None, proposed) == {
        "top1_match": False, "path_match": False}


def test_baseline_end_hexagram_computed_not_trusted():
    baseline = _load_baseline_module()
    # 12 否, moving line 5 -> 35 晋 (same as exam E03).
    assert baseline.compute_end(12, [5])["end_king_wen"] == 35
    assert baseline.compute_end(1, [1])["end_king_wen"] == 44
    # The prompt hands the model exactly the one-line readings.
    prompt = baseline.build_prompt("测试情境")
    assert "测试情境" in prompt
    assert "乾：创造与主导" in prompt
    assert "未济：未成定局" in prompt
    assert "上卦" not in prompt and "动向" not in prompt, (
        "baseline prompt must not contain the richer trigram/paragraph text"
    )


# --- harness runs end-to-end on a 2-scenario subset of the 100 -------------

def _v2_weights_for_test() -> dict:
    if REPORT_V2_PATH.exists():
        report = json.loads(REPORT_V2_PATH.read_text(encoding="utf-8"))
        return report["fitted_weights"]["weights"]
    return dict(DEFAULT_WEIGHTS)


def test_harness_runs_on_two_scenarios_of_exam_100(encoder):
    scenarios = load_exam(EXAM_100_PATH)["scenarios"][:2]
    report = run_exam(encoder, scenarios, reasoner="v2",
                      weights=_v2_weights_for_test())
    assert report["summary"]["n"] == 2
    assert len(report["results"]) == 2
    assert report["baseline"] is None
    for r in report["results"]:
        assert r["id"] in ("F01", "F02")
        assert isinstance(r["top1_match"], bool)
        assert isinstance(r["top3_hit"], bool)
        assert isinstance(r["path_match"], bool)
        assert r["predicted_top1_king_wen"] in range(1, 65)
        assert r["predicted_path"]["moving_lines"]
