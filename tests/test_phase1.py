"""Tests for Phase 1: grounding encoder, resonance reasoning, interpreter,
exam paper, and exam harness.

The embedding-model probes are deliberately LOOSE (top-5, not top-1):
they check that resonance points in the right direction, not that a
heuristic encoder is precise. The exam harness is the real measurement,
and its labels are drafts.
"""

from pathlib import Path

import numpy as np
import pytest

from taiji_field import grounding
from taiji_field.data import load_hexagrams
from taiji_field.documents import READINGS, hexagram_documents
from taiji_field.exam import load_exam, run_exam
from taiji_field.graph import flip_bit, hamming
from taiji_field.interpreter import explain
from taiji_field.reasoning import best_path, enumerate_paths, reason

VALID_BITS = {h.bits for h in load_hexagrams()}
VALID_KW = set(range(1, 65))


@pytest.fixture(scope="module")
def encoder():
    """Real encoder (sentence-transformers). Fails loudly if the model
    cannot load — a silent TF-IDF fallback must never fake a pass here."""
    enc = grounding.GroundingEncoder()
    assert enc.backend.startswith("sentence-transformers"), (
        f"probe tests need the embedding model, got backend {enc.backend!r}"
    )
    return enc


@pytest.fixture(scope="module")
def tfidf_encoder():
    return grounding.TfidfEncoder()


# --- documents ------------------------------------------------------------

def test_documents_cover_all_64():
    docs = hexagram_documents()
    assert len(docs) == 64
    assert {d["king_wen"] for d in docs} == VALID_KW
    assert {d["bits"] for d in docs} == VALID_BITS
    assert len({d["text"] for d in docs}) == 64
    assert all(d["text"].strip() for d in docs)
    assert len(READINGS) == 64


# --- encoder invariants (both backends) -----------------------------------

def _check_ranking(ranked):
    assert len(ranked) == 64
    assert {r["king_wen"] for r in ranked} == VALID_KW
    assert {r["bits"] for r in ranked} == VALID_BITS
    scores = [r["score"] for r in ranked]
    assert all(np.isfinite(s) and -1.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert len({r["king_wen"] for r in ranked[:3]}) == 3


def test_tfidf_encoder_returns_64_ranked_scores(tfidf_encoder):
    _check_ranking(tfidf_encoder.similarities("谈判僵持住了，谁也不让步"))
    top3 = tfidf_encoder.match("谈判僵持住了，谁也不让步", top_k=3)
    assert len(top3) == 3
    assert top3 == tfidf_encoder.similarities("谈判僵持住了，谁也不让步")[:3]


def test_encoder_returns_64_ranked_scores(encoder):
    ranked = encoder.similarities("谈判僵持住了，谁也不让步，卡住了")
    _check_ranking(ranked)
    top3 = encoder.match("谈判僵持住了，谁也不让步，卡住了", top_k=3)
    assert len(top3) == 3
    assert [m["king_wen"] for m in top3] == [m["king_wen"] for m in ranked[:3]]


def test_encoder_rejects_empty_text(encoder):
    with pytest.raises(ValueError):
        encoder.similarities("   ")


def test_doc_embeddings_shape_matches_documents(encoder):
    emb = encoder.doc_embeddings()
    assert emb.shape[0] == 64
    assert emb.shape[1] > 0
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# --- loose resonance probes (real model only) ------------------------------
# Each probe sentence is a plain paraphrase of the situation pattern, not
# a copy of the document text. Top-5, not top-1: resonance is heuristic.

PROBES = [
    ("我想自己创业，带头做一件全新的事，主动开创局面", 1),   # 乾
    ("双方谈判僵持住了，谁也不让步，上下不通，卡住了", 12),  # 否
    ("深夜还在刷题，一坑接一坑，越陷越深", 29),              # 坎
    ("经历一段低谷之后，终于开始重新复苏", 24),              # 复
    ("时机还没到，只能耐心等待，养精蓄锐", 5),               # 需
]


@pytest.mark.parametrize("sentence,king_wen", PROBES)
def test_resonance_probe_ranks_expected_hexagram_top5(encoder, sentence, king_wen):
    top5 = [m["king_wen"] for m in encoder.match(sentence, top_k=5)]
    assert king_wen in top5, f"expected #{king_wen} in top-5, got {top5}"


# --- path expansion --------------------------------------------------------

def test_enumerate_paths_terminates_within_3_steps():
    paths = enumerate_paths("111111", max_steps=3)
    assert 0 < len(paths) <= 6 + 6 * 5 + 6 * 5 * 5  # == 186 simple paths
    assert all(1 <= len(p) <= 3 for p in paths)


def test_every_path_step_is_a_valid_bian_edge():
    for start in ("111111", "000000", "010010", "101010"):
        for path in enumerate_paths(start, max_steps=3):
            visited = {start}
            current = start
            for step in path:
                assert step.from_bits == current
                assert hamming(step.from_bits, step.to_bits) == 1
                assert 1 <= step.line <= 6
                assert flip_bit(step.from_bits, step.line) == step.to_bits
                assert step.to_bits not in visited
                visited.add(step.to_bits)
                current = step.to_bits


def test_best_path_is_valid_and_scored(encoder):
    vec = encoder.encode(["团队连续加班，担子太重，快撑不住了"])[0]
    steps, path_score, endpoint_score = best_path(
        "011110", vec, encoder.doc_embeddings(), encoder.documents
    )
    assert 1 <= len(steps) <= 3
    assert np.isfinite(path_score) and np.isfinite(endpoint_score)
    assert -1.0 <= endpoint_score <= 1.0
    current = "011110"
    for step in steps:
        assert flip_bit(current, step.line) == step.to_bits
        current = step.to_bits


# --- full pipeline + interpreter -------------------------------------------

def test_reason_end_to_end(encoder):
    result = reason("两个部门为项目归属僵持了三周，谁也不让步", encoder)
    assert result.start_king_wen in VALID_KW
    assert result.end_king_wen in VALID_KW
    assert 1 <= len(result.steps) <= 3
    assert len(result.top_matches) == 3
    current = result.start_bits
    for step in result.steps:
        assert flip_bit(current, step.line) == step.to_bits
        current = step.to_bits
    assert current == result.end_bits


def test_interpreter_mentions_start_and_end(encoder):
    result = reason("两个部门为项目归属僵持了三周，谁也不让步", encoder)
    text = explain(result)
    assert result.top_matches[0]["name_zh"] in text
    assert "落点" in text
    assert "动第" in text


# --- exam paper ------------------------------------------------------------

def test_exam_has_30_scenarios_with_consistent_labels():
    exam = load_exam()
    scenarios = exam["scenarios"]
    assert len(scenarios) == 30
    assert len({s["id"] for s in scenarios}) == 30
    assert all(s["label_status"].startswith("draft") for s in scenarios)
    categories = {s["category"] for s in scenarios}
    assert categories == {"工作决策", "谈判", "关系", "创意", "日常"}
    for s in scenarios:
        p = s["proposed"]
        assert p["start_king_wen"] in VALID_KW
        assert p["end_king_wen"] in VALID_KW
        assert 1 <= len(p["moving_lines"]) <= 3
        bits = p["start_bits"]
        for line in p["moving_lines"]:
            bits = flip_bit(bits, line)
        assert bits == p["end_bits"], s["id"]


def test_exam_harness_runs_end_to_end_on_2_scenarios(encoder):
    scenarios = load_exam()["scenarios"][:2]
    report = run_exam(encoder, scenarios)
    assert report["exam_version"] == "v1"
    assert report["baseline"] is None  # LLM baseline deliberately not built
    assert report["summary"]["n"] == 2
    assert len(report["results"]) == 2
    for r in report["results"]:
        assert r["predicted_top1_king_wen"] in VALID_KW
        assert len(r["predicted_top3"]) == 3
        assert isinstance(r["top1_match"], bool)
        assert isinstance(r["top3_hit"], bool)
        assert isinstance(r["path_match"], bool)
        assert 1 <= len(r["predicted_path"]["moving_lines"]) <= 3
    s = report["summary"]
    assert s["top1_matches"] == sum(r["top1_match"] for r in report["results"])
    assert s["top3_hits"] == sum(r["top3_hit"] for r in report["results"])


# --- embedding cache -------------------------------------------------------

def test_embeddings_cache_file_exists_and_loads():
    cache = Path(__file__).resolve().parent.parent / "taiji_field" / "data" / "embeddings.npz"
    assert cache.exists(), "run scripts/build_embeddings.py first"
    data = np.load(cache, allow_pickle=False)
    assert data["embeddings"].shape == (64, data["embeddings"].shape[1])
    assert str(data["model"]) in (
        grounding.DEFAULT_MODEL, grounding.FALLBACK_MODEL)
