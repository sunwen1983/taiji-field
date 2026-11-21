"""Tests for Phase 2: VSA vectorization, continuous lines, learned
dynamics, and the exam v2 harness.

The VSA tests assert the structural properties the design promises:
cosine between two hexagram vectors is ~ ``1 - hamming/6`` (Hamming-1
neighbours most similar, opposite/cuo pairs least similar), and the
zong relation is consistent with the same formula. Tolerances are loose
on purpose — D=10,000 random vectors have small but nonzero noise.
"""

import numpy as np
import pytest

from taiji_field.continuous import interpolate, nearest_hexagram, soft_vector
from taiji_field.data import load_hexagrams
from taiji_field.dynamics import (
    DEFAULT_WEIGHTS,
    fit_weights,
    leave_out_check,
    leave_out_split,
    reason_v2,
    score_features,
)
from taiji_field.exam import load_exam, run_exam_v2
from taiji_field.graph import cuo_bits, hamming, zong_bits
from taiji_field.grounding import GroundingEncoder
from taiji_field.vsa import (
    DIMENSION,
    cosine,
    hexagram_matrix,
    hexagram_vector,
    trigram_vector,
)

HEXAGRAMS = load_hexagrams()
BY_BITS = {h.bits: h for h in HEXAGRAMS}
VALID_KW = set(range(1, 65))


@pytest.fixture(scope="module")
def tfidf_encoder():
    return GroundingEncoder(backend="tfidf")


# --- VSA -------------------------------------------------------------------

def test_hexagram_vector_shape_and_self_similarity():
    v = hexagram_vector("111111")
    assert v.shape == (DIMENSION,)
    assert np.all(np.isfinite(v))
    assert cosine(v, v) == pytest.approx(1.0, abs=1e-6)
    assert cosine(hexagram_vector("000000"), hexagram_vector("000000")) == pytest.approx(1.0, abs=1e-6)


def test_hexagram_matrix_shape():
    m = hexagram_matrix()
    assert m.shape == (64, DIMENSION)
    assert np.all(np.isfinite(m))


def test_hamming_distance_orders_similarity():
    """cos ~ 1 - hamming/6: Hamming-1 > Hamming-3 > Hamming-6."""
    m = hexagram_matrix()
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    sims = (m @ m.T) / (norms @ norms.T)
    by_d: dict[int, list[float]] = {}
    for i, hi in enumerate(HEXAGRAMS):
        for j, hj in enumerate(HEXAGRAMS):
            if j <= i:
                continue
            by_d.setdefault(hamming(hi.bits, hj.bits), []).append(float(sims[i, j]))
    assert set(by_d) == {1, 2, 3, 4, 5, 6}
    means = {d: float(np.mean(v)) for d, v in by_d.items()}
    assert means[1] > means[3] + 0.15
    assert means[3] > means[6] + 0.15
    assert means[1] > 0.6  # ~0.83 by construction
    assert means[6] < 0.2  # ~0 by construction


def test_cuo_pairs_are_least_similar():
    m = hexagram_matrix()
    idx = {h.bits: i for i, h in enumerate(HEXAGRAMS)}
    cuo_sims = []
    for h in HEXAGRAMS:
        if h.bits < cuo_bits(h.bits):  # each pair once
            cuo_sims.append(float(cosine(m[idx[h.bits]], m[idx[cuo_bits(h.bits)]])))
    assert len(cuo_sims) == 32
    assert float(np.mean(cuo_sims)) < 0.1
    # cuo (Hamming-6) is the minimum over all Hamming distances
    d5 = [
        float(cosine(m[idx[a.bits]], m[idx[b.bits]]))
        for a in HEXAGRAMS for b in HEXAGRAMS
        if hamming(a.bits, b.bits) == 5
    ]
    assert float(np.mean(cuo_sims)) < float(np.mean(d5))


def test_zong_consistent_with_hamming_formula():
    """cos(v, v_zong) ~ 1 - hamming(bits, zong_bits)/6 for all 64.

    The 8 self-symmetric hexagrams (zong == itself) have hamming 0,
    hence cosine 1 — the formula covers them without special-casing.
    """
    m = hexagram_matrix()
    idx = {h.bits: i for i, h in enumerate(HEXAGRAMS)}
    for h in HEXAGRAMS:
        zb = zong_bits(h.bits)
        d = hamming(h.bits, zb)
        expected = 1.0 - d / 6.0
        got = float(cosine(m[idx[h.bits]], m[idx[zb]]))
        assert abs(got - expected) < 0.05, (h.king_wen, d, expected, got)


def test_hamming1_similarity_is_near_constant():
    """Every Hamming-1 step has ~the same VSA cosine — so VSA cosine
    cannot rank *which* line to move. This is why dynamics.py measures
    path coherence in the grounding (document) space instead."""
    m = hexagram_matrix()
    idx = {h.bits: i for i, h in enumerate(HEXAGRAMS)}
    vals = []
    for h in HEXAGRAMS:
        for line in range(1, 7):
            bits = list(h.bits)
            bits[line - 1] = "0" if bits[line - 1] == "1" else "1"
            nb = "".join(bits)
            if h.bits < nb:
                vals.append(float(cosine(m[idx[h.bits]], m[idx[nb]])))
    assert len(vals) == 192
    assert float(np.std(vals)) < 0.05
    assert 0.7 < float(np.mean(vals)) < 0.95


def test_trigram_vector_shares_lines_with_hexagram():
    tri = trigram_vector("111", offset=0)
    assert tri.shape == (DIMENSION,)
    # Lower trigram of Qian shares 3 of 6 bound pairs -> cos ~ 3/sqrt(18).
    assert float(cosine(tri, hexagram_vector("111111"))) > 0.6
    # Same bits at a different offset use different position vectors.
    assert float(cosine(tri, trigram_vector("111", offset=3))) < 0.5


# --- continuous lines ------------------------------------------------------

def test_soft_vector_recovers_binary_hexagrams():
    for bits in ("111111", "000000", "101010"):
        thetas = [float(c) for c in bits]
        assert float(cosine(soft_vector(thetas), hexagram_vector(bits))) == pytest.approx(
            1.0, abs=1e-6
        )


def test_interpolation_endpoints_recover_hexagrams():
    steps = interpolate("111111", "000000", steps=10)
    assert len(steps) == 11
    assert steps[0]["nearest_king_wen"] == 1
    assert steps[-1]["nearest_king_wen"] == 2
    assert steps[0]["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert steps[-1]["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert all(s["nearest_king_wen"] in VALID_KW for s in steps)


def test_midpoint_of_qian_and_kun_is_equidistant():
    mid = soft_vector([0.5] * 6)
    cq = float(cosine(mid, hexagram_vector("111111")))
    ck = float(cosine(mid, hexagram_vector("000000")))
    assert abs(cq - ck) < 0.05
    assert cq > 0.5 and ck > 0.5


def test_nearest_hexagram_returns_valid_hexagram():
    vec = soft_vector([0.9, 0.1, 0.5, 0.5, 0.2, 0.8])
    h, score = nearest_hexagram(vec)
    assert h.king_wen in VALID_KW
    assert -1.0 <= score <= 1.0


# --- dynamics --------------------------------------------------------------

def test_score_features_arithmetic():
    f = {"endpoint_resonance": 0.6, "coherence": 0.5, "nuclear_hit": 1.0, "steps": 2}
    w = {"w1": 1.0, "w2": 1.0, "w3": 0.5, "w4": 0.1}
    assert score_features(f, w) == pytest.approx(0.6 + 0.5 + 0.5 - 0.2)
    assert score_features(f, {"w1": 0.0, "w2": 0.0, "w3": 0.0, "w4": 0.0}) == 0.0


def test_fit_weights_runs_on_4_scenarios(tfidf_encoder):
    scenarios = load_exam()["scenarios"][:4]
    fitted = fit_weights(scenarios, tfidf_encoder)
    assert set(fitted["weights"]) == {"w1", "w2", "w3", "w4"}
    assert fitted["n"] == 4
    assert 0 <= fitted["path_matches"] <= 4
    assert 0 <= fitted["end_matches"] <= 4
    # Deterministic: same input, same fit.
    assert fit_weights(scenarios, tfidf_encoder)["weights"] == fitted["weights"]


def test_leave_out_split_is_stratified():
    scenarios = load_exam()["scenarios"]
    a, b = leave_out_split(scenarios)
    assert len(a) == 15 and len(b) == 15
    assert {s["id"] for s in a} | {s["id"] for s in b} == {s["id"] for s in scenarios}
    assert not ({s["id"] for s in a} & {s["id"] for s in b})
    for half in (a, b):
        cats = [s["category"] for s in half]
        assert len(cats) == 15
        assert all(cats.count(c) == 3 for c in set(cats))


def test_leave_out_check_runs_on_10_scenarios(tfidf_encoder):
    scenarios = []
    by_cat: dict[str, list] = {}
    for s in load_exam()["scenarios"]:
        by_cat.setdefault(s["category"], []).append(s)
    for cat in sorted(by_cat):
        scenarios.extend(by_cat[cat][:2])
    assert len(scenarios) == 10  # 5 categories x 2
    loo = leave_out_check(tfidf_encoder, scenarios)
    for key in ("fit_A_test_B", "fit_B_test_A"):
        assert key in loo
        assert 0 <= loo[key]["path_matches"] <= 5
        assert loo[key]["n"] == 5
        assert set(loo[key]["weights"]) == {"w1", "w2", "w3", "w4"}


def test_reason_v2_returns_valid_path(tfidf_encoder):
    result = reason_v2(
        "两个部门为项目归属僵持了三周，谁也不让步",
        tfidf_encoder,
        weights=DEFAULT_WEIGHTS,
    )
    assert result.start_king_wen in VALID_KW
    assert result.end_king_wen in VALID_KW
    assert 1 <= len(result.steps) <= 3
    current = result.start_bits
    for step in result.steps:
        assert step.to_bits not in {current}  # moves somewhere
        assert hamming(step.from_bits, step.to_bits) == 1
        current = step.to_bits
    assert current == result.end_bits


# --- exam v2 harness -------------------------------------------------------

def test_exam_v2_harness_runs_on_2_scenarios(tfidf_encoder):
    scenarios = load_exam()["scenarios"][:2]
    report = run_exam_v2(tfidf_encoder, scenarios)
    assert report["exam_version"] == "v2"
    assert report["baseline"] is None
    assert set(report["fitted_weights"]["weights"]) == {"w1", "w2", "w3", "w4"}
    assert set(report["leave_out"]) == {"fit_A_test_B", "fit_B_test_A"}
    assert report["summary"]["n"] == 2
    assert len(report["results"]) == 2
    for r in report["results"]:
        assert r["predicted_top1_king_wen"] in VALID_KW
        assert isinstance(r["top1_match"], bool)
        assert isinstance(r["top3_hit"], bool)
        assert isinstance(r["path_match"], bool)
        assert 1 <= len(r["predicted_path"]["moving_lines"]) <= 3
    s = report["summary"]
    assert s["top1_matches"] == sum(r["top1_match"] for r in report["results"])
    assert s["path_matches"] == sum(r["path_match"] for r in report["results"])
    # The fitted count must use the same path-match definition as the
    # harness (right start AND right lines) — a fit that counts lines
    # alone inflates the number on wrong starts.
    assert report["fitted_weights"]["path_matches"] == s["path_matches"]
