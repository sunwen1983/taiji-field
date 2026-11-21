"""Learned dynamics, v1 (Phase 2): a parameterized path scorer.

Phase 1's finding: resonance finds "which hexagram is this situation"
half-decently (15/30 top-1), but "which line should move" is almost
always wrong (2/30 path match) — the v1 ranking only looked at the
endpoint, so it had no direction information. This module tests whether
a *learned direction* helps, or whether the data is too thin. Either
outcome is reported honestly (see docs/phase2-notes.md).

Scorer (4 weights, fitted by coarse grid search on the 30 exam
scenarios — "tuned", not "learned", at this sample size):

    score(path) = w1 * endpoint_resonance
                + w2 * path_coherence
                + w3 * nuclear_attraction
                - w4 * steps

- ``endpoint_resonance``: cosine between the endpoint hexagram's
  grounding document and the situation (the Phase 1 term).
- ``path_coherence``: mean cosine between *consecutive states* on the
  path, measured in the grounding (document) space — i.e. does the
  path move through semantically related states? It is deliberately
  NOT measured in VSA space: :mod:`taiji_field.vsa` shows every
  Hamming-1 step has nearly the same VSA cosine (~0.83, std < 0.05),
  so a VSA coherence term would be near-constant and could not rank
  anything. Coherence includes the start -> first-step pair.
- ``nuclear_attraction``: 1.0 if the path passes through the start
  hexagram's nuclear (hu) hexagram at any step after the start, else
  0.0. The nuclear hexagram is the "inner core" of a situation in the
  classical reading; the hypothesis is that evolutions pass through
  the core. Corner case: for Qian (#1) and Kun (#2) the nuclear
  hexagram is the hexagram itself, so no simple path can pass through
  it after the start and the term never fires for them.
- ``steps``: the path length (1..3), penalized as in Phase 1.

Fitting: coarse grid search maximizing path matches on the exam, with
ties broken by (top-1 matches, end-hexagram matches). Top-1 is fixed
by the encoder and does not depend on the weights — it is included in
the tie-break as specified, but it never actually breaks a tie; the
end-match count does the work. The honest test is the leave-out check
(:func:`leave_out_check`): fit on half the exam, test on the other
half, both directions. With 30 single-author draft labels, any
in-sample improvement that does not survive leave-out is overfitting,
and is reported as such.

No LLM calls anywhere here — the scorer is arithmetic over cosine
similarities, and the weights are tuned on 30 draft labels.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from taiji_field.graph import hu_bits
from taiji_field.grounding import GroundingEncoder, cosine_similarity
from taiji_field.reasoning import ReasoningResult, _lookups, enumerate_paths

# Coarse grid for the 4 weights. 3 * 4 * 3 * 4 = 144 combinations.
WEIGHT_GRID: dict[str, list[float]] = {
    "w1": [0.5, 1.0, 2.0],
    "w2": [0.0, 0.5, 1.0, 2.0],
    "w3": [0.0, 0.5, 1.0],
    "w4": [0.0, 0.05, 0.1, 0.2],
}

# Phase-1-equivalent weights: endpoint only, 0.05 per-step penalty.
DEFAULT_WEIGHTS: dict[str, float] = {"w1": 1.0, "w2": 0.0, "w3": 0.0, "w4": 0.05}

_FEATURE_KEYS = ("endpoint_resonance", "coherence", "nuclear_hit", "steps")


@dataclass
class PathFeatures:
    endpoint_resonance: float
    coherence: float
    nuclear_hit: float  # 0.0 or 1.0
    steps: int

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.endpoint_resonance,
                self.coherence,
                self.nuclear_hit,
                float(self.steps),
            ],
            dtype=np.float64,
        )


def score_features(features: dict | PathFeatures, weights: dict[str, float]) -> float:
    """score = w1*endpoint + w2*coherence + w3*nuclear - w4*steps."""
    if isinstance(features, PathFeatures):
        f = {
            "endpoint_resonance": features.endpoint_resonance,
            "coherence": features.coherence,
            "nuclear_hit": features.nuclear_hit,
            "steps": features.steps,
        }
    else:
        f = features
    return (
        weights["w1"] * f["endpoint_resonance"]
        + weights["w2"] * f["coherence"]
        + weights["w3"] * f["nuclear_hit"]
        - weights["w4"] * f["steps"]
    )


def _doc_similarity_matrix(doc_embeddings: np.ndarray) -> np.ndarray:
    """64x64 cosine matrix between hexagram document vectors."""
    emb = np.asarray(doc_embeddings, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    n = emb / np.where(norms == 0, 1.0, norms)
    return n @ n.T


def extract_features(
    start_bits: str,
    situation_vec: np.ndarray,
    doc_embeddings: np.ndarray,
    documents: list[dict],
) -> list[tuple[list, PathFeatures]]:
    """Features for every simple 1..3-step bian path from ``start_bits``.

    Returns a list of ``(path, features)`` in the same order as
    :func:`taiji_field.reasoning.enumerate_paths`.
    """
    doc_by_bits = {d["bits"]: i for i, d in enumerate(documents)}
    doc_sim = _doc_similarity_matrix(doc_embeddings)
    nuclear = hu_bits(start_bits)
    out = []
    for path in enumerate_paths(start_bits, max_steps=3):
        end_bits = path[-1].to_bits
        endpoint = float(
            cosine_similarity(
                doc_embeddings[doc_by_bits[end_bits]: doc_by_bits[end_bits] + 1],
                situation_vec,
            )[0]
        )
        states = [start_bits] + [s.to_bits for s in path]
        coh = float(
            np.mean(
                [
                    doc_sim[doc_by_bits[a], doc_by_bits[b]]
                    for a, b in zip(states, states[1:])
                ]
            )
        )
        hit = 1.0 if nuclear in states[1:] else 0.0
        out.append(
            (
                path,
                PathFeatures(
                    endpoint_resonance=endpoint,
                    coherence=coh,
                    nuclear_hit=hit,
                    steps=len(path),
                ),
            )
        )
    return out


def _prepare_records(scenarios: list[dict], encoder: GroundingEncoder) -> list[dict]:
    """Encode each scenario once and extract path features.

    A record is ``{"scenario", "start_bits", "start_king_wen",
    "features": [(path, PathFeatures), ...]}``.
    """
    doc_embeddings = encoder.doc_embeddings()
    documents = encoder.documents
    records = []
    for s in scenarios:
        ranked = encoder.similarities(s["scenario"])
        top = ranked[0]
        situation_vec = encoder.encode([s["scenario"]])[0]
        records.append(
            {
                "scenario": s,
                "start_bits": top["bits"],
                "start_king_wen": top["king_wen"],
                "features": extract_features(
                    top["bits"], situation_vec, doc_embeddings, documents
                ),
            }
        )
    return records


def _counts_for_weights(records: list[dict], weights: dict[str, float]) -> dict:
    """Path / end / top-1 matches for one weight setting."""
    wvec = np.array(
        [weights["w1"], weights["w2"], weights["w3"], -weights["w4"]],
        dtype=np.float64,
    )
    path_matches = end_matches = top1_matches = 0
    for rec in records:
        feats = np.stack([f.as_array() for _, f in rec["features"]])
        scores = feats @ wvec
        best_path = rec["features"][int(np.argmax(scores))][0]
        proposed = rec["scenario"]["proposed"]
        pred_lines = [step.line for step in best_path]
        # Same definition as the exam harness (exam.py): a path match
        # needs the right start hexagram AND the right moving lines.
        # (An earlier version counted lines only, which inflated the
        # fitted count whenever the lines matched on a wrong start.)
        start_ok = rec["start_king_wen"] == proposed["start_king_wen"]
        if start_ok and pred_lines == proposed["moving_lines"]:
            path_matches += 1
        if start_ok and best_path[-1].to_king_wen == proposed["end_king_wen"]:
            end_matches += 1
        if start_ok:
            top1_matches += 1
    return {
        "path_matches": path_matches,
        "end_matches": end_matches,
        "top1_matches": top1_matches,
        "n": len(records),
    }


def _grid_combinations(grid: dict[str, list[float]] | None = None):
    grid = grid or WEIGHT_GRID
    for w1, w2, w3, w4 in itertools.product(
        grid["w1"], grid["w2"], grid["w3"], grid["w4"]
    ):
        yield {"w1": w1, "w2": w2, "w3": w3, "w4": w4}


def _fit_on_records(records: list[dict], grid: dict[str, list[float]] | None = None) -> dict:
    """Grid-search the weights on precomputed records.

    Objective: maximize path matches; ties broken by (top-1 matches,
    end matches). Top-1 does not depend on the weights, so in practice
    the end-match count breaks ties.
    """
    best_weights: dict[str, float] | None = None
    best_counts: dict | None = None
    best_key = None
    for weights in _grid_combinations(grid):
        counts = _counts_for_weights(records, weights)
        key = (counts["path_matches"], counts["top1_matches"], counts["end_matches"])
        if best_key is None or key > best_key:
            best_key = key
            best_weights = weights
            best_counts = counts
    assert best_weights is not None and best_counts is not None
    return {"weights": best_weights, **best_counts}


def fit_weights(
    scenarios: list[dict],
    encoder: GroundingEncoder,
    grid: dict[str, list[float]] | None = None,
) -> dict:
    """Fit the 4 weights on ``scenarios`` (in-sample, see module docstring)."""
    records = _prepare_records(scenarios, encoder)
    return _fit_on_records(records, grid)


def leave_out_split(scenarios: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stratified split: within each category (sorted by id), the first
    half goes to A, the rest to B. For the 30-scenario exam that is 3/3
    per category; both halves cover all categories."""
    by_cat: dict[str, list[dict]] = {}
    for s in scenarios:
        by_cat.setdefault(s["category"], []).append(s)
    a, b = [], []
    for cat in sorted(by_cat):
        group = sorted(by_cat[cat], key=lambda s: s["id"])
        half = len(group) // 2
        a.extend(group[:half])
        b.extend(group[half:])
    return a, b


def leave_out_check(encoder: GroundingEncoder, scenarios: list[dict]) -> dict:
    """Fit on half, test on the other half, both directions.

    Returns ``{"fit_A_test_B": {...}, "fit_B_test_A": {...}}`` where each
    entry holds the fitted weights and the match counts on the held-out
    half. These are the honest numbers; the full-fit numbers are
    in-sample and optimistic by construction.
    """
    a, b = leave_out_split(scenarios)
    records_a = _prepare_records(a, encoder)
    records_b = _prepare_records(b, encoder)
    fit_a = _fit_on_records(records_a)
    fit_b = _fit_on_records(records_b)
    test_b = _counts_for_weights(records_b, fit_a["weights"])
    test_a = _counts_for_weights(records_a, fit_b["weights"])
    return {
        "fit_A_test_B": {"weights": fit_a["weights"], **test_b},
        "fit_B_test_A": {"weights": fit_b["weights"], **test_a},
    }


def best_path_v2(
    start_bits: str,
    situation_vec: np.ndarray,
    doc_embeddings: np.ndarray,
    documents: list[dict],
    weights: dict[str, float] | None = None,
) -> tuple[list, float, float]:
    """Best path under the parameterized scorer.

    Returns ``(steps, path_score, endpoint_resonance)``.
    """
    weights = weights or DEFAULT_WEIGHTS
    best = None
    best_score = float("-inf")
    best_endpoint = 0.0
    for path, feats in extract_features(
        start_bits, situation_vec, doc_embeddings, documents
    ):
        score = score_features(feats, weights)
        if score > best_score:
            best, best_score, best_endpoint = path, score, feats.endpoint_resonance
    assert best is not None
    return best, best_score, best_endpoint


def reason_v2(
    situation: str,
    encoder: GroundingEncoder | None = None,
    weights: dict[str, float] | None = None,
    max_steps: int = 3,
) -> ReasoningResult:
    """Full v2 pipeline: encode -> match top-1 -> expand -> best path
    under the fitted scorer. Mirrors :func:`taiji_field.reasoning.reason`."""
    if max_steps != 3:
        raise ValueError("reason_v2 currently only supports max_steps=3")
    encoder = encoder or GroundingEncoder()
    ranked = encoder.similarities(situation)
    top = ranked[0]
    situation_vec = encoder.encode([situation])[0]
    steps, path_score, endpoint_score = best_path_v2(
        top["bits"], situation_vec, encoder.doc_embeddings(),
        encoder.documents, weights,
    )
    _, by_bits = _lookups()
    end = by_bits[steps[-1].to_bits]
    return ReasoningResult(
        situation=situation,
        start_king_wen=top["king_wen"],
        start_bits=top["bits"],
        match_score=top["score"],
        top_matches=ranked[:3],
        steps=steps,
        path_score=path_score,
        end_king_wen=end.king_wen,
        end_bits=end.bits,
        endpoint_score=endpoint_score,
    )
