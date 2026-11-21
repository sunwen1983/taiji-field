"""Resonance reasoning v1 (Phase 1): expand from the matched hexagram
along bian (变卦, flip-one-line) edges and pick a path.

Procedure: the situation is matched to its most resonant hexagram
(:mod:`taiji_field.grounding`); then every *simple* path of 1..``max_steps``
bian steps from that start is enumerated (at most 6 + 6*5 + 6*5*5 = 186
paths for 3 steps — trivially cheap) and ranked by

    score(path) = cosine(endpoint document, situation) - STEP_PENALTY * steps

i.e. the best reachable next state that stays close to the situation, with
a small penalty for longer detours. The empty (stay-put) path is NOT a
candidate: the question being asked is "how does this evolve?", and the
start is by construction the most resonant hexagram, so allowing it would
make the answer always "stay" — a degenerate ranking, not a finding.

Honest limits (see docs/phase1-notes.md):
- This is a heuristic ranking, not dynamics. There is no energy function
  yet (that is Phase 2), so "best" means "closest endpoint by cosine
  among 1-3 bian steps", nothing deeper.
- The per-step "why" is template-based (which line moved + the two
  hexagrams' one-line readings). The system does not understand *why* a
  line moves; the template only says *what* moved and what the two states
  mean. Do not mistake the explanation for a reason.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from taiji_field.data import load_hexagrams
from taiji_field.graph import flip_bit
from taiji_field.grounding import GroundingEncoder, cosine_similarity

STEP_PENALTY = 0.05
DEFAULT_MAX_STEPS = 3

_BY_KW: dict[int, object] | None = None
_BY_BITS: dict[str, object] | None = None


def _lookups():
    global _BY_KW, _BY_BITS
    if _BY_KW is None:
        hexagrams = load_hexagrams()
        _BY_KW = {h.king_wen: h for h in hexagrams}
        _BY_BITS = {h.bits: h for h in hexagrams}
    return _BY_KW, _BY_BITS


@dataclass
class PathStep:
    from_bits: str
    to_bits: str
    line: int  # 1..6, L1 = bottom

    @property
    def from_name(self) -> str:
        return _lookups()[1][self.from_bits].name_zh

    @property
    def to_name(self) -> str:
        return _lookups()[1][self.to_bits].name_zh

    @property
    def from_king_wen(self) -> int:
        return _lookups()[1][self.from_bits].king_wen

    @property
    def to_king_wen(self) -> int:
        return _lookups()[1][self.to_bits].king_wen


@dataclass
class ReasoningResult:
    situation: str
    start_king_wen: int
    start_bits: str
    match_score: float
    top_matches: list[dict]
    steps: list[PathStep] = field(default_factory=list)
    path_score: float = 0.0
    end_king_wen: int = 0
    end_bits: str = ""
    endpoint_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "situation": self.situation,
            "start_king_wen": self.start_king_wen,
            "start_bits": self.start_bits,
            "match_score": self.match_score,
            "top_matches": [
                {
                    "king_wen": m["king_wen"],
                    "name_zh": m["name_zh"],
                    "bits": m["bits"],
                    "score": m["score"],
                }
                for m in self.top_matches
            ],
            "steps": [
                {
                    "from_bits": s.from_bits,
                    "from_king_wen": s.from_king_wen,
                    "from_name": s.from_name,
                    "to_bits": s.to_bits,
                    "to_king_wen": s.to_king_wen,
                    "to_name": s.to_name,
                    "line": s.line,
                }
                for s in self.steps
            ],
            "path_score": self.path_score,
            "end_king_wen": self.end_king_wen,
            "end_bits": self.end_bits,
            "endpoint_score": self.endpoint_score,
        }


def enumerate_paths(start_bits: str, max_steps: int = DEFAULT_MAX_STEPS) -> list[list[PathStep]]:
    """All simple bian paths of 1..``max_steps`` steps from ``start_bits``.

    Simple = no hexagram visited twice (flipping a line and flipping it
    straight back is not a path, it is standing still). Terminates: the
    search space is bounded by 6 + 6*5 + 6*5*5 = 186 paths at 3 steps
    (at each step after the first, 5 of the 6 lines lead somewhere new:
    only the line just flipped leads back to the previous node).
    """
    if not (1 <= max_steps <= 3):
        raise ValueError(f"max_steps must be 1..3, got {max_steps}")
    results: list[list[PathStep]] = []

    def walk(current: str, visited: frozenset, path: list[PathStep]):
        if path:
            results.append(list(path))
        if len(path) == max_steps:
            return
        for line in range(1, 7):
            nxt = flip_bit(current, line)
            if nxt in visited:
                continue
            path.append(PathStep(current, nxt, line))
            walk(nxt, visited | {nxt}, path)
            path.pop()

    walk(start_bits, frozenset({start_bits}), [])
    return results


def best_path(start_bits: str, situation_vec: np.ndarray,
              doc_embeddings: np.ndarray, documents: list[dict],
              max_steps: int = DEFAULT_MAX_STEPS,
              step_penalty: float = STEP_PENALTY) -> tuple[list[PathStep], float, float]:
    """Best path by endpoint resonance minus a per-step penalty.

    Returns ``(steps, path_score, endpoint_cosine)``.
    """
    doc_by_bits = {d["bits"]: i for i, d in enumerate(documents)}
    best: list[PathStep] | None = None
    best_score = float("-inf")
    best_endpoint = 0.0
    for path in enumerate_paths(start_bits, max_steps):
        end_bits = path[-1].to_bits
        endpoint = float(
            cosine_similarity(
                doc_embeddings[doc_by_bits[end_bits]: doc_by_bits[end_bits] + 1],
                situation_vec,
            )[0]
        )
        score = endpoint - step_penalty * len(path)
        if score > best_score:
            best, best_score, best_endpoint = path, score, endpoint
    assert best is not None  # max_steps >= 1 always yields paths
    return best, best_score, best_endpoint


def reason(situation: str, encoder: GroundingEncoder | None = None,
           max_steps: int = DEFAULT_MAX_STEPS) -> ReasoningResult:
    """Full v1 pipeline: encode -> match top-1 -> expand -> best path."""
    encoder = encoder or GroundingEncoder()
    ranked = encoder.similarities(situation)
    top = ranked[0]
    situation_vec = encoder.encode([situation])[0]
    steps, path_score, endpoint_score = best_path(
        top["bits"], situation_vec, encoder.doc_embeddings(),
        encoder.documents, max_steps,
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


def describe(result: ReasoningResult) -> str:
    """One-line-per-step template explanation (see interpreter.py)."""
    from taiji_field.interpreter import explain

    return explain(result)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print('usage: python -m taiji_field.reasoning "情境描述……"')
        return 2
    situation = " ".join(args)
    encoder = GroundingEncoder()
    print(f"[encoder backend: {encoder.backend}]")
    result = reason(situation, encoder)
    print(describe(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
