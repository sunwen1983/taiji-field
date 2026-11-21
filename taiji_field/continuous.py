"""Continuous lines (Phase 2): a state is 6 floats in [0,1].

Roadmap section 2: a line is not a 0/1 bit but a yin-yang ratio. Binary
hexagrams are the corners of the continuous cube. The soft vector of a
state is the per-line weighted sum of the two bound pair vectors:

    soft(theta) = sum_i [ theta_i * bound(line_i, yang)
                          + (1 - theta_i) * bound(line_i, yin) ]

At theta_i = 0.5 the two bound vectors partly cancel, so the line's
contribution shrinks — maximal uncertainty is a weaker signal, not a
third value. At the corners (all theta in {0, 1}) the soft vector is
exactly the binary hexagram vector from :mod:`taiji_field.vsa`.

Usage:
    python -m taiji_field.continuous 1 2 [--steps 10]
prints the interpolation path from hexagram #1 (Qian) to #2 (Kun),
reporting the nearest binary hexagram at each step.
"""

from __future__ import annotations

import argparse

import numpy as np

from taiji_field.data import Hexagram, load_hexagrams
from taiji_field.graph import find_hexagram
from taiji_field.vsa import bound_pairs_numpy, hexagram_matrix

_PAIRS: np.ndarray | None = None


def _pairs() -> np.ndarray:
    global _PAIRS
    if _PAIRS is None:
        _PAIRS = bound_pairs_numpy()
    return _PAIRS


def soft_vector(thetas: list[float] | np.ndarray) -> np.ndarray:
    """Soft hexagram vector for 6 yin-yang ratios (index 0 = L1)."""
    thetas = np.asarray(thetas, dtype=np.float64)
    if thetas.shape != (6,):
        raise ValueError(f"thetas must have 6 entries, got shape {thetas.shape}")
    if np.any(thetas < 0) or np.any(thetas > 1):
        raise ValueError(f"thetas must be in [0, 1], got {thetas}")
    pairs = _pairs().astype(np.float64)  # (6, 2, D)
    # theta * yang-pair + (1 - theta) * yin-pair, summed over lines.
    weights = np.stack([1.0 - thetas, thetas], axis=1)  # (6, 2)
    return np.einsum("lv,lvd->d", weights, pairs)


def nearest_hexagram(
    vec: np.ndarray,
    matrix: np.ndarray | None = None,
    hexagrams: list[Hexagram] | None = None,
) -> tuple[Hexagram, float]:
    """Nearest binary hexagram to ``vec`` by VSA cosine."""
    matrix = hexagram_matrix() if matrix is None else matrix
    hexagrams = load_hexagrams() if hexagrams is None else hexagrams
    vec = np.asarray(vec, dtype=np.float64)
    sims = (matrix @ vec) / (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec)
    )
    i = int(np.argmax(sims))
    return hexagrams[i], float(sims[i])


def interpolate(
    bits_a: str, bits_b: str, steps: int = 10
) -> list[dict]:
    """Linear interpolation between two hexagrams.

    Returns ``steps + 1`` states (endpoints included). Each state is a
    dict with ``step``, ``thetas`` (the 6 ratios), ``nearest_king_wen``,
    ``nearest_name``, and ``cosine`` to that nearest binary hexagram.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    a = np.array([float(c) for c in bits_a], dtype=np.float64)
    b = np.array([float(c) for c in bits_b], dtype=np.float64)
    matrix = hexagram_matrix()
    hexagrams = load_hexagrams()
    out = []
    for k in range(steps + 1):
        t = k / steps
        thetas = (1.0 - t) * a + t * b
        vec = soft_vector(thetas)
        h, score = nearest_hexagram(vec, matrix, hexagrams)
        out.append(
            {
                "step": k,
                "t": round(t, 3),
                "thetas": [round(float(x), 3) for x in thetas],
                "nearest_king_wen": h.king_wen,
                "nearest_name": h.name_zh,
                "cosine": round(score, 4),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interpolate between two hexagrams in continuous space."
    )
    parser.add_argument("hex_a", help="first hexagram (number, name, or bits)")
    parser.add_argument("hex_b", help="second hexagram (number, name, or bits)")
    parser.add_argument("--steps", type=int, default=10,
                        help="interpolation intervals (default 10)")
    args = parser.parse_args(argv)
    ha, hb = find_hexagram(args.hex_a), find_hexagram(args.hex_b)
    if ha is None:
        print(f"unknown hexagram: {args.hex_a!r}")
        return 2
    if hb is None:
        print(f"unknown hexagram: {args.hex_b!r}")
        return 2
    print(f"{ha.name_zh} #{ha.king_wen} ({ha.bits}) -> "
          f"{hb.name_zh} #{hb.king_wen} ({hb.bits})")
    for s in interpolate(ha.bits, hb.bits, steps=args.steps):
        print(f"  step {s['step']:>2} t={s['t']:.2f} theta={s['thetas']} "
              f"-> #{s['nearest_king_wen']} {s['nearest_name']} "
              f"(cos {s['cosine']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
