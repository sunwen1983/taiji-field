"""VSA vectorization (Phase 2): hexagrams as hypervectors (torchhd MAP).

Design (roadmap section 3a, section 5 Phase 2):
- Dimension D = 10,000. Base hypervectors: 6 line-position vectors plus
  2 value vectors (yin, yang), drawn once from a fixed seed so the
  space is reproducible.
- A hexagram vector = bundle (elementwise sum, torchhd MAP) of the 6
  bound position-value pairs: ``bind(pos_i, value_bit_i)`` for lines
  L1..L6. A trigram vector is the same construction from 3 lines
  (``offset`` 0 = lower trigram L1-L3, 3 = upper trigram L4-L6).
- Similarity = cosine.

Consequences worth knowing (all verified in tests/test_phase2.py):
- Cosine between two hexagram vectors is ~ ``1 - hamming/6``: Hamming-1
  neighbours sit at ~0.83, Hamming-3 at ~0.5, opposite (cuo, Hamming-6)
  at ~0. So cuo pairs are the *least* similar (orthogonal), not
  anti-parallel: yin/yang are independent random vectors, not negatives
  of each other. "Opposite" here means "shares nothing", which matches
  the classical reading better than "is the negative".
- Because every Hamming-1 step has nearly the SAME cosine (~0.83, std
  < 0.05), VSA cosine cannot rank *which* line to move — it carries no
  directional information. That is why the dynamics scorer in
  :mod:`taiji_field.dynamics` measures path coherence in the grounding
  (document) space, not in VSA space. See that module's docstring.

No LLM calls anywhere here — the vectors are random projections plus
algebra, not learned from text.
"""

from __future__ import annotations

import numpy as np

from taiji_field.data import Hexagram, load_hexagrams

DIMENSION = 10000
SEED = 20261121

_BASE: dict | None = None
_HEX_MATRIX: np.ndarray | None = None
_HEXAGRAMS: list[Hexagram] | None = None


def _base_vectors():
    """The 8 base hypervectors: 6 positions + yin/yang values.

    Returns ``(positions, yin, yang)`` as torch MAPTensors of shape
    ``(6, D)``, ``(D,)``, ``(D,)``. Generated once, fixed seed.
    """
    global _BASE
    if _BASE is None:
        import torch
        import torchhd

        gen = torch.Generator().manual_seed(SEED)
        base = torchhd.random(8, DIMENSION, vsa="MAP", generator=gen)
        _BASE = {
            "positions": base[:6],
            "yin": base[6],
            "yang": base[7],
        }
    return _BASE


def bound_pair(line: int, value: int):
    """``bind(pos_line, value)`` as a torch tensor (line 1..6, value 0/1)."""
    import torchhd

    base = _base_vectors()
    val = base["yang"] if value == 1 else base["yin"]
    return torchhd.bind(base["positions"][line - 1], val)


def bound_pairs_numpy() -> np.ndarray:
    """All 12 bound pairs as a numpy array of shape ``(6, 2, D)``.

    Index ``[line-1, value]``. Used by :mod:`taiji_field.continuous`,
    where per-line weighted sums need the raw pair vectors.
    """
    pairs = np.empty((6, 2, DIMENSION), dtype=np.float32)
    for line in range(1, 7):
        for value in (0, 1):
            pairs[line - 1, value] = bound_pair(line, value).numpy()
    return pairs


def _bundle_numpy(vectors: np.ndarray) -> np.ndarray:
    """torchhd MAP bundle (elementwise sum) of stacked vectors.

    Kept as a helper so the bundle step goes through torchhd, not a
    hand-rolled sum: ``multiset`` on a ``(n, D)`` stack. (Do NOT use
    ``bundle_sequence`` here — it permutes each element before summing,
    which is sequence encoding, not bundling.)
    """
    import torch
    import torchhd

    stack = torch.from_numpy(np.asarray(vectors, dtype=np.float32))
    return torchhd.multiset(stack).numpy().astype(np.float64)


def hexagram_vector(bits: str) -> np.ndarray:
    """VSA vector of one hexagram (bits L1..L6, yang=1, yin=0)."""
    if len(bits) != 6 or set(bits) - {"0", "1"}:
        raise ValueError(f"bits must be a 6-char 0/1 string, got {bits!r}")
    pairs = np.stack(
        [bound_pair(i + 1, int(b)).numpy() for i, b in enumerate(bits)]
    )
    return _bundle_numpy(pairs)


def trigram_vector(bits3: str, offset: int = 0) -> np.ndarray:
    """VSA vector of a 3-line trigram at ``offset`` (0 = L1-L3, 3 = L4-L6)."""
    if len(bits3) != 3 or set(bits3) - {"0", "1"}:
        raise ValueError(f"bits3 must be a 3-char 0/1 string, got {bits3!r}")
    if offset not in (0, 3):
        raise ValueError(f"offset must be 0 or 3, got {offset}")
    pairs = np.stack(
        [bound_pair(offset + i + 1, int(b)).numpy() for i, b in enumerate(bits3)]
    )
    return _bundle_numpy(pairs)


def hexagram_matrix() -> np.ndarray:
    """All 64 hexagram vectors, shape ``(64, D)``, in King Wen order
    (the order :func:`taiji_field.data.load_hexagrams` returns)."""
    global _HEX_MATRIX, _HEXAGRAMS
    if _HEX_MATRIX is None:
        _HEXAGRAMS = load_hexagrams()
        _HEX_MATRIX = np.stack([hexagram_vector(h.bits) for h in _HEXAGRAMS])
    return _HEX_MATRIX


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (or rows, pairwise)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(a @ b / denom)


def similarity_matrix() -> np.ndarray:
    """The 64x64 cosine matrix of all hexagram vectors."""
    m = hexagram_matrix().astype(np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    n = m / np.where(norms == 0, 1.0, norms)
    return n @ n.T
