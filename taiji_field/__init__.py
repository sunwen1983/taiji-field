"""Taiji Field — a token-free, state-dynamics substrate for intelligence.

Phase 0: the 64-hexagram state graph. Phase 1: grounding encoder,
resonance reasoning, template interpreter, exam paper. Phase 2: VSA
vectorization, continuous lines, learned dynamics scorer. No LLM calls
anywhere in this package — that is the point.
"""

from taiji_field.data import Hexagram, load_hexagrams
from taiji_field.documents import hexagram_documents
from taiji_field.dynamics import reason_v2
from taiji_field.graph import (
    build_graph,
    build_nuclear_graph,
    cuo_bits,
    hu_bits,
    zong_bits,
)
from taiji_field.grounding import GroundingEncoder
from taiji_field.reasoning import reason
from taiji_field.vsa import hexagram_vector

__version__ = "0.2.0"

__all__ = [
    "Hexagram",
    "GroundingEncoder",
    "build_graph",
    "build_nuclear_graph",
    "cuo_bits",
    "hexagram_documents",
    "hexagram_vector",
    "hu_bits",
    "load_hexagrams",
    "reason",
    "reason_v2",
    "zong_bits",
]
