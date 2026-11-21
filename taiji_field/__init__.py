"""Taiji Field — a token-free, state-dynamics substrate for intelligence.

Phase 0: the 64-hexagram state graph. Phase 1: grounding encoder,
resonance reasoning, template interpreter, exam paper. No LLM calls
anywhere in this package — that is the point.
"""

from taiji_field.data import Hexagram, load_hexagrams
from taiji_field.documents import hexagram_documents
from taiji_field.graph import (
    build_graph,
    build_nuclear_graph,
    cuo_bits,
    hu_bits,
    zong_bits,
)
from taiji_field.grounding import GroundingEncoder
from taiji_field.reasoning import reason

__version__ = "0.1.0"

__all__ = [
    "Hexagram",
    "GroundingEncoder",
    "build_graph",
    "build_nuclear_graph",
    "cuo_bits",
    "hexagram_documents",
    "hu_bits",
    "load_hexagrams",
    "reason",
    "zong_bits",
]
