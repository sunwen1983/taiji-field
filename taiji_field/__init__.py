"""Taiji Field — a token-free, state-dynamics substrate for intelligence.

Phase 0: the 64-hexagram state graph. No LLM calls anywhere in this
package — that is the point.
"""

from taiji_field.data import Hexagram, load_hexagrams
from taiji_field.graph import (
    build_graph,
    build_nuclear_graph,
    cuo_bits,
    hu_bits,
    zong_bits,
)

__version__ = "0.1.0"

__all__ = [
    "Hexagram",
    "build_graph",
    "build_nuclear_graph",
    "cuo_bits",
    "hu_bits",
    "load_hexagrams",
    "zong_bits",
]
