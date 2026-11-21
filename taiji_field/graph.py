"""The 64-hexagram state graph (Phase 0).

Nodes are the 64 hexagrams, identified by their 6-bit string
(L1..L6, bottom-to-top, yang = 1, yin = 0).

``build_graph`` returns a :class:`networkx.MultiGraph` with three typed,
undirected edge kinds:

- ``bian`` 变卦 — flip exactly one line. The 192 edges of the 6-dimensional
  hypercube Q6 (each node has degree 6). Edge attribute ``line`` (1..6)
  records which line flips.
- ``cuo`` 错卦 — bitwise NOT. Exactly 32 pairs; every hexagram has exactly
  one opposite.
- ``zong`` 综卦 — reverse bit order. 28 pairs between distinct hexagrams
  plus 8 self-symmetric hexagrams (King Wen 1, 2, 27, 28, 29, 30, 61, 62),
  stored as self-loop edges.

``build_nuclear_graph`` returns a separate :class:`networkx.DiGraph` for
the fourth relation, which is directed:

- ``hu`` 互卦 — nuclear hexagram: the lower trigram is lines 2, 3, 4 and
  the upper trigram is lines 3, 4, 5 of the original, i.e. the image is
  ``(b2, b3, b4, b3, b4, b5)``. Every hexagram has exactly one outgoing
  edge; the image has exactly 16 hexagrams, each the image of exactly 4.
  It is the only classical relation that is not reversible (a rank-4
  projection on GF(2)).
"""

from __future__ import annotations

import networkx as nx

from taiji_field.data import load_hexagrams

# King Wen numbers of the eight self-symmetric hexagrams (zong == itself).
SELF_SYMMETRIC_KING_WEN = (1, 2, 27, 28, 29, 30, 61, 62)


def flip_bit(bits: str, line: int) -> str:
    """Flip line ``line`` (1..6, L1 = bottom) of a 6-bit string."""
    if not 1 <= line <= 6:
        raise ValueError(f"line must be 1..6, got {line}")
    idx = line - 1
    flipped = "1" if bits[idx] == "0" else "0"
    return bits[:idx] + flipped + bits[idx + 1:]


def cuo_bits(bits: str) -> str:
    """错卦: bitwise NOT of all six lines."""
    return "".join("1" if b == "0" else "0" for b in bits)


def zong_bits(bits: str) -> str:
    """综卦: reverse the line order (turn the hexagram upside down)."""
    return bits[::-1]


def hu_bits(bits: str) -> str:
    """互卦 (nuclear): lower trigram = lines 2,3,4; upper = lines 3,4,5."""
    b = bits  # b[0] is L1 (bottom), b[5] is L6 (top)
    return b[1] + b[2] + b[3] + b[2] + b[3] + b[4]


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def build_graph() -> nx.MultiGraph:
    """Build the state graph: 64 nodes, typed bian/cuo/zong edges."""
    hexagrams = load_hexagrams()
    graph: nx.MultiGraph = nx.MultiGraph()
    for h in hexagrams:
        graph.add_node(
            h.bits,
            king_wen=h.king_wen,
            name_zh=h.name_zh,
            unicode=h.unicode,
            pinyin=h.pinyin,
            english=h.english,
            bits=h.bits,
            shao_yong=h.shao_yong,
        )
    nodes = sorted(graph.nodes)
    for bits in nodes:
        # bian: flip one line; add each pair once (lexicographic order).
        for line in range(1, 7):
            nbr = flip_bit(bits, line)
            if bits < nbr:
                graph.add_edge(bits, nbr, key="bian", type="bian", line=line)
        # cuo: bitwise NOT; add each pair once.
        opposite = cuo_bits(bits)
        if bits < opposite:
            graph.add_edge(bits, opposite, key="cuo", type="cuo")
        # zong: reverse; self-symmetric hexagrams get a self-loop.
        reversed_bits = zong_bits(bits)
        if reversed_bits == bits:
            graph.add_edge(bits, bits, key="zong", type="zong")
        elif bits < reversed_bits:
            graph.add_edge(bits, reversed_bits, key="zong", type="zong")
    return graph


def build_nuclear_graph() -> nx.DiGraph:
    """Build the directed nuclear (hu) graph: each hexagram -> its nuclear."""
    hexagrams = load_hexagrams()
    graph: nx.DiGraph = nx.DiGraph()
    for h in hexagrams:
        graph.add_node(
            h.bits,
            king_wen=h.king_wen,
            name_zh=h.name_zh,
            unicode=h.unicode,
            pinyin=h.pinyin,
            english=h.english,
            bits=h.bits,
        )
    for h in hexagrams:
        graph.add_edge(h.bits, hu_bits(h.bits), type="hu")
    return graph


def neighbors_by_type(graph: nx.MultiGraph, bits: str) -> dict[str, list]:
    """Group a node's neighbors by edge type.

    Returns ``{"bian": [(line, neighbor), ...], "cuo": [neighbor],
    "zong": [neighbor]}``. For ``bian`` the flipped line is included.
    """
    out: dict[str, list] = {"bian": [], "cuo": [], "zong": []}
    for nbr, edge_dict in graph[bits].items():
        for key, attrs in edge_dict.items():
            if key == "bian":
                out["bian"].append((attrs["line"], nbr))
            else:
                out[key].append(nbr)
    out["bian"].sort()
    return out


def find_hexagram(query: str):
    """Find a hexagram by King Wen number, pinyin, Chinese name, unicode
    symbol, or 6-bit string. Returns a :class:`Hexagram` or ``None``."""
    q = query.strip()
    hexagrams = load_hexagrams()
    # A 6-character 0/1 string is a bit string, not a King Wen number
    # (numbers are 1-64, at most two digits) — check it first.
    if len(q) == 6 and set(q) <= {"0", "1"}:
        for h in hexagrams:
            if h.bits == q:
                return h
        return None
    if q.isdigit():
        n = int(q)
        for h in hexagrams:
            if h.king_wen == n:
                return h
        return None
    lowered = q.lower()
    for h in hexagrams:
        if (
            h.pinyin.lower() == lowered
            or h.name_zh == q
            or h.unicode == q
            or h.english.lower() == lowered
        ):
            return h
    return None
