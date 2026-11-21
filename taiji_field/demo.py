"""Demo CLI: inspect one hexagram's neighbors, grouped by edge type.

Usage:
    python -m taiji_field.demo 24
    python -m taiji_field.demo ䷗
    python -m taiji_field.demo Fu
    python -m taiji_field.demo 100000
"""

from __future__ import annotations

import sys

from taiji_field.data import load_hexagrams
from taiji_field.graph import (
    build_graph,
    build_nuclear_graph,
    find_hexagram,
    hu_bits,
    neighbors_by_type,
)


def _label(graph, bits: str) -> str:
    attrs = graph.nodes[bits]
    return (
        f"{attrs['unicode']} {attrs['name_zh']} "
        f"({attrs['pinyin']}, #{attrs['king_wen']})"
    )


def describe(query: str) -> str:
    hexagram = find_hexagram(query)
    if hexagram is None:
        return f"no hexagram matches {query!r} (try a number 1-64, a name, or bits)"
    graph = build_graph()
    nuclear = build_nuclear_graph()
    by_kw = {h.king_wen: h for h in load_hexagrams()}
    lines = [
        f"{hexagram.unicode} {hexagram.name_zh} "
        f"({hexagram.pinyin}, {hexagram.english}) — "
        f"#{hexagram.king_wen}, bits L1..L6: {hexagram.bits}",
        "",
    ]
    nbrs = neighbors_by_type(graph, hexagram.bits)
    lines.append("变卦 bian — flip one line (6 neighbors):")
    for line, nbr in nbrs["bian"]:
        target = by_kw[graph.nodes[nbr]["king_wen"]]
        lines.append(f"  动第 {line} 爻 -> {_label(graph, nbr)}")
    lines.append("错卦 cuo — opposite (bitwise NOT):")
    for nbr in nbrs["cuo"]:
        lines.append(f"  {_label(graph, nbr)}")
    lines.append("综卦 zong — reversed (turned upside down):")
    for nbr in nbrs["zong"]:
        if nbr == hexagram.bits:
            lines.append(f"  {_label(graph, nbr)} (self-symmetric)")
        else:
            lines.append(f"  {_label(graph, nbr)}")
    lines.append("互卦 hu — nuclear hexagram (directed):")
    target = hu_bits(hexagram.bits)
    lines.append(f"  {_label(nuclear, target)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m taiji_field.demo <number|name|unicode|bits>")
        print("  e.g. python -m taiji_field.demo 24")
        return 2
    print(describe(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
