"""Render the 64-hexagram state graph to docs/hexagram-graph.png.

Layout: a 2D projection of the Q6 hypercube — the first two principal
components of the centered 6-bit vectors — so position reflects bit
structure instead of a random spring layout. Nodes are colored by their
number of yang lines (0-6). Bian (Q6) edges are light gray, cuo edges
red, zong edges blue; a few well-known hexagrams are labeled.

Usage:
    python -m taiji_field.viz [output-path]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Noto Sans CJK SC is installed on this machine and covers Chinese
# characters; DejaVu Sans (matplotlib's default) does not. The Yijing
# hexagram symbols (U+4DC0..) may still fall back to boxes on systems
# without a font covering that block — the demo CLI always shows them
# in full, so the figure keeps pinyin as the primary label.
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]

from taiji_field.graph import build_graph

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "hexagram-graph.png"

# King Wen numbers labeled on the figure, with per-label offsets so
# neighbors that sit close together in the projection (notably Kun #2
# and Fu #24) do not print on top of each other.
LABELED = (1, 2, 11, 12, 24, 63, 64)
LABEL_OFFSETS = {2: (6, 10), 24: (-52, -14)}


def _positions(graph):
    """First two principal components of the centered bit matrix."""
    nodes = sorted(graph.nodes)
    bits = np.array([[int(c) for c in n] for n in nodes], dtype=float)
    centered = bits - bits.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    return {node: (float(x), float(y)) for node, (x, y) in zip(nodes, coords)}


def render(output: str | Path = DEFAULT_OUTPUT) -> Path:
    graph = build_graph()
    pos = _positions(graph)
    nodes = sorted(graph.nodes)
    yang_count = np.array([n.count("1") for n in nodes])

    fig, ax = plt.subplots(figsize=(12, 10))
    # bian edges (Q6 skeleton)
    for u, v, key in graph.edges(keys=True):
        if key != "bian":
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="#cccccc", linewidth=0.6, zorder=1)
    # cuo edges (red) and zong edges (blue)
    for u, v, key in graph.edges(keys=True):
        if key == "cuo":
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#d62728", linewidth=0.8,
                    alpha=0.55, zorder=2)
        elif key == "zong" and u != v:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#1f77b4", linewidth=0.8,
                    alpha=0.55, zorder=2)
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    sc = ax.scatter(xs, ys, c=yang_count, cmap="YlOrBr", s=90,
                    edgecolors="black", linewidths=0.5, zorder=3)
    for n in nodes:
        if graph.nodes[n]["king_wen"] in LABELED:
            x, y = pos[n]
            attrs = graph.nodes[n]
            ax.annotate(
                f"{attrs['name_zh']} {attrs['pinyin']} #{attrs['king_wen']}",
                (x, y), textcoords="offset points",
                xytext=LABEL_OFFSETS.get(attrs["king_wen"], (6, 6)),
                fontsize=11, zorder=4,
            )
    ax.set_title(
        "Taiji Field — 64-hexagram state graph (Phase 0)\n"
        "gray: bian 变卦 (Q6) · red: cuo 错卦 · blue: zong 综卦 · "
        "color: yang-line count",
        fontsize=12,
    )
    ax.set_axis_off()
    fig.colorbar(sc, ax=ax, shrink=0.6, label="yang lines (0-6)")
    fig.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out = render(args[0] if args else DEFAULT_OUTPUT)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
