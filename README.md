# Taiji Field 太极场

> 不是预测下一个词的模型，而是一个"场"：智能发生在状态的演化与共振里，语言只是进出的门。

> Not a model that predicts the next token, but a *field*: intelligence happens
> in the evolution and resonance of states, and language is only the door in
> and out.

Taiji Field is an open research project exploring a token-free, non-probabilistic
substrate for intelligence, inspired by the *I Ching* (*Yijing*, Book of Changes)
and by the Chinese idea of **意会 (yìhuì)** — understanding through resonance
and context, without everything being spelled out.

The working hypothesis: current large language models are an illusion of
understanding — fluent, but modeling only the shadow of thought (language
statistics) rather than thought itself. This project tries a different
substrate: **states, not tokens**. A situation is a configuration of
yin–yang lines (a hexagram, a point on the 6-dimensional hypercube Q₆);
thinking is movement on a graph of such states, along meaningful edges
(changing lines, opposites, reversals, nuclear hexagrams); understanding
is resonance between states.

Full roadmap (Chinese): [docs/roadmap.md](docs/roadmap.md).

## Status

**Phase 0 — skeleton validation (done).** The 64-hexagram state graph is built,
tested, and visualized. See [docs/phase0-notes.md](docs/phase0-notes.md) for
three hand-walked everyday scenarios and an honest assessment of where the
walk feels right and where it feels forced.

No LLM calls anywhere in this codebase — that is the point.

## Install

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Hexagram data comes from the [`zhouyi-6bit`](https://pypi.org/project/zhouyi-6bit/)
package (MIT), loaded from the installed package at runtime; a vendored copy
with attribution lives in `taiji_field/data/` as a fallback.

## Use

```sh
# Run the tests
pytest

# Inspect one hexagram's neighbors, grouped by edge type
python -m taiji_field.demo 24
python -m taiji_field.demo ䷗
python -m taiji_field.demo "Fu"

# Re-render the graph visualization
python -m taiji_field.viz
```

## Conventions (read before touching the code)

- **Yang = 1, yin = 0. Bits are ordered bottom-to-top, L1..L6.** This matches
  `zhouyi-6bit`. The Go library `godcong/yi` uses the *opposite* convention
  (yang = 0) — do not mix the two.
- A hexagram's canonical identity is its 6-bit string, e.g. `111111` (Qian 乾).
- Edge types on the state graph: `bian` 变卦 (flip exactly one line — the Q₆
  hypercube edges), `cuo` 错卦 (bitwise NOT), `zong` 综卦 (reverse bit order),
  `hu` 互卦 (nuclear hexagram; directed edges, see `taiji_field.graph`).

## License

MIT — see [LICENSE](LICENSE).
