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

**Phase 1 — grounding and resonance (done).** A situation is encoded with
`BAAI/bge-small-zh-v1.5` and matched to its most resonant hexagram by cosine
similarity; the match is expanded 1–3 steps along bian (changing-line) edges
and explained with a template interpreter. The exam paper
([docs/exam-v1.md](docs/exam-v1.md), 30 scenarios) and its harness measure
the pipeline against draft labels — the labels are unreviewed, the LLM
baseline is not yet built, and the explanations are templates, not
understanding. See [docs/phase1-notes.md](docs/phase1-notes.md) for the
honest limits and the exam score.

**Phase 2 — learned dynamics (done).** Each hexagram is a hypervector
(`torchhd` MAP, D=10,000); lines are continuous yin–yang ratios in [0,1],
so states interpolate smoothly between the binary corners; and the path
scorer is parameterized (endpoint resonance + path coherence + nuclear
attraction − steps) with weights tuned by grid search on the 30 exam
scenarios. Honest result: in-sample path matches rose 2/30 → 5/30, but
the leave-out check (fit on half, test on the other half, both ways)
gave 2/15 and 0/15 — combined 2/30, exactly Phase 1's number — and the
fitted weights are unstable across halves. So there is **no evidence
the dynamics learned a direction that generalizes**; the in-sample gain
is consistent with overfitting 4 weights on 30 draft labels. That is a
result, not a failure to hide. See [docs/phase2-notes.md](docs/phase2-notes.md).

**Measurement step — LLM baseline, richer documents, 100-scenario exam
(partly done, 2026-11-21).** The baseline runner
(`scripts/run_baseline.py`, DeepSeek `deepseek-chat`, same prompt
information the encoder gets) is written and its offline checks pass,
but the first live call was blocked by a timed-out approval, so **there
is still no baseline number** — see [docs/baseline-v1.md](docs/baseline-v1.md)
for exactly what happened and how to run it. Hexagram documents were
enriched (a short paragraph per hexagram + upper/lower trigram meanings,
[docs/rich-documents.md](docs/rich-documents.md)): on the 30 scenarios
top-1 went 15/30 → 11/30, top-3 20/30 → 16/30, path still 2/30 —
richer documents did not help there. A new 100-scenario exam
(`taiji_field/data/exam_100.json`, 5 categories × 20) was built with
two annotation passes (author reasoning + embedding-resonance retrieval)
and a third pass pending human review — **99/100 scenarios are flagged**,
the two passes fully agree on only 1. The Phase 2 reasoner scores
**10/100 top-1, 21/100 top-3, 1/100 path** on it, with weights fitted on
the 30 (not refit). The honest headline: the 30-scenario 50% top-1 does
not survive a bigger paper (7–10%), so it was likely flattered by the
small, single-author label set. See
[docs/exam-100-notes.md](docs/exam-100-notes.md).

No LLM calls anywhere in the `taiji_field` package — that is the point.
Evaluation-only scripts under `scripts/` (`run_baseline.py`) may call an
LLM through the registered connector; they are never imported by the
package.

## Install

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Note: `sentence-transformers` pulls in PyTorch, and PyPI's default torch
wheel bundles CUDA — expect a multi-GB download on first install even
though everything here runs on CPU. (A CPU-only torch from the PyTorch
index is much smaller; install that first if you prefer, then
`pip install -e ".[dev]"` will reuse it.)

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

# Phase 1: resonate a situation to a hexagram, expand 1-3 bian steps,
# and explain the path (template-based)
python -m taiji_field.reasoning "两个部门为项目归属僵持了三周，谁也不让步"

# Phase 1: run the 30-scenario exam paper, write the JSON report
python -m taiji_field.exam

# Phase 2: interpolate between two hexagrams in continuous space
python -m taiji_field.continuous 1 2 --steps 5

# Phase 2: run the exam with the fitted dynamics scorer (fits weights,
# leave-out check, writes exam_report_v2.json — never overwrites v1)
python -m taiji_field.exam --reasoner v2

# Measurement step: LLM baseline on the 30 (needs DeepSeek connector
# approval; writes taiji_field/data/baseline_v1.json)
python scripts/run_baseline.py

# Measurement step: run the v2 reasoner (weights fitted on the 30,
# not refit) on the 100-scenario exam
python scripts/run_exam_100.py

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
