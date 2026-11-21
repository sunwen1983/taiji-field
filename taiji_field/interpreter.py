"""Interpreter v1 (Phase 1): template-based Chinese explanation of a path.

Lives in its own module (rather than inside reasoning.py) so the
*selection* of a path (reasoning) and the *verbalization* of a path
(interpreter) stay separate: the day the explanation stops being a
template, only this module changes.

The explanation is a template. It states which line moved and quotes the
two hexagrams' authored one-line readings; it does NOT invent a causal
reason for the move. If a sentence in the output sounds like the system
"understands why", that is the template's wording, not a claim — see
docs/phase1-notes.md.
"""

from __future__ import annotations

from taiji_field.data import load_hexagrams
from taiji_field.documents import READINGS

_LINE_CN = {1: "初", 2: "二", 3: "三", 4: "四", 5: "五", 6: "上"}


def _reading(king_wen: int) -> str:
    return READINGS[king_wen]


def explain(result) -> str:
    """Render a :class:`ReasoningResult` as short, plain Chinese."""
    by_kw = {h.king_wen: h for h in load_hexagrams()}
    top = result.top_matches[0]
    lines = [
        f"这个情境最共振的是 {top['name_zh']} #{top['king_wen']}（相似度 {top['score']:.2f}）：{_reading(top['king_wen'])}",
    ]
    if len(result.top_matches) > 1:
        others = "、".join(
            f"{m['name_zh']} #{m['king_wen']}（{m['score']:.2f}）"
            for m in result.top_matches[1:]
        )
        lines.append(f"次近的共振：{others}。")
    for i, step in enumerate(result.steps, 1):
        lines.append(
            f"第 {i} 步：动第 {step.line} 爻（{_LINE_CN[step.line]}爻），"
            f"从 {step.from_name} #{step.from_king_wen} 到 {step.to_name} #{step.to_king_wen}。"
            f"{step.from_name}：{_reading(step.from_king_wen)}"
            f"{step.to_name}：{_reading(step.to_king_wen)}"
        )
    end = by_kw[result.end_king_wen]
    lines.append(
        f"落点是 {end.name_zh} #{end.king_wen}：{_reading(result.end_king_wen)}"
    )
    lines.append(
        "一句话：从"
        f"{result.top_matches[0]['name_zh']}出发，"
        f"经过 {len(result.steps)} 步演化，落到"
        f"{end.name_zh} #{end.king_wen}——这是一条由共振排出来的路，不是系统“想明白了”的结论。"
    )
    return "\n".join(lines)
