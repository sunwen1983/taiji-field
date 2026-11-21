"""Exam harness v1 (Phase 1): run the full pipeline on the exam paper.

For every scenario in ``taiji_field/data/exam_v1.json``: encode the
scenario text, match the top-3 hexagrams, reason out the best 1-3 step
bian path, and compare against the PROPOSED (draft, unreviewed) label.

Writes a JSON report (default ``taiji_field/data/exam_report.json`` —
generated, NOT committed) with, per scenario: predicted top-1/top-3,
predicted path, scores, ``top1_match`` (predicted top-1 == proposed
start), ``top3_hit``, and ``path_match`` (predicted moving lines ==
proposed moving lines).

The report also reserves the baseline slot for the Phase 1 roadmap's
planned LLM baseline ("ask an LLM what happens next, blind review"):
``baseline`` is ``null`` in v1 — the baseline was deliberately NOT built
in this phase (it needs API keys and a blind-review setup). See
docs/phase1-notes.md. When it is built, fill ``baseline`` in with the
same per-scenario fields; do not change the report schema.

Usage:
    python -m taiji_field.exam [--limit N] [--output PATH]
    python scripts/run_exam.py [--limit N] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from taiji_field.grounding import GroundingEncoder
from taiji_field.reasoning import reason

EXAM_PATH = Path(__file__).parent / "data" / "exam_v1.json"
DEFAULT_REPORT = Path(__file__).parent / "data" / "exam_report.json"


def load_exam(path: str | Path = EXAM_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_exam(encoder: GroundingEncoder | None = None,
             scenarios: list[dict] | None = None,
             max_steps: int = 3) -> dict:
    """Run the pipeline on ``scenarios`` (default: the full v1 paper)."""
    encoder = encoder or GroundingEncoder()
    if scenarios is None:
        scenarios = load_exam()["scenarios"]
    results = []
    for s in scenarios:
        proposed = s["proposed"]
        r = reason(s["scenario"], encoder, max_steps=max_steps)
        pred_lines = [step.line for step in r.steps]
        results.append(
            {
                "id": s["id"],
                "category": s["category"],
                "scenario": s["scenario"],
                "proposed_start_king_wen": proposed["start_king_wen"],
                "proposed_start_name": proposed["start_name"],
                "proposed_moving_lines": proposed["moving_lines"],
                "proposed_end_king_wen": proposed["end_king_wen"],
                "predicted_top1_king_wen": r.start_king_wen,
                "predicted_top1_name": r.top_matches[0]["name_zh"],
                "predicted_top1_score": r.top_matches[0]["score"],
                "predicted_top3": [
                    {
                        "king_wen": m["king_wen"],
                        "name_zh": m["name_zh"],
                        "score": m["score"],
                    }
                    for m in r.top_matches
                ],
                "predicted_path": {
                    "start_king_wen": r.start_king_wen,
                    "moving_lines": pred_lines,
                    "end_king_wen": r.end_king_wen,
                    "path_score": r.path_score,
                    "endpoint_score": r.endpoint_score,
                },
                "top1_match": r.start_king_wen == proposed["start_king_wen"],
                "top3_hit": any(
                    m["king_wen"] == proposed["start_king_wen"]
                    for m in r.top_matches
                ),
                "path_match": pred_lines == proposed["moving_lines"]
                and r.start_king_wen == proposed["start_king_wen"],
            }
        )
    n = len(results)
    summary = {
        "n": n,
        "top1_matches": sum(r["top1_match"] for r in results),
        "top3_hits": sum(r["top3_hit"] for r in results),
        "path_matches": sum(r["path_match"] for r in results),
    }
    return {
        "exam_version": "v1",
        "encoder_backend": encoder.backend,
        "encoder_model": getattr(encoder, "model_name", None),
        "label_status": "draft — proposed labels awaiting Wen's review, not ground truth",
        "summary": summary,
        # Baseline slot (roadmap §6): null in v1, filled in when the blind
        # LLM baseline is built. Same per-scenario fields when it lands.
        "baseline": None,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 1 exam paper.")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N scenarios")
    parser.add_argument("--output", type=str, default=str(DEFAULT_REPORT),
                        help="where to write the JSON report")
    args = parser.parse_args(argv)
    encoder = GroundingEncoder()
    print(f"[encoder backend: {encoder.backend}]")
    scenarios = load_exam()["scenarios"]
    if args.limit:
        scenarios = scenarios[: args.limit]
    report = run_exam(encoder, scenarios)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    s = report["summary"]
    print(f"exam v1: {s['top1_matches']}/{s['n']} top-1 matches, "
          f"{s['top3_hits']}/{s['n']} top-3 hits, "
          f"{s['path_matches']}/{s['n']} full path matches")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
