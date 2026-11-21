"""Exam harness v1 (Phase 1) + v2 (Phase 2): run the pipeline on the exam paper.

For every scenario in ``taiji_field/data/exam_v1.json``: encode the
scenario text, match the top-3 hexagrams, reason out the best 1-3 step
bian path, and compare against the PROPOSED (draft, unreviewed) label.

v1 (default) uses the Phase 1 heuristic ranking (endpoint resonance
minus a per-step penalty). v2 (``--reasoner v2``) uses the Phase 2
parameterized dynamics scorer (:mod:`taiji_field.dynamics`); see
:func:`run_exam_v2`, which fits the weights, runs the leave-out check,
and writes the v2 report. The v1 report is never overwritten by a v2
run — v2 writes ``taiji_field/data/exam_report_v2.json``.

Writes a JSON report with, per scenario: predicted top-1/top-3,
predicted path, scores, ``top1_match`` (predicted top-1 == proposed
start), ``top3_hit``, and ``path_match`` (predicted moving lines ==
proposed moving lines).

The report also reserves the baseline slot for the Phase 1 roadmap's
planned LLM baseline ("ask an LLM what happens next, blind review"):
``baseline`` is ``null`` — the baseline was deliberately NOT built
(it needs API keys and a blind-review setup). See
docs/phase1-notes.md. When it is built, fill ``baseline`` in with the
same per-scenario fields; do not change the report schema.

Usage:
    python -m taiji_field.exam [--limit N] [--output PATH]
    python -m taiji_field.exam --reasoner v2 [--limit N] [--output PATH]
    python scripts/run_exam.py [--limit N] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from taiji_field.dynamics import (
    DEFAULT_WEIGHTS,
    fit_weights,
    leave_out_check,
    reason_v2,
)
from taiji_field.grounding import GroundingEncoder
from taiji_field.reasoning import reason

EXAM_PATH = Path(__file__).parent / "data" / "exam_v1.json"
DEFAULT_REPORT = Path(__file__).parent / "data" / "exam_report.json"
DEFAULT_REPORT_V2 = Path(__file__).parent / "data" / "exam_report_v2.json"


def load_exam(path: str | Path = EXAM_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_exam(encoder: GroundingEncoder | None = None,
             scenarios: list[dict] | None = None,
             max_steps: int = 3,
             reasoner: str = "v1",
             weights: dict[str, float] | None = None) -> dict:
    """Run the pipeline on ``scenarios`` (default: the full v1 paper).

    ``reasoner="v1"`` uses the Phase 1 heuristic ranking;
    ``reasoner="v2"`` uses the Phase 2 parameterized dynamics scorer
    with ``weights`` (default :data:`taiji_field.dynamics.DEFAULT_WEIGHTS`,
    the Phase-1-equivalent setting).
    """
    encoder = encoder or GroundingEncoder()
    if scenarios is None:
        scenarios = load_exam()["scenarios"]
    results = []
    for s in scenarios:
        proposed = s["proposed"]
        if reasoner == "v2":
            r = reason_v2(s["scenario"], encoder, weights=weights, max_steps=max_steps)
        elif reasoner == "v1":
            r = reason(s["scenario"], encoder, max_steps=max_steps)
        else:
            raise ValueError(f"unknown reasoner {reasoner!r} (want 'v1' or 'v2')")
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
        "exam_version": "v2" if reasoner == "v2" else "v1",
        "reasoner": (
            "dynamics-v1 (w1*endpoint + w2*coherence + w3*nuclear - w4*steps)"
            if reasoner == "v2"
            else "resonance-v1 (endpoint - 0.05*steps)"
        ),
        "encoder_backend": encoder.backend,
        "encoder_model": getattr(encoder, "model_name", None),
        "label_status": "draft — proposed labels awaiting Wen's review, not ground truth",
        "summary": summary,
        # Baseline slot (roadmap §6): null in v1, filled in when the blind
        # LLM baseline is built. Same per-scenario fields when it lands.
        "baseline": None,
        "results": results,
    }


def run_exam_v2(encoder: GroundingEncoder | None = None,
                scenarios: list[dict] | None = None) -> dict:
    """Phase 2 exam run: fit the dynamics weights, leave-out check,
    then run the full exam with the fitted weights.

    The report's ``summary`` is the IN-SAMPLE result with the fitted
    weights (optimistic by construction). The honest numbers are in
    ``leave_out``: fit on half the exam, test on the other half, both
    directions. If the leave-out path matches do not beat Phase 1's
    2/30, the dynamics did not learn a direction — that is a result,
    not a failure to hide (see docs/phase2-notes.md).
    """
    encoder = encoder or GroundingEncoder()
    if scenarios is None:
        scenarios = load_exam()["scenarios"]
    fitted = fit_weights(scenarios, encoder)
    loo = leave_out_check(encoder, scenarios)
    report = run_exam(encoder, scenarios, reasoner="v2", weights=fitted["weights"])
    report["fitted_weights"] = fitted
    report["leave_out"] = loo
    report["label_status"] = (
        "draft — proposed labels are single-author drafts; the user "
        "waived his review on 2026-11-21 and authorized proceeding on "
        "self-check, so the labels are NOT ground truth and 30 scenarios "
        "are too few for real learning — the weights are tuned, not learned"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the exam paper.")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N scenarios")
    parser.add_argument("--reasoner", choices=["v1", "v2"], default="v1",
                        help="v1: Phase 1 heuristic; v2: Phase 2 dynamics "
                        "(fits weights + leave-out check)")
    parser.add_argument("--output", type=str, default=None,
                        help="where to write the JSON report "
                        "(default: exam_report.json for v1, "
                        "exam_report_v2.json for v2 — v2 never overwrites v1)")
    args = parser.parse_args(argv)
    encoder = GroundingEncoder()
    print(f"[encoder backend: {encoder.backend}]")
    scenarios = load_exam()["scenarios"]
    if args.limit:
        scenarios = scenarios[: args.limit]
    if args.reasoner == "v2":
        report = run_exam_v2(encoder, scenarios)
        out = Path(args.output) if args.output else DEFAULT_REPORT_V2
        s = report["summary"]
        print(f"exam v2 (fitted weights {report['fitted_weights']['weights']}): "
              f"{s['top1_matches']}/{s['n']} top-1, "
              f"{s['top3_hits']}/{s['n']} top-3, "
              f"{s['path_matches']}/{s['n']} path matches (in-sample)")
        loo = report["leave_out"]
        print(f"leave-out: fit-A/test-B "
              f"{loo['fit_A_test_B']['path_matches']}/{loo['fit_A_test_B']['n']} path, "
              f"fit-B/test-A "
              f"{loo['fit_B_test_A']['path_matches']}/{loo['fit_B_test_A']['n']} path")
    else:
        report = run_exam(encoder, scenarios)
        out = Path(args.output) if args.output else DEFAULT_REPORT
        s = report["summary"]
        print(f"exam v1: {s['top1_matches']}/{s['n']} top-1 matches, "
              f"{s['top3_hits']}/{s['n']} top-3 hits, "
              f"{s['path_matches']}/{s['n']} full path matches")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
