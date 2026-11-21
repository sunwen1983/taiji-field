"""Run the Phase 2 reasoner on the 100-scenario exam (EVALUATION ONLY).

Uses the dynamics weights FITTED ON THE 30-SCENARIO EXAM
(``taiji_field/data/exam_report_v2.json``) — the weights are NOT refit
here. Refitting on the 100 would be tuning the dynamics again, which
the measurement brief forbids. If the v2 report is missing, falls back
to :data:`taiji_field.dynamics.DEFAULT_WEIGHTS` and says so.

Writes ``taiji_field/data/exam_report_100.json`` (generated, not
committed — same convention as the v1/v2 reports). ``baseline`` stays
``null``: a 100-scenario LLM baseline is future work.

Usage: python scripts/run_exam_100.py [--limit N] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taiji_field.dynamics import DEFAULT_WEIGHTS
from taiji_field.exam import load_exam, run_exam
from taiji_field.grounding import GroundingEncoder

ROOT = Path(__file__).resolve().parent.parent
EXAM_100_PATH = ROOT / "taiji_field" / "data" / "exam_100.json"
REPORT_V2_PATH = ROOT / "taiji_field" / "data" / "exam_report_v2.json"
DEFAULT_OUTPUT = ROOT / "taiji_field" / "data" / "exam_report_100.json"


def fitted_weights_from_v2() -> tuple[dict, str]:
    """Weights fitted on the 30-scenario exam (Phase 2), with provenance."""
    if REPORT_V2_PATH.exists():
        report = json.loads(REPORT_V2_PATH.read_text(encoding="utf-8"))
        weights = report["fitted_weights"]["weights"]
        return weights, "fitted on exam v1 (30 scenarios), see exam_report_v2.json"
    return dict(DEFAULT_WEIGHTS), (
        "exam_report_v2.json not found; fell back to DEFAULT_WEIGHTS "
        "(Phase-1-equivalent)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the v2 reasoner on exam 100.")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N scenarios")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    encoder = GroundingEncoder()
    print(f"[encoder backend: {encoder.backend}]")
    weights, provenance = fitted_weights_from_v2()
    print(f"[weights: {weights} ({provenance})]")
    scenarios = load_exam(EXAM_100_PATH)["scenarios"]
    if args.limit:
        scenarios = scenarios[: args.limit]
    report = run_exam(encoder, scenarios, reasoner="v2", weights=weights)
    report["exam_version"] = "100-v1"
    report["exam_path"] = "taiji_field/data/exam_100.json"
    report["weights"] = weights
    report["weights_provenance"] = provenance
    report["label_status"] = (
        "draft — pass A (author) + pass B (embedding resonance); pass C "
        "pending human review; NOT ground truth (see docs/exam-100-notes.md)"
    )
    report["baseline"] = None  # 100-scenario LLM baseline is future work
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    s = report["summary"]
    print(f"exam 100 (v2 reasoner): {s['top1_matches']}/{s['n']} top-1, "
          f"{s['top3_hits']}/{s['n']} top-3, {s['path_matches']}/{s['n']} path")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
