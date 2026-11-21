"""LLM baseline runner (EVALUATION ONLY — never imported by taiji_field).

Runs the 30-scenario exam paper (``taiji_field/data/exam_v1.json``) past
an LLM and scores it with the SAME match definitions as the exam harness
(:mod:`taiji_field.exam`): ``top1`` = predicted start hexagram equals the
proposed start; ``path`` = start AND moving lines both match.

Fairness: the LLM gets exactly the information the system gets — the
scenario text plus all 64 hexagrams with their one-line readings from
:mod:`taiji_field.documents` (name + one sentence each). Nothing more:
no trigram breakdown, no paragraphs, no graph relations, no hints about
the proposed labels. One API call per scenario, temperature 0.

Model access uses the registered ``custom.deepseek`` connector's own
mechanism (``dynamic_credentials`` surrogate exchange against
``api.deepseek.com``) — the same pattern as the connector's helper
``deepseek.py``. No API key is read, stored, or passed anywhere here.

Usage:
    python scripts/run_baseline.py [--limit N] [--output PATH] [--resume]

Output (default ``taiji_field/data/baseline_v1.json``):
model name, the full prompt template and hexagram list, per-scenario
predictions (start + moving lines + end + rationale + raw response),
and top-1 / path totals. The end hexagram is COMPUTED here from
start + moving lines (``taiji_field.graph.flip_bit``), never trusted
from the model; the model's own end number is recorded alongside for
comparison. A prediction that fails to parse after one retry is kept
in the file with ``parsed: false`` and counts as a miss — never faked.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request, read_json_response

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from taiji_field.data import load_hexagrams  # noqa: E402
from taiji_field.documents import READINGS  # noqa: E402
from taiji_field.exam import load_exam  # noqa: E402
from taiji_field.graph import flip_bit  # noqa: E402

API = "https://api.deepseek.com"
MODEL = "deepseek-chat"
TEMPERATURE = 0.0
MAX_TOKENS = 400
DEFAULT_OUTPUT = ROOT / "taiji_field" / "data" / "baseline_v1.json"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused, verify the endpoint")


def hexagram_list_text() -> str:
    """All 64 hexagrams, King Wen order: number + name + one-line reading.

    This is exactly the document content the grounding encoder embeds
    (``taiji_field.documents.hexagram_documents``) — name + one sentence.
    """
    lines = []
    for h in load_hexagrams():
        lines.append(f"{h.king_wen} {h.name_zh}：{READINGS[h.king_wen]}")
    return "\n".join(lines)


PROMPT_TEMPLATE = """你是一位《易经》解读者。下面是一个真实情境，以及 64 卦的全部信息（卦名 + 一句现代解读）。请判断：这个情境现在最像哪一卦？它接下来最可能动哪几爻、变成哪一卦？

情境：{scenario}

64 卦（编号 卦名：解读）：
{hexagrams}

只输出 JSON，不要任何其他文字：
{{"start_king_wen": <编号>, "start_name": "<卦名>", "moving_lines": [<爻位，1 到 6 之间，1 到 3 个>], "end_king_wen": <编号>, "rationale": "<一句话理由>"}}

要求：moving_lines 只能是 1 到 3 个爻位；end 必须是 start 按动爻翻转得到的卦。"""


def build_prompt(scenario: str) -> str:
    return PROMPT_TEMPLATE.format(scenario=scenario, hexagrams=hexagram_list_text())


def call_llm(prompt: str) -> str:
    """One chat completion via the custom.deepseek connector mechanism."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    req = urllib.request.Request(
        API + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    add_surrogate_to_request(req, "custom.deepseek", allowed_hosts=["api.deepseek.com"])
    try:
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=120) as resp:
            out = read_json_response(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"DeepSeek POST /v1/chat/completions -> {e.code}: {body}") from e
    return out["choices"][0]["message"]["content"]


def parse_prediction(raw: str) -> dict | None:
    """Parse the model's JSON. Returns None if no valid object is found."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    try:
        start = int(obj["start_king_wen"])
        lines = [int(x) for x in obj["moving_lines"]]
        end = int(obj["end_king_wen"])
        rationale = str(obj.get("rationale", ""))
    except (KeyError, TypeError, ValueError):
        return None
    if not (1 <= start <= 64 and 1 <= end <= 64):
        return None
    if not (1 <= len(lines) <= 3 and all(1 <= x <= 6 for x in lines)):
        return None
    if len(set(lines)) != len(lines):
        return None
    return {
        "start_king_wen": start,
        "start_name": str(obj.get("start_name", "")),
        "moving_lines": lines,
        "end_king_wen": end,
        "rationale": rationale,
    }


def compute_end(start_king_wen: int, moving_lines: list[int]) -> dict:
    """End hexagram computed from start + moving lines (never hand-typed,
    never trusted from the model)."""
    by_kw = {h.king_wen: h for h in load_hexagrams()}
    by_bits = {h.bits: h for h in load_hexagrams()}
    bits = by_kw[start_king_wen].bits
    for line in moving_lines:
        bits = flip_bit(bits, line)
    end = by_bits[bits]
    return {"end_king_wen": end.king_wen, "end_name": end.name_zh, "end_bits": end.bits}


def score_prediction(pred: dict | None, proposed: dict) -> dict:
    """Same match definitions as the exam harness (taiji_field/exam.py).

    ``top1``: predicted start hexagram == proposed start.
    ``path``: start AND moving lines both match (start must be right
    for the lines to count — a right line set on a wrong start is a miss).
    An unparsed prediction scores 0 on both.
    """
    if pred is None:
        return {"top1_match": False, "path_match": False}
    top1 = pred["start_king_wen"] == proposed["start_king_wen"]
    path = top1 and pred["moving_lines"] == proposed["moving_lines"]
    return {"top1_match": bool(top1), "path_match": bool(path)}


def run_baseline(scenarios: list[dict], resume_from: dict | None = None,
                 sleep_s: float = 1.0) -> list[dict]:
    """One LLM call per scenario. ``resume_from`` maps scenario id to an
    already-recorded prediction dict, skipped without a new call."""
    done = dict(resume_from or {})
    predictions = []
    for s in scenarios:
        proposed = s["proposed"]
        if s["id"] in done:
            predictions.append(done[s["id"]])
            continue
        prompt = build_prompt(s["scenario"])
        raw = None
        parsed = None
        for attempt in range(2):  # one retry on unparseable output
            raw = call_llm(prompt)
            parsed = parse_prediction(raw)
            if parsed is not None:
                break
            time.sleep(sleep_s)
        computed = compute_end(parsed["start_king_wen"], parsed["moving_lines"]) if parsed else None
        scores = score_prediction(parsed, proposed)
        predictions.append(
            {
                "id": s["id"],
                "category": s["category"],
                "scenario": s["scenario"],
                "proposed_start_king_wen": proposed["start_king_wen"],
                "proposed_start_name": proposed["start_name"],
                "proposed_moving_lines": proposed["moving_lines"],
                "proposed_end_king_wen": proposed["end_king_wen"],
                "predicted_start_king_wen": parsed["start_king_wen"] if parsed else None,
                "predicted_start_name": parsed["start_name"] if parsed else None,
                "predicted_moving_lines": parsed["moving_lines"] if parsed else None,
                "predicted_end_king_wen_llm": parsed["end_king_wen"] if parsed else None,
                "predicted_end_king_wen_computed": computed["end_king_wen"] if computed else None,
                "predicted_end_name_computed": computed["end_name"] if computed else None,
                "end_agrees_with_llm": (
                    parsed["end_king_wen"] == computed["end_king_wen"] if computed else None
                ),
                "rationale": parsed["rationale"] if parsed else None,
                "parsed": parsed is not None,
                "raw_response": raw,
                **scores,
            }
        )
        print(
            f"{s['id']}: pred #{predictions[-1]['predicted_start_king_wen']} "
            f"lines {predictions[-1]['predicted_moving_lines']} "
            f"top1={scores['top1_match']} path={scores['path_match']}",
            flush=True,
        )
        time.sleep(sleep_s)
    return predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LLM baseline on exam v1.")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N scenarios")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true",
                        help="reuse predictions already in the output file")
    args = parser.parse_args(argv)
    scenarios = load_exam()["scenarios"]
    if args.limit:
        scenarios = scenarios[: args.limit]
    resume_from = None
    out_path = Path(args.output)
    if args.resume and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        resume_from = {p["id"]: p for p in prev.get("predictions", [])}
        print(f"resuming: {len(resume_from)} predictions already recorded")
    predictions = run_baseline(scenarios, resume_from=resume_from)
    summary = {
        "n": len(predictions),
        "top1_matches": sum(p["top1_match"] for p in predictions),
        "path_matches": sum(p["path_match"] for p in predictions),
        "unparsed": sum(1 for p in predictions if not p["parsed"]),
    }
    report = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "method": (
            "One chat completion per scenario. Prompt = scenario text + all 64 "
            "hexagrams (name + one-line reading from taiji_field/documents.py) — "
            "exactly the information the grounding encoder embeds, no more. "
            "End hexagram computed from start + moving lines, not trusted from "
            "the model. Scored with the same top-1/path definitions as "
            "taiji_field/exam.py."
        ),
        "prompt_template": PROMPT_TEMPLATE,
        "hexagram_list": hexagram_list_text(),
        "label_status": "draft — proposed labels awaiting Wen's review, not ground truth",
        "summary": summary,
        "predictions": predictions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"baseline ({MODEL}): {summary['top1_matches']}/{summary['n']} top-1, "
          f"{summary['path_matches']}/{summary['n']} path, "
          f"{summary['unparsed']} unparsed")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
