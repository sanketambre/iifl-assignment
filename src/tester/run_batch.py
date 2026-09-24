#!/usr/bin/env python3
"""Run a file of questions through the agent and write the results to a CSV.

    python -m src.tester.run_batch                        # the 5 sample questions
    python -m src.tester.run_batch --input questions.txt  # one question per line

This is how I check the agent against the real model rather than the test stub.
Each row holds the answer and the signals that produced it, so a run can be read
in a spreadsheet, and results land in src/tester/outputs/. It exits non-zero if a
graded question misses its expected action, so it could gate a CI job.

The pacing exists because Gemini's free tier allows five requests a minute; left
unpaced, a five-question run trips the limit and every answer escalates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.backend import config
from src.backend.main import SupportAgent
from src.backend.query_log import BASE_COLUMNS, flatten

# The shared answer columns, plus what only a graded batch run knows.
COLUMNS = ["id", "expected_action", "matched"] + BASE_COLUMNS


def load_questions(path: str) -> List[Dict[str, Any]]:
    """Accepts the sample JSON format, or a plain one-question-per-line file."""
    if not os.path.exists(path):
        raise SystemExit("Input file not found: {}".format(path))

    if path.endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            items = json.load(handle)
        return [{"id": q.get("id", "Q{}".format(n + 1)), "question": q["question"],
                 "expected": q.get("expected_action", "")} for n, q in enumerate(items)]

    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    return [{"id": "Q{}".format(n + 1), "question": q, "expected": ""}
            for n, q in enumerate(lines)]


def to_row(item: Dict[str, Any], response) -> Dict[str, Any]:
    expected = item["expected"]
    row = {
        "id": item["id"],
        "expected_action": expected,
        # Blank, not a misleading FAIL, when no expectation was supplied.
        "matched": "" if not expected else str(response.action == expected),
    }
    row.update(flatten(response))
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Batch-run questions and write a CSV")
    parser.add_argument("--input", default=str(config.QUESTIONS_FILE))
    parser.add_argument("--out", help="CSV path (default: outputs/results_<timestamp>.csv)")
    parser.add_argument("--delay", type=float, default=13.0,
                        help="Seconds between questions. The free tier allows 5 "
                             "requests/minute, so the default stays just under it.")
    args = parser.parse_args(argv)
    delay = args.delay

    questions = load_questions(args.input)
    out_path = args.out
    if not out_path:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, "results_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M%S")))

    agent = SupportAgent()
    rows = []
    started = time.time()
    print("Running {} question(s) via {}...\n".format(len(questions), config.GEMINI_MODEL))

    for position, item in enumerate(questions):
        if position and delay:
            time.sleep(delay)
        row = to_row(item, agent.answer(item["question"]))
        rows.append(row)
        print("{:<4} {:<9} {:<7} {:<9} {}".format(
            row["id"], row["action"], row["confidence"],
            {"True": "ok", "False": "MISMATCH", "": "-"}[row["matched"]],
            item["question"][:52]))
        if row["error"]:
            print("      error: {}".format(str(row["error"])[:88]))

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    graded = [r for r in rows if r["matched"]]
    passed = sum(1 for r in graded if r["matched"] == "True")
    errors = sum(1 for r in rows if r["error"])

    print("\n" + "-" * 58)
    print("wrote {} row(s) to {}  ({:.1f}s)".format(len(rows), out_path, time.time() - started))
    if graded:
        print("expected action matched: {}/{}".format(passed, len(graded)))
    if errors:
        print("rows with an LLM error: {}".format(errors))
    return 0 if (not graded or passed == len(graded)) and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
