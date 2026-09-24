"""Flatten a response into CSV columns, and append it to the usage log.

Every question asked through the UI lands here with the answer and the signals
behind it, so a real run can be reviewed in a spreadsheet afterwards. I put the
flattening in this file rather than in each caller so the usage log and the
batch runner cannot drift apart.

Two choices worth stating:
  - Logging never breaks a request. If the file cannot be written I print a
    warning and carry on, because a customer should not lose an answer over a
    full disk.
  - This log holds raw customer questions, which in production means PII. It is
    gitignored, and redacting before the write is on my list of things to do
    before this went anywhere near production.
"""

from __future__ import annotations

import csv
import os
import sys
import threading
from datetime import datetime
from typing import Any, Dict

from src.backend import config

LOG_DIR = config.PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "query_log.csv"

# The answer, plus every signal that produced it.
BASE_COLUMNS = ["query", "action", "confidence", "category", "source", "answer",
                "retrieval_score", "retrieval_confidence", "model_confidence",
                "model_grounded", "citation_valid", "insufficient_information",
                "escalation_reason", "model", "latency_ms", "error"]

_DIAGNOSTIC_KEYS = ["retrieval_score", "retrieval_confidence", "model_confidence",
                    "model_grounded", "citation_valid", "insufficient_information",
                    "escalation_reason", "model", "latency_ms", "error"]

COLUMNS = ["timestamp"] + BASE_COLUMNS

# Flask serves requests on threads, so serialise writes.
_lock = threading.Lock()


def flatten(response) -> Dict[str, Any]:
    """One response as a flat dict of BASE_COLUMNS. Shared by both CSV writers."""
    diagnostics = response.diagnostics
    row = {
        "query": response.query,
        "action": response.action,
        "confidence": response.confidence,
        "category": response.category,
        "source": response.source,
        "answer": response.answer,
    }
    for key in _DIAGNOSTIC_KEYS:
        row[key] = diagnostics.get(key, "")
    return row


def log(response, path=None) -> None:
    """Append one response to the usage log. Never raises."""
    target = str(path or LOG_FILE)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds")}
    row.update(flatten(response))

    try:
        with _lock:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            is_new = not os.path.exists(target) or os.path.getsize(target) == 0
            with open(target, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                if is_new:
                    writer.writeheader()
                writer.writerow(row)
    except OSError as exc:
        print("[warn] could not write query log: {}".format(exc), file=sys.stderr)
