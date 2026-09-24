"""The usage log: every question asked through the UI must be recorded."""

from __future__ import annotations

import csv

from src.backend.main import AgentResponse
from src.backend.query_log import COLUMNS, log


def _response(query="Will I be charged a fee?", action="respond"):
    return AgentResponse(
        query=query, category="loan_prepayment", answer="The charge is Nil.",
        source="POL-PREPAY-01 / Foreclosure charges", confidence="high", action=action,
        diagnostics={"retrieval_score": 0.55, "model": "gemini-3.6-flash",
                     "latency_ms": 1200, "citation_valid": True},
    )


def test_writes_a_header_then_one_row_per_question(tmp_path):
    target = tmp_path / "query_log.csv"
    log(_response("first question"), path=target)
    log(_response("second question"), path=target)

    rows = list(csv.DictReader(open(str(target), encoding="utf-8")))
    assert len(rows) == 2
    assert [r["query"] for r in rows] == ["first question", "second question"]
    assert list(rows[0].keys()) == COLUMNS


def test_row_carries_the_answer_and_its_signals(tmp_path):
    target = tmp_path / "query_log.csv"
    log(_response(), path=target)

    row = list(csv.DictReader(open(str(target), encoding="utf-8")))[0]
    assert row["action"] == "respond"
    assert row["confidence"] == "high"
    assert row["source"] == "POL-PREPAY-01 / Foreclosure charges"
    assert row["retrieval_score"] == "0.55"
    assert row["latency_ms"] == "1200"
    assert row["timestamp"]


def test_escalations_are_logged_too(tmp_path):
    target = tmp_path / "query_log.csv"
    log(_response(action="escalate"), path=target)
    assert list(csv.DictReader(open(str(target), encoding="utf-8")))[0]["action"] == "escalate"


def test_a_logging_failure_never_breaks_the_request(tmp_path):
    """A customer must not lose an answer because the log could not be written."""
    unwritable = tmp_path / "missing" / "nested"
    unwritable.parent.mkdir()
    unwritable.parent.chmod(0o500)  # read+execute, no write
    try:
        log(_response(), path=unwritable / "query_log.csv")  # must not raise
    finally:
        unwritable.parent.chmod(0o700)


def test_appending_does_not_duplicate_the_header(tmp_path):
    target = tmp_path / "query_log.csv"
    for _ in range(3):
        log(_response(), path=target)
    lines = open(str(target), encoding="utf-8").read().strip().splitlines()
    assert lines[0].startswith("timestamp")
    assert sum(1 for line in lines if line.startswith("timestamp")) == 1
