"""Shared test setup.

The shipped code has no offline mode: `generate` always calls Gemini. So the
tests substitute their own deterministic stub, which keeps every test offline,
free and repeatable, and means a flaky API or an exhausted quota can never fail
a test run.

I moved the stub here on purpose. It was originally a "mock provider" inside the
backend, which meant test scaffolding was shipping with the product. A stand-in
for the model belongs in the tests that need it.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.backend import config
from src.backend.main import SupportAgent
from src.backend.retrieval import build_index

# The stub cannot judge grounding the way a model does, so it approximates with
# a retrieval-score cutoff. Good enough to exercise the workflow; not an
# evaluation baseline.
STUB_GROUNDING_THRESHOLD = 0.35

_CATEGORY_BY_DOC = {
    "POL-PREPAY-01": "loan_prepayment",
    "POL-KYC-02": "kyc_and_account_update",
    "FAQ-EMI-03": "emi_and_payments",
}


def stub_generate(question, chunks):
    """Stand in for a Gemini call: answer from the top chunk, or report that
    the excerpts are insufficient."""
    if not chunks or chunks[0].score < STUB_GROUNDING_THRESHOLD:
        return {
            "category": "other", "answer": config.ESCALATION_MESSAGE, "source": "",
            "grounded": False, "insufficient_information": True, "self_confidence": "low",
        }

    top = chunks[0]
    body = " ".join(line.strip() for line in top.chunk.text.splitlines() if line.strip())
    return {
        "category": _CATEGORY_BY_DOC.get(top.chunk.doc_id, "other"),
        "answer": "[stub] {}".format(body[:400]),
        "source": top.chunk.citation,
        "grounded": True,
        "insufficient_information": False,
        "self_confidence": "high" if top.score >= config.HIGH_SCORE else "medium",
    }


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every test runs against the stub. Nothing here touches the network."""
    monkeypatch.setattr("src.backend.main.generate", stub_generate)


@pytest.fixture(scope="session")
def index():
    return build_index()


@pytest.fixture(scope="session")
def agent(index):
    return SupportAgent(index=index)


@pytest.fixture(scope="session")
def samples():
    with open(config.QUESTIONS_FILE, encoding="utf-8") as handle:
        return json.load(handle)
