"""The end-to-end workflow: the supplied examples, and every failure case."""

from __future__ import annotations

import pytest

from src.backend.llm import LLMError
from src.backend.main import validate

REQUIRED_FIELDS = {"query", "category", "answer", "source", "confidence", "action"}


# --- the supplied examples --------------------------------------------------

def test_every_sample_produces_the_expected_action(agent, samples):
    for sample in samples:
        response = agent.answer(sample["question"])
        assert response.action == sample["expected_action"], sample["id"]


def test_every_sample_satisfies_the_output_contract(agent, samples):
    for sample in samples:
        response = agent.answer(sample["question"])
        assert validate(response) == [], sample["id"]
        assert REQUIRED_FIELDS.issubset(response.to_dict()), sample["id"]


def test_answered_questions_cite_a_real_section(agent, samples):
    citations = {chunk.citation for chunk in agent.index.chunks}
    for sample in samples:
        response = agent.answer(sample["question"])
        if response.action == "respond":
            assert response.source in citations, sample["id"]


def test_escalations_never_carry_a_source(agent, samples):
    for sample in samples:
        response = agent.answer(sample["question"])
        if response.action == "escalate":
            assert response.source == ""
            assert response.confidence == "low"


# --- failure cases ----------------------------------------------------------

@pytest.mark.parametrize("bad_input", ["", "   ", "\n\t"])
def test_empty_input_escalates(agent, bad_input):
    response = agent.answer(bad_input)
    assert response.action == "escalate"
    assert response.diagnostics["escalation_reason"] == "empty_input"


def test_gibberish_escalates_without_calling_the_model(agent):
    response = agent.answer("qwertyuiop zxcvbnm asdfgh")
    assert response.diagnostics["escalation_reason"] == "no_relevant_policy_found"


def test_out_of_scope_question_escalates(agent):
    response = agent.answer("What is the weather in Mumbai tomorrow?")
    assert response.action == "escalate"


def test_overlong_input_escalates(agent):
    response = agent.answer("foreclosure " * 400)
    assert response.diagnostics["escalation_reason"] == "input_too_long"


def test_llm_failure_escalates_instead_of_raising(agent, monkeypatch):
    def boom(question, chunks):
        raise LLMError("simulated network failure")

    monkeypatch.setattr("src.backend.main.generate", boom)
    response = agent.answer("Will I be charged a foreclosure fee?")
    assert response.action == "escalate"
    assert response.diagnostics["escalation_reason"] == "llm_unavailable"


def test_fabricated_citation_is_rejected(agent, monkeypatch):
    """A hallucinated source must not reach the customer as a confident answer."""
    def fabricate(question, chunks):
        return {
            "category": "loan_prepayment", "answer": "Charges are waived entirely.",
            "source": "POL-DOES-NOT-EXIST / Invented section", "grounded": True,
            "insufficient_information": False, "self_confidence": "high",
        }

    monkeypatch.setattr("src.backend.main.generate", fabricate)
    response = agent.answer("Will I be charged a foreclosure fee?")
    assert response.action == "escalate"
    assert response.diagnostics["citation_valid"] is False


def test_model_reporting_insufficient_information_escalates(agent, monkeypatch):
    def unsure(question, chunks):
        return {
            "category": "loan_prepayment", "answer": "I am not sure.",
            "source": chunks[0].chunk.citation, "grounded": True,
            "insufficient_information": True, "self_confidence": "high",
        }

    monkeypatch.setattr("src.backend.main.generate", unsure)
    response = agent.answer("Will I be charged a foreclosure fee?")
    assert response.action == "escalate"


def test_ungrounded_answer_escalates_even_with_a_valid_citation(agent, monkeypatch):
    def ungrounded(question, chunks):
        return {
            "category": "loan_prepayment", "answer": "Probably no charge.",
            "source": chunks[0].chunk.citation, "grounded": False,
            "insufficient_information": False, "self_confidence": "high",
        }

    monkeypatch.setattr("src.backend.main.generate", ungrounded)
    response = agent.answer("Will I be charged a foreclosure fee?")
    assert response.action == "escalate"


def test_empty_answer_violates_the_contract(agent, monkeypatch):
    def blank(question, chunks):
        return {
            "category": "loan_prepayment", "answer": "   ",
            "source": chunks[0].chunk.citation, "grounded": True,
            "insufficient_information": False, "self_confidence": "high",
        }

    monkeypatch.setattr("src.backend.main.generate", blank)
    response = agent.answer("Will I be charged a foreclosure fee?")
    assert response.action == "escalate"
    assert response.diagnostics["escalation_reason"] == "invalid_model_output"


def test_service_outage_reads_differently_from_a_policy_gap(agent, monkeypatch):
    """A customer must not be told their question is unanswerable when the API
    simply could not be reached."""
    def boom(question, chunks):
        raise LLMError("simulated outage", retryable=True)

    monkeypatch.setattr("src.backend.main.generate", boom)
    outage = agent.answer("Will I be charged a foreclosure fee?")
    gap = agent.answer("What is the weather in Mumbai tomorrow?")
    assert "temporarily unavailable" in outage.answer
    assert outage.answer != gap.answer
