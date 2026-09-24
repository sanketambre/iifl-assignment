"""The LLM boundary: prompt construction, the response schema, error handling.

These tests never hit the network. Where a call is needed, `_call_gemini` is
patched. The live API is covered by `python app.py --check`, deliberately a
manual step so the suite cannot fail because of quota or an outage.
"""

from __future__ import annotations

import pytest

from src.backend import config
from src.backend.llm import LLMError, RESPONSE_SCHEMA, build_prompt, generate


def test_prompt_contains_the_question_and_every_citation(index):
    chunks = index.search("foreclosure charges", 4)
    prompt = build_prompt("Will I be charged?", chunks)
    assert "Will I be charged?" in prompt
    for chunk in chunks:
        assert chunk.chunk.citation in prompt


def test_prompt_is_explicit_when_nothing_was_retrieved():
    assert "no relevant excerpts" in build_prompt("anything", [])


def test_response_schema_matches_the_agreed_contract():
    assert set(RESPONSE_SCHEMA["required"]) == {
        "category", "answer", "source", "grounded",
        "insufficient_information", "self_confidence",
    }
    assert RESPONSE_SCHEMA["properties"]["category"]["enum"] == config.CATEGORIES
    assert RESPONSE_SCHEMA["properties"]["self_confidence"]["enum"] == config.CONFIDENCES


def test_a_missing_api_key_fails_loudly(monkeypatch):
    """There is no offline fallback, so a missing key must say so plainly."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    with pytest.raises(LLMError) as raised:
        generate("any question", [])
    assert "GEMINI_API_KEY" in str(raised.value)


def test_transient_errors_are_retryable_and_permanent_ones_are_not():
    assert LLMError("overloaded", retryable=True).retryable is True
    assert LLMError("bad key").retryable is False


def test_retry_hint_is_carried_on_the_error():
    assert LLMError("rate limited", retryable=True, retry_after=27.0).retry_after == 27.0


def test_generate_gives_up_after_max_retries(monkeypatch):
    """A permanently failing API must raise, not loop forever."""
    calls = []

    def always_fails(question, chunks):
        calls.append(1)
        raise LLMError("still overloaded", retryable=True)

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(config, "RETRY_BACKOFF", 0)  # keep the test fast
    monkeypatch.setattr("src.backend.llm._call_gemini", always_fails)

    with pytest.raises(LLMError):
        generate("question", [])
    assert len(calls) == config.MAX_RETRIES


def test_permanent_errors_are_not_retried(monkeypatch):
    calls = []

    def bad_key(question, chunks):
        calls.append(1)
        raise LLMError("Gemini rejected the API key.")

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr("src.backend.llm._call_gemini", bad_key)

    with pytest.raises(LLMError):
        generate("question", [])
    assert len(calls) == 1


def test_a_successful_call_is_returned_unchanged(monkeypatch):
    payload = {
        "category": "loan_prepayment", "answer": "The charge is Nil.",
        "source": "POL-PREPAY-01 / Foreclosure charges", "grounded": True,
        "insufficient_information": False, "self_confidence": "high",
    }
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr("src.backend.llm._call_gemini", lambda q, c: payload)
    assert generate("question", []) == payload
