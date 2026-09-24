"""The LLM boundary: one function, one implementation.

Everything that knows about Gemini lives in this file, so changing provider is a
one-file change rather than a search through the codebase.

The error handling here is deliberate rather than defensive habit. Building this
I hit overloaded models, rate limits, an exhausted daily quota and a retired
model name, so failures are sorted into transient ones worth retrying with the
server's own suggested delay, and permanent ones where retrying only postpones
telling the user what is actually wrong.

Tests never reach the network path; they patch `generate` with their own stub.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from src.backend import config
from src.backend.retrieval import ScoredChunk

SYSTEM_PROMPT = """You are a customer support assistant for a non-banking financial company.

You answer ONLY from the policy excerpts supplied in the user message.

Rules:
1. Never use outside knowledge. If the excerpts do not contain the answer, set
   insufficient_information to true and do not guess.
2. Partial information is not sufficient information. If the excerpts answer only
   part of the question, answer that part, say what is not covered, and set
   self_confidence to medium.
3. Quote charges and timelines exactly as written. Never round or infer a number
   that is not present.
4. `source` must be the citation string of the excerpt you relied on most, copied
   exactly. If you used nothing, return an empty string.
5. `grounded` is true only if every claim you make is traceable to the excerpts.
6. Never promise an outcome or give pricing, eligibility or legal advice that is
   not written in the excerpts.
7. Under 120 words, plain English, addressed to the customer.
"""

# Handed to Gemini so valid JSON comes back by construction, not by asking nicely.
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "category": {"type": "STRING", "enum": config.CATEGORIES},
        "answer": {"type": "STRING"},
        "source": {"type": "STRING"},
        "grounded": {"type": "BOOLEAN"},
        "insufficient_information": {"type": "BOOLEAN"},
        "self_confidence": {"type": "STRING", "enum": config.CONFIDENCES},
    },
    "required": ["category", "answer", "source", "grounded",
                 "insufficient_information", "self_confidence"],
}


class LLMError(Exception):
    """`retryable` separates transient failures (overload, rate limit) from
    permanent ones (bad key, unknown model). Retrying the latter just delays
    the escalation."""

    def __init__(self, message: str, retryable: bool = False, retry_after: float = 0.0):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


def build_prompt(question: str, chunks: List[ScoredChunk]) -> str:
    excerpts = "\n\n---\n\n".join(
        "[citation: {}]\nSection: {}\n{}".format(c.chunk.citation, c.chunk.section, c.chunk.text)
        for c in chunks
    )
    return "Policy excerpts:\n\n{}\n\n=====\n\nCustomer question: {}".format(
        excerpts or "(no relevant excerpts were found)", question
    )


def _retry_delay(response) -> float:
    """Honour the server's own retry hint on a 429 rather than guessing."""
    try:
        for detail in response.json().get("error", {}).get("details", []):
            delay = detail.get("retryDelay", "")
            if delay.endswith("s"):
                return min(float(delay[:-1]), 60.0)
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    return 0.0


def _call_gemini(question: str, chunks: List[ScoredChunk]) -> Dict[str, Any]:
    import requests

    try:
        # The key travels in a header, never the query string, so it cannot
        # leak into proxy or server access logs.
        response = requests.post(
            config.GEMINI_URL.format(model=config.GEMINI_MODEL),
            headers={"Content-Type": "application/json", "x-goog-api-key": config.GEMINI_API_KEY},
            json={
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": build_prompt(question, chunks)}]}],
                "generationConfig": {
                    "temperature": 0,  # a policy lookup, not creative writing
                    "responseMimeType": "application/json",
                    "responseSchema": RESPONSE_SCHEMA,
                },
            },
            timeout=config.TIMEOUT,
        )
    except Exception as exc:
        raise LLMError("Could not reach Gemini: {}".format(exc), retryable=True)

    status = response.status_code
    if status == 429:
        raise LLMError(
            "Rate limit reached. The free tier allows 5 requests/minute and 20/day per model.",
            retryable=True,
            retry_after=_retry_delay(response),
        )
    if status in (500, 502, 503, 504):
        raise LLMError("Gemini is overloaded (HTTP {}).".format(status), retryable=True)
    if status == 404:
        raise LLMError("Model '{}' not found. Check GEMINI_MODEL in .env.".format(config.GEMINI_MODEL))
    if status in (401, 403):
        raise LLMError("Gemini rejected the API key. Check GEMINI_API_KEY in .env.")
    if status == 402:
        raise LLMError("This project has no credits or free-tier quota. Use a different project.")
    if status != 200:
        raise LLMError("Gemini returned HTTP {}: {}".format(status, response.text[:200]))

    try:
        return json.loads(response.json()["candidates"][0]["content"]["parts"][0]["text"])
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError("Unusable model output: {}".format(exc), retryable=True)


def generate(question: str, chunks: List[ScoredChunk]) -> Dict[str, Any]:
    """Answer the question from the chunks, retrying transient failures."""
    if not config.GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set. Add it to .env before running.")

    error = None
    for attempt in range(config.MAX_RETRIES):
        try:
            return _call_gemini(question, chunks)
        except LLMError as exc:
            error = exc
            if not exc.retryable or attempt == config.MAX_RETRIES - 1:
                break
            time.sleep(exc.retry_after or config.RETRY_BACKOFF * (2 ** attempt))
    raise error
