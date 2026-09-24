"""The workflow: retrieve, generate, then decide whether to trust the result.

This is a fixed pipeline, not an autonomous agent. Code chooses the policy
sections; the model only writes the answer from what it is given.

Three independent signals must agree before we respond:
  1. retrieval  - did the corpus contain anything relevant?
  2. generation - did the model consider the excerpts sufficient and grounded?
  3. citation   - is the source it cited one we actually supplied?
Any one failing drops confidence to low, and low confidence escalates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.backend import config
from src.backend.llm import LLMError, generate
from src.backend.retrieval import PolicyIndex, ScoredChunk, build_index

_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass
class AgentResponse:
    """The six required fields, plus additive diagnostics so the escalation
    decision can be audited without enabling logging."""

    query: str
    category: str
    answer: str
    source: str
    confidence: str
    action: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query, "category": self.category, "answer": self.answer,
            "source": self.source, "confidence": self.confidence,
            "action": self.action, "diagnostics": self.diagnostics,
        }


def validate(response: AgentResponse) -> List[str]:
    """Contract violations. An empty list means the response is well-formed."""
    problems = []
    if response.category not in config.CATEGORIES:
        problems.append("bad category: {}".format(response.category))
    if response.confidence not in config.CONFIDENCES:
        problems.append("bad confidence: {}".format(response.confidence))
    if response.action not in config.ACTIONS:
        problems.append("bad action: {}".format(response.action))
    if not response.answer.strip():
        problems.append("empty answer")
    if response.action == "respond" and not response.source.strip():
        problems.append("respond without a source")
    return problems


def _band(score: float) -> str:
    if score >= config.HIGH_SCORE:
        return "high"
    return "medium" if score >= config.MEDIUM_SCORE else "low"


def _escalate(query: str, reason: str, diagnostics: Optional[dict] = None,
              message: Optional[str] = None) -> AgentResponse:
    payload = {"escalation_reason": reason}
    payload.update(diagnostics or {})
    return AgentResponse(query, "other", message or config.ESCALATION_MESSAGE,
                         "", "low", "escalate", payload)


class SupportAgent:
    def __init__(self, index: Optional[PolicyIndex] = None):
        self.index = index if index is not None else build_index()

    def answer(self, query: str) -> AgentResponse:
        started = time.time()

        if not query or not query.strip():
            return _escalate("", "empty_input")
        query = query.strip()
        if len(query) > 2000:
            return _escalate(query[:200] + "...", "input_too_long")

        found: List[ScoredChunk] = self.index.search(query, config.TOP_K)
        top = found[0].score if found else 0.0
        info = {
            "retrieval_score": round(top, 3),
            "retrieved": [{"citation": f.chunk.citation, "score": round(f.score, 3)} for f in found],
        }

        # Nothing relevant: escalate without paying for a call that cannot succeed.
        if top < config.MIN_SCORE:
            return _escalate(query, "no_relevant_policy_found", info)

        try:
            result = generate(query, found)
        except LLMError as exc:
            info.update({"model": config.GEMINI_MODEL, "error": str(exc)})
            return _escalate(query, "llm_unavailable", info, config.SERVICE_ERROR_MESSAGE)

        # A citation we never supplied means the answer is not grounded,
        # whatever the model claims.
        cited = (result.get("source") or "").strip()
        citation_ok = cited in {f.chunk.citation for f in found}
        if not citation_ok:
            cited = ""

        model_confidence = result.get("self_confidence", "low")
        if not result.get("grounded", False) or not citation_ok:
            model_confidence = "low"
        confidence = min(_band(top), model_confidence, key=lambda c: _RANK.get(c, 0))
        insufficient = bool(result.get("insufficient_information", False))

        info.update({
            "retrieval_confidence": _band(top),
            "model_confidence": result.get("self_confidence", "low"),
            "model_grounded": bool(result.get("grounded", False)),
            "citation_valid": citation_ok,
            "insufficient_information": insufficient,
            "model": config.GEMINI_MODEL,
            "latency_ms": int((time.time() - started) * 1000),
        })

        if insufficient or confidence == "low":
            reason = "model_reported_insufficient_information" if insufficient else "low_confidence"
            info["escalation_reason"] = reason
            return AgentResponse(query, result.get("category", "other"),
                                 config.ESCALATION_MESSAGE, "", "low", "escalate", info)

        response = AgentResponse(query, result.get("category", "other"),
                                 (result.get("answer") or "").strip(), cited,
                                 confidence, "respond", info)

        # Never emit a response that breaks the contract.
        problems = validate(response)
        if problems:
            info["contract_violations"] = problems
            return _escalate(query, "invalid_model_output", info)
        return response
