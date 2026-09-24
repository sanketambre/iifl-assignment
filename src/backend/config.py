"""Settings and the thresholds the agent's decisions depend on."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # env vars still work without python-dotenv installed.
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_DIR = PROJECT_ROOT / "data" / "policies"
QUESTIONS_FILE = PROJECT_ROOT / "data" / "questions.json"

# LLM. A key is required; there is no offline fallback.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
MAX_RETRIES = 3          # transient failures only; permanent ones fail fast.
RETRY_BACKOFF = 1.5      # seconds, doubled each attempt

# Retrieval. The score is a 0-1 measure of how much of the question's
# information a section covers. Thresholds were tuned on the five sample
# questions and would need a real evaluation set before production.
TOP_K = 4
MIN_SCORE = 0.10     # below this we escalate without calling the model
MEDIUM_SCORE = 0.25
HIGH_SCORE = 0.45

CATEGORIES = ["loan_prepayment", "kyc_and_account_update", "emi_and_payments", "other"]
CONFIDENCES = ["high", "medium", "low"]
ACTIONS = ["respond", "escalate"]

ESCALATION_MESSAGE = (
    "I could not answer this confidently from the available policy documents, "
    "so I am routing it to a human support agent."
)

# A service outage is not a policy gap. Saying so honestly matters: the first
# message would tell a customer their question is unanswerable when in fact we
# never managed to ask.
SERVICE_ERROR_MESSAGE = (
    "The answering service is temporarily unavailable, so I am routing this to "
    "a human support agent. Please try again shortly."
)
