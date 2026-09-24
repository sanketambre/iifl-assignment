#!/usr/bin/env python3
"""Local web app for the policy-aware support agent.

    python app.py              # then open http://127.0.0.1:5001
    python app.py --check      # verify the API key and model, then exit

The agent is built once at startup and reused, so the policy index is not
rebuilt on every request.
"""

from __future__ import annotations

import argparse
import sys

from flask import Flask, jsonify, render_template, request

from src.backend import config
from src.backend.llm import LLMError, generate
from src.backend.main import SupportAgent
from src.backend.query_log import LOG_FILE, log

app = Flask(
    __name__,
    template_folder="src/frontend/templates",
    static_folder="src/frontend/static",
)

agent = SupportAgent()


@app.route("/")
def index():
    return render_template("index.html", model=config.GEMINI_MODEL)


@app.route("/ask", methods=["POST"])
def ask():
    """Answer one question. Always returns JSON, never an HTML error page."""
    question = (request.json or {}).get("question", "")
    try:
        response = agent.answer(question)
        log(response)  # every question asked through the UI is recorded
        return jsonify(response.to_dict())
    except Exception as exc:  # the UI should show a message, not a stack trace
        return jsonify({
            "query": question, "category": "other",
            "answer": "Something went wrong: {}".format(exc),
            "source": "", "confidence": "low", "action": "escalate",
            "diagnostics": {"escalation_reason": "server_error"},
        }), 500


def check() -> int:
    """Confirm the key and model work before spending a run on them."""
    print("model: {}\nkey:   {}".format(
        config.GEMINI_MODEL, "set" if config.GEMINI_API_KEY else "not set"))
    try:
        generate("check", [])
        print("\nOK: {} answered successfully.".format(config.GEMINI_MODEL))
        return 0
    except LLMError as exc:
        print("\nFAIL: {}".format(exc))
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the support agent web app")
    parser.add_argument("--check", action="store_true", help="Verify API key and model, then exit")
    # 5000 is taken by macOS ControlCenter (AirPlay Receiver), so default to 5001.
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    if args.check:
        sys.exit(check())

    print("\n  Policy-Aware Support Agent")
    print("  model: {}".format(config.GEMINI_MODEL))
    print("  log:   {}".format(LOG_FILE))
    print("  open:  http://127.0.0.1:{}\n".format(args.port))
    app.run(debug=False, port=args.port)
