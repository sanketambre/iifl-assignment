"""The customer-facing interface.

Written as a support chat rather than a form, because that is what a customer
recognises. The agent's internal vocabulary - escalate, grounded, retrieval
score - never reaches the page; a customer is told in plain words whether they
have an answer or whether a person is picking it up. The diagnostics that
vocabulary comes from are one click away for anyone reviewing the system.
"""

from __future__ import annotations

import streamlit as st

from src.backend import config
from src.backend.main import SupportAgent
from src.backend.query_log import log

SAMPLE_QUESTIONS = [
    "Will I be charged a foreclosure fee if I close my home loan early?",
    "How do I update the mobile number on my loan account?",
    "My EMI auto debit failed last month. What charges will I pay?",
    "Can I move my EMI date to the 20th of every month?",
]

# What the customer is told when the agent hands the question on. The internal
# reason is precise; these are the same facts without the jargon.
HANDOVER_REASON = {
    "empty_input": "It looks like the question came through empty.",
    "input_too_long": "That question was a little too long for me to read.",
    "no_relevant_policy_found": "This isn't covered by the policy documents I have access to.",
    "model_reported_insufficient_information":
        "The policies I have don't fully answer this one.",
    "low_confidence": "I wasn't confident enough that I had the right policy for this.",
    "llm_unavailable": "My answering service is temporarily unavailable.",
    "invalid_model_output": "I couldn't put together an answer I trust.",
}


@st.cache_resource
def get_agent() -> SupportAgent:
    """Built once per session, so the policy index is not rebuilt on every rerun."""
    return SupportAgent()


def show_answer(response) -> None:
    data = response.to_dict()

    if response.action == "respond":
        st.markdown(response.answer)
        document, _, section = response.source.partition(" / ")
        st.caption("Based on {} in policy {}".format(section, document))
    else:
        reason = response.diagnostics.get("escalation_reason", "")
        st.markdown(
            "I'd rather not guess on this one, so I've passed it to one of our "
            "support agents. They'll come back to you."
        )
        st.caption(HANDOVER_REASON.get(reason, "I couldn't answer this confidently."))

    with st.expander("Technical details"):
        st.json(data)


def render() -> None:
    st.set_page_config(page_title="Customer Support", page_icon="*", layout="centered")

    st.title("Customer Support")
    st.caption(
        "Ask about loan prepayment and foreclosure, KYC and account updates, or "
        "EMIs and payments. Every answer is taken from our published policies, "
        "and anything they don't cover goes to a person."
    )

    with st.sidebar:
        st.subheader("About this assistant")
        st.write(
            "It answers only from three policy documents and cites the section "
            "it used. When the documents don't support a confident answer it "
            "hands the question to a human rather than guessing."
        )
        st.divider()
        st.caption("Model: {}".format(config.GEMINI_MODEL))
        if st.button("Clear conversation"):
            st.session_state.history = []
            st.rerun()

    if "history" not in st.session_state:
        st.session_state.history = []

    # Sample questions, shown only on an empty conversation so they do not
    # clutter the page once someone is actually using it.
    asked = None
    if not st.session_state.history:
        st.write("**Try one of these**")
        for index, question in enumerate(SAMPLE_QUESTIONS):
            if st.button(question, key="sample_{}".format(index), use_container_width=True):
                asked = question

    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            show_answer(entry["response"])

    typed = st.chat_input("Ask a question about your loan or account")
    question = typed or asked
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the policy documents..."):
            response = get_agent().answer(question)
            log(response)
        show_answer(response)

    st.session_state.history.append({"question": question, "response": response})
