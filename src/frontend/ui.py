"""The customer-facing interface.

Written as a support chat rather than a form, because that is what a customer
recognises. The agent's internal vocabulary - escalate, grounded, retrieval
score - never reaches the page; a customer is told in plain words whether they
have an answer or whether a person is picking it up. The diagnostics that
vocabulary comes from are one click away for anyone reviewing the system.

Each policy document gets its own page, so a citation on an answer is something
the customer can actually go and read rather than take on trust.
"""

from __future__ import annotations

import re

import streamlit as st

from src.backend import config
from src.backend.main import SupportAgent
from src.backend.query_log import log

SAMPLE_QUESTIONS = [
    "Will I be charged a foreclosure fee if I close my home loan early?",
    "How do I update the mobile number on my loan account?",
    "My EMI auto debit failed last month. What charges will I pay?",
    "Can I move my EMI date to the 20th of every month?",
    "How long does it take to get my NOC after I close the loan?",
    "Which documents are accepted as proof of identity?",
    "Is there a grace period if I pay my EMI a day late?",
    "Can I pay my EMI with a credit card?",
    "How much can I part-prepay without a charge?",
    "How often do I need to complete re-KYC?",
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


@st.cache_resource
def get_documents() -> dict:
    """The indexed sections grouped by document, one entry per policy."""
    documents: dict = {}
    for chunk in get_agent().index.chunks:
        entry = documents.setdefault(chunk.doc_title, {
            "doc_id": chunk.doc_id, "file_name": chunk.file_name, "sections": []})
        entry["sections"].append(chunk)
    return documents


def show_answer(response) -> None:
    if response.action == "respond":
        st.markdown(response.answer)
        document, _, section = response.source.partition(" / ")
        st.caption("Based on {} in policy {}. You can read it from the menu on "
                   "the left.".format(section, document))
    else:
        reason = response.diagnostics.get("escalation_reason", "")
        st.markdown(
            "I'd rather not guess on this one, so I've passed it to one of our "
            "support agents. They'll come back to you."
        )
        st.caption(HANDOVER_REASON.get(reason, "I couldn't answer this confidently."))

    with st.expander("Technical details"):
        st.json(response.to_dict())


def chat_page() -> None:
    st.title("Customer Support")
    st.caption(
        "Ask about loan prepayment and foreclosure, KYC and account updates, or "
        "EMIs and payments. Every answer is taken from our published policies, "
        "and anything they don't cover goes to a person."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.caption("Answers come only from the policies listed above.")
        if st.session_state.history and st.button("Start a new conversation"):
            st.session_state.history = []
            st.rerun()

    # Open on an empty conversation and collapsed afterwards, so the suggestions
    # lead the way in but stay reachable once someone is mid-conversation. Two
    # columns, because ten full-width buttons would push the chat off screen.
    asked = None
    with st.expander("Suggested questions",
                     expanded=not st.session_state.history):
        columns = st.columns(2)
        for index, question in enumerate(SAMPLE_QUESTIONS):
            with columns[index % 2]:
                if st.button(question, key="sample_{}".format(index),
                             use_container_width=True):
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


def document_page(title: str) -> None:
    """One policy, in full.

    Section bodies are written with st.text rather than markdown so the charge
    tables keep the column alignment they have in the PDF.
    """
    entry = get_documents()[title]
    st.title(title)
    st.caption("Policy {} - {} sections".format(entry["doc_id"], len(entry["sections"])))

    path = config.POLICY_DIR / entry["file_name"]
    if path.exists():
        st.download_button("Download the PDF", path.read_bytes(),
                           file_name=entry["file_name"], mime="application/pdf")

    st.divider()
    for chunk in entry["sections"]:
        st.subheader(chunk.section)
        st.text(chunk.text)


def build_navigation():
    """The chat, plus one page per policy document."""
    pages = [st.Page(chat_page, title="Ask a question", default=True, url_path="ask")]

    for title in get_documents():
        # A closure per document, each with its own URL so Streamlit can tell
        # the pages apart.
        def page(bound_title: str = title) -> None:
            document_page(bound_title)

        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        pages.append(st.Page(page, title=title, url_path=slug))

    return st.navigation({"Support": pages[:1], "Policy documents": pages[1:]})
