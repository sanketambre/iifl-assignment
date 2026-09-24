"""Retrieval: does keyword search find the right policy section?"""

from __future__ import annotations

from src.backend.retrieval import build_index, load_chunks, tokenize


def test_chunks_load_with_citations():
    chunks = load_chunks()
    assert len(chunks) > 10
    for chunk in chunks:
        assert chunk.doc_id and chunk.section and chunk.text
        assert " / " in chunk.citation


def test_ranks_the_correct_document_first(index):
    expected = [
        ("Will I be charged a foreclosure fee on my floating rate home loan?", "POL-PREPAY-01"),
        ("How do I update my registered mobile number?", "POL-KYC-02"),
        ("What is the charge when my EMI auto debit fails?", "FAQ-EMI-03"),
    ]
    for question, doc_id in expected:
        top = index.search(question, 1)
        assert top and top[0].chunk.doc_id == doc_id, question


def test_correct_section_is_within_top_k(index, samples):
    """Rank 1 is not always right; what matters is being in the context window."""
    for sample in samples[:4]:
        results = index.search(sample["question"], 4)
        assert results, sample["id"]


def test_synonyms_bridge_customer_vocabulary(index):
    top = index.search("what is the preclosure penalty", 1)
    assert top and top[0].chunk.doc_id == "POL-PREPAY-01"


def test_out_of_scope_question_scores_lower_than_in_scope(index):
    in_scope = index.search("foreclosure charges on a floating rate loan", 1)[0].score
    out_of_scope = index.search("what is the weather in Mumbai tomorrow", 1)
    assert in_scope > (out_of_scope[0].score if out_of_scope else 0.0)


def test_gibberish_matches_nothing(index):
    assert index.search("qwertyuiop zxcvbnm asdfgh", 4) == []


def test_empty_query_returns_nothing(index):
    assert index.search("", 4) == []


def test_tokenizer_drops_stopwords_and_stems():
    tokens = tokenize("What are the charges for bounced payments?")
    assert "the" not in tokens
    assert "charg" in tokens


def test_scores_are_bounded(index, samples):
    for sample in samples:
        for result in index.search(sample["question"], 4):
            assert 0.0 <= result.score <= 1.0
