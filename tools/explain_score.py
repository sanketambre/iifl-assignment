#!/usr/bin/env python3
"""Show how a retrieval score is derived, term by term.

    python tools/explain_score.py "will I be charged a foreclosure fee?"

Prints every query term, how rare it is in the corpus, what that rarity is
worth, and whether the winning section contains it - then the arithmetic that
turns those into the final 0-1 score.

This reaches into PolicyIndex internals on purpose: it is a diagnostic tool,
and the point of keyword scoring is that the derivation can be shown.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend import config
from src.backend.retrieval import build_index, tokenize


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit('Usage: python tools/explain_score.py "your question"')
    question = " ".join(sys.argv[1:])

    index = build_index()
    total_sections = len(index.chunks)
    results = index.search(question, config.TOP_K)
    if not results:
        print("No section matched. The agent escalates without calling the model.")
        return 0

    winner = results[0]
    position = index.chunks.index(winner.chunk)
    winner_tokens = index._tokens[position]

    wanted = set(tokenize(question, expand=True))
    typed = set(tokenize(question))

    print("\nQuestion: {}".format(question))
    print("Corpus:   {} sections\n".format(total_sections))
    print("Winning section: {}\n".format(winner.chunk.citation))

    print("{:<16}{:>10}{:>10}  {:<10}{}".format(
        "term", "sections", "weight", "matched", "source"))
    print("-" * 62)

    matched_mass = 0.0
    total_mass = 0.0
    for term in sorted(wanted, key=lambda t: -index._weight(t)):
        frequency = sum(1 for tokens in index._tokens if term in tokens)
        weight = index._weight(term)
        hit = term in winner_tokens
        total_mass += weight
        if hit:
            matched_mass += weight
        print("{:<16}{:>10}{:>10.2f}  {:<10}{}".format(
            term, frequency, weight, "yes" if hit else "-",
            "typed" if term in typed else "synonym"))

    boosted = bool(wanted & set(tokenize(winner.chunk.section)))
    print("-" * 62)
    print("matched weight        {:.2f}".format(matched_mass))
    print("total query weight    {:.2f}".format(total_mass))
    print("heading boost         {}".format("x1.25 (query words in the heading)"
                                            if boosted else "none"))
    raw = matched_mass / total_mass * (1.25 if boosted else 1.0)
    print("score                 {:.2f} / {:.2f}{} = {:.3f}{}".format(
        matched_mass, total_mass, " x 1.25" if boosted else "", raw,
        "  (capped at 1.0)" if raw > 1 else ""))

    band = ("high" if winner.score >= config.HIGH_SCORE
            else "medium" if winner.score >= config.MEDIUM_SCORE else "low")
    print("\nfinal score {:.3f} -> retrieval confidence '{}'  "
          "(high >= {}, medium >= {}, escalate < {})".format(
              winner.score, band, config.HIGH_SCORE, config.MEDIUM_SCORE, config.MIN_SCORE))

    print("\nAll sections passed to the model:")
    for result in results:
        print("  {:.3f}  {}".format(result.score, result.chunk.citation))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
