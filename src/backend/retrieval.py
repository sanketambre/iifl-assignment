"""Keyword search over the policy documents. No LLM, no vector database.

The corpus is three small documents written in the same vocabulary customers
use, so IDF-weighted keyword matching is accurate enough and every score is
explainable. It fails on paraphrase with no shared words; see the README.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.backend.config import POLICY_DIR

STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "that",
    "this", "have", "has", "was", "will", "can", "would", "what", "when",
    "where", "how", "why", "which", "from", "about", "they", "than", "any",
    "all", "been", "does", "did", "get", "want", "need", "please", "every",
}

# Customer wording mapped onto policy wording. Each entry is here because a
# realistic phrasing would otherwise miss.
SYNONYMS = {
    "preclosure": ["foreclosure", "prepayment"],
    "foreclose": ["foreclosure"],
    "prepay": ["prepayment"],
    "close": ["closure", "foreclosure"],
    "early": ["prepayment", "foreclosure"],
    "penalty": ["charge"],
    "fee": ["charge"],
    "bounce": ["dishonoured", "failed"],
    "failed": ["dishonoured", "bounce"],
    "debit": ["auto-debit", "nach", "mandate"],
    "instalment": ["emi"],
    "installment": ["emi"],
    "phone": ["mobile"],
    "number": ["mobile"],
    "change": ["update"],
    "update": ["change"],
    "late": ["overdue"],
}


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    section: str
    text: str
    file_name: str = ""  # so the UI can offer the source PDF for download

    @property
    def citation(self) -> str:
        return "{} / {}".format(self.doc_id, self.section)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


def _stem(token: str) -> str:
    """Crude suffix stripping; good enough for policy vocabulary."""
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[:-3] + "y" if suffix == "ies" else token[: -len(suffix)]
    return token


def tokenize(text: str, expand: bool = False) -> List[str]:
    """Lowercase, drop stopwords, optionally add synonyms, then stem."""
    tokens: List[str] = []
    for word in re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower()):
        word = word.strip("'-")
        if len(word) < 3 or word in STOPWORDS:
            continue
        tokens.append(word)
        if expand:
            tokens.extend(SYNONYMS.get(word, []))
    return [_stem(t) for t in tokens]


# Sections in the policy PDFs are numbered, as policy documents normally are.
# A PDF carries no structural markup, so headings have to be recognised in the
# extracted text, and the numbering is the most reliable signal available that
# also belongs on the page. Body lists use dashes, so nothing inside a section
# can be mistaken for a heading. Third-party PDFs would need font-size detection
# (pdfplumber) instead.
HEADING_PATTERN = re.compile(r"^(\d{1,2})\.\s+(\S.*)$")
MAX_HEADING_LENGTH = 80


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required to read policy PDFs: pip install -r requirements.txt")

    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise ValueError("Could not read {}: {}".format(path.name, exc))


def load_chunks(policy_dir: Optional[Path] = None) -> List[Chunk]:
    """One chunk per marked section of each policy PDF.

    This is the only function that knows what format the documents are in.
    Everything downstream works on Chunk objects.
    """
    directory = Path(policy_dir) if policy_dir else POLICY_DIR
    if not directory.is_dir():
        raise FileNotFoundError("Policy directory not found: {}".format(directory))

    chunks: List[Chunk] = []
    for path in sorted(directory.glob("*.pdf")):
        lines = _read_pdf(path).splitlines()

        # The header block is written in mixed case so it cannot be mistaken
        # for a section heading.
        meta: Dict[str, str] = {}
        for line in lines[:12]:
            if line.startswith("Document ID:"):
                meta["doc_id"] = line.split(":", 1)[1].strip()
            elif not meta.get("title") and line.strip():
                meta["title"] = line.strip()

        doc_id = meta.get("doc_id", path.stem)
        title = meta.get("title", path.stem)

        section, body = None, []
        for line in lines:
            stripped = line.strip()
            match = HEADING_PATTERN.match(stripped)
            if match and len(stripped) <= MAX_HEADING_LENGTH:
                if section and body:
                    chunks.append(
                        Chunk(doc_id, title, section, "\n".join(body).strip(), path.name))
                # The number is document formatting, not part of the name, so
                # citations read "POL-PREPAY-01 / Foreclosure charges".
                section, body = match.group(2).strip(), []
            elif section is not None and stripped:
                body.append(stripped)
        if section and body:
            chunks.append(Chunk(doc_id, title, section, "\n".join(body).strip(), path.name))

    if not chunks:
        raise ValueError(
            "No policy content found in {}. Run: python tools/make_pdfs.py".format(directory)
        )
    return chunks


class PolicyIndex:
    """In-memory IDF index. Built once at startup."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._tokens = [
            set(tokenize("{} {} {}".format(c.doc_title, c.section, c.text))) for c in chunks
        ]

        frequency: Dict[str, int] = {}
        for tokens in self._tokens:
            for token in tokens:
                frequency[token] = frequency.get(token, 0) + 1

        total = len(chunks)
        self._idf = {t: math.log(1 + total / n) for t, n in frequency.items()}
        # Words absent from the corpus count as maximally informative, which is
        # what makes an out-of-scope question score low instead of accidentally high.
        self._unseen = math.log(1 + total / 0.5)

    def _weight(self, token: str) -> float:
        return self._idf.get(token, self._unseen)

    def search(self, query: str, top_k: int) -> List[ScoredChunk]:
        """Top sections, scored 0-1 by how much of the query they cover."""
        wanted = set(tokenize(query, expand=True))
        total = sum(self._weight(t) for t in wanted)
        if not wanted or total <= 0:
            return []

        results = []
        for chunk, tokens in zip(self.chunks, self._tokens):
            matched = wanted & tokens
            if not matched:
                continue
            mass = sum(self._weight(t) for t in matched)
            if wanted & set(tokenize(chunk.section)):
                mass *= 1.25  # a heading match is a strong topical signal
            results.append(ScoredChunk(chunk, min(mass / total, 1.0)))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


def build_index(policy_dir: Optional[Path] = None) -> PolicyIndex:
    return PolicyIndex(load_chunks(policy_dir))
