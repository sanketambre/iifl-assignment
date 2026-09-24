#!/usr/bin/env python3
"""Generate the policy PDFs in data/policies/ from the Markdown in data/source/.

    python tools/make_pdfs.py

The Markdown is the authoring format; the PDFs are what the agent actually
reads, the same way a real deployment would receive them.

Section headings are written with a leading section sign. PDFs carry no
structural markup, so a reader has to infer where sections begin; because we
control generation here, we mark them explicitly instead of guessing. For PDFs
produced by someone else, heading detection has to come from font size and
position (pdfplumber), which is a good deal less reliable.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
except ImportError:
    sys.exit("fpdf2 is required: pip install -r requirements.txt")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "source"
OUTPUT_DIR = PROJECT_ROOT / "data" / "policies"

HEADING_MARK = "§ "  # section sign, kept in sync with retrieval.load_chunks


def clean(text: str) -> str:
    """Drop Markdown syntax, and anything the core PDF fonts cannot render."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # bold
    text = re.sub(r"`(.+?)`", r"\1", text)         # inline code
    for bad, good in {"’": "'", "‘": "'", "“": '"', "”": '"',
                      "—": "-", "–": "-", "₹": "Rs."}.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "replace").decode("latin-1")


def parse_markdown(raw: str):
    """Return (metadata, [(heading, blocks)]).

    A block is ("para", text), ("item", text) or ("table", [[cells], ...]).
    Grouping happens here so the renderer only has to lay blocks out: the
    Markdown is hard-wrapped, and paragraphs must be rejoined before the PDF
    can wrap them to its own width.
    """
    meta = {}
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            for line in raw[3:end].strip().splitlines():
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
            raw = raw[end + 4:]

    sections, heading, blocks = [], None, []
    paragraph, table = [], []

    def flush():
        if paragraph:
            blocks.append(("para", " ".join(paragraph)))
            paragraph.clear()
        if table:
            blocks.append(("table", list(table)))
            table.clear()

    for line in raw.splitlines():
        if line.startswith(">") or line.startswith("# "):
            continue  # the synthetic-data disclaimer and the H1 title

        if line.startswith("## "):
            flush()
            if heading:
                sections.append((heading, blocks))
            heading, blocks = line[3:].strip(), []
            continue

        if heading is None:
            continue

        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("|"):
            if paragraph:
                flush()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):  # skip |---|---| rules
                table.append(cells)
        elif stripped.startswith("- ") or re.match(r"^\d+\.\s", stripped):
            flush()
            blocks.append(("item", stripped))
        else:
            if table:
                flush()
            paragraph.append(stripped)

    flush()
    if heading:
        sections.append((heading, blocks))
    return meta, sections


def write(pdf, height: float, text: str, indent: float = 0) -> None:
    """Write one wrapped block and return the cursor to the left margin.

    fpdf2 leaves the cursor at the right edge of a multi_cell by default, so the
    next full-width cell would have no horizontal space and raise.
    """
    if indent:
        pdf.set_x(pdf.l_margin + indent)
    pdf.multi_cell(0, height, clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def build_pdf(meta, sections, destination: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    write(pdf, 9, meta.get("title", destination.stem))

    # Mixed case on purpose: these lines must not look like section headings.
    pdf.set_font("Helvetica", "", 9)
    for label, key in (("Document ID", "doc_id"), ("Version", "version"),
                       ("Effective date", "effective_date"), ("Owner", "owner")):
        if key in meta:
            write(pdf, 5, "{}: {}".format(label, meta[key]))

    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    write(pdf, 4, "SYNTHETIC SAMPLE DATA. Written for a technical assignment. "
                  "Not a real policy of any company.")
    pdf.ln(4)

    for heading, blocks in sections:
        pdf.set_font("Helvetica", "B", 12)
        write(pdf, 7, HEADING_MARK + heading)
        pdf.ln(1)

        for kind, content in blocks:
            if kind == "para":
                pdf.set_font("Helvetica", "", 10)
                write(pdf, 5, content)
                pdf.ln(2)
            elif kind == "item":
                pdf.set_font("Helvetica", "", 10)
                write(pdf, 5, content, indent=5)
            else:
                # Courier keeps the columns aligned once the pipes are gone.
                widths = [max(len(clean(row[i])) for row in content)
                          for i in range(len(content[0]))]
                pdf.set_font("Courier", "", 9)
                for index, row in enumerate(content):
                    if index == 1:
                        pdf.ln(1)
                    padded = "  ".join(
                        clean(cell).ljust(widths[position])
                        for position, cell in enumerate(row))
                    write(pdf, 4.5, padded, indent=3)
                pdf.ln(2)
        pdf.ln(2)

    pdf.output(str(destination))


def main() -> int:
    if not SOURCE_DIR.is_dir():
        sys.exit("No source directory at {}. Move the .md files there first.".format(SOURCE_DIR))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(SOURCE_DIR.glob("*.md"))
    if not sources:
        sys.exit("No .md files found in {}".format(SOURCE_DIR))

    for path in sources:
        meta, sections = parse_markdown(path.read_text(encoding="utf-8"))
        destination = OUTPUT_DIR / (path.stem + ".pdf")
        build_pdf(meta, sections, destination)
        print("{:<42} {} section(s)".format(destination.name, len(sections)))

    print("\nWrote {} PDF(s) to {}".format(len(sources), OUTPUT_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
