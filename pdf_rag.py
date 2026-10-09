"""A small, transparent PDF retrieval-augmented generation demo with page citations.

Retrieval is implemented with lightweight TF-IDF cosine similarity. The final
response is deliberately extractive: it presents retrieved evidence and its
source citation instead of inventing an answer with an unconfigured LLM.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

WORD_RE = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Result:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    """Normalize text to lowercase alphanumeric terms."""
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def split_page(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    """Split a page into overlapping, word-boundary-aware chunks."""
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Require max_chars > overlap >= 0")
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        next_start = max(start + 1, end - overlap)
        # Avoid starting the next chunk in the middle of a word.
        space = cleaned.find(" ", next_start)
        start = space + 1 if space != -1 and space < end + 30 else next_start
    return chunks


def chunks_from_pages(pages: Iterable[tuple[int, str]], source: str,
                      max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    """Convert (one-based page number, text) pairs into citation-aware chunks."""
    result: list[Chunk] = []
    for page_number, text in pages:
        if page_number < 1:
            raise ValueError("Page numbers must be one-based positive integers")
        result.extend(Chunk(source, page_number, piece)
                      for piece in split_page(text, max_chars=max_chars, overlap=overlap))
    return result


def load_pdf(path: str | Path) -> list[Chunk]:
    """Extract PDF pages and preserve page numbers for citations."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires pypdf. Install with: pip install -r requirements.txt") from exc
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    pages = ((number, page.extract_text() or "")
             for number, page in enumerate(reader.pages, start=1))
    chunks = chunks_from_pages(pages, pdf_path.name)
    if not chunks:
        raise ValueError(f"No extractable text found in {pdf_path}; scanned PDFs need OCR first")
    return chunks


def _tfidf_vectors(documents: Sequence[Sequence[str]]) -> list[dict[str, float]]:
    """Build normalized TF-IDF vectors for the supplied documents."""
    tokenized = [list(doc) for doc in documents]
    document_frequency: Counter[str] = Counter()
    for terms in tokenized:
        document_frequency.update(set(terms))
    count = len(tokenized)
    vectors: list[dict[str, float]] = []
    for terms in tokenized:
        frequencies = Counter(terms)
        vector = {
            term: (1.0 + math.log(freq)) * (math.log((1 + count) / (1 + document_frequency[term])) + 1.0)
            for term, freq in frequencies.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        vectors.append({term: value / norm for term, value in vector.items()} if norm else {})
    return vectors


def retrieve(question: str, chunks: Sequence[Chunk], top_k: int = 3) -> list[Result]:
    """Rank chunks by TF-IDF cosine similarity to a question."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not chunks or not tokenize(question):
        return []
    vectors = _tfidf_vectors([tokenize(chunk.text) for chunk in chunks])
    query_vector = _tfidf_vectors([tokenize(question)])[0]
    ranked = [Result(chunk, sum(weight * query_vector.get(term, 0.0)
                                for term, weight in vector.items()))
              for chunk, vector in zip(chunks, vectors)]
    ranked = [item for item in ranked if item.score > 0]
    return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]


def format_evidence(question: str, results: Sequence[Result]) -> str:
    """Render retrieved evidence with explicit source and page citations."""
    if not results:
        return f"Question: {question}\n\nNo relevant passages found in the provided PDFs."
    lines = [f"Question: {question}", "", "Retrieved evidence (ranked by relevance):"]
    for rank, result in enumerate(results, start=1):
        citation = f"{result.chunk.source}, p. {result.chunk.page}"
        lines.extend((f"{rank}. {result.chunk.text}", f"   Citation: [{citation}]"))
    lines.extend(("", "This demo returns source-grounded passages, not a generated answer. "
                  "Use the cited pages to verify the evidence."))
    return "\n".join(lines)


def answer_question(question: str, pdf_paths: Sequence[str | Path], top_k: int = 3) -> str:
    """Load PDFs, retrieve relevant passages, and return cited evidence."""
    if not question.strip():
        raise ValueError("Question cannot be empty")
    chunks = [chunk for path in pdf_paths for chunk in load_pdf(path)]
    return format_evidence(question, retrieve(question, chunks, top_k=top_k))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, nargs="+", help="One or more searchable PDF files")
    parser.add_argument("--question", required=True, help="Question to retrieve evidence for")
    parser.add_argument("--top-k", type=int, default=3, help="Number of passages to return (default: 3)")
    args = parser.parse_args(argv)
    try:
        print(answer_question(args.question, args.pdf, top_k=args.top_k))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
