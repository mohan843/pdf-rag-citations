"""Unit tests for the PDF RAG retrieval and citation pipeline."""
import unittest
from pathlib import Path

from pdf_rag import Chunk, chunks_from_pages, format_evidence, load_pdf, retrieve, split_page, tokenize


class TextProcessingTests(unittest.TestCase):
    def test_tokenize_normalizes_case_and_ignores_punctuation(self):
        self.assertEqual(tokenize("AI Agents, RAG-2026!"), ["ai", "agents", "rag-2026"])

    def test_split_page_overlaps_without_losing_text(self):
        text = "one two three four five six seven eight nine ten"
        parts = split_page(text, max_chars=20, overlap=5)
        self.assertGreater(len(parts), 1)
        self.assertIn("one", parts[0])
        self.assertIn("ten", parts[-1])
        self.assertTrue(any(set(part.split()) & set(parts[i + 1].split())
                            for i, part in enumerate(parts[:-1])))

    def test_chunks_preserve_page_citation_metadata(self):
        chunks = chunks_from_pages([(2, "Budget is reviewed every quarter."), (3, "Teams publish reports.")], "handbook.pdf")
        self.assertEqual([(chunk.source, chunk.page) for chunk in chunks], [("handbook.pdf", 2), ("handbook.pdf", 3)])


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("handbook.pdf", 1, "The annual leave policy grants twenty vacation days."),
            Chunk("handbook.pdf", 2, "Security training is required for all employees each year."),
            Chunk("handbook.pdf", 3, "Employees may work remotely up to three days each week."),
        ]

    def test_retrieval_ranks_relevant_passage_first(self):
        results = retrieve("How many vacation days are granted?", self.chunks, top_k=2)
        self.assertEqual(results[0].chunk.page, 1)
        self.assertGreater(results[0].score, 0)

    def test_format_includes_document_and_page_citation(self):
        output = format_evidence("vacation policy", retrieve("vacation policy", self.chunks))
        self.assertIn("[handbook.pdf, p. 1]", output)
        self.assertIn("source-grounded passages", output)

    def test_no_match_is_explicit(self):
        self.assertIn("No relevant passages", format_evidence("quantum telescope", []))


class SamplePdfTests(unittest.TestCase):
    def test_included_pdf_extracts_searchable_pages(self):
        path = Path(__file__).parent / "sample_data" / "sample_company_handbook.pdf"
        chunks = load_pdf(path)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(any("vacation" in chunk.text.lower() for chunk in chunks))
        self.assertEqual({chunk.source for chunk in chunks}, {path.name})


if __name__ == "__main__":
    unittest.main(verbosity=2)
