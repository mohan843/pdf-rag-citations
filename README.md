# PDF RAG with Citations

A compact, reproducible Python mini-project that retrieves evidence from one or more searchable PDFs and prints citations with the source filename and one-based page number. It uses TF-IDF cosine similarity and an extractive evidence response, so results stay traceable without requiring an API key or an LLM dependency.

## Features

- Extracts text from PDFs with `pypdf` while retaining page numbers.
- Splits long pages into overlapping chunks to preserve context.
- Ranks passages with a lightweight TF-IDF cosine retriever.
- Prints explicit citations such as `[sample_company_handbook.pdf, p. 2]`.
- Includes a sample searchable PDF and a standard-library `unittest` suite.

> This is a learning/demo project, not a substitute for checking the cited source. Scanned PDFs require OCR before they contain searchable text. The demo returns cited evidence rather than generating an uncited answer.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/mohan843/pdf-rag-citations.git
cd pdf-rag-citations
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage

Ask a question against the included sample PDF:

```bash
python pdf_rag.py \
  --pdf sample_data/sample_company_handbook.pdf \
  --question "How many vacation days do employees receive?" \
  --top-k 3
```

Pass more PDFs by adding paths after `--pdf`:

```bash
python pdf_rag.py --pdf docs/handbook.pdf docs/policies.pdf \
  --question "What is the remote work policy?"
```

The CLI prints the ranked evidence, citation, and score ordering. A no-match question returns an explicit no-relevant-passages message instead of fabricating an answer.

## Run tests

```bash
python -m unittest -v
```

## Project layout

```text
pdf-rag-citations/
├── pdf_rag.py
├── requirements.txt
├── tests.py
└── sample_data/
    └── sample_company_handbook.pdf
```

## How retrieval works

1. Read each PDF page and assign its one-based page number.
2. Normalize and tokenize text; split long pages into overlapping chunks.
3. Compute TF-IDF vectors for each chunk and the question.
4. Rank by cosine similarity and attach the original filename/page to every result.

No document text leaves your machine. Retrieval quality is intentionally simple; production RAG systems may add semantic embeddings, reranking, OCR, and a carefully grounded generation step.

## License

MIT
