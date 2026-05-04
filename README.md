# PubMed RAG Researcher

## About
PubMed RAG Researcher is a Streamlit app for rapid literature review.
It searches PubMed abstracts, summarizes findings with Gemini, and supports follow-up Q&A grounded in retrieved papers.

## Updated Features
- PubMed search with configurable date range and target paper count.
- Iterative PMID batching to keep searching until enough valid papers are collected.
- Optional free-access filtering behavior through the sidebar toggle.
- Real-time retrieval progress UI (progress bar + per-paper status text).
- Gemini-generated structured summary with:
  - findings synthesis,
  - most relevant paper recommendation,
  - IEEE-style references.
- Vector retrieval for chat using ChromaDB + Google Generative AI embeddings.
- Automatic fallback to stored abstracts if vector retrieval is unavailable.
- Retrieval-mode transparency in UI (shows whether answer used vector search or fallback).
- Session-state chat history and conversation continuity.
- PDF export options:
  - summary only,
  - summary with full follow-up chat log.
- Graceful degradation when ChromaDB is unavailable.

## Retrieval Flow
1. Search PubMed with your topic and date range.
2. Build a local abstract corpus for the current run.
3. Store documents in ChromaDB (when available).
4. For follow-up questions, retrieve top-k semantically similar chunks.
5. Fall back to full stored abstracts if DB retrieval is unavailable.

## Tech Stack
- Python
- Streamlit
- Google Gemini API (`google-generativeai`)
- Metapub (PubMed access)
- ChromaDB (vector store)
- FPDF (PDF generation)
- python-dotenv (env configuration)

## Demo
https://github.com/user-attachments/assets/b21de8d1-b248-47c9-942c-9e269daee2eb

## Setup
1. Clone this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here
NCBI_API_KEY=your_ncbi_key_here
```

`NCBI_API_KEY` is optional but recommended for higher PubMed request throughput.

4. Run the app:

```bash
streamlit run app.py
```

## Testing
This repo currently includes tests for:
- `pdf_generator.py` (smoke + regression coverage)
- `database_manager.py` (ResearchDB behavior and fallbacks)

Recommended (runs both unittest-style and pytest-style tests):

```bash
pytest -v tests
```

If `pytest` is not installed:

```bash
pip install pytest
```

## Current Limitations
- Summaries and Q&A are bounded by abstract availability/quality from PubMed records.
- If embeddings or vector DB initialization fails, chat falls back to raw abstract context.
- Export currently targets PDF only.

## Future Improvements
- Add citation-level source snippets in chat responses.
- Improve ranking with metadata-aware re-ranking (year, study type, relevance score).
- Add trend visualizations (publication count over time by topic).
- Add optional full-text ingestion for open-access papers.
