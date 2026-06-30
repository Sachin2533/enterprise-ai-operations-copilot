# Enterprise AI Operations Copilot

Local AI operations copilot built with Streamlit, FastAPI, LangGraph, SQLite, FAISS RAG, and Groq-backed LLM responses.

For the full architecture and runtime flow, see [WORKFLOW.md](WORKFLOW.md).

## UX Highlights

- Upload PDFs and ask evidence-filtered suggested questions that are answerable from the selected document.
- Delete uploaded documents and refresh the index from the sidebar.
- Use a focused chat workspace with document suggestions centered above the chat input.
- Route requests with optional LLM-based planning and keyword fallback.
- Retrieve document context with hybrid BM25 + FAISS ranking and page-number citations.
- Show document-answer confidence and source/page pills under answers.
- See API, RAG index, document count, and MCP enterprise-tool status in the sidebar.
- Inspect a compact right-side debugger summary, with advanced trace available when needed.
- Select the active PDF so document answers stay scoped to the right uploaded file.
- Chat history shows which document was used for each answer.

## Quick Start

```bash
uvicorn app.main:app --reload
```

```bash
streamlit run ui/streamlit_app.py
```

Document and general LLM answers require `GROQ_API_KEY` to be set in the environment or `.env`.
