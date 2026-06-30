# Enterprise AI Operations Copilot Workflow

This project is a local enterprise assistant with a Streamlit UI, a FastAPI chat API, a LangGraph agent workflow, MCP-like enterprise tools, SQLite memory, and PDF RAG over a FAISS vector index.

## Main Runtime Components

| Layer | Files | Purpose |
| --- | --- | --- |
| UI | `ui/streamlit_app.py` | Upload PDFs, select the active document, show content-aware suggested questions, send chat requests, and display chat history. |
| API | `app/main.py` | FastAPI service with `GET /` health metadata and `POST /chat` for user questions. |
| Agent workflow | `app/agents/langgraph_agent.py` | LangGraph pipeline that plans the query, runs the selected tool or RAG path, saves conversation memory, and returns the final answer. |
| Planner | `app/agents/planner.py` | Optional LLM router with keyword fallback that classifies each query as `document`, `leave`, `employee`, `date`, `calculator`, or `general`. |
| MCP tools | `app/mcp/mcp_server.py`, `app/tools/enterprise_tools.py`, `app/tools/calculator.py`, `app/tools/date_tool.py`, `app/rag/retriever.py`, `app/rag/ingest.py` | In-process MCP-style gateway for employee data, leave balance, calculator, date/time, document answers, document listing, document deletion, and index refresh. |
| RAG | `app/rag/ingest.py`, `app/rag/retriever.py`, `app/rag/suggestions.py` | Builds a FAISS index from PDFs, performs hybrid BM25 + FAISS retrieval, and answers document questions using retrieved context. |
| LLM | `app/llm/general.py` | Calls Groq `llama-3.3-70b-versatile` for general answers or context-grounded RAG responses. |
| Memory | `app/memory/memory.py`, `app/database/database.py` | Stores conversation history and the default employee ID in SQLite. |
| Utility implementations | `app/tools/calculator.py`, `app/tools/date_tool.py` | Implements simple arithmetic and date/time/day logic that is exposed through MCP tools. |

## End-to-End User Workflow

1. Start the FastAPI backend.

   ```bash
   uvicorn app.main:app --reload
   ```

2. Start the Streamlit UI.

   ```bash
   streamlit run ui/streamlit_app.py
   ```

3. The sidebar checks backend health through `GET /health` and shows API, RAG, document, and MCP status.

4. The user uploads a PDF from the Streamlit sidebar.

5. Streamlit writes the file to `data/documents/<filename>.pdf`.

6. Streamlit calls `build_vector_store()` from `app/rag/ingest.py`.

7. The ingest pipeline loads all PDFs from `data/documents`, splits them into chunks, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and saves the FAISS index in `data/faiss_index`.

8. Streamlit extracts text from the selected PDF and displays clickable suggested questions from `app/rag/suggestions.py` only when the underlying terms have evidence in the document.

9. The user can ask a question in the chat input or click a suggested document question.

10. Streamlit sends a `POST` request to `http://127.0.0.1:8000/chat`. The UI sends `debug: true` so the backend returns a live trace of planner, tool, RAG, and memory steps.

   ```json
   {
     "question": "Summarize this resume",
     "document_name": "SACHIN_CV0.pdf"
   }
   ```

11. FastAPI receives the request in `app/main.py` and calls `process_query_langgraph(question, source_filter=document_name)`.

12. LangGraph runs this pipeline:

   ```text
   planner -> tool -> response -> END
   ```

13. The planner node calls `decide_action_with_metadata(query)` and stores the chosen action in the agent state. If `USE_LLM_PLANNER=true` and `GROQ_API_KEY` is available, the LLM planner classifies the request; otherwise the keyword planner is used as a fallback.

14. The tool node executes one of the supported paths:

   | Action | Trigger examples | Execution path |
   | --- | --- | --- |
   | `document` | policy, document, handbook, resume, summarize, skills, projects | MCP tool `document_answer` performs hybrid BM25 + FAISS retrieval, builds cited context, and calls Groq. |
   | `leave` | leave, leaves, vacation, sick leave, casual leave | MCP tool `leave_balance` calls `get_leave_balance(employee_id)`. |
   | `employee` | employee, department, manager, profile | MCP tool `employee_info` calls `get_employee_info(employee_id)`. |
   | `calculator` | `10 + 5`, `20 * 3` | MCP tool `calculator` parses and evaluates a two-number expression. |
   | `date` | date, today, time, day | MCP tool `date_time` returns current date, time, or weekday. |
   | `general` | Any unclassified query | `generate_general_answer(query)` calls Groq without document context. |

15. If the query contains an employee ID like `EMP101`, the tool node saves it to profile memory and returns a confirmation. Future employee and leave queries use that saved ID, falling back to `EMP101` when no ID is saved.

16. The response node saves the user question and final answer to `data/memory.db`.

17. FastAPI returns the answer to Streamlit.

18. Streamlit appends the answer to `st.session_state["chat_history"]`, including the active document name and timestamp, then renders the newest chats first.

19. Streamlit renders confidence and source/page pills below document answers.

20. Streamlit renders a compact latest-request debugger summary in the right-side panel, with the full advanced trace available on demand.

## API Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Basic application metadata. |
| `GET /health` | Returns backend readiness, RAG index status, PDF count, PDF names, and MCP health. |
| `GET /tools` | Lists MCP-style tools and required arguments. |
| `POST /chat` | Runs the LangGraph assistant workflow for a question. |

When `POST /chat` receives `"debug": true`, it returns:

```json
{
  "answer": "Leave balance found for EMP101.",
  "action": "leave",
  "debug_trace": [
    {
      "time": "14:30:00.123",
      "step": "mcp_server.execute",
      "detail": "Calling MCP tool leave_balance with {'employee_id': 'EMP101'}",
      "status": "ok"
    }
  ]
}
```

## Data Workflow

### SQLite Databases

`app/database/database.py` defines two SQLite files:

| Database | Path | Tables |
| --- | --- | --- |
| Enterprise DB | `data/enterprise.db` | `employees`, `leave_balance` |
| Memory DB | `data/memory.db` | `conversations`, `user_profiles` |

`initialize_databases()` creates the tables and seeds one sample employee:

```text
EMP101 / Sachin / AI Team
Casual leave: 8
Sick leave: 5
```

Memory functions call `initialize_memory_database()` before reads and writes, so the memory database self-initializes when the app saves conversations or employee IDs.

### Document Index

The RAG index is stored in:

```text
data/faiss_index/index.faiss
data/faiss_index/index.pkl
```

The index is rebuilt from all PDFs currently present in `data/documents`. Uploading one document can therefore refresh the entire local knowledge base, not just the uploaded file.

The sidebar also includes document management controls. Deleting a document removes the selected PDF from `data/documents` and refreshes the FAISS index. If no PDFs remain, the local FAISS index is cleared.

## Query Routing Details

`app/agents/planner.py` supports two planner modes:

- Default: deterministic keyword and regex routing.
- Optional: LLM classification through Groq when `USE_LLM_PLANNER=true`.

The LLM planner still falls back to keyword routing if the API key is missing, the model call fails, or the model returns an invalid action.

In default keyword mode, a query such as "summarize employee policy" will be routed to `document` because document-related terms have priority.

## RAG Answering Details

`ask_document()` performs these steps:

1. Builds the vector index if `data/faiss_index` is missing and PDFs exist.
2. Requires `GROQ_API_KEY`; without it, document questions return a configuration message.
3. Loads the FAISS index with MiniLM embeddings.
4. Searches for semantic candidates with FAISS.
5. Applies the active document filter when Streamlit sends `document_name`.
6. Scores indexed chunks with local BM25 keyword matching.
7. Combines FAISS and BM25 scores into a hybrid ranking.
8. Optionally prefers likely source files based on query terms such as resume, CV, handbook, policy, manual, or guide.
9. Keeps the top three hybrid-ranked chunks.
10. Rejects weak matches when there is no active source filter and both hybrid score and term overlap look poor.
11. Rejects unsupported document questions before answer generation when retrieved chunks do not contain enough meaningful overlap with the question.
12. Sends the retrieved context to Groq with strict instructions to use only the supplied context and cite source numbers with page numbers.
13. Returns structured metadata for confidence, selected chunk count, retrieval type, and source/page references.

## Important Implementation Notes

- `app/agents/agent.py` contains an older linear version of the agent. The FastAPI app currently uses `app/agents/langgraph_agent.py`.
- The Streamlit app assumes the FastAPI backend is already running at `http://127.0.0.1:8000/chat`.
- Set `COPILOT_API_URL` if the Streamlit UI should call a backend URL other than `http://127.0.0.1:8000`.
- Set `USE_LLM_PLANNER=true` to enable LLM-based planning. Set `PLANNER_MODEL` to override the default planner model.
- General and document LLM answers require `GROQ_API_KEY` in the environment or `.env`.
- The calculator MCP tool intentionally supports only simple two-number expressions.
- The MCP layer is currently an in-process class, not a networked MCP server. It exposes `list_tools()`, `execute()`, and `health_check()` and is surfaced through `GET /tools`, `GET /health`, and the right-side Streamlit debugger.

