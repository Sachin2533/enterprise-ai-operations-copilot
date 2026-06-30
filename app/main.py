from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from app.agents.langgraph_agent import (
    process_query_langgraph
)
from app.database.database import (
    initialize_databases
)
from app.mcp.mcp_server import (
    mcp_server
)
from app.llm.general import (
    generate_general_answer
)

app = FastAPI(
    title="Enterprise AI Operations Copilot",
    version="2.0.0"
)


class ChatRequest(BaseModel):
    question: str
    document_name: str | None = None
    debug: bool = False


class GeneralChatRequest(BaseModel):
    question: str
    debug: bool = False


def _debug_event(
    step: str,
    detail: str,
    status: str = "ok"
) -> dict:

    return {
        "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "step": step,
        "detail": detail,
        "status": status
    }


def _get_pdf_documents():

    documents_path = Path(
        "data/documents"
    )

    if not documents_path.exists():

        return []

    return sorted(
        path.name
        for path in documents_path.glob("*.pdf")
    )


def _is_rag_index_ready():

    index_path = Path(
        "data/faiss_index/index.faiss"
    )

    metadata_path = Path(
        "data/faiss_index/index.pkl"
    )

    return (
        index_path.exists()
        and metadata_path.exists()
    )


@app.on_event("startup")
def startup():

    initialize_databases()


@app.get("/")
def root():

    return {
        "status": "running",
        "application": "Enterprise AI Operations Copilot",
        "architecture": "LangGraph + MCP + RAG"
    }


@app.get("/health")
def health():

    documents = _get_pdf_documents()

    return {
        "status": "ready",
        "application": "Enterprise AI Operations Copilot",
        "rag": {
            "index_ready": _is_rag_index_ready(),
            "document_count": len(documents),
            "documents": documents
        },
        "mcp": mcp_server.health_check()
    }


@app.get("/tools")
def tools():

    return {
        "mcp_tools": mcp_server.list_tools()
    }


@app.post("/chat")
def chat(
    request: ChatRequest
):

    try:

        response = process_query_langgraph(
            request.question,
            source_filter=request.document_name,
            include_debug=request.debug
        )

        if request.debug:

            return {
                "success": True,
                "question": request.question,
                "answer": response["answer"],
                "action": response.get("action"),
                "metadata": response.get(
                    "metadata",
                    {}
                ),
                "debug_trace": response.get(
                    "debug_trace",
                    []
                )
            }

        return {
            "success": True,
            "question": request.question,
            "answer": response
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


@app.post("/general-chat")
def general_chat(
    request: GeneralChatRequest
):

    debug_trace = []

    try:

        if request.debug:

            debug_trace.append(
                _debug_event(
                    "general_tab",
                    "General tab request received; skipping planner, RAG, and MCP tools."
                )
            )
            debug_trace.append(
                _debug_event(
                    "generate_general_answer",
                    "Calling general Groq response path."
                )
            )

        answer = generate_general_answer(
            request.question
        )

        if request.debug:

            debug_trace.append(
                _debug_event(
                    "response_node",
                    "General response completed."
                )
            )

        return {
            "success": True,
            "question": request.question,
            "answer": answer,
            "action": "general",
            "metadata": {},
            "debug_trace": debug_trace
        }

    except Exception as e:

        if request.debug:

            debug_trace.append(
                _debug_event(
                    "general_chat",
                    str(e),
                    status="error"
                )
            )

        return {
            "success": False,
            "error": str(e),
            "debug_trace": debug_trace
        }
