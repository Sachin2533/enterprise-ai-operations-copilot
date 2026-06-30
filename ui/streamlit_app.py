import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(
    page_title="Enterprise AI Operations Copilot",
    page_icon="🤖",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1480px;
    }

    h1, h2, h3 {
        letter-spacing: 0;
    }

    div[data-testid="stSidebar"] {
        background: #f7f9fb;
        border-right: 1px solid #e5eaf0;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5eaf0;
        border-radius: 8px;
        padding: 0.7rem;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #e5eaf0;
        border-radius: 8px;
        padding: 0.35rem 0.5rem;
        background: #ffffff;
    }

    .copilot-header {
        border-bottom: 1px solid #e5eaf0;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }

    .copilot-title {
        font-size: 2rem;
        font-weight: 720;
        color: #18212f;
        margin: 0;
    }

    .copilot-subtitle {
        color: #5d6b7c;
        margin-top: 0.25rem;
        font-size: 0.98rem;
    }

    .workspace-label {
        color: #2d3b4f;
        font-size: 0.95rem;
        font-weight: 650;
        margin-bottom: 0.35rem;
    }

    .pill {
        display: inline-block;
        border: 1px solid #d9e2ec;
        background: #f7f9fb;
        color: #334155;
        border-radius: 999px;
        padding: 0.2rem 0.55rem;
        margin: 0.15rem 0.2rem 0.15rem 0;
        font-size: 0.78rem;
    }

    .debug-summary {
        border: 1px solid #e5eaf0;
        background: #ffffff;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
    }

    .suggestion-title {
        color: #2d3b4f;
        font-size: 0.9rem;
        font-weight: 650;
        text-align: center;
        margin: 1rem 0 0.35rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )

from app.rag.ingest import (
    build_vector_store,
    delete_document,
    refresh_vector_store
)
from app.rag.suggestions import (
    get_suggested_questions
)

API_BASE_URL = os.getenv(
    "COPILOT_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

CHAT_API_URL = f"{API_BASE_URL}/chat"
GENERAL_CHAT_API_URL = f"{API_BASE_URL}/general-chat"
HEALTH_API_URL = f"{API_BASE_URL}/health"

DOCUMENTS_DIR = Path(
    "data/documents"
)


def get_document_names() -> list[str]:

    if not DOCUMENTS_DIR.exists():

        return []

    return [
        path.name
        for path in sorted(
            DOCUMENTS_DIR.glob("*.pdf"),
            key=lambda item: item.stat().st_mtime,
            reverse=True
        )
    ]


@st.cache_data(ttl=300)
def get_cached_suggested_questions(
    document_name: str,
    modified_time: float
) -> list[str]:

    return get_suggested_questions(
        document_name
    )


@st.cache_data(ttl=10)
def get_backend_health():

    try:

        response = requests.get(
            HEALTH_API_URL,
            timeout=5
        )
        response.raise_for_status()

        return response.json()

    except Exception as exc:

        return {
            "status": "offline",
            "error": str(exc)
        }


def call_chat_api(
    question: str,
    document_name: str | None
) -> tuple[str, bool, list[dict], str | None, dict]:

    try:

        response = requests.post(
            CHAT_API_URL,
            json={
                "question": question,
                "document_name": document_name,
                "debug": True
            },
            timeout=120
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("success", True):

            return (
                payload.get(
                    "answer",
                    "No answer returned."
                ),
                True,
                payload.get(
                    "debug_trace",
                    []
                ),
                payload.get("action"),
                payload.get(
                    "metadata",
                    {}
                )
            )

        return (
            "Backend returned an error: "
            f"{payload.get('error', 'Unknown error.')}",
            False,
            payload.get(
                "debug_trace",
                []
            ),
            payload.get("action"),
            payload.get(
                "metadata",
                {}
            )
        )

    except Exception as exc:

        return (
            f"Backend request failed: {exc}",
            False,
            [
                {
                    "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    "step": "requests.post",
                    "detail": str(exc),
                    "status": "error"
                }
            ],
            None,
            {}
        )


def call_general_chat_api(
    question: str
) -> tuple[str, bool, list[dict], str | None, dict]:

    try:

        response = requests.post(
            GENERAL_CHAT_API_URL,
            json={
                "question": question,
                "debug": True
            },
            timeout=120
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("success", True):

            return (
                payload.get(
                    "answer",
                    "No answer returned."
                ),
                True,
                payload.get(
                    "debug_trace",
                    []
                ),
                payload.get("action"),
                payload.get(
                    "metadata",
                    {}
                )
            )

        return (
            "Backend returned an error: "
            f"{payload.get('error', 'Unknown error.')}",
            False,
            payload.get(
                "debug_trace",
                []
            ),
            payload.get("action"),
            payload.get(
                "metadata",
                {}
            )
        )

    except Exception as exc:

        return (
            f"Backend request failed: {exc}",
            False,
            [
                {
                    "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    "step": "requests.post",
                    "detail": str(exc),
                    "status": "error"
                }
            ],
            None,
            {}
        )


def submit_question(
    question: str,
    document_name: str | None = None,
    history_key: str = "chat_history",
    use_active_document: bool = True,
    general_only: bool = False
):

    question = question.strip()

    if not question:

        return

    active_document = (
        document_name
        if document_name is not None
        else st.session_state.get("active_document_name")
        if use_active_document
        else None
    )

    with st.spinner("Generating response..."):

        if general_only:

            answer, success, debug_trace, action, metadata = call_general_chat_api(
                question
            )

        else:

            answer, success, debug_trace, action, metadata = call_chat_api(
                question,
                active_document
            )

    st.session_state["latest_debug_trace"] = debug_trace
    st.session_state["latest_action"] = action

    st.session_state[history_key].append(
        {
            "question": question,
            "answer": answer,
            "document_name": active_document,
            "success": success,
            "action": action,
            "metadata": metadata,
            "debug_trace": debug_trace,
            "created_at": datetime.now().strftime("%H:%M:%S")
        }
    )


def render_answer_metadata(
    metadata: dict | None
):

    if not metadata:

        return

    confidence = metadata.get(
        "confidence"
    )
    retrieval = metadata.get(
        "retrieval",
        {}
    )
    sources = metadata.get(
        "sources",
        []
    )

    pills = []

    if confidence:

        pills.append(
            f"Confidence: {confidence}"
        )

    if retrieval.get("chunks_selected"):

        pills.append(
            f"Chunks: {retrieval['chunks_selected']}"
        )

    for source in sources:

        page = source.get("page")
        label = source.get(
            "file",
            "document"
        )

        if page:

            label = f"{label}, page {page}"

        pills.append(
            label
        )

    if not pills:

        return

    st.markdown(
        " ".join(
            f'<span class="pill">{pill}</span>'
            for pill in pills
        ),
        unsafe_allow_html=True
    )


def render_debug_trace(
    debug_trace: list[dict],
    action: str | None = None
):

    if action:

        st.caption(
            f"Planner action: {action}"
        )

    if not debug_trace:

        st.caption("No debug events captured for this request.")
        return

    for index, event in enumerate(
        debug_trace,
        start=1
    ):

        status = event.get(
            "status",
            "ok"
        )
        marker = {
            "ok": "OK",
            "warning": "WARN",
            "error": "ERROR"
        }.get(
            status,
            status.upper()
        )

        st.code(
            (
                f"{index}. [{event.get('time', '-')}] "
                f"{marker} {event.get('step', '-')}\n"
                f"   {event.get('detail', '')}"
            ),
            language="text"
        )


def render_latest_debugger():

    latest_trace = st.session_state.get(
        "latest_debug_trace",
        []
    )

    st.subheader("Live Debugger")

    st.caption(
        "Latest request trace"
    )

    if latest_trace:

        statuses = [
            event.get(
                "status",
                "ok"
            )
            for event in latest_trace
        ]
        errors = statuses.count("error")
        warnings = statuses.count("warning")
        planner_event = next(
            (
                event
                for event in latest_trace
                if event.get("step") == "planner"
            ),
            {}
        )
        retrieval_event = next(
            (
                event
                for event in latest_trace
                if event.get("step") == "hybrid_retrieval"
            ),
            None
        )
        tool_call_event = next(
            (
                event
                for event in reversed(latest_trace)
                if event.get("step") == "mcp_server.execute"
            ),
            None
        )
        tool_result_event = next(
            (
                event
                for event in reversed(latest_trace)
                if event.get("step") in {
                    "employee_info",
                    "leave_balance",
                    "calculator",
                    "date_time",
                    "document_answer",
                    "list_documents",
                    "rebuild_index",
                    "delete_document"
                }
            ),
            None
        )
        legacy_tool_event = next(
            (
                event
                for event in reversed(latest_trace)
                if event.get("step") in {
                    "calculate",
                    "get_current_date",
                    "get_current_time",
                    "get_day_of_week",
                    "ask_document"
                }
            ),
            None
        )
        tool_events = [
            event
            for event in latest_trace
            if event.get("step") == "mcp_server.execute"
            or event.get("step") in {
                "employee_info",
                "leave_balance",
                "calculator",
                "date_time",
                "document_answer",
                "list_documents",
                "rebuild_index",
                "delete_document"
            }
        ]
        tool_summary = (
            tool_call_event.get("detail", "")
            if tool_call_event
            else legacy_tool_event.get("detail", "")
            if legacy_tool_event
            else "No MCP tool called."
        )

        if tool_result_event:

            tool_summary = (
                f"{tool_summary}<br>{tool_result_event.get('step')}: "
                f"{tool_result_event.get('status', 'ok').upper()}"
            )

        elif tool_events:

            tool_summary = (
                f"{tool_summary}<br>{tool_events[-1].get('step')}: "
                f"{tool_events[-1].get('status', 'ok').upper()}"
            )

        if legacy_tool_event and not tool_call_event:

            tool_summary = (
                f"{tool_summary}<br>Restart backend to show MCP gateway events."
            )
        
        tool_summary = (
            tool_summary.replace(
                "<script",
                "&lt;script"
            )
        )
        llm_event = next(
            (
                event
                for event in latest_trace
                if event.get("step") == "ChatGroq.invoke"
            ),
            None
        )

        st.markdown(
            f"""
            <div class="debug-summary">
                <strong>Planner</strong><br>{planner_event.get('detail', 'No planner event yet.')}<br><br>
                <strong>Tool</strong><br>{tool_summary}<br><br>
                <strong>Retrieval</strong><br>{retrieval_event.get('detail', 'No retrieval step for this request.') if retrieval_event else 'No retrieval step for this request.'}<br><br>
                <strong>LLM</strong><br>{'Called' if llm_event else 'Not called'}<br><br>
                <strong>Status</strong><br>{errors} errors, {warnings} warnings
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(
            "Advanced trace",
            expanded=False
        ):

            render_debug_trace(
                latest_trace,
                st.session_state.get("latest_action")
            )

    else:

        st.info("Run a question to see planner, tool, RAG, and error events.")


def rebuild_knowledge_base():

    try:

        with st.sidebar.status(
            "Updating knowledge base...",
            expanded=False
        ):

            refresh_vector_store()

        get_backend_health.clear()
        st.sidebar.success(
            "Knowledge base updated."
        )

    except Exception as exc:

        st.sidebar.error(
            f"Knowledge base update failed: {exc}"
        )


def render_status_panel():

    health = get_backend_health()

    st.sidebar.subheader("System Status")

    if health.get("status") == "offline":

        st.sidebar.error("API offline")
        st.sidebar.caption(
            health.get(
                "error",
                "Start the FastAPI backend."
            )
        )
        return

    rag = health.get(
        "rag",
        {}
    )
    mcp = health.get(
        "mcp",
        {}
    )

    status_columns = st.sidebar.columns(2)

    status_columns[0].metric(
        "Docs",
        rag.get(
            "document_count",
            0
        )
    )

    status_columns[1].metric(
        "MCP",
        mcp.get(
            "status",
            "unknown"
        ).title()
    )

    if rag.get("index_ready"):

        st.sidebar.success("RAG index ready")

    else:

        st.sidebar.warning("RAG index not ready")

    if mcp.get("status") == "ready":

        st.sidebar.success("Enterprise tools ready")

    else:

        st.sidebar.warning("Enterprise tools need attention")


def render_suggestion_buttons(
    document_name: str | None
):

    if not document_name:

        return

    document_path = DOCUMENTS_DIR / document_name
    modified_time = (
        document_path.stat().st_mtime
        if document_path.exists()
        else 0
    )

    suggestions = get_cached_suggested_questions(
        document_name,
        modified_time
    )

    if not suggestions:

        return

    st.markdown(
        '<div class="suggestion-title">Suggested Questions</div>',
        unsafe_allow_html=True
    )

    for row_start in range(
        0,
        len(suggestions),
        2
    ):

        columns = st.columns(2)

        for offset, suggestion in enumerate(
            suggestions[row_start:row_start + 2]
        ):

            index = row_start + offset

            if columns[offset].button(
                suggestion,
                key=f"suggestion::{document_name}::{index}",
                use_container_width=True
            ):

                submit_question(
                    suggestion,
                    document_name
                )


def render_chat_history(
    history_key: str
):

    for chat in reversed(
        st.session_state[history_key]
    ):

        with st.chat_message("user"):

            st.markdown(chat["question"])

            if chat.get("document_name"):

                st.caption(
                    f"Document: {chat['document_name']} | {chat['created_at']}"
                )

            else:

                st.caption(
                    chat["created_at"]
                )

        with st.chat_message("assistant"):

            if chat.get("success"):

                st.markdown(chat["answer"])
                render_answer_metadata(
                    chat.get("metadata")
                )

            else:

                st.error(chat["answer"])


st.markdown(
    """
    <div class="copilot-header">
        <div class="copilot-title">Enterprise AI Operations Copilot</div>
        <div class="copilot-subtitle">
            Ask operational questions, analyze uploaded documents, and inspect every tool call in the debugger.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if "chat_history" not in st.session_state:

    st.session_state["chat_history"] = []

if "general_chat_history" not in st.session_state:

    st.session_state["general_chat_history"] = []

if st.session_state.get("general_tab_version") != "direct_general_v1":

    st.session_state["general_chat_history"] = []
    st.session_state["general_tab_version"] = "direct_general_v1"

if "active_document_name" not in st.session_state:

    st.session_state["active_document_name"] = None

if "latest_debug_trace" not in st.session_state:

    st.session_state["latest_debug_trace"] = []

if "latest_action" not in st.session_state:

    st.session_state["latest_action"] = None

render_status_panel()

st.sidebar.header("Knowledge Base")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    DOCUMENTS_DIR.mkdir(
        exist_ok=True
    )

    uploaded_bytes = uploaded_file.getvalue()
    file_path = DOCUMENTS_DIR / uploaded_file.name

    file_digest = hashlib.sha256(
        uploaded_bytes
    ).hexdigest()

    indexed_key = f"indexed::{uploaded_file.name}::{file_digest}"

    file_path.write_bytes(
        uploaded_bytes
    )

    st.sidebar.success(
        f"{uploaded_file.name} uploaded."
    )

    st.session_state["active_document_name"] = uploaded_file.name

    if not st.session_state.get(indexed_key):

        rebuild_knowledge_base()
        st.session_state[indexed_key] = True

document_names = get_document_names()

pending_active_document_name = st.session_state.pop(
    "pending_active_document_name",
    None
)

if pending_active_document_name in document_names:

    st.session_state["active_document_name"] = pending_active_document_name

elif pending_active_document_name is None and not document_names:

    st.session_state["active_document_name"] = None

pending_delete_message = st.session_state.pop(
    "pending_delete_message",
    None
)

if pending_delete_message:

    st.sidebar.success(
        pending_delete_message
    )

if document_names:

    active_document_name = st.session_state.get(
        "active_document_name"
    )

    if active_document_name not in document_names:

        active_document_name = document_names[0]
        st.session_state["active_document_name"] = active_document_name

    selected_index = document_names.index(
        active_document_name
    )

    st.sidebar.selectbox(
        "Active document",
        document_names,
        index=selected_index,
        key="active_document_name"
    )

    st.sidebar.caption(
        f"Using {st.session_state['active_document_name']} for document questions."
    )

    with st.sidebar.expander(
        "Document Management",
        expanded=False
    ):

        document_to_delete = st.selectbox(
            "Document to delete",
            document_names,
            index=selected_index,
            key="document_to_delete"
        )

        confirm_delete = st.checkbox(
            "Confirm deletion",
            key="confirm_document_delete"
        )

        if st.button(
            "Delete Document and Refresh Index",
            disabled=not confirm_delete,
            use_container_width=True
        ):

            try:

                deleted_name = delete_document(
                    document_to_delete
                )

                remaining_documents = get_document_names()

                st.session_state["pending_active_document_name"] = (
                    remaining_documents[0]
                    if remaining_documents
                    else None
                )
                st.session_state["pending_delete_message"] = (
                    f"{deleted_name} deleted. Index refreshed."
                )
                get_backend_health.clear()
                get_cached_suggested_questions.clear()
                st.rerun()

            except Exception as exc:

                st.sidebar.error(
                    f"Delete failed: {exc}"
                )

    if st.sidebar.button(
        "Rebuild Vector Store",
        use_container_width=True
    ):

        rebuild_knowledge_base()

else:

    st.session_state["active_document_name"] = None
    st.sidebar.info("Upload a PDF to enable document Q&A.")

if st.sidebar.button(
    "Clear Chat",
    use_container_width=True
):

    st.session_state["chat_history"] = []
    st.session_state["general_chat_history"] = []
    st.session_state["latest_debug_trace"] = []
    st.session_state["latest_action"] = None

chat_column, debugger_column = st.columns(
    [0.66, 0.34],
    gap="large"
)

with chat_column:

    st.markdown(
        '<div class="workspace-label">Conversation</div>',
        unsafe_allow_html=True
    )

    active_document = st.session_state.get(
        "active_document_name"
    )

    document_tab, general_tab = st.tabs(
        [
            "Document",
            "General"
        ]
    )

    with document_tab:

        if active_document:

            st.caption(
                f"Active document: {active_document}"
            )

            render_suggestion_buttons(
                active_document
            )

        else:

            st.caption(
                "Upload or select a PDF in the sidebar to enable document-specific answers."
            )

        document_query = st.chat_input(
            "Ask about the selected document",
            key="document_chat_input"
        )

        if document_query:

            submit_question(
                document_query,
                active_document
            )

        if not st.session_state["chat_history"]:

            st.info(
                "Upload a PDF, click a suggested question, or ask a document question."
            )

        render_chat_history(
            "chat_history"
        )

    with general_tab:

        st.caption(
            "Ask a general question without using the selected PDF."
        )

        general_query = st.chat_input(
            "Ask a general question",
            key="general_chat_input"
        )

        if general_query:

            submit_question(
                general_query,
                None,
                history_key="general_chat_history",
                use_active_document=False,
                general_only=True
            )

        if not st.session_state["general_chat_history"]:

            st.info(
                "Ask a general Groq question here."
            )

        render_chat_history(
            "general_chat_history"
        )

with debugger_column:

    render_latest_debugger()
