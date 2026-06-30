from app.tools.enterprise_tools import (
    get_employee_info,
    get_leave_balance
)
from app.tools.calculator import (
    calculate
)
from app.tools.date_tool import (
    get_current_date,
    get_current_time,
    get_day_of_week
)


DOCUMENTS_DIR = "data/documents"


class MCPServer:
    """
    Simple MCP-like server
    Exposes enterprise tools
    """

    tools = {
        "employee_info": {
            "description": "Get employee profile details by employee ID.",
            "required_args": [
                "employee_id"
            ]
        },
        "calculator": {
            "description": "Run a simple arithmetic calculation.",
            "required_args": [
                "expression"
            ]
        },
        "date_time": {
            "description": "Get current date, time, or weekday.",
            "required_args": [
                "mode"
            ]
        },
        "document_answer": {
            "description": "Answer a question from uploaded document context using RAG.",
            "required_args": [
                "question"
            ]
        },
        "list_documents": {
            "description": "List uploaded PDF documents available for RAG.",
            "required_args": []
        },
        "rebuild_index": {
            "description": "Rebuild the FAISS vector index from uploaded PDFs.",
            "required_args": []
        },
        "delete_document": {
            "description": "Delete an uploaded PDF and refresh the vector index.",
            "required_args": [
                "document_name"
            ]
        },
        "leave_balance": {
            "description": "Get casual and sick leave balance by employee ID.",
            "required_args": [
                "employee_id"
            ]
        }
    }

    def list_tools(
        self
    ):

        return self.tools

    def execute(
        self,
        tool_name: str,
        **kwargs
    ):

        if tool_name not in self.tools:

            return {
                "error": f"Unknown tool: {tool_name}"
            }

        missing_args = [
            arg
            for arg in self.tools[tool_name]["required_args"]
            if kwargs.get(arg) in (None, "")
        ]

        if missing_args:

            return {
                "error": f"Missing required args: {', '.join(missing_args)}"
            }

        employee_id = kwargs.get(
            "employee_id"
        )

        if tool_name == "employee_info":

            return get_employee_info(
                employee_id
            )

        if tool_name == "leave_balance":

            return get_leave_balance(
                employee_id
            )

        if tool_name == "calculator":

            return {
                "result": calculate(
                    kwargs["expression"]
                )
            }

        if tool_name == "date_time":

            mode = kwargs["mode"]

            if mode == "time":

                return {
                    "mode": mode,
                    "value": get_current_time()
                }

            if mode == "day":

                return {
                    "mode": mode,
                    "value": get_day_of_week()
                }

            return {
                "mode": "date",
                "value": get_current_date()
            }

        if tool_name == "document_answer":

            from app.rag.retriever import (
                ask_document
            )

            return ask_document(
                kwargs["question"],
                source_filter=kwargs.get("source_filter"),
                debug_trace=kwargs.get("debug_trace"),
                return_metadata=True
            )

        if tool_name == "list_documents":

            from pathlib import Path

            documents_path = Path(DOCUMENTS_DIR)

            if not documents_path.exists():

                return {
                    "documents": []
                }

            return {
                "documents": sorted(
                    path.name
                    for path in documents_path.glob("*.pdf")
                )
            }

        if tool_name == "rebuild_index":

            from app.rag.ingest import (
                refresh_vector_store
            )

            refresh_vector_store()

            return {
                "status": "rebuilt"
            }

        if tool_name == "delete_document":

            from app.rag.ingest import (
                delete_document
            )

            deleted_name = delete_document(
                kwargs["document_name"]
            )

            return {
                "status": "deleted",
                "document_name": deleted_name
            }

        return {
            "error": f"Unhandled tool: {tool_name}"
        }

    def health_check(
        self
    ):

        sample_employee = self.execute(
            "employee_info",
            employee_id="EMP101"
        )

        sample_leave = self.execute(
            "leave_balance",
            employee_id="EMP101"
        )

        return {
            "status": "ready" if sample_employee and sample_leave else "degraded",
            "tools": list(self.tools.keys()),
            "sample_employee_available": bool(sample_employee),
            "sample_leave_available": bool(sample_leave)
        }


mcp_server = MCPServer()
