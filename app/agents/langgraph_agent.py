import re
from datetime import datetime

from langgraph.graph import (
    StateGraph,
    END
)

from app.agents.state import (
    AgentState
)

from app.agents.planner import (
    decide_action_with_metadata
)

from app.mcp.mcp_server import (
    mcp_server
)

from app.llm.general import (
    generate_general_answer
)

from app.agents.response_formatter import (
    format_copilot_response
)

from app.memory.memory import (
    save_conversation,
    get_employee_id,
    save_employee_id
)


EMPLOYEE_ID_PATTERN = re.compile(
    r"\bemp\s*[-_ ]?\s*(\d+)\b",
    re.IGNORECASE
)


def add_debug_event(
    state: AgentState,
    step: str,
    detail: str,
    status: str = "ok"
):

    state.setdefault(
        "debug_trace",
        []
    ).append(
        {
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "step": step,
            "detail": detail,
            "status": status
        }
    )


def execute_mcp_tool(
    state: AgentState,
    tool_name: str,
    **kwargs
):

    display_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key != "debug_trace"
    }

    add_debug_event(
        state,
        "mcp_server.execute",
        f"Calling MCP tool {tool_name} with {display_kwargs}"
    )

    result = mcp_server.execute(
        tool_name,
        **kwargs
    )

    if isinstance(result, dict) and result.get("error"):

        add_debug_event(
            state,
            tool_name,
            f"MCP tool failed: {result['error']}",
            status="error"
        )

    else:

        add_debug_event(
            state,
            tool_name,
            "MCP tool completed successfully."
        )

    return result


# ==================================
# Planner Node
# ==================================

def planner_node(
    state: AgentState
):

    query = state["query"]

    add_debug_event(
        state,
        "planner_node",
        "Received query and started action planning."
    )

    planner_result = (
        decide_action_with_metadata(query)
    )

    state["action"] = planner_result["action"]

    if (
        state.get("source_filter")
        and state["action"] == "general"
    ):

        state["action"] = "document"
        planner_result = {
            "method": "document_context_override",
            "reason": "An active document is selected for this general query."
        }

    add_debug_event(
        state,
        "planner",
        (
            f"Selected action: {state['action']} "
            f"via {planner_result['method']}. "
            f"{planner_result['reason']}"
        )
    )

    return state


# ==================================
# Tool Node
# ==================================

def tool_node(
    state: AgentState
):

    query = state["query"]

    action = state["action"]

    employee_id = (
        get_employee_id()
        or "EMP101"
    )

    employee_match = EMPLOYEE_ID_PATTERN.search(
        query
    )

    if employee_match:

        employee_id = (
            f"EMP{employee_match.group(1)}"
        )

        save_employee_id(
            employee_id
        )

        add_debug_event(
            state,
            "save_employee_id",
            f"Saved employee ID from query: {employee_id}"
        )

        if action not in {
            "employee",
            "leave"
        }:

            state["result"] = (
                format_copilot_response(
                    summary=f"Employee ID {employee_id} saved successfully.",
                    key_findings=[
                        f"Future employee and leave queries will use {employee_id}."
                    ],
                    recommended_next_actions=[
                        "Ask for leave balance or employee details."
                    ]
                )
            )

            return state

    if action == "leave":

        result = execute_mcp_tool(
            state,
            "leave_balance",
            employee_id=employee_id
        )

        if result and not result.get("error"):

            state["result"] = (
                format_copilot_response(
                    summary=f"Leave balance found for {employee_id}.",
                    key_findings=[
                        f"Casual Leave: {result['casual_leave']}",
                        f"Sick Leave: {result['sick_leave']}"
                    ],
                    recommended_next_actions=[
                        "Use this balance before planning time off."
                    ]
                )
            )

        else:

            add_debug_event(
                state,
                "leave_balance",
                "MCP tool returned no record.",
                status="warning"
            )

            state["result"] = format_copilot_response(
                summary=f"No leave balance found for {employee_id}.",
                key_findings=[
                    "The MCP leave balance tool returned no record."
                ],
                recommended_next_actions=[
                    "Confirm the employee ID and try again."
                ]
            )

    # Employee Tool

    elif action == "employee":

        result = execute_mcp_tool(
            state,
            "employee_info",
            employee_id=employee_id
        )

        if result and not result.get("error"):

            state["result"] = format_copilot_response(
                summary=f"Employee information found for {employee_id}.",
                key_findings=[
                    f"Employee ID: {result['employee_id']}",
                    f"Name: {result['name']}",
                    f"Department: {result['department']}"
                ],
                recommended_next_actions=[
                    "Ask for leave balance if you need availability details."
                ]
            )

        else:

            add_debug_event(
                state,
                "employee_info",
                "MCP tool returned no record.",
                status="warning"
            )

            state["result"] = format_copilot_response(
                summary=f"No employee found for {employee_id}.",
                key_findings=[
                    "The MCP employee information tool returned no record."
                ],
                recommended_next_actions=[
                    "Confirm the employee ID and try again."
                ]
            )

    # Calculator

    elif action == "calculator":

        result = execute_mcp_tool(
            state,
            "calculator",
            expression=query
        )

        state["result"] = format_copilot_response(
            summary="Calculation completed.",
            key_findings=[
                f"Result: {result.get('result')}"
            ],
            recommended_next_actions=[
                "Ask another calculation if needed."
            ]
        )

    # Date Tool

    elif action == "date":

        lower_query = query.lower()

        if "time" in lower_query:

            result = execute_mcp_tool(
                state,
                "date_time",
                mode="time"
            )
            value = result.get("value")
            state["result"] = format_copilot_response(
                summary=f"Current time: {value}",
                key_findings=[
                    f"Time: {value}"
                ]
            )

        elif re.search(
            r"\b(day|weekday)\b",
            lower_query
        ):

            result = execute_mcp_tool(
                state,
                "date_time",
                mode="day"
            )
            value = result.get("value")
            state["result"] = format_copilot_response(
                summary=f"Today is {value}.",
                key_findings=[
                    f"Day: {value}"
                ]
            )

        else:

            result = execute_mcp_tool(
                state,
                "date_time",
                mode="date"
            )
            value = result.get("value")
            state["result"] = format_copilot_response(
                summary=f"Today's date is {value}.",
                key_findings=[
                    f"Date: {value}"
                ]
            )

    # RAG

    elif action == "document":

        document_response = execute_mcp_tool(
            state,
            "document_answer",
            question=query,
            source_filter=state.get("source_filter"),
            debug_trace=state.get("debug_trace")
        )

        if isinstance(document_response, dict) and document_response.get("error"):

            state["result"] = (
                f"Document tool failed: {document_response['error']}"
            )

        else:

            state["result"] = document_response["answer"]
            state["metadata"] = document_response.get(
                "metadata",
                {}
            )

    else:

        add_debug_event(
            state,
            "generate_general_answer",
            "Calling general LLM fallback."
        )

        state["result"] = (
            generate_general_answer(query)
        )

    return state


# ==================================
# Response Node
# ==================================

def response_node(
    state: AgentState
):

    add_debug_event(
        state,
        "save_conversation",
        "Saving question and answer to memory database."
    )

    save_conversation(
        state["query"],
        state["result"]
    )

    add_debug_event(
        state,
        "response_node",
        "Workflow completed."
    )

    return state


# ==================================
# LangGraph Workflow
# ==================================

workflow = StateGraph(
    AgentState
)

workflow.add_node(
    "planner",
    planner_node
)

workflow.add_node(
    "tool",
    tool_node
)

workflow.add_node(
    "response",
    response_node
)

workflow.set_entry_point(
    "planner"
)

workflow.add_edge(
    "planner",
    "tool"
)

workflow.add_edge(
    "tool",
    "response"
)

workflow.add_edge(
    "response",
    END
)

graph = workflow.compile()


# ==================================
# Public Function
# ==================================

def process_query_langgraph(
    query: str,
    source_filter: str | None = None,
    include_debug: bool = False
):

    result = graph.invoke(
        {
            "query": query,
            "action": "",
            "result": "",
            "metadata": {},
            "source_filter": source_filter,
            "debug_trace": []
        }
    )

    if include_debug:

        return {
            "answer": result["result"],
            "debug_trace": result.get(
                "debug_trace",
                []
            ),
            "action": result.get("action"),
            "metadata": result.get(
                "metadata",
                {}
            )
        }

    return result["result"]
