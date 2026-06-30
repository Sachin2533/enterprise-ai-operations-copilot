
import re

from app.agents.planner import (
    decide_action
)

from app.memory.memory import (
    save_conversation,
    get_employee_id,
    save_employee_id
)

from app.mcp.mcp_server import (
    mcp_server
)

from app.tools.calculator import (
    calculate
)

from app.tools.date_tool import (
    get_current_date,
    get_current_time,
    get_day_of_week
)

from app.rag.retriever import (
    ask_document
)

from app.llm.general import (
    generate_general_answer
)

from app.agents.response_formatter import (
    format_copilot_response
)

EMPLOYEE_ID_PATTERN = re.compile(
    r"\bemp\s*[-_ ]?\s*(\d+)\b",
    re.IGNORECASE
)


def process_query(
    query: str
):

    # -------------------------
    # Save Employee ID if provided
    # Example:
    # My employee id is EMP101
    # -------------------------

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

        return (
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

    # -------------------------
    # Get Stored Employee ID
    # -------------------------

    employee_id = (
        get_employee_id()
        or "EMP101"
    )

    action = decide_action(query)

    # -------------------------
    # Leave Queries
    # -------------------------

    if action == "leave":

        result = mcp_server.execute(
            "leave_balance",
            employee_id=employee_id
        )

        if result:

            answer = format_copilot_response(
                summary=f"Leave balance found for {employee_id}.",
                key_findings=[
                    f"Casual Leave: {result['casual_leave']}",
                    f"Sick Leave: {result['sick_leave']}"
                ],
                recommended_next_actions=[
                    "Use this balance before planning time off."
                ]
            )

        else:

            answer = format_copilot_response(
                summary=f"No leave balance found for {employee_id}.",
                key_findings=[
                    "The MCP leave balance tool returned no record."
                ],
                recommended_next_actions=[
                    "Confirm the employee ID and try again."
                ]
            )

    # -------------------------
    # Employee Queries
    # -------------------------

    elif action == "employee":

        result = mcp_server.execute(
            "employee_info",
            employee_id=employee_id
        )

        if result:

            answer = format_copilot_response(
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

            answer = format_copilot_response(
                summary=f"No employee found for {employee_id}.",
                key_findings=[
                    "The MCP employee information tool returned no record."
                ],
                recommended_next_actions=[
                    "Confirm the employee ID and try again."
                ]
            )

    # -------------------------
    # Calculator
    # -------------------------

    elif action == "calculator":

        answer = format_copilot_response(
            summary="Calculation completed.",
            key_findings=[
                f"Result: {calculate(query)}"
            ],
            recommended_next_actions=[
                "Ask another calculation if needed."
            ]
        )

    # -------------------------
    # Date / Time
    # -------------------------

    elif action == "date":

        lower_query = query.lower()

        if "time" in lower_query:

            value = get_current_time()
            answer = format_copilot_response(
                summary=f"Current time: {value}",
                key_findings=[
                    f"Time: {value}"
                ]
            )

        elif "day" in lower_query:

            value = get_day_of_week()
            answer = format_copilot_response(
                summary=f"Today is {value}.",
                key_findings=[
                    f"Day: {value}"
                ]
            )

        else:

            value = get_current_date()
            answer = format_copilot_response(
                summary=f"Today's date is {value}.",
                key_findings=[
                    f"Date: {value}"
                ]
            )

    # -------------------------
    # Document Search (RAG)
    # -------------------------

    elif action == "document":

        answer = ask_document(
            query
        )

    # -------------------------
    # Fallback
    # -------------------------

    else:

        answer = generate_general_answer(query)

    save_conversation(
        query,
        answer
    )

    return answer
