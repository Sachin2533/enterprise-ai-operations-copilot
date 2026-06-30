import os
import re

from dotenv import load_dotenv

load_dotenv()

VALID_ACTIONS = {
    "document",
    "leave",
    "employee",
    "date",
    "calculator",
    "general"
}


def _matches_any(
    query: str,
    keywords: list[str]
) -> bool:
    return any(
        re.search(
            rf"\b{re.escape(keyword)}\b",
            query
        )
        for keyword in keywords
    )


def decide_action_keyword(query: str):

    query = query.lower().strip()

    document_keywords = [
        "policy",
        "document",
        "handbook",
        "guidelines",
        "resume",
        "cv",
        "contract",
        "agreement",
        "manual",
        "sop",
        "research paper",
        "paper",
        "uploaded",
        "summarize",
        "summary",
        "extract",
        "explain section",
        "mentioned",
        "included",
        "technologies",
        "experience",
        "details",
        "provided",
        "candidate",
        "skills",
        "projects",
        "clauses",
        "methodology",
        "findings",
        "limitations"
    ]

    if _matches_any(query, document_keywords):
        return "document"

    # Leave queries
    if _matches_any(
        query,
        [
            "leave",
            "leaves",
            "vacation",
            "sick leave",
            "casual leave"
        ]
    ):
        return "leave"

    # Employee queries
    if _matches_any(
        query,
        [
            "employee",
            "department",
            "manager",
            "profile"
        ]
    ):
        return "employee"

    # Date queries
    if _matches_any(
        query,
        [
            "date",
            "today",
            "time",
            "day"
        ]
    ):
        return "date"

    # Calculator
    math_pattern = (
        r"^\s*-?\d+(\.\d+)?\s*[\+\-\*/]\s*-?\d+(\.\d+)?\s*$"
    )

    if re.match(math_pattern, query):
        return "calculator"

    return "general"


def _llm_planner_enabled() -> bool:

    return os.getenv(
        "USE_LLM_PLANNER",
        ""
    ).lower() in {
        "1",
        "true",
        "yes",
        "on"
    }


def decide_action_with_metadata(query: str) -> dict:

    keyword_action = decide_action_keyword(
        query
    )

    if not _llm_planner_enabled():

        return {
            "action": keyword_action,
            "method": "keyword",
            "reason": "USE_LLM_PLANNER is not enabled."
        }

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        return {
            "action": keyword_action,
            "method": "keyword_fallback",
            "reason": "GROQ_API_KEY is missing."
        }

    prompt = f"""
Classify the user's enterprise assistant request into exactly one action.

Allowed actions:
- document: questions about uploaded PDFs, policies, resumes, manuals, summaries, sections, clauses, skills, projects, or document content.
- leave: leave balance, vacation, sick leave, casual leave, time off.
- employee: employee profile, department, manager, employee details.
- date: current date, time, today, day of week.
- calculator: simple arithmetic.
- general: anything else.

Return only the action name.

User request:
{query}
"""

    try:

        from langchain_groq import ChatGroq

        llm = ChatGroq(
            api_key=api_key,
            model=os.getenv(
                "PLANNER_MODEL",
                "llama-3.1-8b-instant"
            ),
            temperature=0
        )

        response = llm.invoke(
            prompt
        )

        action = response.content.strip().lower()

        action = re.sub(
            r"[^a-z_]",
            "",
            action
        )

        if action in VALID_ACTIONS:

            return {
                "action": action,
                "method": "llm",
                "reason": "LLM planner classified the request."
            }

        return {
            "action": keyword_action,
            "method": "keyword_fallback",
            "reason": f"LLM returned invalid action: {response.content!r}"
        }

    except Exception as exc:

        return {
            "action": keyword_action,
            "method": "keyword_fallback",
            "reason": f"LLM planner failed: {exc}"
        }


def decide_action(query: str):

    return decide_action_with_metadata(
        query
    )["action"]
