import os

from dotenv import load_dotenv

from app.agents.response_formatter import (
    format_copilot_response
)

load_dotenv()

GENERAL_MODEL = "llama-3.3-70b-versatile"


def generate_general_answer(question: str, context: str | None = None) -> str:
    prompt = (
        "You are an Enterprise AI Operations Copilot.\n"
        "Answer the user's question accurately and concisely.\n"
        "Use the provided context only if it is relevant.\n"
        "If the context is not relevant, answer from general knowledge.\n"
        "Do not use a fixed report template.\n"
        "Start with the direct answer in plain language.\n"
        "Use short bullets only when they make the answer easier to scan.\n"
        "Add a next step only when it is genuinely useful.\n"
    )

    if context:
        prompt += f"\nContext:\n{context}\n"

    prompt += f"\nQuestion:\n{question}\n"

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return format_copilot_response(
            summary="General knowledge answers are not configured yet.",
            key_findings=[
                "GROQ_API_KEY is missing."
            ],
            recommended_next_actions=[
                "Set GROQ_API_KEY to enable general LLM reasoning."
            ]
        )

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            api_key=api_key,
            model=GENERAL_MODEL
        )

        response = llm.invoke(prompt)
        return response.content

    except Exception:
        return format_copilot_response(
            summary="The general LLM service is not reachable right now.",
            key_findings=[
                "The request to the external model provider failed."
            ],
            recommended_next_actions=[
                "Check the API key, network connection, and provider status."
            ]
        )
