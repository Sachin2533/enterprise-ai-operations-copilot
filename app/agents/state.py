from typing import TypedDict


class AgentState(TypedDict, total=False):
    query: str
    action: str
    result: str
    metadata: dict
    source_filter: str | None
    debug_trace: list[dict]
