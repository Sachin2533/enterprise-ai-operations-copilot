def format_copilot_response(
    summary: str,
    key_findings: list[str] | None = None,
    detailed_explanation: str | None = None,
    recommended_next_actions: list[str] | None = None,
    suggested_follow_up_questions: list[str] | None = None
) -> str:
    sections = [
        summary.strip()
    ]

    if key_findings:

        if len(key_findings) == 1:

            sections.append(
                key_findings[0]
            )

        else:

            sections.append(
                "\n".join(
                    f"- {item}"
                    for item in key_findings
                )
            )

    if detailed_explanation:

        sections.append(
            detailed_explanation.strip()
        )

    if suggested_follow_up_questions:

        sections.append(
            "You can also ask:\n"
            + "\n".join(
                    f"- {item}"
                    for item in suggested_follow_up_questions
                )
        )

    return "\n\n".join(sections)
