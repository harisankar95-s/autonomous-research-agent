def compile_prompt(
    role_prompt: str,
    project_brief: str,
    facts: str = "",
    knowledge: list[str] | None = None
) -> str:
    if knowledge is None:
        knowledge = []

    sections = [role_prompt, f"PROJECT DESCRIPTION:\n{project_brief}"]

    if facts:
        sections.append(f"KNOWN FACTS:\n{facts}")

    if knowledge:
        knowledge_text = "\n".join(f"- {item}" for item in knowledge)
        sections.append(f"RELEVANT PRIOR FINDINGS:\n{knowledge_text}")

    return "\n\n".join(sections)