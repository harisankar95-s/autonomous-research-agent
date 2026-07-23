from src.prompt_building.skill_catalog import get_skill_catalog, get_always_load_bodies


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

    for body in get_always_load_bodies():
        sections.append(f"METHODOLOGY STANDARDS:\n{body}")

    skill_catalog = get_skill_catalog()
    if skill_catalog:
        sections.append(
            "AVAILABLE SKILLS (call load_skill with the exact name to load "
            f"full instructions):\n{skill_catalog}"
        )

    return "\n\n".join(sections)