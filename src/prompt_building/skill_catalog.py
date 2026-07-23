from pathlib import Path

SKILLS_DIR = Path(__file__).parent.parent / "skills"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        raise ValueError("Skill file missing frontmatter (expected leading '---')")

    _, frontmatter_block, body = text.split("---", 2)

    metadata = {}
    for line in frontmatter_block.strip().splitlines():
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip()

    return metadata, body.strip()


def _load_all_skills() -> dict[str, dict]:
    skills = {}
    for path in sorted(SKILLS_DIR.glob("*.md")):
        metadata, body = _parse_frontmatter(path.read_text())
        name = metadata.get("name", path.stem)
        skills[name] = {
            "description": metadata.get("description", ""),
            "body": body,
            "metadata": metadata,
        }
    return skills


def _is_always_load(metadata: dict) -> bool:
    return metadata.get("always_load", "").strip().lower() == "true"


def get_skill_catalog() -> str:
    skills = _load_all_skills()
    lines = [
        f"- {name}: {info['description']}"
        for name, info in skills.items()
        if not _is_always_load(info["metadata"])
    ]
    return "\n".join(lines)


def get_always_load_bodies() -> list[str]:
    skills = _load_all_skills()
    return [
        info["body"]
        for info in skills.values()
        if _is_always_load(info["metadata"])
    ]


def get_skill_body(name: str) -> str:
    skills = _load_all_skills()
    if name not in skills:
        available = ", ".join(skills.keys())
        return f"Error: no skill named '{name}'. Available skills: {available}"
    return skills[name]["body"]