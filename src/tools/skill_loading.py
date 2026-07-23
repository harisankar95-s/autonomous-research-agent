from src.tools.base import Tool
from src.prompt_building.skill_catalog import get_skill_body
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_load_skill_tool() -> Tool:
    def load_skill(name: str) -> str:
        result = get_skill_body(name)
        logger.info(f"Skill loaded via tool | name={name}")
        return result

    return Tool(
        name="load_skill",
        description=(
            "Load the full instructions for a named skill from the skill "
            "catalog listed in your system prompt. Call this when a skill's "
            "description indicates it applies to your current task."
        ),
        parameters={
            "name": "the exact skill name as shown in the catalog"
        },
        func=load_skill
    )