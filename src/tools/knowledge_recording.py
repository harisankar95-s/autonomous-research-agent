from src.tools.base import Tool
from src.memory.manager import KnowledgeStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_record_observation_tool(knowledge_store: KnowledgeStore, dataset_id: str) -> Tool:
    async def record_observation(content: str, confidence_score: float) -> str:
        obs_id = await knowledge_store.save_observation(
            dataset_id=dataset_id,
            content=content,
            confidence_score=confidence_score
        )
        logger.info(f"Observation recorded via tool | dataset_id={dataset_id} | id={obs_id}")
        return f"Observation recorded with id {obs_id}"

    return Tool(
        name="record_observation",
        description=(
            "Record a specific finding, hypothesis, or observation about the "
            "dataset as you discover it. Use this for anything you notice that "
            "might matter later - a skewed column, a possible data quality "
            "issue, a hypothesis worth testing further. Provide your confidence "
            "in this observation as a number between 0 and 1."
        ),
        parameters={
            "content": "the observation or finding, as a clear sentence",
            "confidence_score": "your confidence in this observation, from 0 to 1"
        },
        func=record_observation
    )