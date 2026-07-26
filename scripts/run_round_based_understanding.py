import asyncio
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.llm.gemini_client import GeminiClient
from src.llm.gemini_embeddings import GeminiEmbeddingClient
from src.agent.round_based_understanding import run_round_based_understanding
from src.utils.config import config
from src.utils.logger import get_logger
from src.observability.langfuse_client import init_langfuse
from langfuse import get_client

logger = get_logger(__name__)

TABLE_NAME = "turbine_data"
# Dev/prod toggle: TABLE_NAME always points at the real data (every analysis
# reads the real 789,120-row table either way) - DATASET_ID is only the
# memory namespace, and stays separate while this redesign is being
# developed and live-tested so nothing here touches the real accumulated
# knowledge under dataset_id="turbine_data". Flip this to TABLE_NAME only
# for the deliberate, final production run once the dev namespace has been
# verified clean.
DATASET_ID = "turbine_data_dev"
PROJECT_BRIEF = (
    "This is sensor data from a fleet of 10 wind turbines. We want to identify "
    "unusual or anomalous sensor behavior — readings or patterns that deviate "
    "from normal operation and could indicate a developing issue."
)


async def main():
    init_langfuse()
    engine = create_engine(config.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()

    llm_client = GeminiClient()
    embedding_client = GeminiEmbeddingClient()

    dataset_id = DATASET_ID

    try:
        logger.info(f"Starting round-based understanding | dataset_id={dataset_id}")
        result = await run_round_based_understanding(
            llm_client=llm_client,
            embedding_client=embedding_client,
            db_session=db_session,
            table_name=TABLE_NAME,
            dataset_id=dataset_id,
            project_brief=PROJECT_BRIEF
        )

        print("\n=== ROUND-BASED UNDERSTANDING RESULT ===\n")
        print(json.dumps(result, indent=2))
    finally:
        db_session.close()
        get_client().flush()


if __name__ == "__main__":
    asyncio.run(main())
