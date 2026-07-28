import asyncio
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.llm.gemini_client import GeminiClient
from src.llm.gemini_embeddings import GeminiEmbeddingClient
from src.agent.modeling import create_modeling_agent
from src.utils.config import config
from src.utils.logger import get_logger
from src.observability.langfuse_client import init_langfuse
from langfuse import get_client

logger = get_logger(__name__)

TABLE_NAME = "turbine_data"
# Dev/prod toggle: TABLE_NAME always points at the real data - DATASET_ID is
# only the memory namespace, and stays separate from the real "turbine_data"
# id while this agent is being developed and tested, so nothing here reads
# or writes the real accumulated knowledge under that id. Flip this to
# TABLE_NAME only for a deliberate, final production run.
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
    temp_files = []

    try:
        logger.info(f"Creating modeling agent | dataset_id={DATASET_ID}")
        agent = await create_modeling_agent(
            llm_client=llm_client,
            embedding_client=embedding_client,
            db_session=db_session,
            table_name=TABLE_NAME,
            dataset_id=DATASET_ID,
            project_brief=PROJECT_BRIEF,
            temp_files=temp_files
        )

        logger.info("Running agent")
        result = await agent.run("Engineer features, train, and validate a model.")

        print("\n=== FINAL ANSWER ===\n")
        print(result)
    finally:
        for path in temp_files:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        logger.info(f"Cleaned up {len(temp_files)} temp files")

        db_session.close()

        get_client().flush()


if __name__ == "__main__":
    asyncio.run(main())
