import asyncio
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.llm.gemini_client import GeminiClient
from src.llm.gemini_embeddings import GeminiEmbeddingClient
from src.agent.data_understanding import create_data_understanding_agent
from src.utils.config import config
from src.utils.logger import get_logger
from src.observability.langfuse_client import init_langfuse
from langfuse import get_client

logger = get_logger(__name__)

TABLE_NAME = "titanic"
PROJECT_BRIEF = (
    "This is data about passengers on the Titanic. We want to understand "
    "what factors were associated with whether someone survived or not."
)


async def main():
    init_langfuse()
    engine = create_engine(config.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()

    llm_client = GeminiClient()
    embedding_client = GeminiEmbeddingClient()

    dataset_id = str(uuid.uuid4())
    temp_files = []

    logger.info(f"Creating Data Understanding agent | dataset_id={dataset_id}")
    agent = await create_data_understanding_agent(
        llm_client=llm_client,
        embedding_client=embedding_client,
        db_session=db_session,
        table_name=TABLE_NAME,
        dataset_id=dataset_id,
        project_brief=PROJECT_BRIEF,
        temp_files=temp_files
    )

    logger.info("Running agent")
    result = await agent.run("Analyze this dataset and record your findings.")

    print("\n=== FINAL ANSWER ===\n")
    print(result)

    import os
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