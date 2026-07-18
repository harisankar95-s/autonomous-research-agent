import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.memory.manager import FactStore, KnowledgeStore
from src.llm.gemini_embeddings import GeminiEmbeddingClient
from src.utils.config import config


async def test_fact_store_saves_facts():
    engine = create_engine(config.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()

    fact_store = FactStore(db_session)
    dataset_id = str(uuid.uuid4())

    fact_store.save_facts(
        dataset_id=dataset_id,
        inferred_task_type="regression",
        candidate_targets=[
            {"column": "price", "confidence": 0.9, "reason": "numeric outcome variable"}
        ]
    )

    db_session.close()


async def test_knowledge_store_saves_and_retrieves():
    engine = create_engine(config.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()

    embedding_client = GeminiEmbeddingClient()
    knowledge_store = KnowledgeStore(embedding_client, db_session)
    dataset_id = str(uuid.uuid4())

    first_id = await knowledge_store.save_observation(
        dataset_id=dataset_id,
        content="Column area is highly skewed, log transform may help.",
        confidence_score=0.7
    )

    second_id = await knowledge_store.save_observation(
        dataset_id=dataset_id,
        content="After removing outliers, column area is no longer skewed.",
        confidence_score=0.9,
        supersedes_id=first_id
    )

    results = await knowledge_store.get_relevant_observations(
        dataset_id=dataset_id,
        query="is the area column skewed?"
    )

    assert len(results) > 0
    assert any("skewed" in r["content"].lower() for r in results)
    assert second_id != first_id

    db_session.close()