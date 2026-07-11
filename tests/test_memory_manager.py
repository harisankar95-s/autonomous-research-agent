import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.memory.manager import MemoryManager
from src.llm.gemini_client import GeminiClient
from src.llm.gemini_embeddings import GeminiEmbeddingClient
from src.utils.config import config


@pytest.mark.asyncio
async def test_save_and_retrieve_session():
    engine = create_engine(config.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()

    llm_client = GeminiClient()
    embedding_client = GeminiEmbeddingClient()
    memory = MemoryManager(llm_client, embedding_client, db_session)

    await memory.save_session(
        summary="User asked about renewable energy trends in solar power.",
        model="gemini-2.0-flash",
        total_tokens=150
    )

    results = await memory.get_relevant_context("Tell me about solar energy")

    assert len(results) > 0
    assert "solar" in results[0].lower()

    db_session.close()