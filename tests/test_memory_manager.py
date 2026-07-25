import pytest

from src.memory.manager import MemoryManager
from src.memory.models import Session
from src.llm.gemini_client import GeminiClient
from src.llm.gemini_embeddings import GeminiEmbeddingClient


@pytest.mark.asyncio
async def test_save_and_retrieve_session(db_session):
    summary = "User asked about renewable energy trends in solar power. [test]"

    llm_client = GeminiClient()
    embedding_client = GeminiEmbeddingClient()
    memory = MemoryManager(llm_client, embedding_client, db_session)

    await memory.save_session(
        summary=summary,
        model="gemini-2.0-flash",
        total_tokens=150
    )

    results = await memory.get_relevant_context("Tell me about solar energy")

    assert len(results) > 0
    assert "solar" in results[0].lower()

    db_session.query(Session).filter_by(summary=summary).delete()
    db_session.commit()