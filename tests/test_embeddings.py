import pytest
from src.llm.gemini_embeddings import GeminiEmbeddingClient


@pytest.mark.asyncio
async def test_gemini_generate_embedding():
    client = GeminiEmbeddingClient()
    response = await client.generate_embedding("This is a test")

    assert response is not None
    assert isinstance(response, list)
    assert len(response ) == 3072

