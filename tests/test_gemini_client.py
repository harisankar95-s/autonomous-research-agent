import pytest
from src.agent.gemini_client import GeminiClient

@pytest.mark.asyncio
async def test_gemini_client_sends_message():
    client = GeminiClient()
    response = await client.send_message("Say hello in one word")
    
    assert response.content is not None
    assert len(response.content) > 0
    assert response.model == "gemini-2.5-flash"
    assert response.total_tokens > 0
    assert response.response_time_ms > 0