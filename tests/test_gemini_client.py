import httpx
import pytest
from src.llm.gemini_client import GeminiClient

@pytest.mark.asyncio
async def test_gemini_client_sends_message():
    client = GeminiClient()
    response = await client.send_message("Say hello in one word")

    assert response.content is not None
    assert len(response.content) > 0
    assert response.model is not None
    assert response.total_tokens > 0
    assert response.response_time_ms > 0


def test_parse_response_reads_all_parts_not_just_first():
    client = GeminiClient()
    raw = {
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "Here's my reasoning before calling a tool."},
                    {"functionCall": {"name": "fetch_data", "args": {"sql": "SELECT 1"}}},
                ]
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
    }
    response = httpx.Response(200, json=raw, request=httpx.Request("POST", "https://example.com"))

    parsed = client._parse_response(response, response_time_ms=1.0)

    assert parsed.tool_name == "fetch_data"
    assert parsed.tool_args == {"sql": "SELECT 1"}
    assert "reasoning" in parsed.content
    assert len(parsed.raw_parts) == 2