import time
import asyncio
import random
import httpx

from src.llm.client import BaseEmbeddingClient
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GeminiEmbeddingClient(BaseEmbeddingClient):
    
    def __init__(self):
        self.api_key = config.gemini_api_key
        self.model = config.gemini_embedding_model
        self.base_url = config.gemini_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def generate_embedding(self, text: str) -> list[float]:
        url = f"{self.base_url}/{self.model}:embedContent?key={self.api_key}"
        body = {
                    "model": f"models/{self.model}",
                    "content": {
                        "parts": [{"text": text}]
                    }
                }
        logger.info(f"Generating embedding | model={self.model}")
        # NOTE: unlike GeminiClient, this has no retry logic for transient
        # failures (429/503). Deferred deliberately when this client was
        # first built. CI hit a real 503 on 2026-07-18, confirming this is
        # a real gap, not just theoretical — worth adding exponential
        # backoff (same pattern as GeminiClient._send_with_retry) before
        # this client sees production traffic.
        response = await self.client.post(url, json=body)
        data     =  self._parse_response(response)
        return data
    
    def _parse_response(self, response: httpx.Response) -> list[float]:
        data = response.json()
        return data["embedding"]["values"]