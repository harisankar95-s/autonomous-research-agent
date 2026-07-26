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
        response = await self._send_with_retry(url, body)
        data     =  self._parse_response(response)
        return data

    async def _send_with_retry(self, url: str, body: dict) -> httpx.Response:
        # Same retry pattern as GeminiClient._send_with_retry - this client
        # previously had none, which surfaced as a real (not just
        # theoretical) KeyError on a 429 response body during live testing.
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await self.client.post(url, json=body)

                if response.status_code in (429, 500, 503):
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Status {response.status_code} | attempt {attempt + 1}/{max_retries} | waiting {wait:.1f}s")
                    await asyncio.sleep(wait)
                    last_error = response.status_code
                    continue

                return response

            except httpx.TransportError as e:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Network error ({type(e).__name__}) | attempt {attempt + 1}/{max_retries} | "
                    f"waiting {wait:.1f}s"
                )
                await asyncio.sleep(wait)
                last_error = f"{type(e).__name__}: {e}"
                continue

        else:
            raise Exception(f"Failed after {max_retries} attempts | last_error={last_error}")

    def _parse_response(self, response: httpx.Response) -> list[float]:
        data = response.json()
        return data["embedding"]["values"]