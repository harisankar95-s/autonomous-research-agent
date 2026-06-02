import time
import asyncio
import random
import httpx

from src.llm.client import BaseLLMClient, LLMResponse
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GeminiClient(BaseLLMClient):

    def __init__(self):
        self.api_key  = config.gemini_api_key
        self.model    = config.gemini_model
        self.base_url = config.gemini_url
        self.client   = httpx.AsyncClient(timeout =30.0)

    def get_model_name(self) ->str:
        return self.model
    
    async def send_message(self, message: str) -> LLMResponse:
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        body = self._build_request_body(message)
        logger.info(f"Sending message to Gemini | model={self.model}")
        start_time = time.time()
        response = await self._send_with_retry(url, body)
        response_time_ms = (time.time() - start_time) * 1000
        return self._parse_response(response, response_time_ms)
    
    def _build_request_body(self, message: str) -> dict:
        return {
            "contents": [
                {
                    "parts": [
                        {"text": message}
                    ]
                }
            ]
        }
    async def _send_with_retry(self,url: str,body:dict) ->httpx.Response:
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await self.client.post(url, json=body)
                
                if response.status_code == 429 or response.status_code == 500:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Status {response.status_code} | attempt {attempt + 1}/{max_retries} | waiting {wait:.1f}s")
                    await asyncio.sleep(wait)
                    last_error = response.status_code
                    continue

                return response
                
            except httpx.TimeoutException:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Timeout | attempt {attempt + 1}/{max_retries} | waiting {wait:.1f}s")
                await asyncio.sleep(wait)
                last_error = "timeout"
                continue

        else:
            raise Exception(f"Failed after {max_retries} attempts | last_error={last_error}")
                
    
    def _parse_response(self, response: httpx.Response, response_time_ms: float) -> LLMResponse:
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        logger.debug(f"Parsed response | tokens={usage.get('totalTokenCount', 0)} | finish={data['candidates'][0].get('finishReason', 'STOP')}")
        return LLMResponse(
            content=text,
            model=self.model,
            response_time_ms=response_time_ms,
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
            total_tokens=usage.get("totalTokenCount", 0),
            finish_reason=data["candidates"][0].get("finishReason", "STOP")
        )

    