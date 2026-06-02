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
    
    async def send_message(
            self, 
            message: str,
            system_prompt: str = "",
            tools: list | None = None,
            conversation_history :list | None = None
            ) -> LLMResponse:
        if tools is None:
            tools = []
        if conversation_history is None:
            conversation_history = []
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        body = self._build_request_body(message, system_prompt, tools, conversation_history)
        logger.info(f"Sending message to Gemini | model={self.model}")
        start_time = time.time()
        response = await self._send_with_retry(url, body)
        response_time_ms = (time.time() - start_time) * 1000
        return self._parse_response(response, response_time_ms)
    
    
    def _build_request_body(
            self,
            message: str,
            system_prompt: str = "",
            tools: list | None = None,
            conversation_history: list | None = None
        ) -> dict:
            if message:
                contents = conversation_history + [{"role": "user", "parts": [{"text": message}]}]
            else:
                contents = conversation_history
            body = {"contents": contents}
            
            if system_prompt:
                body["system_instruction"] = {
                    "parts": [{"text": system_prompt}]
                }
            if tools:
                body["tools"] = [
                    {
                        "function_declarations": [
                            {
                                "name": tool["name"],
                                "description": tool["description"],
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        param: {"type": "string", "description": desc}
                                        for param, desc in tool["parameters"].items()
                                    }
                                }
                            }
                            for tool in tools
                        ]
                    }
                ]
            return body
                
    async def _send_with_retry(self,url: str,body:dict) ->httpx.Response:
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
        logger.debug(f"Raw response: {data}")
        usage = data.get("usageMetadata", {})
        candidate = data["candidates"][0]
        part = candidate["content"]["parts"][0]
        
        if "functionCall" in part:
            tool_name = part["functionCall"]["name"]
            tool_args = part["functionCall"]["args"]
            content = ""
            finish_reason = "TOOL_CALLS"
        else:
            content = part["text"]
            tool_name = None
            tool_args = None
            finish_reason = candidate.get("finishReason", "STOP")
        
        logger.debug(f"Parsed response | tokens={usage.get('totalTokenCount', 0)} | finish={finish_reason}")
        
        return LLMResponse(
            content=content,
            model=self.model,
            response_time_ms=response_time_ms,
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
            total_tokens=usage.get("totalTokenCount", 0),
            finish_reason=finish_reason,
            tool_name=tool_name,
            tool_args=tool_args,
            raw_parts=candidate["content"]["parts"]
        )