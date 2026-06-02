import abc
from pydantic import BaseModel

from src.utils.logger import get_logger

logger = get_logger(__name__)

class LLMResponse(BaseModel):
    content : str
    model : str
    response_time_ms : float
    prompt_tokens : int = 0
    completion_tokens : int = 0
    total_tokens : int = 0
    finish_reason: str = "stop"
    tool_name : str | None = None
    tool_args: dict | None = None


class BaseLLMClient(abc.ABC):
    @abc.abstractmethod
    async def send_message(
        self,
        message: str,
        system_prompt: str = "",
        tools: list | None = None,
        conversation_history: list | None = None
    ) -> LLMResponse:
        pass

    @abc.abstractmethod
    def get_model_name(self) -> str:
        pass



