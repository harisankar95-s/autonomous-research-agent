from src.llm.client import BaseLLMClient, LLMResponse
from src.tools.base import ToolRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ReactAgent:
    def __init__(self,llm_client:BaseLLMClient,tool_registry:ToolRegistry,system_prompt: str = ""):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.max_iterations = 10
        self.conversation_history = []
        
    async def run(self, query: str, context: list[str] | None = None) -> str:
        logger.info(f"Start ReAct loop | query={query}")
        
        if context is None:
            context = []
        
        if context:
            memory_text = "\n".join(f"- {item}" for item in context)
            full_system_prompt = f"{self.system_prompt}\n\nRelevant memory:\n{memory_text}"
        else:
            full_system_prompt = self.system_prompt
        
        self.conversation_history.append({"role": "user", "parts": [{"text": query}]})
        
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            response = await self.llm_client.send_message(
                message="",
                system_prompt=full_system_prompt,
                tools=self.tool_registry.get_all_schemas(),
                conversation_history=self.conversation_history
            )
            
            if response.tool_name:
                logger.info(f"Tool call | tool={response.tool_name} | args={response.tool_args}")
                tool = self.tool_registry.get_tool(response.tool_name)
                tool_result = tool.func(**response.tool_args)
                logger.info(f"Tool result received | tool={response.tool_name}")
                
                self.conversation_history.append({
                    "role": "model",
                    "parts": response.raw_parts
                })
                self.conversation_history.append({
                    "role": "user",
                    "parts": [{"functionResponse": {"name": response.tool_name, "response": {"output": tool_result}}}]
                })
                
            elif response.finish_reason == "STOP":
                logger.info(f"Agent finished | iterations={iteration + 1}")
                self.conversation_history.append({
                    "role": "model",
                    "parts": response.raw_parts
                })
                return response.content
        
        return "Max iterations reached without a final answer"