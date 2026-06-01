from typing import Callable, Any
from pydantic import BaseModel,ConfigDict
from src.utils.logger import get_logger

logger = get_logger(__name__)

class Tool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name : str
    description : str
    parameters : dict
    func : Callable

class ToolRegistry():
    def __init__(self):
        self.tools: dict[str, Tool] = {}
    def register(self, tool: Tool)-> None:
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    def get_tool(self, name: str) -> Tool:
        tool = self.tools.get(name)
        if tool is None:
            logger.warning(f"Tool not found: {name}")
        else:
            logger.debug(f"Retrieved tool: {name}")
        return tool
    def get_all_schemas(self) -> list:
        logger.debug(f"Returning {len(self.tools)} tool schemas")
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]




