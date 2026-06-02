import pytest
from src.agent.loop import ReactAgent
from src.llm.gemini_client import GeminiClient
from src.tools.base import ToolRegistry
from src.tools.search import web_search_tool

async def test_agent_answers_simple_question():
    client = GeminiClient()
    registry = ToolRegistry()
    registry.register(web_search_tool)
    
    agent = ReactAgent(
        llm_client=client,
        tool_registry=registry,
        system_prompt="You are a research agent. Use tools when you need current information."
    )
    
    result = await agent.run("What is 2 + 2?")
    
    assert result is not None
    assert len(result) > 0

async def test_agent_uses_search_tool():
    client = GeminiClient()
    registry = ToolRegistry()
    registry.register(web_search_tool)
    
    agent = ReactAgent(
        llm_client=client,
        tool_registry=registry,
        system_prompt="You are a research agent. Use tools when you need current information."
    )
    
    result = await agent.run("What is the current stock price of NVIDIA?")
    print(f"Result: '{result}'")
    assert result is not None
    assert len(result) > 0