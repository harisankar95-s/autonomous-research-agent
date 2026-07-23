import pytest
from src.agent.loop import ReactAgent
from src.llm.client import BaseLLMClient, LLMResponse
from src.llm.gemini_client import GeminiClient
from src.tools.base import Tool, ToolRegistry
from src.tools.search import web_search_tool


class _ScriptedLLMClient(BaseLLMClient):
    """Returns a fixed sequence of responses, ignoring actual input - for
    deterministically testing ReactAgent's own control flow without hitting
    a real LLM."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get_model_name(self):
        return "scripted"

    async def send_message(self, message="", system_prompt="", tools=None, conversation_history=None):
        response = self._responses[self.calls]
        self.calls += 1
        return response


def _tool_call_response(tool_name, tool_args=None):
    tool_args = tool_args or {}
    return LLMResponse(
        content="",
        model="scripted",
        response_time_ms=1.0,
        finish_reason="TOOL_CALLS",
        tool_name=tool_name,
        tool_args=tool_args,
        raw_parts=[{"functionCall": {"name": tool_name, "args": tool_args}}],
    )


def _stop_response(content):
    return LLMResponse(
        content=content,
        model="scripted",
        response_time_ms=1.0,
        finish_reason="STOP",
        raw_parts=[{"text": content}],
    )

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


async def test_agent_handles_unknown_tool_gracefully():
    client = _ScriptedLLMClient([
        _tool_call_response("nonexistent_tool"),
        _stop_response("nudge me"),
        _stop_response("done"),
    ])
    registry = ToolRegistry()
    agent = ReactAgent(llm_client=client, tool_registry=registry, system_prompt="test")

    result = await agent.run("do something")

    assert result == "done"
    assert client.calls == 3


async def test_agent_handles_tool_exception_gracefully():
    def failing_tool():
        raise ValueError("boom")

    tool = Tool(
        name="failing_tool",
        description="always raises",
        parameters={},
        func=failing_tool,
    )
    client = _ScriptedLLMClient([
        _tool_call_response("failing_tool"),
        _stop_response("nudge me"),
        _stop_response("done"),
    ])
    registry = ToolRegistry()
    registry.register(tool)
    agent = ReactAgent(llm_client=client, tool_registry=registry, system_prompt="test")

    result = await agent.run("do something")

    assert result == "done"
    assert client.calls == 3


async def test_agent_nudges_before_finishing_without_recording_facts():
    def record_dataset_facts():
        return "Facts recorded successfully."

    tool = Tool(
        name="record_dataset_facts",
        description="records facts",
        parameters={},
        func=record_dataset_facts,
    )
    client = _ScriptedLLMClient([
        _stop_response("too early"),
        _tool_call_response("record_dataset_facts"),
        _stop_response("done"),
    ])
    registry = ToolRegistry()
    registry.register(tool)
    agent = ReactAgent(llm_client=client, tool_registry=registry, system_prompt="test")

    result = await agent.run("do something")

    assert result == "done"
    assert client.calls == 3