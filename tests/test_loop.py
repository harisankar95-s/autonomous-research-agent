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
        _stop_response("done"),
    ])
    registry = ToolRegistry()
    agent = ReactAgent(llm_client=client, tool_registry=registry, system_prompt="test")

    result = await agent.run("do something")

    assert result == "done"
    assert client.calls == 2


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
        _stop_response("done"),
    ])
    registry = ToolRegistry()
    registry.register(tool)
    agent = ReactAgent(llm_client=client, tool_registry=registry, system_prompt="test")

    result = await agent.run("do something")

    assert result == "done"
    assert client.calls == 2


def _brief_tool(complete: bool):
    def finalize_modeling_brief():
        if complete:
            return {"text": "Modeling brief finalized - all required fields are complete.", "complete": True}
        return {"text": "Brief saved, but still missing: feature_set.", "complete": False}

    return Tool(
        name="finalize_modeling_brief",
        description="finalizes the modeling brief",
        parameters={},
        func=finalize_modeling_brief,
    )


async def test_agent_nudges_before_finishing_without_complete_brief():
    client = _ScriptedLLMClient([
        _stop_response("too early"),
        _tool_call_response("finalize_modeling_brief"),
        _stop_response("done"),
    ])
    registry = ToolRegistry()
    registry.register(_brief_tool(complete=True))
    agent = ReactAgent(
        llm_client=client, tool_registry=registry, system_prompt="test",
        completion_tool_name="finalize_modeling_brief"
    )

    result = await agent.run("do something")

    assert result == "done"
    assert client.calls == 3


async def test_agent_finishes_immediately_when_brief_already_complete():
    client = _ScriptedLLMClient([
        _tool_call_response("finalize_modeling_brief"),
        _stop_response("done"),
    ])
    registry = ToolRegistry()
    registry.register(_brief_tool(complete=True))
    agent = ReactAgent(
        llm_client=client, tool_registry=registry, system_prompt="test",
        completion_tool_name="finalize_modeling_brief"
    )

    result = await agent.run("do something")

    assert result == "done"
    assert client.calls == 2


async def test_agent_stops_after_max_nudges_even_if_brief_stays_incomplete():
    client = _ScriptedLLMClient([
        _stop_response("attempt 1"),
        _stop_response("attempt 2"),
        _stop_response("attempt 3"),
        _stop_response("attempt 4"),
    ])
    registry = ToolRegistry()
    registry.register(_brief_tool(complete=False))
    agent = ReactAgent(
        llm_client=client, tool_registry=registry, system_prompt="test",
        completion_tool_name="finalize_modeling_brief"
    )

    result = await agent.run("do something")

    assert result == "attempt 4"
    assert client.calls == 4


def _image_tool(name, filename):
    def make_plot():
        return {"text": "plot saved", "images": [{"data": "ZmFrZV9wbmdfYnl0ZXM=", "filename": filename}]}

    return Tool(name=name, description="produces a plot", parameters={}, func=make_plot)


async def test_older_images_are_pruned_from_conversation_history():
    client = _ScriptedLLMClient([
        _tool_call_response("make_plot_a"),
        _tool_call_response("make_plot_b"),
        _tool_call_response("finalize_modeling_brief"),
        _stop_response("done"),
    ])
    registry = ToolRegistry()
    registry.register(_image_tool("make_plot_a", "figure_a.png"))
    registry.register(_image_tool("make_plot_b", "figure_b.png"))
    registry.register(_brief_tool(complete=True))
    agent = ReactAgent(
        llm_client=client, tool_registry=registry, system_prompt="test",
        completion_tool_name="finalize_modeling_brief"
    )

    await agent.run("do something")

    all_parts = [part for turn in agent.conversation_history for part in turn["parts"]]
    inline_data_parts = [p for p in all_parts if "inlineData" in p]
    placeholder_texts = [p["text"] for p in all_parts if "text" in p and "previously shown" in p["text"]]

    assert len(inline_data_parts) == 1
    assert "figure_a.png" in placeholder_texts[0]


def _completion_tool(name: str, complete: bool):
    def finalize():
        if complete:
            return {"text": f"{name} finalized - all required fields are complete.", "complete": True}
        return {"text": f"{name} saved, but still missing required fields.", "complete": False}

    return Tool(name=name, description=f"finalizes via {name}", parameters={}, func=finalize)


async def test_agent_nudges_before_finishing_without_complete_model_result():
    client = _ScriptedLLMClient([
        _stop_response("too early"),
        _tool_call_response("finalize_model_result"),
        _stop_response("done"),
    ])
    registry = ToolRegistry()
    registry.register(_completion_tool("finalize_model_result", complete=True))
    agent = ReactAgent(
        llm_client=client, tool_registry=registry, system_prompt="test",
        completion_tool_name="finalize_model_result"
    )

    result = await agent.run("do something")

    assert result == "done"
    assert client.calls == 3


async def test_agent_finishes_immediately_when_model_result_already_complete():
    client = _ScriptedLLMClient([
        _tool_call_response("finalize_model_result"),
        _stop_response("done"),
    ])
    registry = ToolRegistry()
    registry.register(_completion_tool("finalize_model_result", complete=True))
    agent = ReactAgent(
        llm_client=client, tool_registry=registry, system_prompt="test",
        completion_tool_name="finalize_model_result"
    )

    result = await agent.run("do something")

    assert result == "done"
    assert client.calls == 2


async def test_agent_has_no_completion_gate_by_default():
    client = _ScriptedLLMClient([_stop_response("done")])
    registry = ToolRegistry()
    registry.register(_completion_tool("finalize_modeling_brief", complete=False))
    agent = ReactAgent(llm_client=client, tool_registry=registry, system_prompt="test")

    result = await agent.run("do something")

    assert result == "done"
    assert client.calls == 1


async def test_agent_completion_gate_inactive_if_tool_not_registered():
    client = _ScriptedLLMClient([_stop_response("done")])
    registry = ToolRegistry()
    agent = ReactAgent(
        llm_client=client, tool_registry=registry, system_prompt="test",
        completion_tool_name="finalize_model_result"
    )

    result = await agent.run("do something")

    assert result == "done"
    assert client.calls == 1