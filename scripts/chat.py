from src.agent.loop import ReactAgent
from src.llm.gemini_client import GeminiClient
from src.tools.search import web_search_tool
from src.tools.base import ToolRegistry
import asyncio



async def main():
    llm = GeminiClient()
    registry = ToolRegistry()
    registry.register(web_search_tool)
    system_prompt = "You are a helpful research assistant. Use the web_search " \
    "tool when you need current information, real-time data, or facts " \
    "you're not certain about. Always answer clearly and cite what you found when relevant."
    agent = ReactAgent(
    llm_client=llm,
    tool_registry=registry,
    system_prompt=system_prompt)

    while True :
        user_input = input("You: ")
        if user_input == "exit":
            break
        response = await agent.run(user_input)
        print(response)

if __name__ == "__main__":
    asyncio.run(main())

