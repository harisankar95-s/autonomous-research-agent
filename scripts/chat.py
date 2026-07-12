from src.agent.loop import ReactAgent
from src.llm.gemini_client import GeminiClient
from src.tools.search import web_search_tool
from src.tools.base import ToolRegistry
import asyncio
from src.llm.gemini_embeddings import GeminiEmbeddingClient
from src.utils.config import config
from src.memory.manager import MemoryManager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker



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
    engine = create_engine(config.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()
    embedding_client = GeminiEmbeddingClient()
    memory = MemoryManager(llm, embedding_client, db_session)

    conversation_log = []
    while True :
        user_input = input("You: ")
        if user_input == "exit":
            if conversation_log:
                summary = await memory.summarize_conversation(conversation_log)
                token_num_placeholde = 0
                model_name = llm.get_model_name()
                await memory.save_session(
                    summary=summary,
                    model=model_name,
                    total_tokens=token_num_placeholde
                )

            break
        context = await memory.get_relevant_context(user_input)
        response = await agent.run(user_input,context=context)
        conversation_dict = {}
        conversation_dict["query"] = user_input
        conversation_dict["answer"] = response
        conversation_log.append(conversation_dict)
        print(response)

if __name__ == "__main__":
    asyncio.run(main())

