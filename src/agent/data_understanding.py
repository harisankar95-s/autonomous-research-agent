from src.agent.loop import ReactAgent
from src.llm.client import BaseLLMClient, BaseEmbeddingClient
from sqlalchemy.orm import Session as DBSession
from src.tools.base import ToolRegistry
from src.memory.manager import FactStore, KnowledgeStore
from src.prompt_building.compiler import compile_prompt
from src.prompts.data_understanding import ROLE_PROMPT
from src.tools.sql_query import make_fetch_data_tool
from src.tools.code_execution import make_execute_python_code_tool
from src.tools.fact_recording import make_record_facts_tool
from src.tools.knowledge_recording import make_record_observation_tool


async def create_data_understanding_agent(
    llm_client: BaseLLMClient,
    embedding_client: BaseEmbeddingClient,
    db_session: DBSession,
    table_name: str,
    dataset_id: str,
    project_brief: str,
    temp_files: list
) -> ReactAgent:
    fact_store = FactStore(db_session)
    knowledge_store = KnowledgeStore(embedding_client, db_session)

    existing_facts = fact_store.get_facts(dataset_id)
    facts_text = ""
    if existing_facts:
        facts_text = (
            f"Task type: {existing_facts.inferred_task_type}\n"
            f"Candidate targets: {existing_facts.candidate_targets}"
        )

    existing_observations = await knowledge_store.get_relevant_observations(
        dataset_id=dataset_id,
        query=project_brief
    )
    knowledge_list = [obs["content"] for obs in existing_observations]

    registry = ToolRegistry()
    registry.register(make_fetch_data_tool(table_name, temp_files))
    registry.register(make_execute_python_code_tool(temp_files))
    registry.register(make_record_facts_tool(fact_store, dataset_id))
    registry.register(make_record_observation_tool(knowledge_store, dataset_id))

    system_prompt = compile_prompt(
        role_prompt=ROLE_PROMPT,
        project_brief=project_brief,
        facts=facts_text,
        knowledge=knowledge_list
    )

    return ReactAgent(
        llm_client=llm_client,
        tool_registry=registry,
        system_prompt=system_prompt
    )