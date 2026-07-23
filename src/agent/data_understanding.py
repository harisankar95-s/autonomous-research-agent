from src.agent.loop import ReactAgent
from src.llm.client import BaseLLMClient, BaseEmbeddingClient
from sqlalchemy.orm import Session as DBSession
from src.tools.base import ToolRegistry
from src.memory.manager import FactStore, KnowledgeStore, ModelingBriefStore, ImageStore, compute_missing_fields
from src.prompt_building.compiler import compile_prompt
from src.prompts.data_understanding import ROLE_PROMPT
from src.tools.sql_query import make_fetch_data_tool
from src.tools.code_execution import make_execute_python_code_tool
from src.tools.fact_recording import make_record_facts_tool
from src.tools.knowledge_recording import make_record_observation_tool
from src.tools.skill_loading import make_load_skill_tool
from src.tools.modeling_brief import make_finalize_modeling_brief_tool


def _describe_brief(brief) -> str:
    return (
        f"Label status: {brief.label_status} (column: {brief.label_column})\n"
        f"Label notes: {brief.label_notes}\n"
        f"Feature set: {brief.feature_set}\n"
        f"Preprocessing rules: {brief.preprocessing_rules}\n"
        f"Validation strategy: {brief.validation_strategy}\n"
        f"Confidence: {brief.confidence}"
    )


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
    brief_store = ModelingBriefStore(db_session)
    image_store = ImageStore(db_session)

    existing_facts = fact_store.get_facts(dataset_id)
    facts_text = ""
    if existing_facts:
        facts_text = (
            f"Task type: {existing_facts.inferred_task_type}\n"
            f"Candidate targets: {existing_facts.candidate_targets}"
        )
        if existing_facts.schema_notes:
            facts_text += f"\nSchema notes: {existing_facts.schema_notes}"

    existing_observations = await knowledge_store.get_relevant_observations(
        dataset_id=dataset_id,
        query=project_brief
    )
    knowledge_list = [obs["content"] for obs in existing_observations]

    existing_brief = brief_store.get_brief(dataset_id)
    missing_fields = compute_missing_fields(existing_brief)
    if existing_brief and not missing_fields:
        readiness_text = (
            f"A complete modeling brief already exists for this dataset:\n"
            f"{_describe_brief(existing_brief)}\n\n"
            f"Only call finalize_modeling_brief again if new analysis "
            f"contradicts something above."
        )
    elif existing_brief:
        readiness_text = (
            f"An incomplete modeling brief exists for this dataset:\n"
            f"{_describe_brief(existing_brief)}\n\n"
            f"STILL MISSING: {'; '.join(missing_fields)}"
        )
    else:
        readiness_text = (
            f"No modeling brief exists yet for this dataset.\n"
            f"STILL MISSING: {'; '.join(missing_fields)}"
        )

    registry = ToolRegistry()
    registry.register(make_fetch_data_tool(table_name, temp_files))
    registry.register(make_execute_python_code_tool(temp_files, dataset_id, image_store))
    registry.register(make_record_facts_tool(fact_store, dataset_id))
    registry.register(make_record_observation_tool(knowledge_store, dataset_id))
    registry.register(make_load_skill_tool())
    registry.register(make_finalize_modeling_brief_tool(brief_store, dataset_id))

    system_prompt = compile_prompt(
        role_prompt=ROLE_PROMPT,
        project_brief=project_brief,
        facts=facts_text,
        knowledge=knowledge_list,
        readiness=readiness_text
    )

    return ReactAgent(
        llm_client=llm_client,
        tool_registry=registry,
        system_prompt=system_prompt
    )