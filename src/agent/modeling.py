from src.agent.loop import ReactAgent
from src.llm.client import BaseLLMClient, BaseEmbeddingClient
from sqlalchemy.orm import Session as DBSession
from src.tools.base import ToolRegistry
from src.memory.manager import (
    FactStore, KnowledgeStore, ModelingBriefStore, ImageStore, ModelResultStore,
    compute_missing_fields, compute_missing_model_result_fields
)
from src.prompt_building.compiler import compile_prompt
from src.prompts.modeling import ROLE_PROMPT
from src.tools.sql_query import make_fetch_data_tool, fetch_table_columns, fetch_row_count, detect_entity_columns
from src.tools.code_execution import make_execute_python_code_tool
from src.tools.skill_loading import make_load_skill_tool
from src.tools.model_result import make_finalize_model_result_tool
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _describe_brief(brief) -> str:
    return (
        f"Label status: {brief.label_status} (column: {brief.label_column})\n"
        f"Label notes: {brief.label_notes}\n"
        f"Feature set: {brief.feature_set}\n"
        f"Preprocessing rules: {brief.preprocessing_rules}\n"
        f"Entity heterogeneity notes: {brief.entity_heterogeneity_notes}\n"
        f"Validation anchor: {brief.validation_anchor}\n"
        f"Validation strategy: {brief.validation_strategy}\n"
        f"Confidence: {brief.confidence}"
    )


def _describe_model_result(result) -> str:
    return (
        f"Algorithm: {result.algorithm}\n"
        f"Algorithm rationale: {result.algorithm_rationale}\n"
        f"Applied feature engineering: {result.applied_feature_engineering}\n"
        f"Model path: {result.model_path}\n"
        f"Validation results: {result.validation_results}\n"
        f"Confidence: {result.confidence}\n"
        f"Limitations notes: {result.limitations_notes}"
    )


async def create_modeling_agent(
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
    result_store = ModelResultStore(db_session)

    # Same self-healing rationale as create_data_understanding_agent:
    # re-captured fresh every run rather than trusting a possibly-stale
    # DatasetFacts row.
    columns = fetch_table_columns(table_name)
    row_count = fetch_row_count(table_name)
    entity_columns = detect_entity_columns(table_name, columns, row_count)
    fact_store.set_columns(dataset_id, columns, row_count, entity_columns)

    brief = brief_store.get_brief(dataset_id)
    # Row/entity-coverage gaps depend on a live query_log and aren't
    # persisted - deliberately excluded here too, same reasoning as the
    # initial readiness check in create_data_understanding_agent. This
    # agent consumes a brief that was already validated complete when it
    # was finalized; it only needs to confirm the structural fields are
    # still populated, not re-litigate row coverage.
    brief_gaps = compute_missing_fields(brief, columns=columns)
    if brief is None or brief_gaps:
        raise ValueError(
            f"Cannot start the modeling agent for dataset_id='{dataset_id}': "
            f"its modeling brief is missing or incomplete. Missing: "
            f"{brief_gaps if brief_gaps else ['a modeling brief']}. Run the "
            f"data-understanding agent to completion first."
        )

    column_list = ", ".join(f'{c["name"]} ({c["type"]})' for c in columns)
    facts_text = f"Confirmed columns ({len(columns)} total, {row_count} rows): {column_list}"
    if entity_columns:
        entity_summary = "; ".join(
            f'{e["column"]} ({len(e["distinct_values"])} distinct: {e["distinct_values"][:10]}'
            f'{"..." if len(e["distinct_values"]) > 10 else ""})'
            for e in entity_columns
        )
        facts_text += f"\nCandidate entity/grouping columns: {entity_summary}"
    facts_text += f"\n\nComplete modeling brief for this dataset:\n{_describe_brief(brief)}"

    existing_observations = await knowledge_store.get_relevant_observations(
        dataset_id=dataset_id,
        query=project_brief
    )
    knowledge_list = [obs["content"] for obs in existing_observations]

    existing_result = result_store.get_model_result(dataset_id)
    result_gaps = compute_missing_model_result_fields(existing_result, brief)
    if existing_result and not result_gaps:
        readiness_text = (
            f"A complete model result already exists for this dataset:\n"
            f"{_describe_model_result(existing_result)}\n\n"
            f"Only call finalize_model_result again if new work changes "
            f"something above."
        )
    elif existing_result:
        readiness_text = (
            f"An incomplete model result exists for this dataset:\n"
            f"{_describe_model_result(existing_result)}\n\n"
            f"STILL MISSING: {'; '.join(result_gaps)}"
        )
    else:
        readiness_text = (
            f"No model result exists yet for this dataset.\n"
            f"STILL MISSING: {'; '.join(result_gaps)}"
        )

    registry = ToolRegistry()
    registry.register(make_fetch_data_tool(table_name, temp_files))
    registry.register(make_execute_python_code_tool(
        temp_files, dataset_id, image_store,
        enable_model_artifacts=True,
        timeout_seconds=120,
        mem_limit="2g"
    ))
    registry.register(make_load_skill_tool())
    registry.register(make_finalize_model_result_tool(result_store, dataset_id, brief=brief))

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
        system_prompt=system_prompt,
        completion_tool_name="finalize_model_result"
    )
