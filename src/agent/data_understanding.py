from src.agent.loop import ReactAgent
from src.llm.client import BaseLLMClient, BaseEmbeddingClient
from sqlalchemy.orm import Session as DBSession
from src.tools.base import ToolRegistry
from src.memory.manager import FactStore, KnowledgeStore, ModelingBriefStore, ImageStore, compute_missing_fields
from src.prompt_building.compiler import compile_prompt
from src.prompts.data_understanding import ROLE_PROMPT
from src.tools.sql_query import make_fetch_data_tool, fetch_table_columns, fetch_row_count, detect_entity_columns
from src.tools.code_execution import make_execute_python_code_tool
from src.tools.fact_recording import make_record_facts_tool
from src.tools.knowledge_recording import make_record_observation_tool
from src.tools.skill_loading import make_load_skill_tool
from src.tools.modeling_brief import make_finalize_modeling_brief_tool
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
    temp_files: list,
    round_number: int = 1,
    query_log: list[str] | None = None,
    blind_findings: list[str] | None = None
) -> ReactAgent:
    fact_store = FactStore(db_session)
    knowledge_store = KnowledgeStore(embedding_client, db_session)
    brief_store = ModelingBriefStore(db_session)
    image_store = ImageStore(db_session)

    # Captured deterministically via SQL, not transcribed by the LLM - this
    # is the ground truth completeness gets checked against. Re-captured
    # every run (cheap) so it self-heals against schema/data drift rather
    # than trusting a stale snapshot.
    columns = fetch_table_columns(table_name)
    row_count = fetch_row_count(table_name)
    entity_columns = detect_entity_columns(table_name, columns, row_count)
    fact_store.set_columns(dataset_id, columns, row_count, entity_columns)
    logger.info(
        f"Schema captured | dataset_id={dataset_id} | columns={len(columns)} | "
        f"row_count={row_count} | entity_columns={[e['column'] for e in entity_columns]}"
    )

    # A single round's own query_log can't tell a fresh run "row coverage was
    # already established" - but within one multi-round orchestration, the
    # underlying table doesn't change between rounds, so a caller (e.g.
    # round_based_understanding) can pass the SAME list across all its
    # rounds to carry that evidence forward instead of losing it every round.
    if query_log is None:
        query_log = []
    entity_values = [v for e in entity_columns for v in e["distinct_values"]]

    column_list = ", ".join(f'{c["name"]} ({c["type"]})' for c in columns)
    schema_text = f"Confirmed columns ({len(columns)} total, {row_count} rows): {column_list}"
    if entity_columns:
        entity_summary = "; ".join(
            f'{e["column"]} ({len(e["distinct_values"])} distinct: {e["distinct_values"][:10]}'
            f'{"..." if len(e["distinct_values"]) > 10 else ""})'
            for e in entity_columns
        )
        schema_text += f"\nCandidate entity/grouping columns: {entity_summary}"

    existing_facts = fact_store.get_facts(dataset_id)
    facts_text = schema_text
    if existing_facts and existing_facts.inferred_task_type:
        facts_text += (
            f"\nTask type: {existing_facts.inferred_task_type}\n"
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
    # Row-coverage gaps (COUNT(*)/GROUP BY evidence) depend on query_log,
    # which only reflects this run's own activity and isn't persisted across
    # runs - so those are deliberately left out of the *initial* readiness
    # text and instead enforced live by finalize_modeling_brief using this
    # run's own query_log as it accumulates.
    missing_fields = compute_missing_fields(existing_brief, columns=columns)
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

    if round_number > 1 and existing_brief:
        # Unlike the initial readiness block above, this reflects the
        # query_log carried over from earlier rounds in this same
        # orchestration run, so it's safe to include row/entity-coverage
        # gaps here - there's a real, honest answer for whether they were
        # established earlier in this session.
        round_gaps = compute_missing_fields(
            existing_brief, columns=columns, entity_columns=entity_columns, query_log=query_log
        )
        gap_text = "; ".join(round_gaps) if round_gaps else (
            "none - all column and row/entity coverage checks already satisfied"
        )
        readiness_text += (
            f"\n\nThis is round {round_number} of a multi-round analysis process. "
            f"Before adding anything new, pick ONE existing finding above (an "
            f"observation or a modeling-brief field) and deliberately re-verify it "
            f"against the data - confirm it or correct it, and record what you did "
            f"either way. Then focus your remaining effort on what's still missing: "
            f"{gap_text}. When you call finalize_modeling_brief, pass the FULL "
            f"feature_set - every column already covered in the brief above plus "
            f"anything new this round, not just the newly-covered ones; omitting a "
            f"previously-covered column looks like your analysis regressed."
        )

        if blind_findings:
            findings_text = "\n".join(f"- {f}" for f in blind_findings)
            readiness_text += (
                f"\n\nAn independent blind re-analysis (with no access to your "
                f"prior findings) just found the following, which are not yet "
                f"reflected in your own analysis:\n{findings_text}\n\n"
                f"Verify each of these against the data yourself, and make sure "
                f"whichever hold up are properly reflected in your feature_set "
                f"and recorded observations before finalizing again."
            )

    registry = ToolRegistry()
    registry.register(make_fetch_data_tool(table_name, temp_files, query_log))
    registry.register(make_execute_python_code_tool(temp_files, dataset_id, image_store))
    registry.register(make_record_facts_tool(fact_store, dataset_id))
    registry.register(make_record_observation_tool(knowledge_store, dataset_id, entity_values))
    registry.register(make_load_skill_tool())
    registry.register(make_finalize_modeling_brief_tool(
        brief_store, dataset_id,
        columns=columns,
        entity_columns=entity_columns,
        query_log=query_log
    ))

    system_prompt = compile_prompt(
        role_prompt=ROLE_PROMPT,
        project_brief=project_brief,
        facts=facts_text,
        knowledge=knowledge_list,
        readiness=readiness_text
    )

    agent = ReactAgent(
        llm_client=llm_client,
        tool_registry=registry,
        system_prompt=system_prompt,
        completion_tool_name="finalize_modeling_brief"
    )
    # Exposed so a multi-round orchestrator can re-check row/entity coverage
    # against this round's own query activity after the run completes -
    # query_log isn't persisted, so this is the only way to see it later.
    agent.query_log = query_log
    return agent