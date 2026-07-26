import os

from sqlalchemy.orm import Session as DBSession

from src.agent.data_understanding import create_data_understanding_agent
from src.llm.client import BaseLLMClient, BaseEmbeddingClient
from src.memory.manager import (
    KnowledgeStore,
    ModelingBriefStore,
    UnderstandingRoundStore,
    compute_missing_fields,
)
from src.tools.sql_query import detect_entity_columns, fetch_row_count, fetch_table_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Two consecutive quiet rounds, not one - a single round with nothing new
# doesn't rule out that the next one finds something, since round guidance
# rotates through the gap list rather than exhausting it in one pass.
CONVERGENCE_STREAK = 2
DEFAULT_MAX_ROUNDS = 10
# A blind check that finds something new triggers reconciliation and another
# pass through the round loop, then another blind check - this bounds how
# many times that outer cycle can repeat. A cost ceiling, not the primary
# stopping mechanism: the primary mechanism is a blind check that genuinely
# agrees nothing is left.
DEFAULT_MAX_BLIND_CYCLES = 5


def _covered_columns(brief) -> set:
    if brief is None:
        return set()
    return {f.get("column") for f in brief.feature_set if isinstance(f, dict)}


async def _run_one_round(
    llm_client: BaseLLMClient,
    embedding_client: BaseEmbeddingClient,
    db_session: DBSession,
    table_name: str,
    dataset_id: str,
    project_brief: str,
    round_number: int,
    query_log: list[str],
    blind_findings: list[str] | None = None
):
    temp_files: list = []
    try:
        agent = await create_data_understanding_agent(
            llm_client=llm_client,
            embedding_client=embedding_client,
            db_session=db_session,
            table_name=table_name,
            dataset_id=dataset_id,
            project_brief=project_brief,
            temp_files=temp_files,
            round_number=round_number,
            query_log=query_log,
            blind_findings=blind_findings
        )
        await agent.run("Analyze this dataset and record your findings.")
        return agent
    finally:
        for path in temp_files:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


async def _run_round_loop(
    llm_client: BaseLLMClient,
    embedding_client: BaseEmbeddingClient,
    db_session: DBSession,
    table_name: str,
    dataset_id: str,
    project_brief: str,
    columns: list,
    entity_columns: list,
    entity_values: list[str],
    query_log: list[str],
    round_store: UnderstandingRoundStore,
    start_round: int,
    max_rounds: int,
    blind_findings: list[str] | None = None
) -> tuple[int, bool]:
    """Runs rounds (numbering continues from start_round) until
    CONVERGENCE_STREAK consecutive quiet rounds, or max_rounds rounds within
    THIS invocation - max_rounds is a fresh budget each time this is called,
    since each reconciliation cycle deserves its own room to work through
    whatever the blind check just surfaced. Returns (last_round_number,
    converged)."""
    knowledge_store = KnowledgeStore(embedding_client, db_session)
    brief_store = ModelingBriefStore(db_session)

    round_number = start_round
    quiet_streak = 0
    converged = False
    rounds_this_cycle = 0

    while rounds_this_cycle < max_rounds:
        round_number += 1
        rounds_this_cycle += 1
        logger.info(f"Starting understanding round {round_number} | dataset_id={dataset_id}")

        covered_before = _covered_columns(brief_store.get_brief(dataset_id))
        obs_before = knowledge_store.count_observations(dataset_id)

        await _run_one_round(
            llm_client, embedding_client, db_session, table_name, dataset_id,
            project_brief, round_number, query_log,
            blind_findings=blind_findings if rounds_this_cycle == 1 else None
        )

        brief_after = brief_store.get_brief(dataset_id)
        covered_after = _covered_columns(brief_after)
        new_columns_covered = sorted(covered_after - covered_before)
        new_observations_count = knowledge_store.count_observations(dataset_id) - obs_before
        remaining_gaps = compute_missing_fields(
            brief_after, columns=columns, entity_columns=entity_columns, query_log=query_log
        )

        converged_this_round = (
            len(new_columns_covered) == 0
            and new_observations_count == 0
            and len(remaining_gaps) == 0
        )
        quiet_streak = quiet_streak + 1 if converged_this_round else 0
        converged = quiet_streak >= CONVERGENCE_STREAK

        round_store.record_round(
            dataset_id=dataset_id,
            round_number=round_number,
            new_columns_covered=new_columns_covered,
            new_observations_count=new_observations_count,
            row_gaps_remaining=len(remaining_gaps),
            converged=converged
        )

        if converged:
            logger.info(f"Round loop converged at round {round_number} | dataset_id={dataset_id}")
            break

    if not converged:
        logger.warning(
            f"Round loop hit its max_rounds={max_rounds} budget without converging | "
            f"dataset_id={dataset_id} | last_round={round_number}"
        )

    return round_number, converged


def _compare_briefs(real_brief, blind_brief) -> dict:
    if real_brief is None or blind_brief is None:
        return {"label_status_match": None, "feature_overlap": None}

    real_cols = _covered_columns(real_brief)
    blind_cols = _covered_columns(blind_brief)
    union = len(real_cols | blind_cols) or 1

    return {
        "label_status_match": real_brief.label_status == blind_brief.label_status,
        "feature_overlap": len(real_cols & blind_cols) / union
    }


async def _diff_blind_findings(
    knowledge_store: KnowledgeStore,
    dataset_id: str,
    blind_dataset_id: str,
    real_brief,
    blind_brief,
    entity_values: list[str]
) -> dict:
    """What the blind check found that the main dataset doesn't already
    have: columns covered in the blind brief but not the real one, and
    observations that survive the same duplicate check save_observation
    uses (reusing each row's already-computed embedding rather than
    re-embedding its text)."""
    new_columns = sorted(_covered_columns(blind_brief) - _covered_columns(real_brief))

    new_observations = []
    for obs in knowledge_store.get_observations(blind_dataset_id):
        duplicate = await knowledge_store.find_duplicate(
            dataset_id, obs.content, obs.embedding, entity_values
        )
        if duplicate is None:
            new_observations.append(obs)

    return {"new_columns": new_columns, "new_observations": new_observations}


async def _reconcile_blind_findings(
    knowledge_store: KnowledgeStore,
    dataset_id: str,
    new_observations: list,
    entity_values: list[str]
) -> list[str]:
    """Copies the blind check's genuinely-new observations into the main
    dataset - they get naturally re-deduped against main's current state on
    the way in, since save_observation runs the same check again. Returns
    the merged texts so the caller can surface them explicitly to the next
    round (retrieval-similarity search alone isn't a strong enough
    guarantee that a just-merged finding will actually get shown)."""
    merged_texts = []
    for obs in new_observations:
        await knowledge_store.save_observation(
            dataset_id=dataset_id,
            content=obs.content,
            confidence_score=obs.confidence_score,
            entity_values=entity_values
        )
        merged_texts.append(obs.content)
    return merged_texts


async def run_round_based_understanding(
    llm_client: BaseLLMClient,
    embedding_client: BaseEmbeddingClient,
    db_session: DBSession,
    table_name: str,
    dataset_id: str,
    project_brief: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_blind_cycles: int = DEFAULT_MAX_BLIND_CYCLES
) -> dict:
    """Sequential 'boosting' rounds, each told what prior rounds found and
    what's still missing, until CONVERGENCE_STREAK consecutive rounds find
    nothing new. Then a
    blind round - fresh dataset_id, no accumulated context - independently
    re-checks the same data. If it finds something the sequential chain
    missed, that finding is merged into the real dataset and the round loop
    resumes to absorb it, then the blind check runs again. Only a blind
    check that itself finds nothing new ends the process - that's the real
    definition of done, not just "the chain stopped finding things by its
    own account." max_rounds and max_blind_cycles are cost ceilings, not the
    primary stopping mechanism."""
    knowledge_store = KnowledgeStore(embedding_client, db_session)
    brief_store = ModelingBriefStore(db_session)
    round_store = UnderstandingRoundStore(db_session)

    columns = fetch_table_columns(table_name)
    row_count = fetch_row_count(table_name)
    entity_columns = detect_entity_columns(table_name, columns, row_count)
    entity_values = [v for e in entity_columns for v in e["distinct_values"]]

    blind_dataset_id = f"{dataset_id}__blind_check"
    query_log: list[str] = []
    round_number = 0
    blind_cycle = 0
    fully_converged = False
    comparison = {"label_status_match": None, "feature_overlap": None}
    blind_findings: list[str] | None = None

    while True:
        round_number, round_loop_converged = await _run_round_loop(
            llm_client, embedding_client, db_session, table_name, dataset_id, project_brief,
            columns, entity_columns, entity_values, query_log, round_store,
            start_round=round_number, max_rounds=max_rounds, blind_findings=blind_findings
        )
        blind_findings = None

        if not round_loop_converged:
            # Hit its own max_rounds budget without converging - stop rather
            # than run an uninformative blind check against an admittedly
            # incomplete state.
            break

        if blind_cycle >= max_blind_cycles:
            logger.warning(
                f"Reached max_blind_cycles={max_blind_cycles} | dataset_id={dataset_id}"
            )
            break
        blind_cycle += 1

        logger.info(f"Running blind check | blind_dataset_id={blind_dataset_id} | cycle={blind_cycle}")
        await _run_one_round(
            llm_client, embedding_client, db_session, table_name, blind_dataset_id,
            project_brief, round_number=1, query_log=[]
        )

        real_brief = brief_store.get_brief(dataset_id)
        blind_brief = brief_store.get_brief(blind_dataset_id)
        comparison = _compare_briefs(real_brief, blind_brief)
        diff = await _diff_blind_findings(
            knowledge_store, dataset_id, blind_dataset_id, real_brief, blind_brief, entity_values
        )

        if not diff["new_columns"] and not diff["new_observations"]:
            fully_converged = True
            logger.info(
                f"Blind check agrees - nothing new found | dataset_id={dataset_id} | "
                f"blind_cycle={blind_cycle}"
            )
            break

        logger.info(
            f"Blind check found new information - reconciling | dataset_id={dataset_id} | "
            f"new_columns={diff['new_columns']} | new_observations={len(diff['new_observations'])}"
        )
        merged_texts = await _reconcile_blind_findings(
            knowledge_store, dataset_id, diff["new_observations"], entity_values
        )
        blind_findings = merged_texts + [
            f"column '{c}' was covered by the blind check but is not yet in your feature_set"
            for c in diff["new_columns"]
        ]

    return {
        "rounds_run": round_number,
        "blind_cycles": blind_cycle,
        "converged": fully_converged,
        "blind_check": {"blind_dataset_id": blind_dataset_id, **comparison}
    }
