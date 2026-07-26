from src.memory.manager import FactStore, KnowledgeStore
from src.llm.gemini_embeddings import GeminiEmbeddingClient


async def test_fact_store_saves_facts(db_session, dataset_id):
    fact_store = FactStore(db_session)

    fact_store.save_facts(
        dataset_id=dataset_id,
        inferred_task_type="regression",
        candidate_targets=[
            {"column": "price", "confidence": 0.9, "reason": "numeric outcome variable"}
        ]
    )


async def test_knowledge_store_saves_and_retrieves(db_session, dataset_id):
    embedding_client = GeminiEmbeddingClient()
    knowledge_store = KnowledgeStore(embedding_client, db_session)

    first_id = await knowledge_store.save_observation(
        dataset_id=dataset_id,
        content="Column area is highly skewed, log transform may help.",
        confidence_score=0.7
    )

    second_id = await knowledge_store.save_observation(
        dataset_id=dataset_id,
        content="After removing outliers, column area is no longer skewed.",
        confidence_score=0.9,
        supersedes_id=first_id
    )

    results = await knowledge_store.get_relevant_observations(
        dataset_id=dataset_id,
        query="is the area column skewed?"
    )

    assert len(results) > 0
    assert any("skewed" in r["content"].lower() for r in results)
    assert second_id != first_id


async def test_knowledge_store_skips_near_duplicate_observation(db_session, dataset_id):
    embedding_client = GeminiEmbeddingClient()
    knowledge_store = KnowledgeStore(embedding_client, db_session)

    first_id = await knowledge_store.save_observation(
        dataset_id=dataset_id,
        content="Found sentinel values (0, -1, -999) in the grid current sensors.",
        confidence_score=0.9
    )

    second_id = await knowledge_store.save_observation(
        dataset_id=dataset_id,
        content="The sensor data contains sentinel values (0, -1, -999) in the grid current columns.",
        confidence_score=0.9
    )

    assert second_id == first_id

    results = await knowledge_store.get_relevant_observations(
        dataset_id=dataset_id,
        query="sentinel values"
    )
    assert len(results) == 1


async def test_knowledge_store_does_not_dedupe_different_entities(db_session, dataset_id):
    embedding_client = GeminiEmbeddingClient()
    knowledge_store = KnowledgeStore(embedding_client, db_session)
    entity_values = ["T02", "T04"]

    first_id = await knowledge_store.save_observation(
        dataset_id=dataset_id,
        content="Turbine T02 shows a clear developing nacelle cooling anomaly starting in December.",
        confidence_score=0.95,
        entity_values=entity_values
    )

    second_id = await knowledge_store.save_observation(
        dataset_id=dataset_id,
        content="Turbine T04 shows a clear developing gearbox anomaly starting in November.",
        confidence_score=0.9,
        entity_values=entity_values
    )

    # Structurally similar sentences about different turbines must not merge
    assert second_id != first_id


async def test_knowledge_store_stores_structured_metadata(db_session, dataset_id):
    embedding_client = GeminiEmbeddingClient()
    knowledge_store = KnowledgeStore(embedding_client, db_session)
    entity_values = ["T02", "T04"]

    obs_id = await knowledge_store.save_observation(
        dataset_id=dataset_id,
        content="Turbine T02 shows a nacelle cooling anomaly, distinct from T04's baseline.",
        confidence_score=0.9,
        entity_values=entity_values,
        analysis_type="thermal",
        evidence=["query: GROUP BY System_Name on TEMP_NACELLE", "nacelle_temp.png"]
    )

    saved = {o.id: o for o in knowledge_store.get_observations(dataset_id)}[obs_id]

    # entities is auto-derived from content, not agent-supplied - both T02
    # and T04 are literally named, even though the dedup guard's own
    # _mentioned_entity would treat this as ambiguous (two entities).
    assert set(saved.entities) == {"T02", "T04"}
    assert saved.analysis_type == "thermal"
    assert saved.evidence == ["query: GROUP BY System_Name on TEMP_NACELLE", "nacelle_temp.png"]


async def test_knowledge_store_structured_metadata_defaults_to_none(db_session, dataset_id):
    embedding_client = GeminiEmbeddingClient()
    knowledge_store = KnowledgeStore(embedding_client, db_session)

    obs_id = await knowledge_store.save_observation(
        dataset_id=dataset_id,
        content="No grouping column mentioned here at all.",
        confidence_score=0.5
    )

    saved = {o.id: o for o in knowledge_store.get_observations(dataset_id)}[obs_id]

    assert saved.entities == []
    assert saved.analysis_type is None
    assert saved.evidence is None
