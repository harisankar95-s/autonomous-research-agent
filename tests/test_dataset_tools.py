import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.memory.manager import FactStore, KnowledgeStore
from src.llm.gemini_embeddings import GeminiEmbeddingClient
from src.tools.fact_recording import make_record_facts_tool
from src.tools.knowledge_recording import make_record_observation_tool
from src.utils.config import config


def test_record_facts_tool_writes_to_correct_dataset():
    engine = create_engine(config.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()

    fact_store = FactStore(db_session)
    dataset_id = str(uuid.uuid4())

    tool = make_record_facts_tool(fact_store, dataset_id)

    result = tool.func(
        inferred_task_type="classification",
        candidate_targets=[{"column": "churned", "confidence": 0.95, "reason": "explicitly binary outcome"}]
    )

    assert "recorded" in result.lower()

    saved = fact_store.get_facts(dataset_id)
    assert saved is not None
    assert saved.inferred_task_type == "classification"

    db_session.close()


def test_record_facts_tool_updates_existing_facts_instead_of_raising():
    engine = create_engine(config.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()

    fact_store = FactStore(db_session)
    dataset_id = str(uuid.uuid4())

    tool = make_record_facts_tool(fact_store, dataset_id)

    tool.func(
        inferred_task_type="classification",
        candidate_targets=[{"column": "churned", "confidence": 0.6, "reason": "first guess"}]
    )

    result = tool.func(
        inferred_task_type="regression",
        candidate_targets=[{"column": "tenure", "confidence": 0.9, "reason": "revised conclusion"}]
    )

    assert "recorded" in result.lower()

    saved = fact_store.get_facts(dataset_id)
    assert saved is not None
    assert saved.inferred_task_type == "regression"
    assert saved.candidate_targets[0]["column"] == "tenure"

    db_session.close()


def test_record_facts_tool_saves_and_preserves_schema_notes():
    engine = create_engine(config.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()

    fact_store = FactStore(db_session)
    dataset_id = str(uuid.uuid4())

    tool = make_record_facts_tool(fact_store, dataset_id)

    tool.func(
        inferred_task_type="classification",
        candidate_targets=[],
        schema_notes='Column "System_Name" is mixed-case and must be double-quoted in SQL.'
    )

    saved = fact_store.get_facts(dataset_id)
    assert saved.schema_notes == 'Column "System_Name" is mixed-case and must be double-quoted in SQL.'

    # A later call that omits schema_notes should not erase what's already known
    tool.func(
        inferred_task_type="classification",
        candidate_targets=[]
    )

    saved = fact_store.get_facts(dataset_id)
    assert saved.schema_notes == 'Column "System_Name" is mixed-case and must be double-quoted in SQL.'

    db_session.close()


async def test_record_observation_tool_writes_to_correct_dataset():
    engine = create_engine(config.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()

    embedding_client = GeminiEmbeddingClient()
    knowledge_store = KnowledgeStore(embedding_client, db_session)
    dataset_id = str(uuid.uuid4())

    tool = make_record_observation_tool(knowledge_store, dataset_id)

    result = await tool.func(
        content="Tenure column shows a strong relationship with churn.",
        confidence_score=0.8
    )

    assert "recorded" in result.lower()

    results = await knowledge_store.get_relevant_observations(dataset_id, "tenure and churn relationship")
    assert len(results) > 0

    db_session.close()