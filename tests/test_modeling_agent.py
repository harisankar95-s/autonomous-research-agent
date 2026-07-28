import uuid

import pytest
from sqlalchemy import create_engine, text

from src.agent.modeling import create_modeling_agent
from src.memory.manager import ModelingBriefStore
from src.utils.config import config


@pytest.fixture
def minimal_table():
    """A throwaway table just real enough for schema introspection to
    succeed - these tests are about the brief-completeness precondition,
    not the table's own contents, so it doesn't need realistic data."""
    table_name = f"test_minimal_{uuid.uuid4().hex[:8]}"
    engine = create_engine(config.database_url)
    with engine.begin() as conn:
        conn.execute(text(f'CREATE TABLE "{table_name}" ("a" TEXT)'))
        conn.execute(text(f'INSERT INTO "{table_name}" ("a") VALUES (:v)'), [{"v": "x"}])
        conn.execute(text(f'GRANT SELECT ON "{table_name}" TO dataset_reader'))

    yield table_name

    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))


async def test_create_modeling_agent_raises_when_brief_missing(db_session, dataset_id, minimal_table):
    with pytest.raises(ValueError, match="missing or incomplete"):
        await create_modeling_agent(
            llm_client=None,
            embedding_client=None,
            db_session=db_session,
            table_name=minimal_table,
            dataset_id=dataset_id,
            project_brief="test project",
            temp_files=[]
        )


async def test_create_modeling_agent_raises_when_brief_incomplete(db_session, dataset_id, minimal_table):
    brief_store = ModelingBriefStore(db_session)
    brief_store.save_brief(
        dataset_id=dataset_id,
        label_status="undetermined",
        label_column=None,
        label_notes="first pass, not finished",
        feature_set=[],
        preprocessing_rules=[],
        validation_strategy="",
        confidence=0.5
    )

    with pytest.raises(ValueError, match="missing or incomplete"):
        await create_modeling_agent(
            llm_client=None,
            embedding_client=None,
            db_session=db_session,
            table_name=minimal_table,
            dataset_id=dataset_id,
            project_brief="test project",
            temp_files=[]
        )
