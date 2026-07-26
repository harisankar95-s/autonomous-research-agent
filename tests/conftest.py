import os
import shutil
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.memory.manager import ANALYSIS_IMAGES_DIR
from src.memory.models import AnalysisImage, DatasetFacts, ModelingBrief, Observation, UnderstandingRound
from src.utils.config import config


@pytest.fixture
def db_session():
    engine = create_engine(config.database_url)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def dataset_id(db_session):
    """A dataset_id scoped to a single test, prefixed so it's unmistakably
    test data. Tests run against the real dev database (there's no separate
    test database), so every row and file created under this id is deleted
    on teardown - otherwise the shared database accumulates permanent debris
    from every test run, which is exactly what happened before this fixture
    existed."""
    generated_id = f"test-{uuid.uuid4()}"
    yield generated_id

    # Round-based tests derive a second id (f"{id}__blind_check") for the
    # blind validation round - clean up both under this one fixture rather
    # than making every round-based test repeat its own teardown.
    for scoped_id in (generated_id, f"{generated_id}__blind_check"):
        db_session.query(Observation).filter_by(dataset_id=scoped_id).delete()
        db_session.query(ModelingBrief).filter_by(dataset_id=scoped_id).delete()
        db_session.query(DatasetFacts).filter_by(dataset_id=scoped_id).delete()
        db_session.query(AnalysisImage).filter_by(dataset_id=scoped_id).delete()
        db_session.query(UnderstandingRound).filter_by(dataset_id=scoped_id).delete()
        db_session.commit()

        image_dir = os.path.join(ANALYSIS_IMAGES_DIR, scoped_id)
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
