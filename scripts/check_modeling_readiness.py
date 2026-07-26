import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.memory.manager import FactStore, ModelingBriefStore, compute_missing_fields
from src.utils.config import config


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.check_modeling_readiness <dataset_id>")
        sys.exit(1)

    dataset_id = sys.argv[1]

    engine = create_engine(config.database_url)
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()

    facts = FactStore(db_session).get_facts(dataset_id)
    columns = facts.columns if facts else None

    brief_store = ModelingBriefStore(db_session)
    brief = brief_store.get_brief(dataset_id)
    # Row/entity-coverage gaps need a live query_log from an actual run to
    # check honestly - not available here, so this checks column coverage
    # and field completeness only.
    missing = compute_missing_fields(brief, columns=columns)

    if not missing:
        print(f"READY: modeling brief for '{dataset_id}' is complete.")
    else:
        status = "incomplete" if brief else "does not exist"
        print(f"NOT READY: modeling brief for '{dataset_id}' {status}.")
        print("Missing:")
        for item in missing:
            print(f"  - {item}")

    db_session.close()
    sys.exit(0 if not missing else 1)


if __name__ == "__main__":
    main()
