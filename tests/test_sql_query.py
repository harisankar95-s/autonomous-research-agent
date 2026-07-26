import uuid

import pytest
from sqlalchemy import create_engine, text

from src.tools.sql_query import (
    make_fetch_data_tool,
    fetch_table_columns,
    fetch_row_count,
    detect_entity_columns,
)
from src.utils.config import config

TURBINES = [f"T{i:02d}" for i in range(1, 11)]
ROWS_PER_TURBINE = 30


@pytest.fixture
def fleet_table():
    """A throwaway table shaped like the real sensor-fleet data these tools
    were built against (a text entity column, a timestamp, a numeric
    reading), created fresh in every environment - these tests used to
    assert against the real turbine_data table directly, which only exists
    in the local dev database, not CI's freshly-created one. Queried through
    dataset_reader_url just like the real thing, so a GRANT is required
    since the creating role and the reader role differ."""
    table_name = f"test_fleet_{uuid.uuid4().hex[:8]}"
    engine = create_engine(config.database_url)
    rows = [
        {"name": t, "power": float(i)}
        for t in TURBINES
        for i in range(ROWS_PER_TURBINE)
    ]
    with engine.begin() as conn:
        conn.execute(text(
            f'CREATE TABLE "{table_name}" ('
            f'"System_Name" TEXT, "Timestamp" TIMESTAMP, "ACTIVE_POWER" FLOAT)'
        ))
        conn.execute(
            text(
                f'INSERT INTO "{table_name}" '
                f'("System_Name", "Timestamp", "ACTIVE_POWER") '
                f'VALUES (:name, now(), :power)'
            ),
            rows
        )
        conn.execute(text(f'GRANT SELECT ON "{table_name}" TO dataset_reader'))

    yield table_name

    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))


def test_fetch_table_columns_returns_real_schema(fleet_table):
    columns = fetch_table_columns(fleet_table)

    names = {c["name"] for c in columns}
    assert "System_Name" in names
    assert "Timestamp" in names
    assert "ACTIVE_POWER" in names
    assert all("type" in c for c in columns)


def test_fetch_row_count_returns_positive_int(fleet_table):
    row_count = fetch_row_count(fleet_table)

    assert isinstance(row_count, int)
    assert row_count > 0


def test_detect_entity_columns_finds_system_name(fleet_table):
    columns = fetch_table_columns(fleet_table)
    row_count = fetch_row_count(fleet_table)

    entity_columns = detect_entity_columns(fleet_table, columns, row_count)

    entity_names = {e["column"] for e in entity_columns}
    assert "System_Name" in entity_names

    system_name_entity = next(e for e in entity_columns if e["column"] == "System_Name")
    assert len(system_name_entity["distinct_values"]) == len(TURBINES)


def test_fetch_data_tool_appends_to_query_log(fleet_table):
    temp_files = []
    query_log = []
    tool = make_fetch_data_tool(fleet_table, temp_files, query_log)

    tool.func(sql=f'SELECT COUNT(*) FROM "{fleet_table}"')

    assert len(query_log) == 1
    assert "count(*)" in query_log[0].lower()


def test_fetch_data_tool_works_without_query_log(fleet_table):
    temp_files = []
    tool = make_fetch_data_tool(fleet_table, temp_files)

    response = tool.func(sql=f'SELECT COUNT(*) FROM "{fleet_table}"')

    assert "Fetched" in response
