from src.tools.sql_query import (
    make_fetch_data_tool,
    fetch_table_columns,
    fetch_row_count,
    detect_entity_columns,
)


def test_fetch_table_columns_returns_real_schema():
    columns = fetch_table_columns("turbine_data")

    names = {c["name"] for c in columns}
    assert "System_Name" in names
    assert "Timestamp" in names
    assert "ACTIVE_POWER" in names
    assert all("type" in c for c in columns)


def test_fetch_row_count_returns_positive_int():
    row_count = fetch_row_count("turbine_data")

    assert isinstance(row_count, int)
    assert row_count > 0


def test_detect_entity_columns_finds_system_name():
    columns = fetch_table_columns("turbine_data")
    row_count = fetch_row_count("turbine_data")

    entity_columns = detect_entity_columns("turbine_data", columns, row_count)

    entity_names = {e["column"] for e in entity_columns}
    assert "System_Name" in entity_names

    system_name_entity = next(e for e in entity_columns if e["column"] == "System_Name")
    assert len(system_name_entity["distinct_values"]) == 10


def test_fetch_data_tool_appends_to_query_log():
    temp_files = []
    query_log = []
    tool = make_fetch_data_tool("turbine_data", temp_files, query_log)

    tool.func(sql='SELECT COUNT(*) FROM "turbine_data"')

    assert len(query_log) == 1
    assert "count(*)" in query_log[0].lower()


def test_fetch_data_tool_works_without_query_log():
    temp_files = []
    tool = make_fetch_data_tool("turbine_data", temp_files)

    response = tool.func(sql='SELECT COUNT(*) FROM "turbine_data"')

    assert "Fetched" in response
