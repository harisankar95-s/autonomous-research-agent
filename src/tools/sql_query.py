import tempfile
import pandas as pd
from sqlalchemy import create_engine, text

from src.tools.base import Tool
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

TEXT_TYPES = ("text", "character varying", "varchar", "character", "char", "name")
ENTITY_MAX_DISTINCT = 1000
ENTITY_MAX_FRACTION = 0.05


def fetch_table_columns(table_name: str) -> list[dict]:
    """Deterministically capture the real column list via information_schema,
    rather than trusting the LLM to transcribe it correctly into prose."""
    engine = create_engine(config.dataset_reader_url)
    query = text(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = :table_name ORDER BY ordinal_position"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"table_name": table_name}).fetchall()
    return [{"name": r[0], "type": r[1]} for r in rows]


def fetch_row_count(table_name: str) -> int:
    """The true row count, established directly, not from a sample."""
    engine = create_engine(config.dataset_reader_url)
    with engine.connect() as conn:
        return conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()


def detect_entity_columns(table_name: str, columns: list[dict], row_count: int) -> list[dict]:
    """Candidate grouping/entity columns (e.g. a turbine or store id) - text
    columns with distinct-value counts low relative to row count, so a
    category/identifier rather than free text. Captures the full distinct
    value list for each, so completeness checks can verify every entity was
    actually considered, not just a sample of them."""
    if row_count == 0:
        return []

    engine = create_engine(config.dataset_reader_url)
    entity_columns = []
    with engine.connect() as conn:
        for col in columns:
            if col["type"].lower() not in TEXT_TYPES:
                continue
            name = col["name"]
            distinct_count = conn.execute(
                text(f'SELECT COUNT(DISTINCT "{name}") FROM "{table_name}"')
            ).scalar()
            if distinct_count is None or distinct_count <= 1:
                continue
            if distinct_count > ENTITY_MAX_DISTINCT or distinct_count > row_count * ENTITY_MAX_FRACTION:
                continue
            values = conn.execute(
                text(f'SELECT DISTINCT "{name}" FROM "{table_name}" ORDER BY "{name}"')
            ).fetchall()
            entity_columns.append({
                "column": name,
                "distinct_values": [v[0] for v in values]
            })
    return entity_columns


def make_fetch_data_tool(table_name: str, temp_files: list, query_log: list | None = None) -> Tool:
    engine = create_engine(config.dataset_reader_url)

    def fetch_data(sql: str) -> str:
        logger.info(f"Fetching data | table={table_name} | sql={sql}")
        if query_log is not None:
            query_log.append(sql)

        try:
            df = pd.read_sql(sql, engine)
        except Exception as e:
            logger.warning(f"Fetch failed | error={str(e)[:200]}")
            return f"Query failed: {str(e)}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        temp_files.append(temp_path)

        logger.info(f"Data fetched | rows={len(df)} | file={temp_path}")
        return (
            f"Fetched {len(df)} rows, {len(df.columns)} columns. "
            f"Saved to {temp_path}. Use execute_python_code to read this file "
            f"with pandas and perform any analysis or calculation - do not "
            f"attempt to compute anything from this message yourself."
        )

    return Tool(
        name="fetch_data",
        description=(
            f"Fetch data from the {table_name} table by writing any SQL "
            "SELECT query. This does not return the data itself - it saves "
            "the result to a file and tells you the file path and row count. "
            "You must then use execute_python_code to read that file with "
            "pandas and perform any actual analysis, calculation, or "
            "aggregation. Never try to reason about or compute answers from "
            "the row/column counts alone."
        ),
        parameters={"sql": f"any SELECT query against the {table_name} table"},
        func=fetch_data
    )
