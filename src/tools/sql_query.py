import tempfile
import pandas as pd
from sqlalchemy import create_engine

from src.tools.base import Tool
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def make_fetch_data_tool(table_name: str, temp_files: list) -> Tool:
    engine = create_engine(config.dataset_reader_url)

    def fetch_data(sql: str) -> str:
        logger.info(f"Fetching data | table={table_name} | sql={sql}")

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