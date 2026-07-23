import pandas as pd
from sqlalchemy import create_engine

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

FILE_PATH = "data/fleet_10_turbines_renamed.parquet"
TABLE_NAME = "turbine_data"


def main():
    logger.info(f"Reading parquet file | path={FILE_PATH}")
    df = pd.read_parquet(FILE_PATH)
    logger.info(f"Loaded dataframe | rows={len(df)} | columns={len(df.columns)}")

    engine = create_engine(config.database_url)

    logger.info(f"Writing to Postgres | table={TABLE_NAME}")
    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
    logger.info("Load complete")


if __name__ == "__main__":
    main()