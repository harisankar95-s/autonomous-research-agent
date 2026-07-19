from langfuse import Langfuse
from src.utils.config import config


def init_langfuse() -> None:
    Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host,
    )