from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

    gemini_api_key: str
    gemini_model: str
    gemini_url: str


@lru_cache
def get_settings() -> Settings:
    return Settings()


config = get_settings()