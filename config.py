from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    anthropic_api_key: str
    news_api_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
    )


settings = Settings()
