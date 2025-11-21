from typing import ClassVar, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PINECONE_API_KEY: Optional[str] = Field(default=None)
    PINECONE_INDEX_NAME: Optional[str] = Field(default=None)

    GOOGLE_API_KEY: Optional[str] = Field(default=None)
    GOOGLE_CSE_ID: Optional[str] = Field(default=None)

    GROQ_API_KEY: Optional[str] = Field(default=None)

    NEO4J_URI: Optional[str] = Field(default=None)
    NEO4J_USER: Optional[str] = Field(default=None)
    NEO4J_PASSWORD: Optional[str] = Field(default=None)

    # Logging system (Redis + SQLite)
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    LOG_DB_PATH: str = Field(default="logs.db", description="SQLite database path for persistent logs")

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
