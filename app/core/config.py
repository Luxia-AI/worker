from typing import ClassVar, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PINECONE_API_KEY: Optional[str] = Field(default=None)
    PINECONE_INDEX_NAME: Optional[str] = Field(default=None)

    GOOGLE_API_KEY: Optional[str] = Field(default=None)
    GOOGLE_CSE_ID: Optional[str] = Field(default=None)

    GROQ_API_KEY: Optional[str] = Field(default=None)

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
