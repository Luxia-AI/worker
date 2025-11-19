from typing import ClassVar, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    pinecone_api_key: Optional[str] = Field(default=None)
    pinecone_index_name: Optional[str] = Field(default=None)

    google_api_key: Optional[str] = Field(default=None)
    google_cse_id: Optional[str] = Field(default=None)

    groq_api_key: Optional[str] = Field(default=None)

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
