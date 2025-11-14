from typing import Optional

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    pinecone_api_key: Optional[str] = Field(default=None)
    pinecone_index_name: Optional[str] = Field(default=None)

    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()
