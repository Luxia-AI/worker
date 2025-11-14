from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional

class Settings(BaseSettings):
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    pinecone_secret_key: str
    pinecone_org_name: Optional[str] = None
    pinecone_org_id: Optional[str] = None
    pinecone_project_name: Optional[str] = None
    pinecone_project_id: Optional[str] = None

    model_config = ConfigDict(env_file=".env", extra="ignore")

settings = Settings()