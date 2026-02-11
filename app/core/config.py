from typing import ClassVar, Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PINECONE_API_KEY: Optional[str] = Field(default=None)
    PINECONE_INDEX_NAME: Optional[str] = Field(default=None)
    PINECONE_CLOUD: str = Field(default="aws")
    PINECONE_REGION: str = Field(default="us-east-1")

    GOOGLE_API_KEY: Optional[str] = Field(default=None)
    GOOGLE_CSE_ID: Optional[str] = Field(default=None)

    # Serper.dev API for fallback search (2500 free/month)
    SERPER_API_KEY: Optional[str] = Field(default=None)

    GROQ_API_KEY: Optional[str] = Field(default=None)
    GROQ_ORG_ID: Optional[str] = Field(default=None)
    GROQ_PROJECT_ID: Optional[str] = Field(default=None)
    GROQ_API_KEY_FALLBACK: Optional[str] = Field(default=None)
    GROQ_ORG_ID_FALLBACK: Optional[str] = Field(default=None)
    GROQ_PROJECT_ID_FALLBACK: Optional[str] = Field(default=None)
    LUXIA_CONFIDENCE_MODE: bool = Field(default=False)

    NEO4J_URI: Optional[str] = Field(default=None)
    # Accept both NEO4J_USER and legacy NEO4J_USERNAME env vars
    NEO4J_USER: Optional[str] = Field(default=None, validation_alias=AliasChoices("NEO4J_USER", "NEO4J_USERNAME"))
    NEO4J_PASSWORD: Optional[str] = Field(default=None)

    # Logging system (Redis + SQLite)
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    LOG_DB_PATH: str = Field(default="logs.db", description="SQLite database path for persistent logs")

    # Kafka Configuration
    KAFKA_BOOTSTRAP: str = Field(default="kafka:29092", description="Kafka bootstrap servers")
    KAFKA_SECURITY_PROTOCOL: str = Field(
        default="PLAINTEXT",
        description="Kafka security protocol (PLAINTEXT or SASL_SSL for Azure Event Hubs)",
    )
    KAFKA_SASL_MECHANISM: str = Field(default="PLAIN", description="Kafka SASL mechanism")
    KAFKA_SASL_USERNAME: str = Field(
        default="",
        description="Kafka SASL username ($ConnectionString for Azure Event Hubs)",
    )
    KAFKA_SASL_PASSWORD: str = Field(
        default="",
        description="Kafka SASL password (connection string for Azure Event Hubs)",
    )

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(env_file=".env", extra="ignore")

    def get_kafka_config(self) -> dict:
        """Build Kafka client configuration with optional SASL/SSL for Azure Event Hubs."""
        import ssl

        config = {
            "bootstrap_servers": self.KAFKA_BOOTSTRAP,
        }

        # Azure Event Hubs requires SASL_SSL
        if self.KAFKA_SECURITY_PROTOCOL == "SASL_SSL":
            ssl_context = ssl.create_default_context()
            config.update(
                {
                    "security_protocol": "SASL_SSL",
                    "sasl_mechanism": self.KAFKA_SASL_MECHANISM,
                    "sasl_plain_username": self.KAFKA_SASL_USERNAME,
                    "sasl_plain_password": self.KAFKA_SASL_PASSWORD,
                    "ssl_context": ssl_context,
                }
            )

        return config


settings = Settings()
