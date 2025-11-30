from typing import ClassVar, Literal, Optional

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

    # LLM Configuration
    LLM_SERVICE_TYPE: Literal["groq", "ollama", "hybrid"] = Field(
        default="hybrid", description="LLM service type: 'groq', 'ollama', or 'hybrid'"
    )
    LLM_CONFIDENCE_THRESHOLD: float = Field(default=0.7, description="Confidence threshold for LLM responses")
    OLLAMA_HOST: str = Field(default="ollama", description="Ollama server hostname")
    OLLAMA_PORT: int = Field(default=11434, description="Ollama server port")
    OLLAMA_MODEL: str = Field(default="mistral", description="Ollama model name to use")

    # Logging system (Redis + SQLite)
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    LOG_DB_PATH: str = Field(default="logs.db", description="SQLite database path for persistent logs")

    KAFKA_BOOTSTRAP: str = Field(default="localhost:9092", description="Kafka bootstrap servers")
    JOBS_TOPIC: str = Field(default="jobs.to_worker", description="Kafka topic for incoming jobs")
    RESULTS_TOPIC: str = Field(default="jobs.results", description="Kafka topic for job results")
    PROGRESS_TOPIC: str = Field(default="jobs.progress", description="Kafka topic for job progress updates")
    DLQ_TOPIC: str = Field(default="jobs.worker_failed", description="Kafka topic for dead letter queue")

    # Worker configuration
    WORKER_GROUP_ID: str = Field(default="worker-group-1", description="Consumer group ID for this worker")
    MAX_JOB_ATTEMPTS: int = Field(default=3, description="Maximum number of retry attempts for failed jobs")
    CONSUMER_TIMEOUT_MS: int = Field(default=1000, description="Consumer session timeout in milliseconds")

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# Module-level constants for backward compatibility with consumer.py
GROUP_ID = settings.WORKER_GROUP_ID
MAX_ATTEMPTS = settings.MAX_JOB_ATTEMPTS
JOBS_TOPIC = settings.JOBS_TOPIC
RESULTS_TOPIC = settings.RESULTS_TOPIC
PROGRESS_TOPIC = settings.PROGRESS_TOPIC
DLQ_TOPIC = settings.DLQ_TOPIC
