from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os


class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "ShopWise-Ecommerce-AI"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8002
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Database Settings
    DATABASE_URL: str = "postgresql://ecom_user:ecom_pass@localhost:5432/ecom_db"
    SQLITE_FALLBACK_URL: str = "sqlite:///./data/ecommerce.db"

    # Vector Store Settings
    VECTOR_STORE_DIR: str = "./data/vector_store"
    VECTOR_COLLECTION_NAME: str = "ecom_catalog_policies"

    # Embeddings Configuration
    EMBEDDING_PROVIDER: str = "mock"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # LLM Generation Configuration
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "openai/gpt-oss-20b"
    GROQ_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024

    # Ingestion & Retrieval
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.15

    # Middleware & Security
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    CORS_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
