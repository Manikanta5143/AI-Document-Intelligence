from functools import lru_cache
from typing import Optional
from urllib.parse import quote_plus
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables and .env file.

    Uses Pydantic's BaseSettings for automatic environment parsing, validation,
    and type safety.
    """

    # Application settings
    app_name: str = "AI Insurance Document Intelligence Assistant"
    environment: str = "development"
    app_version: str = "1.0.0"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    # MySQL Database Configuration
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "insurance_ai_db"
    database_url_override: Optional[str] = None

    @property
    def database_url(self) -> str:
        """Construct a valid MySQL SQLAlchemy connection URL.

        Handles special characters in username or password using urllib.parse.quote_plus.
        """
        if self.database_url_override:
            return self.database_url_override

        user = quote_plus(self.mysql_user) if self.mysql_user else "root"
        password_str = f":{quote_plus(self.mysql_password)}" if self.mysql_password else ""
        return f"mysql+pymysql://{user}{password_str}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"

    # AWS & S3 Configuration (Phase 3)
    aws_profile: Optional[str] = None
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    s3_bucket_name: Optional[str] = None
    s3_prefix: str = "documents"

    # Document Chunking Settings (Phase 3)
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Amazon Bedrock & Embedding Settings (Phase 4)
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimension: int = 1024

    # ChromaDB Vector Store Settings (Phase 4)
    chroma_persist_directory: str = "./data/chromadb"
    chroma_collection_name: str = "insurance_documents"

    # Amazon Bedrock LLM Settings (Phase 5 - Nova Lite)
    bedrock_model_id: str = "amazon.nova-lite-v1:0"
    bedrock_max_tokens: int = 2048
    bedrock_temperature: float = 0.0

    # RAG Pipeline Settings (Phase 6)
    rag_top_k: int = 5

    # Snowflake Analytics Configuration (Phase 7)
    snowflake_account: Optional[str] = None
    snowflake_user: Optional[str] = None
    snowflake_password: Optional[str] = None
    snowflake_warehouse: Optional[str] = "COMPUTE_WH"
    snowflake_database: Optional[str] = "INSURANCE_DB"
    snowflake_schema: Optional[str] = "PUBLIC"
    snowflake_role: Optional[str] = "ACCOUNTADMIN"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
