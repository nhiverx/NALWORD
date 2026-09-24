"""
Application configuration — loaded from environment variables.

Copy .env.example to .env and fill in values before running in production.
All secrets are read from env; nothing is hardcoded.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Pydantic-settings model — auto-reads from environment / .env file.
    All fields have safe defaults for local development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------
    app_env: str = "development"   # "development" | "staging" | "production"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    secret_key: str = "change-me-in-production"

    # ------------------------------------------------------------------
    # Amazon Bedrock  (Claude via AWS)
    # ------------------------------------------------------------------
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None          # only for temp credentials

    # Model ID — Claude 3 Sonnet via Bedrock (change to Opus/Haiku as needed)
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    bedrock_max_tokens: int = 4096
    bedrock_temperature: float = 0.2                 # low temp for factual, grounded content

    # ------------------------------------------------------------------
    # LiteLLM proxy
    # ------------------------------------------------------------------
    litellm_proxy_url: Optional[str] = None          # e.g. "http://localhost:4000"
    litellm_api_key: Optional[str] = None
    # When set, all Bedrock calls route through LiteLLM proxy instead of direct SDK
    use_litellm_proxy: bool = False

    # ------------------------------------------------------------------
    # Embeddings — llama.cpp server
    # ------------------------------------------------------------------
    llama_cpp_server_url: str = "http://localhost:8080"
    embedding_model_path: Optional[str] = None       # path to GGUF model file
    embedding_dim: int = 768                          # dimensions for the chosen model
    embedding_batch_size: int = 32

    # ------------------------------------------------------------------
    # FAISS vector index
    # ------------------------------------------------------------------
    faiss_index_path: str = "data/faiss_index"       # directory for .index + .meta files
    faiss_top_k: int = 5                              # default retrieval count
    faiss_nprobe: int = 10                            # IVF probe count (tunable)

    # ------------------------------------------------------------------
    # PostgreSQL (metadata + session storage)
    # ------------------------------------------------------------------
    database_url: str = "postgresql://hitl:hitl@localhost:5432/hitl_pipeline"
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # ------------------------------------------------------------------
    # Knowledge base
    # ------------------------------------------------------------------
    kb_json_path: str = "data/knowledge_base.json"   # fallback for demo / dev
    kb_documents_dir: str = "data/documents"          # ingestion source directory

    # ------------------------------------------------------------------
    # Security / audit
    # ------------------------------------------------------------------
    audit_log_path: str = "logs/audit.jsonl"
    require_hitl_approval: bool = True               # if False, skip HITL (TEST ONLY)
    max_generation_attempts: int = 5                 # max retries per step

    # ------------------------------------------------------------------
    # Content validation gate (pre-generation filter)
    # ------------------------------------------------------------------
    enable_content_validation: bool = True
    max_source_age_days: int = 1825                  # 5-year freshness window for KB sources

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def bedrock_available(self) -> bool:
        return bool(self.aws_access_key_id and self.aws_secret_access_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton. Call once at startup."""
    return Settings()
