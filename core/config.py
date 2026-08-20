"""
core/config.py
──────────────
Centralised settings loaded from .env via pydantic-settings.
Import `settings` everywhere instead of reading env vars directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenAI ───────────────────────────────────────────
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", alias="OPENAI_MODEL")
    openai_temperature: float = Field(0.2, alias="OPENAI_TEMPERATURE")

    # ── Tavily ───────────────────────────────────────────
    tavily_api_key: str = Field(..., alias="TAVILY_API_KEY")

    # ── ChromaDB ─────────────────────────────────────────
    chroma_persist_dir: str = Field("./data/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("research_docs", alias="CHROMA_COLLECTION_NAME")

    # ── Agent behaviour ──────────────────────────────────
    max_search_results: int = Field(8, alias="MAX_SEARCH_RESULTS")
    max_critique_loops: int = Field(3, alias="MAX_CRITIQUE_LOOPS")
    min_quality_score: float = Field(7.0, alias="MIN_QUALITY_SCORE")
    max_tokens_per_agent: int = Field(4000, alias="MAX_TOKENS_PER_AGENT")

    # ── Paths ────────────────────────────────────────────
    reports_dir: str = Field("./data/reports", alias="REPORTS_DIR")

    # ── Logging ──────────────────────────────────────────
    log_level: str = Field("INFO", alias="LOG_LEVEL")


# Singleton — import this everywhere
settings = Settings()