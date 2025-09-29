# backend/app/deps.py
import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, field_validator, model_validator


class Settings(BaseSettings):
    # --- Core ---
    ENV: str = Field(default="development")  # development | production

    # --- Database ---
    DATABASE_URL: str

    # --- Ollama (local) ---
    OLLAMA_URL: str = "http://localhost:11434"

    # --- Embeddings ---
    EMBED_PROVIDER: str = "local"            # local | openai
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DIM: int = 384                     # 384 local, 1536 openai(prod)

    # --- LLM (general) ---
    LLM_PROVIDER: str = "ollama"             # ollama | openai
    LLM_MODEL: str = Field(
        default="qwen2.5:3b-instruct",
        validation_alias=AliasChoices("LLM_MODEL", "llm_model"),
    )
    LLM_NUM_PREDICT: int = Field(
        default=128,
        validation_alias=AliasChoices("LLM_NUM_PREDICT", "llm_num_predict"),
    )

    # --- OpenAI ---
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4.1-mini"
    LLM_MAX_TOKENS: int = 800

    # --- CORS ---
    FRONTEND_ORIGIN: Optional[str] = None

    # .env locally; real env on Render overrides automatically
    model_config = SettingsConfigDict(
        env_file=".env" if os.getenv("ENV") != "production" else ".env.prod",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- Validators / Normalizers ----------

    @field_validator("EMBED_PROVIDER", "LLM_PROVIDER", mode="before")
    @classmethod
    def _lowercase_providers(cls, v: str) -> str:
        return v.lower().strip() if isinstance(v, str) else v

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def _normalize_database_url(cls, v: str) -> str:
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("DATABASE_URL is not set")
        url = v.strip()

        # Ensure driver hint for SQLAlchemy (safe even if psycopg2 not explicit)
        if url.startswith("postgresql://") and "+psycopg2" not in url:
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)

        # Force SSL for Render external hosts
        if "render.com" in url and "sslmode=" not in url:
            url += ("&" if "?" in url else "?") + "sslmode=require"

        return url

    @model_validator(mode="after")
    def _coerce_embed_dim(self):
        """
        Keep EMBED_DIM consistent with well-known models so table schemas match.
        """
        model = (self.EMBED_MODEL or "").lower()
        provider = (self.EMBED_PROVIDER or "").lower()

        # SentenceTransformers MiniLM (common local)
        if "all-minilm-l6-v2" in model and self.EMBED_DIM != 384:
            self.EMBED_DIM = 384

        # OpenAI text-embedding-3-small
        if provider == "openai" and "text-embedding-3-small" in model and self.EMBED_DIM != 1536:
            self.EMBED_DIM = 1536

        # OpenAI text-embedding-3-large (dimension 3072)
        if provider == "openai" and "text-embedding-3-large" in model and self.EMBED_DIM != 3072:
            self.EMBED_DIM = 3072

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Keep import compatibility with existing code:
settings = get_settings()
