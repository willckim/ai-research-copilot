# backend/app/deps.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from typing import Optional

class Settings(BaseSettings):
    # --- Core ---
    ENV: str = "development"  # development | production

    # --- Database ---
    DATABASE_URL: str

    # --- Ollama (local) ---
    OLLAMA_URL: str = "http://localhost:11434"

    # --- Embeddings ---
    EMBED_PROVIDER: str = "local"   # local | openai
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DIM: int = 384            # 384 local, 1536 openai prod

    # --- LLM (general) ---
    LLM_PROVIDER: str = "ollama"    # ollama | openai
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

    # Load from .env or .env.prod depending on ENV
    model_config = SettingsConfigDict(
        env_file=".env" if os.getenv("ENV") != "production" else ".env.prod",
        extra="ignore"
    )

settings = Settings()
