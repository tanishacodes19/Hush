"""Configuration settings for Hush backend."""

import json
import os
from typing import Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment or defaults."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Decider mode: 'classifier' for production trained model, 'stub' for dev keyword heuristics
    HUSH_DECIDER_MODE: str = "classifier"
    HUSH_MODEL_PATH: str | None = None

    # Generation mode: 'openai' for hosted LLM, 'bedrock' for Amazon Bedrock, 'stub' for offline mock
    HUSH_GENERATION_MODE: str = "openai"

    # Hosted OpenAI-compatible LLM configuration (never hardcoded, loaded from environment or .env)
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_TIMEOUT_SECONDS: float = 15.0

    # AWS Bedrock configuration (loaded from environment or .env, never hardcoded)
    AWS_REGION: str | None = None
    BEDROCK_MODEL_ID: str | None = None
    BEDROCK_TIMEOUT_SECONDS: float = 15.0

    # CORS origins
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Allow CORS_ORIGINS to be provided as a comma-separated string, JSON array, or list."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


settings = Settings()
