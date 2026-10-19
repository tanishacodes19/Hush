"""Configuration settings for Hush backend."""

import os
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

    # Decider mode: 'stub' for development, 'real' for teammate hush_gate integration
    HUSH_DECIDER_MODE: str = "stub"

    # Generation mode: 'stub' for offline mock generation, 'bedrock' for Amazon Bedrock
    HUSH_GENERATION_MODE: str = "stub"

    # AWS Bedrock configuration (loaded from environment or .env, never hardcoded)
    AWS_REGION: str | None = None
    BEDROCK_MODEL_ID: str | None = None
    BEDROCK_TIMEOUT_SECONDS: float = 15.0

    # CORS origins
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


settings = Settings()
