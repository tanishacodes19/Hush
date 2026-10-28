"""Configuration settings for Hush backend."""

import json
import os
from typing import Any
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment or defaults."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    HUSH_ENV: str = "production"
    ENVIRONMENT: str = "production"
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

    # Limits and bounds
    HUSH_MAX_MESSAGE_LENGTH: int = 4096
    HUSH_MAX_REQUEST_BODY_BYTES: int = 65536
    HUSH_MAX_CONVERSATIONS: int = 1000

    # Rate limiting
    HUSH_RATE_LIMIT_ENABLED: bool = True
    HUSH_RATE_LIMIT_REQUESTS: int = 60
    HUSH_RATE_LIMIT_WINDOW_SECONDS: float = 60.0
    HUSH_RATE_LIMIT_MAX_CLIENTS: int = 10000

    # Proxy headers
    HUSH_TRUST_PROXY_HEADERS: bool = False
    HUSH_TRUSTED_PROXY_HOPS: int = 1

    # CORS origins
    CORS_ORIGINS: list[str] = Field(default_factory=list)

    @field_validator("HUSH_TRUSTED_PROXY_HOPS", mode="before")
    @classmethod
    def validate_trusted_proxy_hops(cls, v: Any) -> int:
        try:
            val = int(v)
        except (ValueError, TypeError):
            raise ValueError(f"HUSH_TRUSTED_PROXY_HOPS must be a positive integer (>= 1), got: {v!r}")
        if val < 1:
            raise ValueError(f"HUSH_TRUSTED_PROXY_HOPS must be a positive integer (>= 1), got: {val}")
        return val

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

    @model_validator(mode="after")
    def validate_environment_and_cors(self) -> "Settings":
        # Synchronize HUSH_ENV and ENVIRONMENT
        env = (self.HUSH_ENV or "").strip().lower()
        if not env:
            env = (self.ENVIRONMENT or "production").strip().lower()
        self.HUSH_ENV = env
        self.ENVIRONMENT = env

        if self.HUSH_ENV == "development":
            if not self.CORS_ORIGINS:
                self.CORS_ORIGINS = [
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                ]
        else:
            # Production mode: CORS_ORIGINS must be set and non-empty
            if not self.CORS_ORIGINS or all(not str(origin).strip() for origin in self.CORS_ORIGINS):
                raise ValueError(
                    "CORS_ORIGINS must be set to explicit allowed origins in production mode (HUSH_ENV != 'development')."
                )
            if "*" in self.CORS_ORIGINS or any(str(origin).strip() == "*" for origin in self.CORS_ORIGINS):
                raise ValueError(
                    "Wildcard CORS_ORIGINS='*' is not permitted in production mode. Specify explicit origin domains."
                )
        return self


settings = Settings(HUSH_ENV=os.environ.get("HUSH_ENV", "development"))
