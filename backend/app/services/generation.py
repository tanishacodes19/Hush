"""Text generation service interface for Hush.

Provides generation logic for SPEAK (reply) and ASK (clarification question).
Supports hosted OpenAI-compatible LLM generation mode, Amazon Bedrock mode,
and explicit development stub mode.
"""

from typing import Protocol
from app.config import settings
from app.schemas import Message


class GenerationConfigError(ValueError):
    """Raised when required LLM provider configuration or credentials are missing."""
    pass


class GenerationServiceError(RuntimeError):
    """Raised when an error occurs during an LLM generation invocation."""
    pass


class GenerationService(Protocol):
    """Protocol defining the generation service interface."""

    def generate_reply(self, context: list[Message]) -> str:
        """Generate a response when decision is SPEAK."""
        ...

    def generate_clarification(self, context: list[Message]) -> str:
        """Generate a single focused clarifying question when decision is ASK."""
        ...


class DevelopmentStubGenerator:
    """Explicit development stub generator for testing without AWS Bedrock or OpenAI.

    All generated text is explicitly prefixed with '[SIMULATED DEMO]'
    so it cannot be mistaken for real LLM output.
    """

    def generate_reply(self, context: list[Message]) -> str:
        last_msg = context[-1].text if context else "conversation"
        return f"[SIMULATED DEMO] Generated reply addressing: '{last_msg}'"

    def generate_clarification(self, context: list[Message]) -> str:
        last_msg = context[-1].text if context else "conversation"
        return f"[SIMULATED DEMO] Clarifying question: What specific details can you share about '{last_msg}'?"


def get_generator(mode: str | None = None) -> GenerationService:
    """Factory returning active generation service based on configuration."""
    active_mode = (mode or settings.HUSH_GENERATION_MODE).lower()
    if active_mode in ("openai", "hosted", "llm"):
        from app.services.openai_service import OpenAIService
        return OpenAIService()
    if active_mode == "bedrock":
        from app.services.bedrock import BedrockService
        return BedrockService()
    if active_mode in ("stub", "development"):
        return DevelopmentStubGenerator()
    raise GenerationConfigError(
        f"Unsupported generation mode '{active_mode}'. Expected 'openai', 'bedrock', or 'stub'."
    )
