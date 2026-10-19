"""Text generation service interface for Hush.

Provides generation logic for SPEAK (reply) and ASK (clarification question).
Supports development stub mode and real Amazon Bedrock generation mode.
"""

from typing import Protocol
from app.config import settings
from app.schemas import Message


class GenerationService(Protocol):
    """Protocol defining the generation service interface."""

    def generate_reply(self, context: list[Message]) -> str:
        """Generate a response when decision is SPEAK."""
        ...

    def generate_clarification(self, context: list[Message]) -> str:
        """Generate a single focused clarifying question when decision is ASK."""
        ...


class DevelopmentStubGenerator:
    """Explicit development stub generator for testing without AWS Bedrock.

    All generated text is explicitly prefixed with '[Development Stub]'
    so it cannot be mistaken for real LLM output.
    """

    def generate_reply(self, context: list[Message]) -> str:
        last_msg = context[-1].text if context else "conversation"
        return f"[Development Stub] Generated reply addressing: '{last_msg}'"

    def generate_clarification(self, context: list[Message]) -> str:
        last_msg = context[-1].text if context else "conversation"
        return f"[Development Stub] Clarifying question: What specific details can you share about '{last_msg}'?"


def get_generator(mode: str | None = None) -> GenerationService:
    """Factory returning active generation service based on configuration."""
    active_mode = (mode or settings.HUSH_GENERATION_MODE).lower()
    if active_mode == "bedrock":
        from app.services.bedrock import BedrockService
        return BedrockService()
    return DevelopmentStubGenerator()
