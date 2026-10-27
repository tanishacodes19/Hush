"""Amazon Bedrock generation client service using Bedrock Runtime Converse API.

Never hardcodes AWS credentials, regions, or model IDs.
Reads configuration from environment variables or Settings.
"""

from typing import Any, Optional
import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings
from app.schemas import Message
from app.services.generation import GenerationConfigError, GenerationServiceError


class BedrockConfigError(GenerationConfigError):
    """Raised when required AWS or Bedrock configuration is missing."""
    pass


class BedrockServiceError(GenerationServiceError):
    """Raised when an error occurs during an Amazon Bedrock API invocation."""
    pass


class BedrockService:
    """Service wrapping Amazon Bedrock Runtime Converse API for text generation."""

    def __init__(
        self,
        region: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.region = region or settings.AWS_REGION
        self.model_id = model_id or settings.BEDROCK_MODEL_ID
        self.timeout_seconds = timeout_seconds or settings.BEDROCK_TIMEOUT_SECONDS
        self._client = client

    def _get_client(self) -> Any:
        """Lazily initialize boto3 bedrock-runtime client with configured timeouts."""
        if self._client is not None:
            return self._client

        if not self.region:
            raise BedrockConfigError(
                "AWS region is not configured. Set AWS_REGION in your environment or .env file."
            )
        if not self.model_id:
            raise BedrockConfigError(
                "Bedrock model ID is not configured. Set BEDROCK_MODEL_ID in your environment or .env file."
            )

        boto_config = Config(
            connect_timeout=self.timeout_seconds,
            read_timeout=self.timeout_seconds,
            retries={"max_attempts": 2, "mode": "standard"},
        )

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            config=boto_config,
        )
        return self._client

    def _format_context(self, context: list[Message]) -> str:
        """Format conversation history into a structured transcript for Bedrock."""
        if not context:
            return "No previous messages in conversation."
        lines = [f"{msg.speaker}: {msg.text}" for msg in context]
        return "\n".join(lines)

    def _invoke_converse(
        self,
        prompt: str,
        system_instruction: str,
        max_tokens: int = 512,
        temperature: float = 0.5,
    ) -> str:
        """Call Bedrock Runtime Converse API and extract output text."""
        if not self.model_id:
            raise BedrockConfigError(
                "Bedrock model ID is not configured. Set BEDROCK_MODEL_ID in your environment or .env file."
            )

        client = self._get_client()

        try:
            response = client.converse(
                modelId=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
                system=[{"text": system_instruction}],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                },
            )

            # Parse content from Converse API response
            output_msg = response.get("output", {}).get("message", {})
            content_blocks = output_msg.get("content", [])
            if not content_blocks:
                raise BedrockServiceError("Bedrock Converse API returned an empty content block.")

            first_text = content_blocks[0].get("text", "").strip()
            if not first_text:
                raise BedrockServiceError("Bedrock Converse API returned empty text.")

            return first_text

        except (ClientError, BotoCoreError) as exc:
            raise BedrockServiceError(
                f"Amazon Bedrock Converse invocation failed: {exc}"
            ) from exc

    def generate_reply(self, context: list[Message]) -> str:
        """Generate a helpful group chat reply when the decision is SPEAK."""
        transcript = self._format_context(context)
        prompt = (
            f"Here is the recent conversation transcript:\n\n"
            f"{transcript}\n\n"
            f"Please provide a helpful, concise contribution to the conversation."
        )
        system_instruction = (
            "You are Hush, a helpful and restrained conversational AI participant in a team chat. "
            "Write a concise, helpful, and direct contribution. Avoid conversational filler or repeating questions."
        )
        return self._invoke_converse(
            prompt=prompt,
            system_instruction=system_instruction,
            max_tokens=512,
            temperature=0.5,
        )

    def generate_clarification(self, context: list[Message]) -> str:
        """Generate one short, focused clarifying question when the decision is ASK."""
        transcript = self._format_context(context)
        prompt = (
            f"Here is the recent conversation transcript:\n\n"
            f"{transcript}\n\n"
            f"Ask exactly ONE short, focused clarifying question to resolve ambiguity and unblock the conversation."
        )
        system_instruction = (
            "You are Hush, a restrained conversational AI assistant. "
            "Generate exactly one short, focused clarifying question to gather missing information. "
            "Do not attempt to answer the question yourself. Do not include introductory or polite filler."
        )
        return self._invoke_converse(
            prompt=prompt,
            system_instruction=system_instruction,
            max_tokens=150,
            temperature=0.3,
        )
