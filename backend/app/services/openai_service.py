"""Hosted OpenAI-compatible LLM generation client service.

Communicates with OpenAI or any OpenAI-compatible API (e.g. Groq, OpenRouter, Mistral,
or local Ollama/vLLM) using HTTP POST to /v1/chat/completions.

Never hardcodes API keys, secrets, or provider URLs.
All credentials and configurations are loaded via environment variables or Settings.
"""

from typing import Any, Optional
from urllib.parse import urlparse
import httpx

from app.config import settings
from app.schemas import Message
from app.services.generation import GenerationConfigError, GenerationServiceError


class OpenAIService:
    """Service wrapping OpenAI-compatible Chat Completions API for text generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model_id = model_id or settings.OPENAI_MODEL_ID
        self.base_url = self._validate_and_normalize_base_url(base_url or settings.OPENAI_BASE_URL)
        self.timeout_seconds = timeout_seconds or settings.OPENAI_TIMEOUT_SECONDS
        self._client = client

    @staticmethod
    def _validate_and_normalize_base_url(url: str) -> str:
        """Validate and normalize base URL to prevent insecure or accidental credential disclosure."""
        clean = (url or "").strip().rstrip("/")
        if not clean:
            raise GenerationConfigError("OpenAI base URL cannot be empty.")
        parsed = urlparse(clean)
        if parsed.scheme not in ("http", "https"):
            raise GenerationConfigError(
                f"Invalid OpenAI base URL scheme '{parsed.scheme}'. Must be 'http' or 'https'."
            )
        if not parsed.netloc:
            raise GenerationConfigError(
                f"Invalid OpenAI base URL '{clean}'. Hostname is missing."
            )
        hostname = (parsed.hostname or "").lower()
        # Security: Disallow cleartext HTTP for remote endpoints to prevent credential leakage
        if parsed.scheme == "http" and hostname not in ("localhost", "127.0.0.1", "::1", "testserver"):
            raise GenerationConfigError(
                "Insecure plain HTTP base URL is not permitted for remote providers. Use HTTPS."
            )
        return clean

    def _get_client(self) -> httpx.Client:
        """Lazily initialize or return httpx.Client with configured timeouts and safe redirect policy."""
        if not self.api_key or not self.api_key.strip():
            raise GenerationConfigError(
                "OpenAI API key is not configured. Set OPENAI_API_KEY in your environment or .env file."
            )
        if not self.model_id or not self.model_id.strip():
            raise GenerationConfigError(
                "OpenAI model ID is not configured. Set OPENAI_MODEL_ID in your environment or .env file."
            )

        if self._client is None:
            # follow_redirects=False ensures credentials are never forwarded to an unintended host
            self._client = httpx.Client(timeout=self.timeout_seconds, follow_redirects=False)
        return self._client

    def _format_context(self, context: list[Message]) -> str:
        """Format conversation history into a structured transcript."""
        if not context:
            return "No previous messages in conversation."
        lines = [f"{msg.speaker}: {msg.text}" for msg in context]
        return "\n".join(lines)

    def _invoke_chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.5,
    ) -> str:
        """Invoke chat completions endpoint and extract generated text."""
        client = self._get_client()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model_id.strip(),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = client.post(url, json=payload, headers=headers)
            if 300 <= response.status_code < 400:
                raise GenerationServiceError(
                    f"OpenAI API returned unexpected redirect (HTTP {response.status_code}). "
                    "Redirects are disallowed to prevent credential exposure."
                )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code == 401:
                raise GenerationConfigError(
                    "OpenAI authentication failed: invalid or expired API key."
                ) from exc
            if status_code == 429:
                raise GenerationServiceError(
                    "OpenAI rate limit exceeded or quota exhausted. Please retry shortly."
                ) from exc
            if status_code >= 500:
                raise GenerationServiceError(
                    f"OpenAI provider service error (HTTP {status_code})."
                ) from exc
            raise GenerationServiceError(
                f"OpenAI API request failed with status code {status_code}."
            ) from exc
        except httpx.TimeoutException as exc:
            raise GenerationServiceError(
                f"OpenAI API request timed out after {self.timeout_seconds}s."
            ) from exc
        except httpx.RequestError as exc:
            raise GenerationServiceError(
                "OpenAI API network connection error. Check connectivity and provider URL."
            ) from exc
        except Exception as exc:
            raise GenerationServiceError(f"Unexpected error in OpenAI response parsing: {exc}") from exc

        choices = data.get("choices", [])
        if not choices:
            raise GenerationServiceError("OpenAI API returned an empty choices list.")

        first_choice = choices[0]
        message_obj = first_choice.get("message", {})
        content = message_obj.get("content", "")
        if content is None or not content.strip():
            raise GenerationServiceError("OpenAI API returned empty text content.")

        return content.strip()

    def generate_reply(self, context: list[Message]) -> str:
        """Generate a helpful group chat reply when the decision is SPEAK."""
        transcript = self._format_context(context)
        system_instruction = (
            "You are Hush, a helpful and restrained conversational AI participant in a team chat. "
            "Write a concise, helpful, and direct contribution. Avoid conversational filler or repeating questions."
        )
        prompt = (
            f"Here is the recent conversation transcript:\n\n"
            f"{transcript}\n\n"
            f"Please provide a helpful, concise contribution to the conversation."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]
        return self._invoke_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.5,
        )

    def generate_clarification(self, context: list[Message]) -> str:
        """Generate one short, focused clarifying question when the decision is ASK."""
        transcript = self._format_context(context)
        system_instruction = (
            "You are Hush, a restrained conversational AI assistant. "
            "Generate exactly one short, focused clarifying question to gather missing information. "
            "Do not attempt to answer the question yourself. Do not include introductory or polite filler."
        )
        prompt = (
            f"Here is the recent conversation transcript:\n\n"
            f"{transcript}\n\n"
            f"Ask exactly ONE short, focused clarifying question to resolve ambiguity and unblock the conversation."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]
        return self._invoke_chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.3,
        )
