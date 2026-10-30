"""Unit and integration test suite for hosted OpenAI-compatible LLM generation.

All tests remain offline by mocking HTTP/provider calls.
Verifies:
1. Missing API key produces actionable GenerationConfigError (HTTP 500), never silently falling back to stub.
2. SPEAK routes through generate_reply and formats context correctly.
3. ASK routes through generate_clarification and requests a single focused question.
4. SILENT strictly never invokes the generator.
5. Provider errors (401, 429, 500, timeout, network error) are caught and mapped without exposing secrets.
6. Generator factory respects configuration modes and rejects unsupported modes.
"""

from unittest.mock import MagicMock, patch
import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.schemas import Message
from app.services.generation import (
    DevelopmentStubGenerator,
    GenerationConfigError,
    GenerationServiceError,
    get_generator,
)
from app.services.bedrock import BedrockService
from app.services.openai_service import OpenAIService


# ---------------------------------------------------------------------------
# 1. OpenAIService Unit Tests
# ---------------------------------------------------------------------------

def test_openai_service_missing_api_key_raises_config_error():
    """Missing OPENAI_API_KEY raises GenerationConfigError."""
    service = OpenAIService(api_key="")
    with pytest.raises(GenerationConfigError) as exc_info:
        service.generate_reply([Message(speaker="Alice", text="Hello")])

    assert "OpenAI API key is not configured" in str(exc_info.value)
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_openai_service_speak_success():
    """generate_reply sends structured transcript and returns parsed output text."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Binary search is O(log n) because it divides the search space in half each step.",
                }
            }
        ]
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_response

    service = OpenAIService(
        api_key="test-sk-key-never-exposed",
        model_id="gpt-4o-mini",
        client=mock_client,
    )

    context = [
        Message(speaker="Alice", text="What is the complexity of binary search?"),
    ]
    reply = service.generate_reply(context)

    assert reply == "Binary search is O(log n) because it divides the search space in half each step."
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args[1]

    assert call_kwargs["json"]["model"] == "gpt-4o-mini"
    assert call_kwargs["json"]["max_tokens"] == 512
    assert len(call_kwargs["json"]["messages"]) == 2
    assert "You are Hush" in call_kwargs["json"]["messages"][0]["content"]
    assert "Alice: What is the complexity" in call_kwargs["json"]["messages"][1]["content"]


def test_openai_service_ask_success():
    """generate_clarification requests a single focused question."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Could you provide the SQL query and the exact database error message?",
                }
            }
        ]
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_response

    service = OpenAIService(
        api_key="test-sk-key-never-exposed",
        model_id="gpt-4o-mini",
        client=mock_client,
    )

    context = [
        Message(speaker="Bob", text="The query is crashing in staging."),
    ]
    clarification = service.generate_clarification(context)

    assert clarification == "Could you provide the SQL query and the exact database error message?"
    call_kwargs = mock_client.post.call_args[1]
    assert call_kwargs["json"]["max_tokens"] == 150
    assert "exactly one short, focused clarifying question" in call_kwargs["json"]["messages"][0]["content"]


def test_openai_service_custom_base_url_and_model():
    """Configuring custom base URL and model routes correctly (e.g. Groq/OpenRouter/vLLM)."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Groq response."}}]
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_response

    service = OpenAIService(
        api_key="gsk-mock-key",
        model_id="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1/",
        client=mock_client,
    )

    result = service.generate_reply([Message(speaker="User", text="Hi")])
    assert result == "Groq response."

    url_called = mock_client.post.call_args[0][0]
    assert url_called == "https://api.groq.com/openai/v1/chat/completions"
    assert mock_client.post.call_args[1]["json"]["model"] == "llama-3.3-70b-versatile"


def test_openai_service_401_auth_error_sanitized():
    """HTTP 401 raises GenerationConfigError without leaking secret API key."""
    fake_key = "sk-super-secret-secret-token"
    mock_client = MagicMock(spec=httpx.Client)
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(401, request=req)
    mock_client.post.side_effect = httpx.HTTPStatusError("401 Unauthorized", request=req, response=resp)

    service = OpenAIService(api_key=fake_key, client=mock_client)
    with pytest.raises(GenerationConfigError) as exc_info:
        service.generate_reply([Message(speaker="User", text="Hi")])

    err_msg = str(exc_info.value)
    assert "authentication failed" in err_msg.lower()
    # Crucial security assertion: API key must NEVER be printed in error message
    assert fake_key not in err_msg


def test_openai_service_429_rate_limit():
    """HTTP 429 raises GenerationServiceError mentioning rate limits."""
    mock_client = MagicMock(spec=httpx.Client)
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    mock_client.post.side_effect = httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp)

    service = OpenAIService(api_key="sk-test", client=mock_client)
    with pytest.raises(GenerationServiceError) as exc_info:
        service.generate_reply([Message(speaker="User", text="Hi")])

    assert "rate limit" in str(exc_info.value).lower()


def test_openai_service_timeout():
    """TimeoutException raises GenerationServiceError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")

    service = OpenAIService(api_key="sk-test", client=mock_client)
    with pytest.raises(GenerationServiceError) as exc_info:
        service.generate_reply([Message(speaker="User", text="Hi")])

    assert "timed out" in str(exc_info.value).lower()


def test_openai_service_empty_completion_error():
    """Empty choices list raises GenerationServiceError."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_response

    service = OpenAIService(api_key="sk-test", client=mock_client)
    with pytest.raises(GenerationServiceError) as exc_info:
        service.generate_reply([Message(speaker="User", text="Hi")])

    assert "empty choices" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 2. Generator Factory Tests
# ---------------------------------------------------------------------------

def test_generator_factory_modes():
    """get_generator instantiates correct service based on mode."""
    openai_gen = get_generator("openai")
    assert isinstance(openai_gen, OpenAIService)

    bedrock_gen = get_generator("bedrock")
    assert isinstance(bedrock_gen, BedrockService)

    stub_gen = get_generator("stub")
    assert isinstance(stub_gen, DevelopmentStubGenerator)

    with pytest.raises(GenerationConfigError) as exc_info:
        get_generator("unsupported_provider")
    assert "Unsupported generation mode" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. API Integration with OpenAIService (Mocked)
# ---------------------------------------------------------------------------

def test_api_integration_openai_speak_mocked():
    """When decision is SPEAK, OpenAIService.generate_reply provides the answer."""
    client = TestClient(app)

    mock_service = MagicMock(spec=OpenAIService)
    mock_service.generate_reply.return_value = "QuickSort has worst-case complexity of O(n^2)."
    mock_service.generate_clarification = MagicMock()

    with patch("app.routes.chat.get_generator", return_value=mock_service):
        payload = {
            "conversation_id": "conv-openai-speak",
            "message": {
                "message_id": "m201",
                "speaker_id": "alice",
                "text": "@hush What is the time complexity of quicksort in the worst case?",
            },
        }
        resp = client.post("/api/chat/message", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "SPEAK"
        assert data["reply"] == "QuickSort has worst-case complexity of O(n^2)."
        assert data["generation_used"] is True

        mock_service.generate_reply.assert_called_once()
        mock_service.generate_clarification.assert_not_called()


def test_api_integration_openai_ask_mocked():
    """When decision is ASK, OpenAIService.generate_clarification provides clarifying question."""
    client = TestClient(app)

    mock_service = MagicMock(spec=OpenAIService)
    mock_service.generate_clarification.return_value = "Which database engine and migration tool are you running?"
    mock_service.generate_reply = MagicMock()

    with patch("app.routes.chat.get_generator", return_value=mock_service):
        payload = {
            "conversation_id": "conv-openai-ask",
            "message": {
                "message_id": "m202",
                "speaker_id": "bob",
                "text": "@hush fix this query, it failed",
            },
        }
        resp = client.post("/api/chat/message", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "ASK"
        assert data["reply"] == "Which database engine and migration tool are you running?"
        assert data["generation_used"] is True

        mock_service.generate_clarification.assert_called_once()
        mock_service.generate_reply.assert_not_called()


def test_api_integration_openai_silent_never_invokes_provider():
    """When decision is SILENT, generator is strictly never called."""
    client = TestClient(app)

    mock_service = MagicMock(spec=OpenAIService)
    mock_service.generate_reply = MagicMock()
    mock_service.generate_clarification = MagicMock()

    with patch("app.routes.chat.get_generator", return_value=mock_service):
        payload = {
            "conversation_id": "conv-openai-silent",
            "message": {
                "message_id": "m203",
                "speaker_id": "carol",
                "text": "Heading to grab coffee now. Anyone want anything?",
            },
        }
        resp = client.post("/api/chat/message", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "SILENT"
        assert data["reply"] is None
        assert data["generation_used"] is False

        mock_service.generate_reply.assert_not_called()
        mock_service.generate_clarification.assert_not_called()


def test_api_integration_missing_credentials_fails_with_500():
    """When credentials are missing in production mode, chat API returns 500 configuration error."""
    client = TestClient(app)

    # Ensure API key is None
    with patch.object(settings, "OPENAI_API_KEY", None), patch.object(settings, "HUSH_GENERATION_MODE", "openai"):
        payload = {
            "conversation_id": "conv-missing-creds",
            "message": {
                "message_id": "m204",
                "speaker_id": "alice",
                "text": "@hush What is the time complexity of binary search?",
            },
        }
        resp = client.post("/api/chat/message", json=payload)

        assert resp.status_code == 500
        data = resp.json()
        assert "Generation configuration error" in data["detail"]
        assert "OPENAI_API_KEY" in data["detail"]


def test_api_integration_provider_error_fails_with_502():
    """When the provider fails, chat API returns 502 Bad Gateway without leaking internals."""
    client = TestClient(app)

    mock_service = MagicMock(spec=OpenAIService)
    mock_service.generate_reply.side_effect = GenerationServiceError("OpenAI API request timed out after 15.0s.")

    with patch("app.routes.chat.get_generator", return_value=mock_service):
        payload = {
            "conversation_id": "conv-prov-err",
            "message": {
                "message_id": "m205",
                "speaker_id": "alice",
                "text": "@hush What is the time complexity of binary search?",
            },
        }
        resp = client.post("/api/chat/message", json=payload)

        assert resp.status_code == 502
        data = resp.json()
        assert "Generation provider error" in data["detail"]
        assert "timed out" in data["detail"]


# ---------------------------------------------------------------------------
# 4. Security & URL Validation Tests
# ---------------------------------------------------------------------------

def test_openai_service_insecure_remote_http_rejected():
    """Cleartext HTTP on remote domains is rejected to prevent credential sniffing."""
    with pytest.raises(GenerationConfigError) as exc_info:
        OpenAIService(api_key="sk-test", base_url="http://insecure-remote-api.com/v1")
    assert "Insecure plain HTTP base URL is not permitted" in str(exc_info.value)


def test_openai_service_invalid_scheme_rejected():
    """Non-HTTP(S) schemes like ftp:// or file:// are strictly rejected."""
    with pytest.raises(GenerationConfigError) as exc_info:
        OpenAIService(api_key="sk-test", base_url="ftp://api.openai.com/v1")
    assert "Invalid OpenAI base URL scheme" in str(exc_info.value)


def test_openai_service_missing_hostname_rejected():
    """Malformed URLs without a hostname are rejected."""
    with pytest.raises(GenerationConfigError) as exc_info:
        OpenAIService(api_key="sk-test", base_url="just_a_string")
    assert "Invalid OpenAI base URL" in str(exc_info.value)


def test_openai_service_local_http_allowed():
    """Plain HTTP is permitted for local endpoints (localhost, 127.0.0.1, Ollama)."""
    s_local = OpenAIService(api_key="sk-test", base_url="http://localhost:11434/v1")
    assert s_local.base_url == "http://localhost:11434/v1"

    s_ip = OpenAIService(api_key="sk-test", base_url="http://127.0.0.1:8000/v1")
    assert s_ip.base_url == "http://127.0.0.1:8000/v1"


def test_openai_service_redirect_rejected_to_prevent_credential_leakage():
    """Redirects (HTTP 301/302) are rejected to prevent Authorization credential forwarding."""
    mock_client = MagicMock(spec=httpx.Client)
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(301, request=req, headers={"Location": "https://attacker.com/steal"})
    mock_client.post.return_value = resp

    service = OpenAIService(api_key="sk-secret-key-12345", client=mock_client)
    with pytest.raises(GenerationServiceError) as exc_info:
        service.generate_reply([Message(speaker="User", text="Hi")])

    err_msg = str(exc_info.value)
    assert "redirects are disallowed" in err_msg.lower()
    assert "sk-secret-key-12345" not in err_msg


def test_api_integration_legacy_format_with_openai_generator():
    """Legacy 7-field response shape is strictly preserved when using OpenAIService."""
    client = TestClient(app)

    mock_service = MagicMock(spec=OpenAIService)
    mock_service.generate_reply.return_value = "Merge sort has worst-case complexity of O(n log n)."

    with patch("app.routes.chat.get_generator", return_value=mock_service):
        payload = {
            "conversation_id": "conv-legacy-openai",
            "user_id": "user-dave",
            "message": "@hush What is the time complexity of merge sort in the worst case?",
        }
        resp = client.post("/api/chat/message", json=payload)

        assert resp.status_code == 200
        data = resp.json()

        # Strict seven-field legacy contract (PLAN.md Section 7)
        expected_keys = {
            "conversation_id",
            "action",
            "confidence",
            "reply",
            "generation_used",
            "fallback_used",
            "decision_latency_ms",
        }
        assert set(data.keys()) == expected_keys
        assert data["action"] == "SPEAK"
        assert data["reply"] == "Merge sort has worst-case complexity of O(n log n)."
        assert data["generation_used"] is True
        assert isinstance(data["confidence"], float)
        assert data["decision_latency_ms"] >= 0.0


def test_offline_socket_blocking_guard():
    """Verify that socket fixture blocks outbound external connections during tests."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(RuntimeError, match="network isolation violation"):
        s.connect(("8.8.8.8", 53))


