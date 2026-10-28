"""End-to-end and unit tests for Hush backend vertical slice and Bedrock integration.

Verifies:
- GET /health endpoint with decider and generation modes
- POST /api/chat/message for SPEAK, ASK, and SILENT actions in stub mode
- POST /api/chat/message with Bedrock generation mode (SPEAK, ASK, and SILENT)
- CRITICAL: SILENT makes zero generation calls under BOTH stub and Bedrock modes
- Explicit failure handling for missing Bedrock configuration (500) and service errors (502)
- Bounded in-memory conversation state
- Explicit failure when real decider mode is selected without hush_gate
- Zero mock confidence/latency fabrication in development stub mode
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.schemas import Action
from app.services.bedrock import BedrockConfigError, BedrockServiceError
from app.services.conversation import ConversationManager, conversation_manager
from app.services.decider import DevelopmentStubDecider, RealHushGateDecider, get_decider
from app.services.generation import DevelopmentStubGenerator, get_generator


client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_conversation_state():
    """Ensure in-memory conversation history is cleared before each test."""
    conversation_manager.clear_all()
    yield
    conversation_manager.clear_all()


def test_health_check_endpoint():
    """Verify GET /health returns 200 with status and modes."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"status", "generation_mode", "decider_mode"}
    assert data["status"] == "ok"
    assert data["decider_mode"] == settings.HUSH_DECIDER_MODE
    assert data["generation_mode"] == settings.HUSH_GENERATION_MODE
    assert response.headers.get("x-hush-decider-mode") == settings.HUSH_DECIDER_MODE
    assert response.headers.get("x-hush-generation-mode") == settings.HUSH_GENERATION_MODE


def test_chat_message_speak_action_stub_mode():
    """Verify direct mention triggers SPEAK and returns a generated reply in stub mode."""
    payload = {
        "conversation_id": "room-1",
        "user_id": "alex",
        "message": "Hush, what is the plan?",
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["conversation_id"] == "room-1"
    assert data["action"] == "SPEAK"
    assert data["generation_used"] is True
    assert data["reply"] is not None
    assert "[SIMULATED DEMO]" in data["reply"]
    # Stub mode avoids fabricating confidence or latency
    assert data["confidence"] is None
    assert data["decision_latency_ms"] is None


def test_chat_message_ask_action_stub_mode():
    """Verify question without mention triggers ASK and returns a clarifying question in stub mode."""
    payload = {
        "conversation_id": "room-1",
        "user_id": "sam",
        "message": "The deployment is failing with an error. Can anyone help?",
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["conversation_id"] == "room-1"
    assert data["action"] == "ASK"
    assert data["generation_used"] is True
    assert data["reply"] is not None
    assert "[SIMULATED DEMO]" in data["reply"]
    assert data["confidence"] is None
    assert data["decision_latency_ms"] is None


def test_chat_message_silent_action_stub_mode():
    """Verify normal banter triggers SILENT: reply is None, generation_used is False."""
    payload = {
        "conversation_id": "room-1",
        "user_id": "taylor",
        "message": "Heading to grab coffee now.",
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["conversation_id"] == "room-1"
    assert data["action"] == "SILENT"
    assert data["reply"] is None
    assert data["generation_used"] is False
    assert data["confidence"] is None
    assert data["decision_latency_ms"] is None


def test_critical_silent_makes_zero_generation_calls():
    """CRITICAL REQUIREMENT: When action is SILENT, generator service must NEVER be called."""
    mock_gen = MagicMock()
    mock_gen.generate_reply = MagicMock()
    mock_gen.generate_clarification = MagicMock()

    with patch("app.routes.chat.get_generator", return_value=mock_gen):
        payload = {
            "conversation_id": "room-silent-test",
            "user_id": "jordan",
            "message": "Sounds good, see you in 10 minutes.",
        }
        response = client.post("/api/chat/message", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["action"] == "SILENT"
        assert data["generation_used"] is False
        assert data["reply"] is None

        # Verify zero calls to text generation
        mock_gen.generate_reply.assert_not_called()
        mock_gen.generate_clarification.assert_not_called()
        assert mock_gen.generate_reply.call_count == 0
        assert mock_gen.generate_clarification.call_count == 0


def test_bedrock_mode_routes_speak_to_generate_reply():
    """When Bedrock mode is active, SPEAK routes through BedrockService.generate_reply."""
    mock_bedrock = MagicMock()
    mock_bedrock.generate_reply.return_value = "Real generated reply from Bedrock."

    with patch("app.routes.chat.get_generator", return_value=mock_bedrock):
        payload = {
            "conversation_id": "room-bedrock-speak",
            "user_id": "alex",
            "message": "Hush, what is the plan?",
        }
        response = client.post("/api/chat/message", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["action"] == "SPEAK"
        assert data["generation_used"] is True
        assert data["reply"] == "Real generated reply from Bedrock."
        mock_bedrock.generate_reply.assert_called_once()
        mock_bedrock.generate_clarification.assert_not_called()


def test_bedrock_mode_routes_ask_to_generate_clarification():
    """When Bedrock mode is active, ASK routes through BedrockService.generate_clarification."""
    mock_bedrock = MagicMock()
    mock_bedrock.generate_clarification.return_value = "Which deployment failed?"

    with patch("app.routes.chat.get_generator", return_value=mock_bedrock):
        payload = {
            "conversation_id": "room-bedrock-ask",
            "user_id": "sam",
            "message": "Can someone help with the error?",
        }
        response = client.post("/api/chat/message", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["action"] == "ASK"
        assert data["generation_used"] is True
        assert data["reply"] == "Which deployment failed?"
        mock_bedrock.generate_clarification.assert_called_once()
        mock_bedrock.generate_reply.assert_not_called()


def test_bedrock_mode_silent_strictly_never_calls_bedrock():
    """CRITICAL: Even in Bedrock generation mode, SILENT strictly never invokes Bedrock."""
    mock_bedrock = MagicMock()

    with patch("app.routes.chat.get_generator", return_value=mock_bedrock):
        payload = {
            "conversation_id": "room-bedrock-silent",
            "user_id": "taylor",
            "message": "Going for coffee.",
        }
        response = client.post("/api/chat/message", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["action"] == "SILENT"
        assert data["generation_used"] is False
        assert data["reply"] is None

        # Absolute zero calls to Bedrock
        mock_bedrock.generate_reply.assert_not_called()
        mock_bedrock.generate_clarification.assert_not_called()
        assert mock_bedrock.generate_reply.call_count == 0
        assert mock_bedrock.generate_clarification.call_count == 0


def test_bedrock_missing_configuration_fails_with_500():
    """Missing AWS region or model ID returns clear 500 configuration error."""
    mock_bedrock = MagicMock()
    mock_bedrock.generate_reply.side_effect = BedrockConfigError(
        "AWS region is not configured. Set AWS_REGION in your environment."
    )

    with patch("app.routes.chat.get_generator", return_value=mock_bedrock):
        payload = {
            "conversation_id": "room-config-err",
            "user_id": "alex",
            "message": "Hush, help!",
        }
        response = client.post("/api/chat/message", json=payload)
        assert response.status_code == 500
        data = response.json()
        assert "Bedrock configuration error" in data["detail"]
        assert "AWS_REGION" in data["detail"]


def test_bedrock_service_error_fails_with_502():
    """Bedrock runtime/API failure returns clear 502 Bad Gateway."""
    mock_bedrock = MagicMock()
    mock_bedrock.generate_reply.side_effect = BedrockServiceError(
        "Amazon Bedrock Converse invocation failed: ThrottlingException"
    )

    with patch("app.routes.chat.get_generator", return_value=mock_bedrock):
        payload = {
            "conversation_id": "room-service-err",
            "user_id": "alex",
            "message": "Hush, help!",
        }
        response = client.post("/api/chat/message", json=payload)
        assert response.status_code == 502
        data = response.json()
        assert "Bedrock generation error" in data["detail"]
        assert "ThrottlingException" in data["detail"]


def test_real_decider_mode_fails_clearly_when_hush_gate_unavailable():
    """If HUSH_DECIDER_MODE is 'real' but hush_gate is missing, must fail with 503 instead of silent stub fallback."""
    with patch("app.routes.chat.get_decider", return_value=RealHushGateDecider()):
        payload = {
            "conversation_id": "room-fail-test",
            "user_id": "alex",
            "message": "Hush, are you there?",
        }
        response = client.post("/api/chat/message", json=payload)
        assert response.status_code == 503
        data = response.json()
        assert "hush_gate" in data["detail"]
        assert "Failing explicitly" in data["detail"]


def test_bounded_conversation_history():
    """Verify conversation history in memory is bounded to max_history."""
    mgr = ConversationManager(max_history=5)
    for i in range(10):
        mgr.add_message("test-room", f"user-{i}", f"Message number {i}")

    history = mgr.get_history("test-room")
    assert len(history) == 5
    assert history[0].text == "Message number 5"
    assert history[-1].text == "Message number 9"
