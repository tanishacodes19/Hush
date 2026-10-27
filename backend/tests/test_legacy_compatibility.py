"""Legacy API compatibility test suite for Hush.

Locks down the existing legacy request format, response schema contract,
validation rules, and SPEAK/ASK/SILENT routing behavior before Phase 1
schema extensions.

Legacy request format:
    {
        "conversation_id": "legacy-room-1",
        "user_id": "user-alex",
        "message": "Hush, what is the plan?"
    }

Guarantees:
- Existing legacy requests remain fully supported.
- Exact response contract fields and types are strictly preserved.
- Validation rejects missing, empty, or unexpected extra fields.
- Decision routing executes the correct generator paths for SPEAK and ASK.
- SILENT structurally bypasses text generation (zero generator calls).
- In-memory state and telemetry counters are properly isolated.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Action, DecisionResult
from app.services.conversation import conversation_manager
from app.services.metrics import metrics_tracker

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_conversation_and_metrics():
    """Ensure in-memory conversation history and metrics are isolated per test."""
    conversation_manager.clear_all()
    metrics_tracker.reset()
    yield
    conversation_manager.clear_all()
    metrics_tracker.reset()


# ---------------------------------------------------------------------------
# A. Successful Legacy Request & Contract Shape
# ---------------------------------------------------------------------------

def test_legacy_request_success_status_and_keys():
    """A standard legacy request returns HTTP 200 and matches the expected response keys."""
    payload = {
        "conversation_id": "legacy-room-1",
        "user_id": "user-alex",
        "message": "Hush, what is the plan?",
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 200

    data = response.json()
    expected_fields = {
        "conversation_id",
        "action",
        "confidence",
        "reply",
        "generation_used",
        "fallback_used",
        "decision_latency_ms",
    }
    assert set(data.keys()) == expected_fields
    assert data["conversation_id"] == "legacy-room-1"
    assert data["action"] in {"SPEAK", "SILENT", "ASK"}
    assert isinstance(data["generation_used"], bool)
    assert isinstance(data["fallback_used"], bool)


def test_legacy_request_requires_no_conversation_aware_fields():
    """Legacy requests succeed without any modern conversation-aware fields.

    Fields such as 'message_id', 'speaker_id', 'display_name', 'role',
    'mentions', 'reply_to', or 'history' MUST NOT be required.
    """
    payload = {
        "conversation_id": "room-minimal",
        "user_id": "user-simple",
        "message": "Just a plain message without extra metadata.",
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "room-minimal"


# ---------------------------------------------------------------------------
# B. Response Contract & Types
# ---------------------------------------------------------------------------

def test_legacy_response_field_types_on_speak():
    """Verify types and values of all ChatMessageResponse fields on SPEAK."""
    payload = {
        "conversation_id": "room-types-speak",
        "user_id": "user-alex",
        "message": "Hush, can you help us?",
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["conversation_id"], str)
    assert data["action"] == "SPEAK"
    assert data["confidence"] is None or isinstance(data["confidence"], (float, int))
    assert isinstance(data["reply"], str)
    assert data["generation_used"] is True
    assert isinstance(data["fallback_used"], bool)
    assert data["decision_latency_ms"] is None or isinstance(data["decision_latency_ms"], (float, int))


def test_legacy_response_field_types_on_silent():
    """Verify types and invariant values of ChatMessageResponse fields on SILENT."""
    payload = {
        "conversation_id": "room-types-silent",
        "user_id": "user-taylor",
        "message": "Coffee break time.",
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["conversation_id"], str)
    assert data["action"] == "SILENT"
    assert data["reply"] is None
    assert data["generation_used"] is False
    assert isinstance(data["fallback_used"], bool)


# ---------------------------------------------------------------------------
# C. Validation Behavior
# ---------------------------------------------------------------------------

def test_legacy_request_rejects_missing_conversation_id():
    """Missing 'conversation_id' results in 422 Unprocessable Entity."""
    payload = {
        "user_id": "user-alex",
        "message": "Hello",
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 422


def test_legacy_request_rejects_missing_user_id():
    """Missing 'user_id' results in 422 Unprocessable Entity."""
    payload = {
        "conversation_id": "room-1",
        "message": "Hello",
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 422


def test_legacy_request_rejects_missing_message():
    """Missing 'message' results in 422 Unprocessable Entity."""
    payload = {
        "conversation_id": "room-1",
        "user_id": "user-alex",
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 422


def test_legacy_request_rejects_empty_payload():
    """Empty JSON body {} results in 422 Unprocessable Entity."""
    response = client.post("/api/chat/message", json={})
    assert response.status_code == 422


def test_legacy_request_rejects_empty_strings():
    """Empty strings violate min_length=1 and result in 422 Unprocessable Entity."""
    # Empty conversation_id
    resp1 = client.post("/api/chat/message", json={
        "conversation_id": "",
        "user_id": "user-1",
        "message": "Hello",
    })
    assert resp1.status_code == 422

    # Empty user_id
    resp2 = client.post("/api/chat/message", json={
        "conversation_id": "room-1",
        "user_id": "",
        "message": "Hello",
    })
    assert resp2.status_code == 422

    # Empty message
    resp3 = client.post("/api/chat/message", json={
        "conversation_id": "room-1",
        "user_id": "user-1",
        "message": "",
    })
    assert resp3.status_code == 422


def test_legacy_request_rejects_extra_fields():
    """ChatMessageRequest has extra='forbid'; undeclared fields result in 422."""
    payload = {
        "conversation_id": "room-extra",
        "user_id": "user-1",
        "message": "Hello",
        "mentions": ["hush"],  # Undeclared field in legacy schema
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 422


def test_legacy_request_rejects_non_dict_payload():
    """Non-dictionary payloads result in 422 Unprocessable Entity."""
    response = client.post("/api/chat/message", json=["not", "a", "dict"])
    assert response.status_code == 422


def test_legacy_request_rejects_invalid_field_types():
    """Invalid data types (e.g., list for message) result in 422 Unprocessable Entity."""
    payload = {
        "conversation_id": "room-invalid-type",
        "user_id": "user-1",
        "message": ["not", "a", "string"],
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# D. SPEAK, ASK, and SILENT Routing (Decoupled with Mocks)
# ---------------------------------------------------------------------------

def test_legacy_routing_speak_invokes_generate_reply():
    """When decider selects SPEAK, generate_reply is called and reply is returned."""
    mock_decider = MagicMock()
    mock_decider.decide.return_value = DecisionResult(
        action=Action.SPEAK,
        confidence=0.88,
        fallback_used=False,
        decision_latency_ms=15.2,
    )

    mock_generator = MagicMock()
    mock_generator.generate_reply.return_value = "Mocked reply from generator."
    mock_generator.generate_clarification.return_value = "Should not be called."

    with (
        patch("app.routes.chat.get_decider", return_value=mock_decider),
        patch("app.routes.chat.get_generator", return_value=mock_generator),
    ):
        payload = {
            "conversation_id": "room-route-speak",
            "user_id": "user-alex",
            "message": "Any message evaluated as SPEAK",
        }
        response = client.post("/api/chat/message", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["action"] == "SPEAK"
        assert data["reply"] == "Mocked reply from generator."
        assert data["generation_used"] is True
        assert data["confidence"] == 0.88
        assert data["decision_latency_ms"] == 15.2

        # Verify exact generator path
        mock_generator.generate_reply.assert_called_once()
        mock_generator.generate_clarification.assert_not_called()
        assert mock_generator.generate_reply.call_count == 1
        assert mock_generator.generate_clarification.call_count == 0


def test_legacy_routing_ask_invokes_generate_clarification():
    """When decider selects ASK, generate_clarification is called and reply is returned."""
    mock_decider = MagicMock()
    mock_decider.decide.return_value = DecisionResult(
        action=Action.ASK,
        confidence=0.72,
        fallback_used=False,
        decision_latency_ms=18.4,
    )

    mock_generator = MagicMock()
    mock_generator.generate_clarification.return_value = "Which deployment environment?"
    mock_generator.generate_reply.return_value = "Should not be called."

    with (
        patch("app.routes.chat.get_decider", return_value=mock_decider),
        patch("app.routes.chat.get_generator", return_value=mock_generator),
    ):
        payload = {
            "conversation_id": "room-route-ask",
            "user_id": "user-sam",
            "message": "Any message evaluated as ASK",
        }
        response = client.post("/api/chat/message", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["action"] == "ASK"
        assert data["reply"] == "Which deployment environment?"
        assert data["generation_used"] is True
        assert data["confidence"] == 0.72
        assert data["decision_latency_ms"] == 18.4

        # Verify exact generator path
        mock_generator.generate_clarification.assert_called_once()
        mock_generator.generate_reply.assert_not_called()
        assert mock_generator.generate_clarification.call_count == 1
        assert mock_generator.generate_reply.call_count == 0


def test_legacy_routing_silent_strictly_never_calls_generator():
    """CRITICAL INVARIANT: On SILENT, generator is NEVER called under any method."""
    mock_decider = MagicMock()
    mock_decider.decide.return_value = DecisionResult(
        action=Action.SILENT,
        confidence=0.96,
        fallback_used=False,
        decision_latency_ms=6.1,
    )

    mock_generator = MagicMock()
    mock_generator.generate_reply = MagicMock()
    mock_generator.generate_clarification = MagicMock()

    with (
        patch("app.routes.chat.get_decider", return_value=mock_decider),
        patch("app.routes.chat.get_generator", return_value=mock_generator),
    ):
        payload = {
            "conversation_id": "room-route-silent",
            "user_id": "user-jordan",
            "message": "Any message evaluated as SILENT",
        }
        response = client.post("/api/chat/message", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["action"] == "SILENT"
        assert data["reply"] is None
        assert data["generation_used"] is False
        assert data["confidence"] == 0.96
        assert data["decision_latency_ms"] == 6.1

        # Absolute zero calls to generation
        mock_generator.generate_reply.assert_not_called()
        mock_generator.generate_clarification.assert_not_called()
        assert mock_generator.generate_reply.call_count == 0
        assert mock_generator.generate_clarification.call_count == 0


# ---------------------------------------------------------------------------
# E. State Management & Conversation History
# ---------------------------------------------------------------------------

def test_legacy_request_updates_conversation_history_appropriately():
    """User message is appended to history; assistant message is appended only on SPEAK/ASK."""
    # 1. Send message resulting in SPEAK
    client.post("/api/chat/message", json={
        "conversation_id": "room-history",
        "user_id": "user-alex",
        "message": "Hush, are you listening?",
    })
    history_after_speak = conversation_manager.get_history("room-history")
    assert len(history_after_speak) == 2
    assert history_after_speak[0].speaker == "user-alex"
    assert history_after_speak[0].text == "Hush, are you listening?"
    assert history_after_speak[1].speaker == "Hush"
    assert "[Development Stub]" in history_after_speak[1].text

    # 2. Send message resulting in SILENT
    client.post("/api/chat/message", json={
        "conversation_id": "room-history",
        "user_id": "user-taylor",
        "message": "Let us grab some coffee.",
    })
    history_after_silent = conversation_manager.get_history("room-history")
    assert len(history_after_silent) == 3
    assert history_after_silent[2].speaker == "user-taylor"
    assert history_after_silent[2].text == "Let us grab some coffee."
    # No fourth message appended because action was SILENT
