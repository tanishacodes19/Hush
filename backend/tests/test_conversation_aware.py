"""Tests for Phase 1 Step 2: Conversation-Aware Schema & Request Adapter.

Verifies:
- Acceptance of conversation-aware payloads (both nested and flat formats)
- Required field validation and rejection of missing/empty/whitespace fields
- Optional metadata handling (omitted vs supplied)
- Metadata preservation during request normalization
- Rejection of unexpected fields and ambiguous mixtures of legacy/new fields
- Transparent routing to generator (SPEAK), clarification (ASK), and zero-call bypass (SILENT)
- In-memory conversation state compatibility
- Strict preservation of the 7-field ChatMessageResponse contract
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    Action,
    ChatMessageRequest,
    ConversationAwareChatRequest,
    DecisionResult,
    FlatConversationAwareChatRequest,
)
from app.services.conversation import conversation_manager
from app.services.metrics import metrics_tracker
from app.services.request_adapter import detect_request_format, normalize_request

client = TestClient(app)

EXPECTED_RESPONSE_KEYS = {
    "conversation_id",
    "message_id",
    "addressed_to_hush",
    "address_method",
    "action",
    "confidence",
    "confidence_kind",
    "reply",
    "generation_used",
    "decision",
}


@pytest.fixture(autouse=True)
def reset_state():
    """Ensure in-memory history and metrics are isolated per test."""
    conversation_manager.clear_all()
    metrics_tracker.reset()
    yield
    conversation_manager.clear_all()
    metrics_tracker.reset()


# ---------------------------------------------------------------------------
# A. Conversation-Aware Schema Acceptance & Response Contract
# ---------------------------------------------------------------------------

def test_conversation_aware_nested_payload_accepted():
    """Valid nested conversation-aware payload is accepted with exact 7 response fields."""
    payload = {
        "conversation_id": "room-ca-1",
        "message": {
            "message_id": "m100",
            "speaker_id": "u-alex",
            "display_name": "alex",
            "text": "Hush, what is the plan?",
            "mentions": ["hush"],
            "reply_to": None,
            "ts": 1730000000.0,
        },
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert set(data.keys()) == EXPECTED_RESPONSE_KEYS
    assert data["conversation_id"] == "room-ca-1"
    assert data["action"] == "SPEAK"
    assert data["generation_used"] is True
    assert data["reply"] is not None
    assert "[Development Stub]" in data["reply"]


def test_conversation_aware_flat_payload_accepted():
    """Valid flat conversation-aware payload is accepted with exact 7 response fields."""
    payload = {
        "conversation_id": "room-ca-flat",
        "message_id": "m200",
        "speaker_id": "u-sam",
        "message": "Hush, can you check this error?",
        "display_name": "sam",
        "mentions": ["hush"],
        "reply_to": "m100",
        "timestamp": 1730000050.0,
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert set(data.keys()) == EXPECTED_RESPONSE_KEYS
    assert data["conversation_id"] == "room-ca-flat"
    assert data["action"] == "SPEAK"
    assert data["generation_used"] is True


def test_conversation_aware_optional_fields_omitted():
    """Optional fields (display_name, mentions, reply_to, ts) can be cleanly omitted."""
    payload = {
        "conversation_id": "room-ca-minimal",
        "message": {
            "message_id": "m300",
            "speaker_id": "u-taylor",
            "text": "Heading to grab coffee.",
        },
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert set(data.keys()) == EXPECTED_RESPONSE_KEYS
    assert data["conversation_id"] == "room-ca-minimal"
    assert data["action"] == "SILENT"
    assert data["reply"] is None
    assert data["generation_used"] is False


# ---------------------------------------------------------------------------
# B. Validation Behavior & Rejection Rules
# ---------------------------------------------------------------------------

def test_conversation_aware_rejects_missing_required_fields():
    """Missing message_id, speaker_id, text, or conversation_id returns 422."""
    # Missing message_id
    resp1 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {"speaker_id": "u1", "text": "hello"},
    })
    assert resp1.status_code == 422

    # Missing speaker_id
    resp2 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {"message_id": "m1", "text": "hello"},
    })
    assert resp2.status_code == 422

    # Missing text
    resp3 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {"message_id": "m1", "speaker_id": "u1"},
    })
    assert resp3.status_code == 422

    # Missing conversation_id
    resp4 = client.post("/api/chat/message", json={
        "message": {"message_id": "m1", "speaker_id": "u1", "text": "hello"},
    })
    assert resp4.status_code == 422


def test_conversation_aware_rejects_empty_and_whitespace_only_strings():
    """Empty or whitespace-only strings for identifiers and message text return 422."""
    # Whitespace message_id
    resp1 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {"message_id": "   ", "speaker_id": "u1", "text": "hello"},
    })
    assert resp1.status_code == 422

    # Whitespace speaker_id
    resp2 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {"message_id": "m1", "speaker_id": "   ", "text": "hello"},
    })
    assert resp2.status_code == 422

    # Whitespace text
    resp3 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {"message_id": "m1", "speaker_id": "u1", "text": "   "},
    })
    assert resp3.status_code == 422

    # Whitespace conversation_id
    resp4 = client.post("/api/chat/message", json={
        "conversation_id": "   ",
        "message": {"message_id": "m1", "speaker_id": "u1", "text": "hello"},
    })
    assert resp4.status_code == 422

    # Whitespace display_name (when supplied)
    resp5 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {"message_id": "m1", "speaker_id": "u1", "text": "hello", "display_name": "  "},
    })
    assert resp5.status_code == 422

    # Whitespace reply_to (when supplied)
    resp6 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {"message_id": "m1", "speaker_id": "u1", "text": "hello", "reply_to": "  "},
    })
    assert resp6.status_code == 422


def test_conversation_aware_rejects_unexpected_fields():
    """Unexpected extra fields at top-level or inside message payload return 422."""
    # Top-level unexpected field
    resp1 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {"message_id": "m1", "speaker_id": "u1", "text": "hello"},
        "extra_field": "disallowed",
    })
    assert resp1.status_code == 422

    # Nested unexpected field
    resp2 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {
            "message_id": "m1",
            "speaker_id": "u1",
            "text": "hello",
            "extra_nested": 123,
        },
    })
    assert resp2.status_code == 422


def test_conversation_aware_timestamp_validation():
    """Timestamps must be non-negative numbers."""
    # Negative timestamp
    resp1 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {"message_id": "m1", "speaker_id": "u1", "text": "hello", "ts": -1.0},
    })
    assert resp1.status_code == 422

    # String timestamp
    resp2 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {"message_id": "m1", "speaker_id": "u1", "text": "hello", "ts": "not-a-number"},
    })
    assert resp2.status_code == 422


def test_conversation_aware_mentions_validation():
    """Mentions list must contain non-empty string IDs."""
    # Mention with non-string item
    resp1 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {"message_id": "m1", "speaker_id": "u1", "text": "hello", "mentions": [123]},
    })
    assert resp1.status_code == 422

    # Mention with empty string
    resp2 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "message": {"message_id": "m1", "speaker_id": "u1", "text": "hello", "mentions": [""]},
    })
    assert resp2.status_code == 422


def test_reject_ambiguous_mixed_legacy_and_new_format():
    """Mixing legacy fields (user_id) with conversation-aware fields is strictly rejected (422)."""
    # user_id mixed with speaker_id
    resp1 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "user_id": "u1",
        "speaker_id": "s1",
        "message": "hello",
    })
    assert resp1.status_code == 422

    # user_id mixed with message_id
    resp2 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "user_id": "u1",
        "message_id": "m1",
        "message": "hello",
    })
    assert resp2.status_code == 422

    # user_id mixed with nested message dict
    resp3 = client.post("/api/chat/message", json={
        "conversation_id": "c1",
        "user_id": "u1",
        "message": {"message_id": "m1", "speaker_id": "s1", "text": "hello"},
    })
    assert resp3.status_code == 422


# ---------------------------------------------------------------------------
# C. Adapter Behavior & Metadata Normalization
# ---------------------------------------------------------------------------

def test_adapter_legacy_normalization_does_not_fabricate_metadata():
    """Legacy request normalizes user_id to speaker_id without inventing fake metadata."""
    legacy_req = ChatMessageRequest(
        conversation_id="room-leg",
        user_id="user-alex",
        message="Hello world",
    )
    normalized = normalize_request(legacy_req)

    assert normalized.conversation_id == "room-leg"
    assert normalized.speaker_id == "user-alex"
    assert normalized.text == "Hello world"
    assert normalized.format == "legacy"
    # Metadata MUST NOT be fabricated
    assert normalized.message_id is None
    assert normalized.display_name is None
    assert normalized.mentions == []
    assert normalized.reply_to is None
    assert normalized.ts is None


def test_adapter_conversation_aware_preserves_all_metadata():
    """Conversation-aware request preserves all provided metadata accurately."""
    ca_req = ConversationAwareChatRequest(
        conversation_id="room-ca-meta",
        message={
            "message_id": "m-preserve-1",
            "speaker_id": "u-sam",
            "display_name": "Samantha",
            "text": "Review needed",
            "mentions": ["u-alex", "hush"],
            "reply_to": "m-prev-0",
            "ts": 1730001234.5,
        },
    )
    normalized = normalize_request(ca_req)

    assert normalized.conversation_id == "room-ca-meta"
    assert normalized.speaker_id == "u-sam"
    assert normalized.text == "Review needed"
    assert normalized.format == "conversation_aware"
    assert normalized.message_id == "m-preserve-1"
    assert normalized.display_name == "Samantha"
    assert normalized.mentions == ["u-alex", "hush"]
    assert normalized.reply_to == "m-prev-0"
    assert normalized.ts == 1730001234.5


def test_adapter_detect_request_format_direct():
    """detect_request_format correctly identifies formats and raises ValueError on mixed/invalid."""
    # Legacy dict
    assert detect_request_format({
        "conversation_id": "c1",
        "user_id": "u1",
        "message": "hi",
    }) == "legacy"

    # Nested conversation-aware dict
    assert detect_request_format({
        "conversation_id": "c1",
        "message": {"message_id": "m1", "speaker_id": "s1", "text": "hi"},
    }) == "conversation_aware"

    # Flat conversation-aware dict
    assert detect_request_format({
        "conversation_id": "c1",
        "message_id": "m1",
        "speaker_id": "s1",
        "message": "hi",
    }) == "conversation_aware"

    # Mixed legacy + new fields raises ValueError
    with pytest.raises(ValueError, match="Ambiguous request"):
        detect_request_format({
            "conversation_id": "c1",
            "user_id": "u1",
            "speaker_id": "s1",
            "message": "hi",
        })

    # Non-dict payload raises ValueError
    with pytest.raises(ValueError, match="must be a dictionary"):
        detect_request_format(["not", "dict"])


# ---------------------------------------------------------------------------
# D. Decision Routing & SILENT Invariant under Conversation-Aware Payloads
# ---------------------------------------------------------------------------

def test_conversation_aware_routing_speak_path():
    """Mocked SPEAK decider calls generator.generate_reply with conversation-aware input."""
    mock_decider = MagicMock()
    mock_decider.decide.return_value = DecisionResult(
        action=Action.SPEAK,
        confidence=0.91,
        fallback_used=False,
        decision_latency_ms=14.0,
    )
    mock_generator = MagicMock()
    mock_generator.generate_reply.return_value = "Mocked reply for conversation-aware."
    mock_generator.generate_clarification.return_value = "Should not be called."

    with (
        patch("app.routes.chat.get_decider", return_value=mock_decider),
        patch("app.routes.chat.get_generator", return_value=mock_generator),
    ):
        payload = {
            "conversation_id": "room-mock-speak",
            "message": {"message_id": "m1", "speaker_id": "u1", "text": "Any text"},
        }
        response = client.post("/api/chat/message", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["action"] == "SPEAK"
        assert data["reply"] == "Mocked reply for conversation-aware."
        assert data["generation_used"] is True
        mock_generator.generate_reply.assert_called_once()
        mock_generator.generate_clarification.assert_not_called()


def test_conversation_aware_routing_ask_path():
    """Mocked ASK decider calls generator.generate_clarification with conversation-aware input."""
    mock_decider = MagicMock()
    mock_decider.decide.return_value = DecisionResult(
        action=Action.ASK,
        confidence=0.78,
        fallback_used=False,
        decision_latency_ms=16.0,
    )
    mock_generator = MagicMock()
    mock_generator.generate_clarification.return_value = "What is the reproduction step?"
    mock_generator.generate_reply.return_value = "Should not be called."

    with (
        patch("app.routes.chat.get_decider", return_value=mock_decider),
        patch("app.routes.chat.get_generator", return_value=mock_generator),
    ):
        payload = {
            "conversation_id": "room-mock-ask",
            "message": {"message_id": "m2", "speaker_id": "u2", "text": "Bug report"},
        }
        response = client.post("/api/chat/message", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["action"] == "ASK"
        assert data["reply"] == "What is the reproduction step?"
        assert data["generation_used"] is True
        mock_generator.generate_clarification.assert_called_once()
        mock_generator.generate_reply.assert_not_called()


def test_conversation_aware_routing_silent_invariant_never_calls_generator():
    """CRITICAL INVARIANT: SILENT makes zero generator calls under conversation-aware payloads."""
    mock_decider = MagicMock()
    mock_decider.decide.return_value = DecisionResult(
        action=Action.SILENT,
        confidence=0.99,
        fallback_used=False,
        decision_latency_ms=5.0,
    )
    mock_generator = MagicMock()
    mock_generator.generate_reply = MagicMock()
    mock_generator.generate_clarification = MagicMock()

    with (
        patch("app.routes.chat.get_decider", return_value=mock_decider),
        patch("app.routes.chat.get_generator", return_value=mock_generator),
    ):
        payload = {
            "conversation_id": "room-mock-silent",
            "message": {"message_id": "m3", "speaker_id": "u3", "text": "Random chat"},
        }
        response = client.post("/api/chat/message", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["action"] == "SILENT"
        assert data["reply"] is None
        assert data["generation_used"] is False

        # Strictly zero calls
        mock_generator.generate_reply.assert_not_called()
        mock_generator.generate_clarification.assert_not_called()
        assert mock_generator.generate_reply.call_count == 0
        assert mock_generator.generate_clarification.call_count == 0


def test_conversation_aware_updates_history_compatibly():
    """Conversation-aware requests correctly populate speaker and text in in-memory history."""
    payload = {
        "conversation_id": "room-history-ca",
        "message": {
            "message_id": "m500",
            "speaker_id": "user-jordan",
            "text": "Hush, check this out.",
        },
    }
    client.post("/api/chat/message", json=payload)

    history = conversation_manager.get_history("room-history-ca")
    assert len(history) == 2
    assert history[0].speaker == "user-jordan"
    assert history[0].text == "Hush, check this out."
    assert history[1].speaker == "Hush"
    assert "[Development Stub]" in history[1].text
