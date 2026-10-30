"""Tests for Phase 1 Step 4: Confidence Semantics & Decision Telemetry.

Verifies:
- Explicit confidence semantics (calibrated_probability, uncalibrated_score, none).
- Semantic validation preventing dishonest pairings (none with score, or score kind with None).
- Development stub emits confidence=None and confidence_kind="none" (no fabricated confidence).
- Addressing telemetry (addressed_to_hush, address_method) across all five methods:
  mention, reply, name, request, none.
- Decision metadata exposure (method, fallback_used, latency_ms).
- Legacy request shape continues to receive strictly the 7-field ChatMessageResponse contract.
- Conversation-aware request receives the rich 10-field ConversationAwareChatMessageResponse.
- Metrics telemetry (/api/metrics) reports address method breakdown, addressed/unaddressed
  counts, and confidence kind distributions.
- Metrics reset (/api/metrics/reset) clears all new telemetry counters.
- SILENT invariant strictly maintained across all response formats.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas import (
    Action,
    ChatMessageResponse,
    ConversationAwareChatMessageResponse,
    DecisionMetadata,
    DecisionResult,
)
from app.services.conversation import conversation_manager
from app.services.decider import DevelopmentStubDecider
from app.services.metrics import metrics_tracker

client = TestClient(app)

EXPECTED_LEGACY_KEYS = {
    "conversation_id",
    "action",
    "confidence",
    "reply",
    "generation_used",
    "fallback_used",
    "decision_latency_ms",
}

EXPECTED_CONVERSATION_AWARE_KEYS = {
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
# A. Confidence Semantics & Schema Validation
# ---------------------------------------------------------------------------

def test_confidence_semantics_default_none_when_confidence_is_none():
    """DecisionResult defaults confidence_kind to 'none' when confidence is None."""
    decision = DecisionResult(action=Action.SILENT, confidence=None)
    assert decision.confidence is None
    assert decision.confidence_kind == "none"
    assert decision.method == "rules"


def test_confidence_semantics_default_uncalibrated_when_score_provided():
    """DecisionResult defaults confidence_kind to 'uncalibrated_score' when score is provided without explicit kind."""
    decision = DecisionResult(action=Action.SPEAK, confidence=0.82)
    assert decision.confidence == 0.82
    assert decision.confidence_kind == "uncalibrated_score"


def test_confidence_semantics_calibrated_probability_explicit():
    """DecisionResult accepts explicit 'calibrated_probability' with numeric confidence."""
    decision = DecisionResult(
        action=Action.SPEAK,
        confidence=0.88,
        confidence_kind="calibrated_probability",
        method="calibrated_clf",
    )
    assert decision.confidence == 0.88
    assert decision.confidence_kind == "calibrated_probability"
    assert decision.method == "calibrated_clf"


def test_confidence_semantics_rejects_none_kind_with_numeric_confidence():
    """Schemas reject pairing confidence_kind='none' with a non-null numeric score."""
    with pytest.raises(ValidationError) as exc:
        DecisionResult(
            action=Action.SPEAK,
            confidence=0.90,
            confidence_kind="none",
        )
    assert "confidence must be None when confidence_kind is 'none'" in str(exc.value)

    with pytest.raises(ValidationError) as exc2:
        ConversationAwareChatMessageResponse(
            conversation_id="room-1",
            message_id="m1",
            addressed_to_hush=True,
            address_method="mention",
            action=Action.SPEAK,
            confidence=0.90,
            confidence_kind="none",
            reply="Hello",
            generation_used=True,
            decision=DecisionMetadata(method="rules"),
        )
    assert "confidence must be None when confidence_kind is 'none'" in str(exc2.value)


def test_confidence_semantics_rejects_score_kind_with_none_confidence():
    """Schemas reject pairing confidence_kind != 'none' with confidence=None."""
    with pytest.raises(ValidationError) as exc1:
        DecisionResult(
            action=Action.SPEAK,
            confidence=None,
            confidence_kind="calibrated_probability",
        )
    assert "confidence must not be None when confidence_kind is 'calibrated_probability'" in str(exc1.value)

    with pytest.raises(ValidationError) as exc2:
        DecisionResult(
            action=Action.SPEAK,
            confidence=None,
            confidence_kind="uncalibrated_score",
        )
    assert "confidence must not be None when confidence_kind is 'uncalibrated_score'" in str(exc2.value)


def test_development_stub_decider_emits_honest_none_confidence():
    """DevelopmentStubDecider never fabricates high confidence or claims calibrated probability."""
    decider = DevelopmentStubDecider()
    from app.schemas import Message

    msgs = [Message(speaker="u1", text="Hush, can you check this?")]
    result = decider.decide(msgs)

    assert result.confidence is None
    assert result.confidence_kind == "none"
    assert result.method == "rules"
    assert result.decision_latency_ms is None


# ---------------------------------------------------------------------------
# B. Addressing Telemetry Across Five Addressing Methods
# ---------------------------------------------------------------------------

def test_telemetry_address_method_mention():
    """Explicit participant mention (@hush) produces address_method='mention' and addressed_to_hush=True."""
    payload = {
        "conversation_id": "room-tel-1",
        "message": {
            "message_id": "m-mention-1",
            "speaker_id": "u-alex",
            "text": "Can you check the deploy?",
            "mentions": ["hush"],
        },
    }
    resp = client.post("/api/chat/message", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert set(data.keys()) == EXPECTED_CONVERSATION_AWARE_KEYS
    assert data["addressed_to_hush"] is True
    assert data["address_method"] == "mention"
    assert data["confidence_kind"] == "none"
    assert data["confidence"] is None
    assert data["decision"]["method"] == "rules"
    assert data["decision"]["fallback_used"] is False


def test_telemetry_address_method_reply():
    """Replying to a known assistant message produces address_method='reply' and addressed_to_hush=True."""
    conversation_manager.add_message(
        conversation_id="room-tel-2",
        speaker="Hush",
        text="The build completed successfully.",
        message_id="hush-reply-target",
        speaker_id="hush",
        role="assistant",
    )

    payload = {
        "conversation_id": "room-tel-2",
        "message": {
            "message_id": "m-reply-1",
            "speaker_id": "u-sam",
            "text": "Thanks, what about the tests?",
            "reply_to": "hush-reply-target",
        },
    }
    resp = client.post("/api/chat/message", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["addressed_to_hush"] is True
    assert data["address_method"] == "reply"
    assert data["action"] == "SPEAK"


def test_telemetry_address_method_name():
    """Direct name invocation in text without metadata produces address_method='name' and addressed_to_hush=True."""
    payload = {
        "conversation_id": "room-tel-3",
        "message": {
            "message_id": "m-name-1",
            "speaker_id": "u-taylor",
            "text": "Hush, are the logs available yet?",
        },
    }
    resp = client.post("/api/chat/message", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["addressed_to_hush"] is True
    assert data["address_method"] == "name"
    assert data["action"] == "SPEAK"


def test_telemetry_address_method_request():
    """Direct role request ('bot, can you help...') produces address_method='request' and addressed_to_hush=True."""
    payload = {
        "conversation_id": "room-tel-4",
        "message": {
            "message_id": "m-req-1",
            "speaker_id": "u-jordan",
            "text": "assistant, please run the smoke tests.",
        },
    }
    resp = client.post("/api/chat/message", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["addressed_to_hush"] is True
    assert data["address_method"] == "request"
    assert data["action"] == "SPEAK"


def test_telemetry_address_method_none_and_unaddressed():
    """Human banter without signals produces address_method='none' and addressed_to_hush=False."""
    payload = {
        "conversation_id": "room-tel-5",
        "message": {
            "message_id": "m-none-1",
            "speaker_id": "u-casey",
            "text": "Grabbing some lunch, back in 20.",
        },
    }
    resp = client.post("/api/chat/message", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["addressed_to_hush"] is False
    assert data["address_method"] == "none"
    assert data["action"] == "SILENT"
    assert data["reply"] is None
    assert data["generation_used"] is False


# ---------------------------------------------------------------------------
# C. Contract Boundary: Legacy vs. Conversation-Aware Response Shapes
# ---------------------------------------------------------------------------

def test_legacy_response_shape_strictly_preserved():
    """Legacy requests continue to receive the exact 7-field legacy response contract."""
    payload = {
        "conversation_id": "room-legacy-contract",
        "user_id": "user-alex",
        "message": "Hush, what is the status?",
    }
    resp = client.post("/api/chat/message", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert set(data.keys()) == EXPECTED_LEGACY_KEYS
    assert "addressed_to_hush" not in data
    assert "address_method" not in data
    assert "confidence_kind" not in data
    assert "decision" not in data
    assert data["conversation_id"] == "room-legacy-contract"
    assert data["action"] == "SPEAK"
    assert data["generation_used"] is True


def test_conversation_aware_response_shape_has_full_telemetry():
    """Conversation-aware requests receive the 10-field response contract with nested decision."""
    payload = {
        "conversation_id": "room-ca-contract",
        "message": {
            "message_id": "m-ca-1",
            "speaker_id": "u-alex",
            "text": "Hush, what is the status?",
        },
    }
    resp = client.post("/api/chat/message", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert set(data.keys()) == EXPECTED_CONVERSATION_AWARE_KEYS
    assert data["message_id"] == "m-ca-1"
    assert data["addressed_to_hush"] is True
    assert data["address_method"] == "name"
    assert data["confidence_kind"] == "none"
    assert data["confidence"] is None
    assert isinstance(data["decision"], dict)
    assert set(data["decision"].keys()) == {"method", "fallback_used", "latency_ms"}
    assert data["decision"]["method"] == "rules"
    assert data["decision"]["fallback_used"] is False


# ---------------------------------------------------------------------------
# D. Metrics Endpoint Telemetry Breakdowns & Reset
# ---------------------------------------------------------------------------

def test_metrics_telemetry_address_methods_and_confidence_breakdown():
    """GET /api/metrics tracks address method breakdown, addressed counts, and confidence kinds."""
    # 1. mention -> addressed
    client.post("/api/chat/message", json={
        "conversation_id": "room-m-1",
        "message": {
            "message_id": "m1",
            "speaker_id": "u1",
            "text": "Hello there",
            "mentions": ["hush"],
        },
    })

    # 2. name -> addressed
    client.post("/api/chat/message", json={
        "conversation_id": "room-m-1",
        "message": {
            "message_id": "m2",
            "speaker_id": "u2",
            "text": "Hush, check this please",
        },
    })

    # 3. unaddressed banter -> unaddressed, method none
    client.post("/api/chat/message", json={
        "conversation_id": "room-m-1",
        "message": {
            "message_id": "m3",
            "speaker_id": "u3",
            "text": "Nice weather outside today",
        },
    })

    # 4. legacy request -> addressed via name
    client.post("/api/chat/message", json={
        "conversation_id": "room-m-1",
        "user_id": "u4",
        "message": "Hush, help!",
    })

    metrics = client.get("/api/metrics").json()
    assert metrics["messages_evaluated"] == 4

    # Addressing telemetry counts
    assert metrics["addressed_messages"] == 3
    assert metrics["unaddressed_messages"] == 1
    assert metrics["addressed_messages"] + metrics["unaddressed_messages"] == 4

    # Address method breakdown
    methods = metrics["address_methods"]
    assert methods["mention"] == 1
    assert methods["reply"] == 0
    assert methods["name"] == 2
    assert methods["request"] == 0
    assert methods["none"] == 1
    assert sum(methods.values()) == 4

    # Confidence kind distribution (all stub/rules so far -> none)
    kinds = metrics["confidence_kinds"]
    assert kinds["calibrated_probability"] == 0
    assert kinds["uncalibrated_score"] == 0
    assert kinds["none"] == 4
    assert sum(kinds.values()) == 4


def test_metrics_reset_clears_all_new_telemetry_counters():
    """POST /api/metrics/reset clears address_methods, addressed counts, and confidence_kinds."""
    # Send message to populate counters
    client.post("/api/chat/message", json={
        "conversation_id": "room-reset-test",
        "message": {
            "message_id": "m-reset-1",
            "speaker_id": "u1",
            "text": "Hush, test",
            "mentions": ["hush"],
        },
    })

    pre_reset = client.get("/api/metrics").json()
    assert pre_reset["messages_evaluated"] == 1
    assert pre_reset["addressed_messages"] == 1
    assert pre_reset["address_methods"]["mention"] == 1

    # Reset
    reset_resp = client.post("/api/metrics/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "ok"

    post_reset = client.get("/api/metrics").json()
    assert post_reset["messages_evaluated"] == 0
    assert post_reset["addressed_messages"] == 0
    assert post_reset["unaddressed_messages"] == 0
    assert post_reset["address_methods"] == {
        "mention": 0,
        "reply": 0,
        "name": 0,
        "request": 0,
        "none": 0,
    }
    assert post_reset["confidence_kinds"] == {
        "calibrated_probability": 0,
        "uncalibrated_score": 0,
        "none": 0,
    }


# ---------------------------------------------------------------------------
# E. SILENT Invariant Verification
# ---------------------------------------------------------------------------

def test_silent_invariant_with_mocked_generator():
    """SILENT decisions strictly make zero generator calls on conversation-aware requests."""
    mock_gen = MagicMock()
    mock_gen.generate_reply = MagicMock()
    mock_gen.generate_clarification = MagicMock()

    with patch("app.routes.chat.get_generator", return_value=mock_gen):
        resp = client.post("/api/chat/message", json={
            "conversation_id": "room-silent-check",
            "message": {
                "message_id": "m-sil-1",
                "speaker_id": "u-user",
                "text": "Heading offline for the day.",
            },
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["action"] == "SILENT"
        assert data["reply"] is None
        assert data["generation_used"] is False
        assert data["addressed_to_hush"] is False
        assert data["address_method"] == "none"

        mock_gen.generate_reply.assert_not_called()
        mock_gen.generate_clarification.assert_not_called()
