"""Comprehensive unit and integration tests for Hush Addressing Engine (Phase 1, Step 3).

Verifies the Two-Tier Addressing Policy from PLAN.md Section 1:
- Tier 1 (Authoritative metadata):
  1. Explicit mention of Hush's ID -> addressed (Tier 1)
  2. Reply to known Hush-authored message -> addressed (Tier 1)
  3. Reply to known human message -> not addressed (metadata wins over text)
  4. Reply to unknown message ID -> not addressed (unknown target policy)
  5. Metadata indicating a different recipient prevents text fallback from overriding it
  6. Mention and reply metadata conflicts follow plan policy (if either fires, addressed)
- Tier 2 (Conservative text fallback):
  7. Standalone "Hush" invocation is detected
  8. Case and surrounding punctuation are handled appropriately
  9. "Hush, the baby is asleep" is not incorrectly treated as an assistant request
  10. "I told Hush about the issue" is not treated as an invocation
  11. Request directed to another named person is not treated as addressed to Hush
  12. Display-name similarity does not override stable participant IDs
  13. Messages without a clear request remain unaddressed
- Integration & regressions:
  14. Legacy requests still work with existing 7-field contract
  15. Conversation-aware requests still work
  16. SILENT strictly invokes neither generator method (zero calls verified via mocks)
  17. SPEAK and ASK retain correct generator routing
  18. Conversation history and reply-target resolution work together end-to-end
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Action, DecisionResult, Message
from app.services.addressing import resolve_addressing
from app.services.conversation import conversation_manager
from app.services.metrics import metrics_tracker

client = TestClient(app)

EXPECTED_LEGACY_RESPONSE_KEYS = {
    "conversation_id",
    "action",
    "confidence",
    "reply",
    "generation_used",
    "fallback_used",
    "decision_latency_ms",
}

EXPECTED_CONVERSATION_AWARE_RESPONSE_KEYS = {
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
EXPECTED_RESPONSE_KEYS = EXPECTED_CONVERSATION_AWARE_RESPONSE_KEYS


@pytest.fixture(autouse=True)
def reset_state():
    """Ensure in-memory history and metrics are clean before and after each test."""
    conversation_manager.clear_all()
    metrics_tracker.reset()
    yield
    conversation_manager.clear_all()
    metrics_tracker.reset()


# ---------------------------------------------------------------------------
# Tier 1: Authoritative Metadata Unit Tests
# ---------------------------------------------------------------------------

def test_explicit_mention_hush_is_addressed():
    """1. Message explicitly mentioning Hush's ID is addressed to Hush (Tier 1)."""
    result = resolve_addressing(
        message_text="Can someone help?",
        mentions=["hush"],
        reply_to=None,
    )
    assert result.addressed_to_hush is True
    assert result.address_method == "mention"
    assert result.tier == 1


def test_reply_to_known_hush_message_is_addressed():
    """2. Reply targeting a known Hush-authored message is addressed to Hush (Tier 1)."""
    history = [
        Message(
            speaker="Hush",
            text="How can I assist your team?",
            message_id="msg-hush-01",
            speaker_id="hush",
            role="assistant",
        ),
        Message(
            speaker="u-bob",
            text="Good morning team.",
            message_id="msg-bob-01",
            speaker_id="u-bob",
            role="human",
        ),
    ]
    result = resolve_addressing(
        message_text="Thanks, could you check the deployment logs?",
        mentions=[],
        reply_to="msg-hush-01",
        conversation_history=history,
    )
    assert result.addressed_to_hush is True
    assert result.address_method == "reply"
    assert result.tier == 1


def test_reply_to_known_human_message_not_addressed_even_with_hush_in_text():
    """3. Reply to human is NOT addressed to Hush, even if text contains 'Hush' (metadata wins)."""
    history = [
        Message(
            speaker="u-sam",
            text="Has anyone seen the logs?",
            message_id="msg-sam-01",
            speaker_id="u-sam",
            role="human",
        )
    ]
    result = resolve_addressing(
        message_text="hush, check what Sam said",
        mentions=[],
        reply_to="msg-sam-01",
        conversation_history=history,
    )
    assert result.addressed_to_hush is False
    assert result.address_method == "none"
    assert result.tier == 1
    assert "human" in result.reason.lower()


def test_reply_to_unknown_message_id_follows_unknown_target_policy():
    """4. Reply targeting an unknown/pruned message ID is NOT addressed to Hush."""
    history = [
        Message(
            speaker="u-alex",
            text="Hello world.",
            message_id="msg-alex-01",
            speaker_id="u-alex",
            role="human",
        )
    ]
    result = resolve_addressing(
        message_text="Replying to an unknown message",
        mentions=[],
        reply_to="msg-nonexistent-99",
        conversation_history=history,
    )
    assert result.addressed_to_hush is False
    assert result.address_method == "none"
    assert result.tier == 1
    assert "unknown" in result.reason.lower()


def test_metadata_mentioning_other_recipient_prevents_text_fallback():
    """5. Mentions pointing to another participant prevent text analysis from firing."""
    result = resolve_addressing(
        message_text="Hush, can you check this?",
        mentions=["u-alex"],  # Directed to Alex, not Hush
        reply_to=None,
    )
    assert result.addressed_to_hush is False
    assert result.address_method == "none"
    assert result.tier == 1
    assert "other participant" in result.reason.lower()


def test_metadata_conflict_mention_hush_and_reply_to_human():
    """6a. If mentions includes Hush, message is addressed even if reply_to is a human."""
    history = [
        Message(
            speaker="u-alex",
            text="Question for the team.",
            message_id="msg-alex-02",
            speaker_id="u-alex",
            role="human",
        )
    ]
    # PLAN.md Section 1: "If either fires, the message is addressed to Hush"
    result = resolve_addressing(
        message_text="Replying to Alex but explicitly mentioning @Hush",
        mentions=["hush"],
        reply_to="msg-alex-02",
        conversation_history=history,
    )
    assert result.addressed_to_hush is True
    assert result.address_method == "mention"
    assert result.tier == 1


def test_metadata_conflict_reply_to_hush_and_mention_human():
    """6b. If reply_to targets Hush, message is addressed even if another human is mentioned."""
    history = [
        Message(
            speaker="Hush",
            text="What is the error code?",
            message_id="msg-hush-02",
            speaker_id="hush",
            role="assistant",
        )
    ]
    result = resolve_addressing(
        message_text="Alex mentioned the error was 500",
        mentions=["u-alex"],
        reply_to="msg-hush-02",
        conversation_history=history,
    )
    assert result.addressed_to_hush is True
    assert result.address_method == "reply"
    assert result.tier == 1


# ---------------------------------------------------------------------------
# Tier 2: Conservative Text Fallback Unit Tests
# ---------------------------------------------------------------------------

def test_standalone_hush_invocation_detected():
    """7. Standalone 'Hush' invocation is detected server-side (Tier 2)."""
    r1 = resolve_addressing(message_text="Hush, what is the plan?")
    assert r1.addressed_to_hush is True
    assert r1.address_method == "name"
    assert r1.tier == 2

    r2 = resolve_addressing(message_text="Hush")
    assert r2.addressed_to_hush is True
    assert r2.address_method == "name"
    assert r2.tier == 2


def test_case_and_punctuation_handling():
    """8. Case variations and surrounding punctuation are normalized correctly."""
    variations = [
        "hey Hush, can you check this?",
        "HELLO HUSH: status report please",
        "@hush - what is the status?",
        "Can you help, Hush?",
        "ok hush!",
    ]
    for text in variations:
        res = resolve_addressing(message_text=text)
        assert res.addressed_to_hush is True, f"Failed for text: '{text}'"
        assert res.address_method == "name"
        assert res.tier == 2


def test_hush_baby_asleep_false_positive_rejected():
    """9. 'Hush, the baby is asleep' is not treated as an assistant invocation."""
    sentences = [
        "Hush, the baby is asleep.",
        "hush, the baby is sleeping",
        "hush, be quiet please",
        "hush now",
        "please hush",
    ]
    for text in sentences:
        res = resolve_addressing(message_text=text)
        assert res.addressed_to_hush is False, f"False positive on: '{text}'"
        assert res.address_method == "none"
        assert res.tier == 2


def test_told_hush_third_person_rejected():
    """10. 'I told Hush about the issue' is not treated as an invocation."""
    sentences = [
        "I told Hush about the issue.",
        "We asked Hush earlier today.",
        "She talked to Hush yesterday.",
        "He messaged Hush about the outage.",
        "We heard from Hush on Slack.",
        "Let us blame Hush for that.",
    ]
    for text in sentences:
        res = resolve_addressing(message_text=text)
        assert res.addressed_to_hush is False, f"False positive on: '{text}'"
        assert res.address_method == "none"
        assert res.tier == 2


def test_request_directed_to_another_named_person_rejected():
    """11. Request directed to another human is not treated as addressed to Hush."""
    sentences = [
        "Alex, can you fix this?",
        "Sam: please check the build pipeline.",
        "Taylor, what do you think?",
    ]
    for text in sentences:
        res = resolve_addressing(message_text=text)
        assert res.addressed_to_hush is False, f"False positive on: '{text}'"
        assert res.address_method == "none"
        assert res.tier == 2


def test_display_name_similarity_does_not_override_stable_ids():
    """12. A human participant named 'Hush' is not confused with the assistant in Tier 1."""
    # Participant has display_name 'Hush' but ID 'u-hush-fan'
    result = resolve_addressing(
        message_text="Hello friend",
        mentions=["u-hush-fan"],  # Match is strictly on ID, not display name
        reply_to=None,
    )
    assert result.addressed_to_hush is False
    assert result.address_method == "none"
    assert result.tier == 1


def test_direct_role_request_detected():
    """Direct request naming the assistant's role is detected (address_method='request')."""
    requests = [
        "Can the bot help with this issue?",
        "Could the assistant summarize the thread?",
        "bot, can you check the deployment?",
        "assistant, please assist with this",
    ]
    for text in requests:
        res = resolve_addressing(message_text=text)
        assert res.addressed_to_hush is True, f"Failed to detect role request: '{text}'"
        assert res.address_method == "request"
        assert res.tier == 2


def test_messages_without_signals_unaddressed():
    """13. Ordinary banter and statements remain unaddressed."""
    sentences = [
        "Lunch at noon.",
        "I am grabbing a coffee now.",
        "A hush fell over the room.",
        "The deployment failed. Can anyone on the team help?",
    ]
    for text in sentences:
        res = resolve_addressing(message_text=text)
        assert res.addressed_to_hush is False
        assert res.address_method == "none"


# ---------------------------------------------------------------------------
# Integration & Regression Tests
# ---------------------------------------------------------------------------

def test_integration_reply_to_hush_routes_to_speak():
    """14 & 18. End-to-end: Replying to a prior Hush message routes to SPEAK."""
    # 1. First message to Hush
    resp1 = client.post("/api/chat/message", json={
        "conversation_id": "room-thread-1",
        "message": {
            "message_id": "m1",
            "speaker_id": "u-alex",
            "text": "Hush, are you ready?",
        },
    })
    assert resp1.status_code == 200
    assert resp1.json()["action"] == "SPEAK"

    # 2. Follow-up reply targeting Hush's reply ("hush-m1")
    resp2 = client.post("/api/chat/message", json={
        "conversation_id": "room-thread-1",
        "message": {
            "message_id": "m2",
            "speaker_id": "u-sam",
            "text": "Great, please proceed.",
            "reply_to": "hush-m1",
        },
    })
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert set(data2.keys()) == EXPECTED_RESPONSE_KEYS
    assert data2["action"] == "SPEAK"
    assert data2["generation_used"] is True
    assert data2["reply"] is not None


def test_integration_reply_to_human_with_hush_in_text_routes_to_silent():
    """15 & 17. End-to-end: Replying to human with 'hush' in text routes to SILENT."""
    # Seed human message in history
    conversation_manager.add_message(
        conversation_id="room-thread-2",
        speaker="u-alex",
        text="Should we pause?",
        message_id="msg-human-10",
        speaker_id="u-alex",
        role="human",
    )

    # Human-directed reply mentioning 'hush'
    resp = client.post("/api/chat/message", json={
        "conversation_id": "room-thread-2",
        "message": {
            "message_id": "msg-human-11",
            "speaker_id": "u-sam",
            "text": "hush, let us wait for the manager.",
            "reply_to": "msg-human-10",
        },
    })
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == EXPECTED_CONVERSATION_AWARE_RESPONSE_KEYS
    assert data["action"] == "SILENT"
    assert data["reply"] is None
    assert data["generation_used"] is False


def test_integration_silent_makes_zero_generator_calls_verified_with_mocks():
    """16 & 17. CRITICAL INVARIANT: SILENT makes strictly zero calls to generator."""
    mock_gen = MagicMock()
    mock_gen.generate_reply = MagicMock()
    mock_gen.generate_clarification = MagicMock()

    with patch("app.routes.chat.get_generator", return_value=mock_gen):
        # Ordinary human statement
        response = client.post("/api/chat/message", json={
            "conversation_id": "room-silent-mock",
            "message": {
                "message_id": "m-coffee",
                "speaker_id": "u-jordan",
                "text": "Heading to grab some tea.",
            },
        })
        assert response.status_code == 200
        data = response.json()

        assert data["action"] == "SILENT"
        assert data["reply"] is None
        assert data["generation_used"] is False

        mock_gen.generate_reply.assert_not_called()
        mock_gen.generate_clarification.assert_not_called()
        assert mock_gen.generate_reply.call_count == 0
        assert mock_gen.generate_clarification.call_count == 0


def test_integration_legacy_requests_remain_fully_functional():
    """14 & 16. Legacy request shape (conversation_id, user_id, message) continues working."""
    payload = {
        "conversation_id": "room-legacy-verify",
        "user_id": "user-alex",
        "message": "Hush, what is the plan?",
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert set(data.keys()) == EXPECTED_LEGACY_RESPONSE_KEYS
    assert data["conversation_id"] == "room-legacy-verify"
    assert data["action"] == "SPEAK"
    assert data["generation_used"] is True
    assert data["reply"] is not None
