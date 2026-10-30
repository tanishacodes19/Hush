"""Focused schema validation tests for Hush Pydantic v2 schemas.

Verifies constraints specified in PLAN.md:
- Action enum contains only SPEAK, SILENT, ASK
- Confidence is bounded in [0.0, 1.0]
- When action is SILENT, reply MUST be None and generation_used MUST be False
- Required fields and extra-field rejection
"""

import pytest
from pydantic import ValidationError

from app.schemas import Action, ChatMessageRequest, ChatMessageResponse, DecisionResult, Message


def test_action_enum_values():
    """Verify Action enum values match PLAN.md Section 1."""
    assert Action.SPEAK.value == "SPEAK"
    assert Action.SILENT.value == "SILENT"
    assert Action.ASK.value == "ASK"
    assert set(Action) == {Action.SPEAK, Action.SILENT, Action.ASK}


def test_action_enum_invalid():
    """Invalid actions must raise ValueError."""
    with pytest.raises(ValueError):
        Action("IGNORE")


def test_chat_message_request_valid():
    """Valid request parsing."""
    req = ChatMessageRequest(
        conversation_id="demo-room",
        user_id="user-1",
        message="Does anyone know why the deployment failed?",
    )
    assert req.conversation_id == "demo-room"
    assert req.user_id == "user-1"
    assert req.message == "Does anyone know why the deployment failed?"


def test_chat_message_request_rejects_empty_and_extra_fields():
    """Empty strings and unknown extra fields must fail validation."""
    with pytest.raises(ValidationError):
        ChatMessageRequest(conversation_id="", user_id="u1", message="hello")

    with pytest.raises(ValidationError):
        ChatMessageRequest(conversation_id="c1", user_id="u1", message="hello", unexpected_field="bad")


def test_decision_result_valid():
    """Valid decision result parsing."""
    res = DecisionResult(
        action=Action.SILENT,
        confidence=0.91,
        fallback_used=False,
        decision_latency_ms=118.0,
    )
    assert res.action == Action.SILENT
    assert res.confidence == 0.91
    assert res.fallback_used is False
    assert res.decision_latency_ms == 118.0


def test_decision_result_confidence_bounds():
    """Confidence must be between 0.0 and 1.0 when provided."""
    with pytest.raises(ValidationError):
        DecisionResult(action=Action.SPEAK, confidence=1.05)

    with pytest.raises(ValidationError):
        DecisionResult(action=Action.SPEAK, confidence=-0.1)


def test_decision_result_confidence_none():
    """Confidence may be None when unavailable from the model."""
    res = DecisionResult(action=Action.SILENT, confidence=None)
    assert res.confidence is None

    res_default = DecisionResult(action=Action.SILENT)
    assert res_default.confidence is None


def test_chat_message_response_speak_valid():
    """Valid SPEAK response with generation."""
    resp = ChatMessageResponse(
        conversation_id="demo-room",
        action=Action.SPEAK,
        confidence=0.89,
        reply="The deployment failed due to a missing environment variable.",
        generation_used=True,
        fallback_used=False,
        decision_latency_ms=105.0,
    )
    assert resp.action == Action.SPEAK
    assert resp.generation_used is True
    assert resp.reply is not None


def test_chat_message_response_ask_valid():
    """Valid ASK response matching PLAN.md Section 7 example."""
    resp = ChatMessageResponse(
        conversation_id="demo-room",
        action=Action.ASK,
        confidence=0.84,
        reply="Do you have the deployment log or error message?",
        generation_used=True,
        fallback_used=False,
        decision_latency_ms=121.0,
    )
    assert resp.action == Action.ASK
    assert resp.reply == "Do you have the deployment log or error message?"
    assert resp.generation_used is True


def test_chat_message_response_silent_valid():
    """Valid SILENT response: reply must be None, generation_used must be False."""
    resp = ChatMessageResponse(
        conversation_id="demo-room",
        action=Action.SILENT,
        confidence=0.95,
        reply=None,
        generation_used=False,
        fallback_used=False,
        decision_latency_ms=85.0,
    )
    assert resp.action == Action.SILENT
    assert resp.reply is None
    assert resp.generation_used is False


def test_chat_message_response_silent_rejects_generation_used():
    """SILENT response must reject generation_used=True."""
    with pytest.raises(ValidationError, match="generation_used must be False when action is SILENT"):
        ChatMessageResponse(
            conversation_id="demo-room",
            action=Action.SILENT,
            confidence=0.95,
            reply=None,
            generation_used=True,
        )


def test_chat_message_response_silent_rejects_non_null_reply():
    """SILENT response must reject a non-None reply."""
    with pytest.raises(ValidationError, match="reply must be None when action is SILENT"):
        ChatMessageResponse(
            conversation_id="demo-room",
            action=Action.SILENT,
            confidence=0.95,
            reply="Should not be here",
            generation_used=False,
        )


def test_chat_message_response_confidence_none():
    """ChatMessageResponse allows confidence=None when unavailable."""
    resp = ChatMessageResponse(
        conversation_id="demo-room",
        action=Action.SILENT,
        confidence=None,
        reply=None,
        generation_used=False,
    )
    assert resp.confidence is None

    resp_default = ChatMessageResponse(
        conversation_id="demo-room",
        action=Action.SILENT,
        reply=None,
        generation_used=False,
    )
    assert resp_default.confidence is None


def test_message_valid():
    """Conversation history message parsing."""
    msg = Message(speaker="alex", text="The build is broken again.")
    assert msg.speaker == "alex"
    assert msg.text == "The build is broken again."
