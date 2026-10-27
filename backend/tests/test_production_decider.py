"""Comprehensive test suite for Hush production restraint classifier decider.

Verifies:
1. Model loading, validation, and working-directory independence.
2. Strict error handling (FileNotFoundError, ValueError) without silent fallback to stub.
3. Production chat API routing with default configuration (HUSH_DECIDER_MODE='classifier').
4. Realization of all three actions (SPEAK, ASK, SILENT) on realistic inputs.
5. Strict restraint invariant: SILENT makes zero generation calls and returns reply=None.
6. Addressing invariant safeguards: addressed messages never return SILENT.
7. Legacy ChatMessageRequest compatibility and telemetry honesty (uncalibrated_score).
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.schemas import Action, Message
from app.services.decider import (
    DEFAULT_MODEL_PATH,
    DevelopmentStubDecider,
    RealHushGateDecider,
    TrainedRestraintClassifierDecider,
    clear_decider_cache,
    get_decider,
    get_production_decider,
    resolve_model_path,
)


@pytest.fixture(autouse=True)
def reset_decider():
    """Ensure decider cache is cleared before and after each test."""
    clear_decider_cache()
    yield
    clear_decider_cache()


# ---------------------------------------------------------------------------
# 1. Model Loading and Path Independence Tests
# ---------------------------------------------------------------------------

def test_model_loading_and_cwd_independence(tmp_path):
    """Verify TrainedRestraintClassifierDecider loads properly regardless of cwd."""
    # Default resolution
    resolved = resolve_model_path()
    assert resolved.exists(), f"Default model path does not exist: {resolved}"
    assert resolved == DEFAULT_MODEL_PATH.resolve()

    # Instantiate decider from current dir
    decider = TrainedRestraintClassifierDecider()
    assert decider.model is not None
    assert decider.model.is_fitted is True
    assert decider.model.num_features == 22
    assert len(decider.model.weights[0]) == 22

    # Change current working directory to a temporary directory
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # Verify resolution still finds repository data/models/restraint_classifier_v1.json
        resolved_from_tmp = resolve_model_path()
        assert resolved_from_tmp.exists()
        assert resolved_from_tmp == DEFAULT_MODEL_PATH.resolve()

        # Instantiate new decider from inside temporary directory
        decider_from_tmp = TrainedRestraintClassifierDecider()
        assert decider_from_tmp.model.is_fitted is True
    finally:
        os.chdir(original_cwd)


def test_resolve_model_path_custom_and_relative(tmp_path):
    """Verify resolve_model_path with explicit absolute and relative paths."""
    # Absolute path
    custom_abs = tmp_path / "custom_model.json"
    custom_abs.write_text("{}", encoding="utf-8")
    assert resolve_model_path(custom_abs) == custom_abs.resolve()

    # Relative to repo root
    repo_rel = Path("data/models/restraint_classifier_v1.json")
    resolved = resolve_model_path(repo_rel)
    assert resolved.exists()
    assert resolved == DEFAULT_MODEL_PATH.resolve()


def test_missing_model_raises_file_not_found_no_stub_fallback(tmp_path):
    """Missing model artifact must raise FileNotFoundError and never fall back to stub."""
    nonexistent = tmp_path / "nonexistent_model.json"
    with pytest.raises(FileNotFoundError) as exc_info:
        TrainedRestraintClassifierDecider(model_path=nonexistent)

    assert "Trained restraint classifier artifact not found" in str(exc_info.value)


def test_corrupt_json_model_raises_value_error(tmp_path):
    """Corrupt JSON in model file must raise ValueError."""
    corrupt_file = tmp_path / "corrupt_model.json"
    corrupt_file.write_text("{this is not valid json", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        TrainedRestraintClassifierDecider(model_path=corrupt_file)

    assert "Corrupted or invalid JSON" in str(exc_info.value)


def test_missing_keys_in_model_raises_value_error(tmp_path):
    """Model JSON missing required keys must raise ValueError."""
    incomplete_file = tmp_path / "incomplete_model.json"
    incomplete_file.write_text(json.dumps({"weights": {}}), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        TrainedRestraintClassifierDecider(model_path=incomplete_file)

    assert "missing required keys" in str(exc_info.value)


def test_unfitted_model_raises_value_error(tmp_path):
    """Model JSON with is_fitted=False must raise ValueError."""
    unfitted_file = tmp_path / "unfitted_model.json"
    unfitted_data = {
        "weights": {},
        "biases": {},
        "is_fitted": False,
        "class_labels": ["SILENT", "SPEAK", "ASK"],
        "feature_names": ["f1"],
    }
    unfitted_file.write_text(json.dumps(unfitted_data), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        TrainedRestraintClassifierDecider(model_path=unfitted_file)

    assert "model is not fitted" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. Decider Factory & Routing Tests
# ---------------------------------------------------------------------------

def test_get_decider_factory_default_is_classifier():
    """Default mode in settings is 'classifier', returning TrainedRestraintClassifierDecider."""
    assert settings.HUSH_DECIDER_MODE == "classifier"
    decider = get_decider()
    assert isinstance(decider, TrainedRestraintClassifierDecider)


def test_get_decider_explicit_modes():
    """Explicit mode parameters return appropriate adapters."""
    stub = get_decider(mode="stub")
    assert isinstance(stub, DevelopmentStubDecider)

    real = get_decider(mode="real")
    assert isinstance(real, RealHushGateDecider)

    classifier = get_decider(mode="classifier")
    assert isinstance(classifier, TrainedRestraintClassifierDecider)


def test_singleton_get_production_decider():
    """get_production_decider returns singleton when no custom path is given."""
    d1 = get_production_decider()
    d2 = get_production_decider()
    assert d1 is d2


# ---------------------------------------------------------------------------
# 3. Decision Logic and Three Action Realization Tests
# ---------------------------------------------------------------------------

def test_production_decider_action_speak():
    """Directly addressed, clear technical question produces SPEAK."""
    decider = get_production_decider()
    messages = [
        Message(speaker="Alice", text="@hush What is the time complexity of binary search?")
    ]
    result = decider.decide(messages)

    assert result.action == Action.SPEAK
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence_kind == "uncalibrated_score"
    assert result.method == "ml_restraint_classifier"
    assert result.fallback_used is False
    assert result.decision_latency_ms is not None
    assert result.decision_latency_ms >= 0.0


def test_production_decider_action_ask():
    """Directly addressed, ambiguous/underspecified request produces ASK."""
    decider = get_production_decider()
    messages = [
        Message(speaker="Alice", text="@hush can you fix this query? it gave an error")
    ]
    result = decider.decide(messages)

    assert result.action == Action.ASK
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence_kind == "uncalibrated_score"
    assert result.method == "ml_restraint_classifier"
    assert result.fallback_used is False


def test_production_decider_action_silent():
    """Unaddressed casual conversation / banter produces SILENT."""
    decider = get_production_decider()
    messages = [
        Message(speaker="Alice", text="Anyone want to grab tacos for lunch?"),
        Message(speaker="Bob", text="I'm in! Let's go at 12:30."),
    ]
    result = decider.decide(messages)

    assert result.action == Action.SILENT
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence_kind == "uncalibrated_score"
    assert result.method == "ml_restraint_classifier"
    assert result.fallback_used is False


def test_production_decider_empty_messages_returns_silent():
    """Empty conversation history returns default SILENT decision."""
    decider = get_production_decider()
    result = decider.decide([])

    assert result.action == Action.SILENT
    assert result.confidence is None
    assert result.confidence_kind == "none"
    assert result.method == "ml_restraint_classifier"


# ---------------------------------------------------------------------------
# 4. Invariant Tests: SILENT Zero Generation and Addressing Safeguard
# ---------------------------------------------------------------------------

def test_production_api_silent_zero_generation_calls(monkeypatch):
    """When the production classifier decides SILENT, generation must never be invoked."""
    client = TestClient(app)

    mock_gen = MagicMock()
    mock_gen.generate_reply = MagicMock()
    mock_gen.generate_clarification = MagicMock()

    with patch("app.routes.chat.get_generator", return_value=mock_gen):
        payload = {
            "conversation_id": "conv-silent-test",
            "message": {
                "message_id": "m100",
                "speaker_id": "alice",
                "text": "Heading out for lunch now, see everyone later.",
            },
        }
        resp = client.post("/api/chat/message", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "SILENT"
        assert data["reply"] is None
        assert data["generation_used"] is False

        # Invariant: exactly 0 calls to generator
        mock_gen.generate_reply.assert_not_called()
        mock_gen.generate_clarification.assert_not_called()


def test_production_api_speak_invokes_reply_generator():
    """When the production classifier decides SPEAK, generate_reply is called."""
    client = TestClient(app)

    mock_gen = MagicMock()
    mock_gen.generate_reply = MagicMock(return_value="Binary search is O(log n).")
    mock_gen.generate_clarification = MagicMock()

    with patch("app.routes.chat.get_generator", return_value=mock_gen):
        payload = {
            "conversation_id": "conv-speak-test",
            "message": {
                "message_id": "m101",
                "speaker_id": "alice",
                "text": "@hush What is the time complexity of binary search?",
            },
        }
        resp = client.post("/api/chat/message", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "SPEAK"
        assert data["reply"] == "Binary search is O(log n)."
        assert data["generation_used"] is True

        mock_gen.generate_reply.assert_called_once()
        mock_gen.generate_clarification.assert_not_called()


def test_production_api_ask_invokes_clarification_generator():
    """When the production classifier decides ASK, generate_clarification is called."""
    client = TestClient(app)

    mock_gen = MagicMock()
    mock_gen.generate_reply = MagicMock()
    mock_gen.generate_clarification = MagicMock(return_value="Could you share the SQL query and error message?")

    with patch("app.routes.chat.get_generator", return_value=mock_gen):
        payload = {
            "conversation_id": "conv-ask-test",
            "message": {
                "message_id": "m102",
                "speaker_id": "alice",
                "text": "@hush can you fix this query? it gave an error",
            },
        }
        resp = client.post("/api/chat/message", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "ASK"
        assert data["reply"] == "Could you share the SQL query and error message?"
        assert data["generation_used"] is True

        mock_gen.generate_clarification.assert_called_once()
        mock_gen.generate_reply.assert_not_called()


def test_addressing_safeguard_never_silent():
    """Addressed messages must never resolve to SILENT even if underlying model predicted it."""
    decider = get_production_decider()

    # Simulate an edge case where raw model would predict SILENT
    with patch.object(
        decider.model,
        "decide_with_telemetry",
        return_value={
            "action": "SILENT",
            "confidence": 0.55,
            "confidence_kind": "uncalibrated_score",
            "probabilities": {"SILENT": 0.55, "SPEAK": 0.35, "ASK": 0.10},
            "latency_ms": 0.2,
        },
    ):
        messages = [
            Message(speaker="Alice", text="@hush help me with this task")
        ]
        result = decider.decide(messages)

        # Addressing invariant must have overridden SILENT to SPEAK (since speak_p 0.35 >= ask_p 0.10)
        assert result.action != Action.SILENT
        assert result.action == Action.SPEAK


# ---------------------------------------------------------------------------
# 5. Production API Integration & Compatibility Tests
# ---------------------------------------------------------------------------

def test_production_api_headers_and_telemetry():
    """Verify HTTP response headers and decision telemetry under default production mode."""
    client = TestClient(app)

    payload = {
        "conversation_id": "conv-prod-headers",
        "message": {
            "message_id": "m103",
            "speaker_id": "alice",
            "text": "Heading out for lunch now.",
        },
    }
    resp = client.post("/api/chat/message", json=payload)

    assert resp.status_code == 200
    assert resp.headers.get("x-hush-decider-mode") == "classifier"

    data = resp.json()
    assert data["decision"]["method"] == "ml_restraint_classifier"
    assert data["confidence_kind"] == "uncalibrated_score"
    assert isinstance(data["confidence"], float)
    assert data["decision"]["latency_ms"] >= 0.0


def test_production_api_legacy_request_compatibility():
    """Verify legacy ChatMessageRequest payload shape works seamlessly with production decider."""
    client = TestClient(app)

    legacy_payload = {
        "conversation_id": "conv-legacy-prod",
        "user_id": "user-42",
        "message": "@hush What is the time complexity of binary search?",
    }
    mock_gen = MagicMock()
    mock_gen.generate_reply = MagicMock(return_value="Binary search is O(log n).")
    mock_gen.generate_clarification = MagicMock()

    with patch("app.routes.chat.get_generator", return_value=mock_gen):
        resp = client.post("/api/chat/message", json=legacy_payload)

        assert resp.status_code == 200
        data = resp.json()

        # Legacy response shape fields
        assert "conversation_id" in data
        assert "action" in data
        assert "confidence" in data
        assert "reply" in data
        assert "generation_used" in data
        assert "fallback_used" in data
        assert "decision_latency_ms" in data

        assert data["action"] == "SPEAK"
        assert isinstance(data["confidence"], float)
        assert data["generation_used"] is True
        assert data["decision_latency_ms"] >= 0.0


def test_production_decider_determinism():
    """Multiple decisions on identical messages yield identical output."""
    decider = get_production_decider()
    messages = [
        Message(speaker="Alice", text="@hush What is the time complexity of binary search?")
    ]

    r1 = decider.decide(messages)
    r2 = decider.decide(messages)

    assert r1.action == r2.action
    assert r1.confidence == r2.confidence
    assert r1.confidence_kind == r2.confidence_kind
    assert r1.method == r2.method


# ---------------------------------------------------------------------------
# 6. CORS and Environment Configuration Tests
# ---------------------------------------------------------------------------

def test_cors_origins_parsing_various_formats():
    """Verify Settings parses comma-separated, single string, and JSON list for CORS_ORIGINS."""
    from app.config import Settings

    # Comma-separated format
    s_csv = Settings(CORS_ORIGINS="https://app.vercel.app, https://hush.example.com")
    assert s_csv.CORS_ORIGINS == ["https://app.vercel.app", "https://hush.example.com"]

    # Single string format (like backend/.env.example)
    s_single = Settings(CORS_ORIGINS="https://app.vercel.app")
    assert s_single.CORS_ORIGINS == ["https://app.vercel.app"]

    # JSON array string format
    s_json = Settings(CORS_ORIGINS='["https://app.vercel.app", "http://localhost:5173"]')
    assert s_json.CORS_ORIGINS == ["https://app.vercel.app", "http://localhost:5173"]

    # Native list format
    s_list = Settings(CORS_ORIGINS=["https://app.vercel.app"])
    assert s_list.CORS_ORIGINS == ["https://app.vercel.app"]


def test_cors_preflight_allowed_and_disallowed_origins():
    """Verify CORS preflight handling for allowed and disallowed origins."""
    client = TestClient(app)

    # Allowed origin preflight
    resp_allowed = client.options(
        "/api/chat/message",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert resp_allowed.status_code == 200
    assert resp_allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert resp_allowed.headers.get("access-control-allow-credentials") == "true"

    # Disallowed origin preflight
    resp_disallowed = client.options(
        "/api/chat/message",
        headers={
            "Origin": "https://malicious-site.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert resp_disallowed.status_code == 400
    assert "access-control-allow-origin" not in resp_disallowed.headers

