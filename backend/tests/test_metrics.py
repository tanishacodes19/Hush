"""Unit and integration tests for Hush metrics tracking and /api/metrics endpoints.

Verifies:
- GET /api/metrics initial state
- Counter updates on SPEAK, ASK, and SILENT
- Mutually exclusive category counting
- Invariant: SILENT increments calls avoided and NEVER attempts generation
- POST /api/metrics/reset behavior
- Generation failure tracking (failed count vs succeeded count)
- Preflight, health checks, and reset calls do NOT pollute message counts
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Action
from app.services.bedrock import BedrockServiceError
from app.services.conversation import conversation_manager
from app.services.metrics import metrics_tracker


client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset conversation and metrics before and after each test."""
    conversation_manager.clear_all()
    metrics_tracker.reset()
    yield
    conversation_manager.clear_all()
    metrics_tracker.reset()


def test_metrics_initial_state():
    """Verify initial metrics state has zero counts."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()

    assert data["messages_evaluated"] == 0
    assert data["decisions"]["SPEAK"] == 0
    assert data["decisions"]["SILENT"] == 0
    assert data["decisions"]["ASK"] == 0
    assert data["generation_calls_attempted"] == 0
    assert data["generation_calls_succeeded"] == 0
    assert data["generation_calls_failed"] == 0
    assert data["generation_calls_avoided"] == 0
    assert data["call_reduction_rate_pct"] == 0.0
    assert data["decision_latency"]["count"] == 0
    assert data["generation_latency"]["count"] == 0


def test_metrics_updated_on_chat_messages_with_category_exclusivity():
    """Verify counters update correctly and action categories are mutually exclusive."""
    # 1. Send message triggering SPEAK (direct address)
    client.post("/api/chat/message", json={
        "conversation_id": "room-m",
        "user_id": "alex",
        "message": "Hush, what is the status?",
    })

    metrics_1 = client.get("/api/metrics").json()
    assert metrics_1["messages_evaluated"] == 1
    assert metrics_1["decisions"]["SPEAK"] == 1
    assert metrics_1["decisions"]["SILENT"] == 0
    assert metrics_1["decisions"]["ASK"] == 0
    assert metrics_1["generation_calls_attempted"] == 1
    assert metrics_1["generation_calls_succeeded"] == 1
    assert metrics_1["generation_calls_avoided"] == 0
    assert metrics_1["call_reduction_rate_pct"] == 0.0

    # 2. Send message triggering SILENT (banter)
    client.post("/api/chat/message", json={
        "conversation_id": "room-m",
        "user_id": "sam",
        "message": "Getting coffee now.",
    })

    metrics_2 = client.get("/api/metrics").json()
    assert metrics_2["messages_evaluated"] == 2
    assert metrics_2["decisions"]["SPEAK"] == 1
    assert metrics_2["decisions"]["SILENT"] == 1
    assert metrics_2["decisions"]["ASK"] == 0
    # Crucial invariant: SILENT must not increment generation attempts
    assert metrics_2["generation_calls_attempted"] == 1
    assert metrics_2["generation_calls_avoided"] == 1
    assert metrics_2["call_reduction_rate_pct"] == 50.0

    # 3. Send message triggering ASK (question)
    client.post("/api/chat/message", json={
        "conversation_id": "room-m",
        "user_id": "taylor",
        "message": "Can someone help with the error?",
    })

    metrics_3 = client.get("/api/metrics").json()
    assert metrics_3["messages_evaluated"] == 3
    assert metrics_3["decisions"]["SPEAK"] == 1
    assert metrics_3["decisions"]["SILENT"] == 1
    assert metrics_3["decisions"]["ASK"] == 1
    # Category exclusivity: sum of decisions equals messages_evaluated
    total_decisions = (
        metrics_3["decisions"]["SPEAK"]
        + metrics_3["decisions"]["SILENT"]
        + metrics_3["decisions"]["ASK"]
    )
    assert total_decisions == metrics_3["messages_evaluated"]
    assert metrics_3["generation_calls_attempted"] == 2
    assert metrics_3["generation_calls_succeeded"] == 2
    assert metrics_3["generation_calls_avoided"] == 1
    assert round(metrics_3["call_reduction_rate_pct"], 1) == 33.3


def test_metrics_reset_endpoint():
    """Verify POST /api/metrics/reset clears all telemetry counters."""
    # Send messages first
    client.post("/api/chat/message", json={
        "conversation_id": "room-r",
        "user_id": "alex",
        "message": "Hush, are you there?",
    })
    client.post("/api/chat/message", json={
        "conversation_id": "room-r",
        "user_id": "sam",
        "message": "Just chatting.",
    })

    pre_reset = client.get("/api/metrics").json()
    assert pre_reset["messages_evaluated"] == 2

    # Reset metrics
    reset_resp = client.post("/api/metrics/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "ok"

    post_reset = client.get("/api/metrics").json()
    assert post_reset["messages_evaluated"] == 0
    assert post_reset["decisions"]["SPEAK"] == 0
    assert post_reset["decisions"]["SILENT"] == 0
    assert post_reset["decisions"]["ASK"] == 0
    assert post_reset["generation_calls_attempted"] == 0
    assert post_reset["generation_calls_succeeded"] == 0
    assert post_reset["generation_calls_avoided"] == 0


def test_generation_failure_updates_failed_counter():
    """When text generation fails, generation_calls_failed must be incremented."""
    mock_bedrock = MagicMock()
    mock_bedrock.generate_reply.side_effect = BedrockServiceError("ThrottlingException")

    with patch("app.routes.chat.get_generator", return_value=mock_bedrock):
        response = client.post("/api/chat/message", json={
            "conversation_id": "room-fail",
            "user_id": "alex",
            "message": "Hush, help!",
        })
        assert response.status_code == 502

    metrics = client.get("/api/metrics").json()
    assert metrics["messages_evaluated"] == 1
    assert metrics["decisions"]["SPEAK"] == 1
    assert metrics["generation_calls_attempted"] == 1
    assert metrics["generation_calls_succeeded"] == 0
    assert metrics["generation_calls_failed"] == 1


def test_health_checks_and_resets_do_not_pollute_metrics():
    """Calling /health, OPTIONS, or /metrics does not increment message evaluation counters."""
    client.get("/health")
    client.get("/health")
    client.get("/api/metrics")
    client.options("/api/chat/message")

    metrics = client.get("/api/metrics").json()
    assert metrics["messages_evaluated"] == 0
    assert metrics["generation_calls_attempted"] == 0
