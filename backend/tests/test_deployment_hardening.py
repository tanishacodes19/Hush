"""Tests for Deployment Hardening features in Hush.

Covers:
- Metrics reset endpoint security (403 when not development)
- Input limits (message length 422, request body size 413)
- Conversation memory bound (oldest-first eviction, 20-turn preserved)
- In-memory rate limiting & stale entry cleanup (429 Too Many Requests)
- Client IP extraction & trusted proxy headers (right-counted, spoof-resistant)
- CORS configuration validation (production failure on empty origins)
- Health endpoint allowlist (only status, generation_mode, decider_mode)
- Stub mode honesty ([SIMULATED DEMO] prefix, no implicit stub fallback)
"""

import time
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.config import Settings, settings
from app.main import app
from app.schemas import Action, Message
from app.services.conversation import ConversationManager
from app.services.generation import DevelopmentStubGenerator, get_generator
from app.services.rate_limiter import RateLimiter, get_client_ip, rate_limiter

client = TestClient(app)


# ============================================================================
# 1. Metrics Reset Endpoint
# ============================================================================

def test_metrics_reset_disabled_in_production(monkeypatch):
    """POST /api/metrics/reset must return 403 Forbidden in production mode."""
    monkeypatch.setattr(settings, "HUSH_ENV", "production")
    response = client.post("/api/metrics/reset")
    assert response.status_code == 403
    assert "development mode" in response.json()["detail"]


def test_metrics_reset_enabled_in_development(monkeypatch):
    """POST /api/metrics/reset must return 200 OK in development mode."""
    monkeypatch.setattr(settings, "HUSH_ENV", "development")
    response = client.post("/api/metrics/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ============================================================================
# 2. Input Limits (Message Length & Request Body Size)
# ============================================================================

def test_oversized_message_rejected_with_422(monkeypatch):
    """Message exceeding HUSH_MAX_MESSAGE_LENGTH must be rejected with HTTP 422."""
    monkeypatch.setattr(settings, "HUSH_MAX_MESSAGE_LENGTH", 100)
    oversized_text = "x" * 105
    payload = {
        "conversation_id": "room-limits",
        "user_id": "alice",
        "message": oversized_text,
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 422
    assert "exceeds limit" in str(response.json())


def test_valid_message_length_accepted(monkeypatch):
    """Message within HUSH_MAX_MESSAGE_LENGTH must be accepted."""
    monkeypatch.setattr(settings, "HUSH_MAX_MESSAGE_LENGTH", 200)
    payload = {
        "conversation_id": "room-limits",
        "user_id": "alice",
        "message": "Valid short message.",
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 200


def test_oversized_request_body_rejected_with_413(monkeypatch):
    """Request body exceeding HUSH_MAX_REQUEST_BODY_BYTES must be rejected with HTTP 413."""
    monkeypatch.setattr(settings, "HUSH_MAX_REQUEST_BODY_BYTES", 500)
    # Construct payload larger than 500 bytes
    payload = {
        "conversation_id": "room-body-limit",
        "user_id": "alice",
        "message": "x" * 600,
    }
    response = client.post("/api/chat/message", json=payload)
    assert response.status_code == 413
    assert "Request body exceeds maximum size" in response.json()["detail"]


def test_oversized_request_body_with_misleading_content_length_rejected(monkeypatch):
    """Request body exceeding HUSH_MAX_REQUEST_BODY_BYTES is rejected based on received bytes, even with misleading Content-Length."""
    monkeypatch.setattr(settings, "HUSH_MAX_REQUEST_BODY_BYTES", 500)
    raw_bytes = b'{"conversation_id": "room-body-limit", "user_id": "alice", "message": "' + (b'x' * 600) + b'"}'
    # Content-Length is intentionally set to a small value (10), but actual body received is ~670 bytes
    response = client.post(
        "/api/chat/message",
        content=raw_bytes,
        headers={"Content-Type": "application/json", "Content-Length": "10"},
    )
    assert response.status_code == 413
    assert "Request body exceeds maximum size" in response.json()["detail"]


# ============================================================================
# 3. Conversation Memory Bound
# ============================================================================

def test_conversation_manager_total_capacity_and_oldest_eviction():
    """ConversationManager must evict the oldest conversation when capacity is reached."""
    mgr = ConversationManager(max_history=5, max_conversations=3)

    # Add messages to 3 conversations
    mgr.add_message("room-1", "user1", "Hello in room 1")
    mgr.add_message("room-2", "user2", "Hello in room 2")
    mgr.add_message("room-3", "user3", "Hello in room 3")

    assert len(mgr._conversations) == 3
    assert "room-1" in mgr._conversations
    assert "room-2" in mgr._conversations
    assert "room-3" in mgr._conversations

    # Adding a 4th room must evict room-1 (the oldest)
    mgr.add_message("room-4", "user4", "Hello in room 4")
    assert len(mgr._conversations) == 3
    assert "room-1" not in mgr._conversations
    assert "room-2" in mgr._conversations
    assert "room-3" in mgr._conversations
    assert "room-4" in mgr._conversations


def test_conversation_manager_preserves_20_turn_limit():
    """ConversationManager must continue to cap turn history per conversation at max_history."""
    mgr = ConversationManager(max_history=5, max_conversations=10)
    for i in range(10):
        mgr.add_message("room-turns", f"user-{i}", f"Message {i}")

    history = mgr.get_history("room-turns")
    assert len(history) == 5
    assert history[0].text == "Message 5"
    assert history[-1].text == "Message 9"


# ============================================================================
# 4. In-Memory Rate Limiting
# ============================================================================

def test_rate_limiter_exceeded_returns_429(monkeypatch):
    """Burst of requests exceeding rate limit threshold must return HTTP 429."""
    rate_limiter.clear()
    monkeypatch.setattr(rate_limiter, "max_requests", 3)
    monkeypatch.setattr(rate_limiter, "window_seconds", 60.0)

    payload = {"conversation_id": "room-rl", "user_id": "alice", "message": "hello"}

    for _ in range(3):
        resp = client.post("/api/chat/message", json=payload)
        assert resp.status_code == 200

    # 4th request exceeds threshold -> 429
    blocked_resp = client.post("/api/chat/message", json=payload)
    assert blocked_resp.status_code == 429
    assert blocked_resp.json()["detail"] == "Too many requests. Please try again later."


def test_rate_limiter_window_expiry_resets_allowance():
    """Rate limit allowance resets after window_seconds has elapsed."""
    limiter = RateLimiter(max_requests=2, window_seconds=10.0)
    t0 = 1000.0

    assert limiter.is_allowed("client-1", now=t0) is True
    assert limiter.is_allowed("client-1", now=t0 + 1.0) is True
    # 3rd request at t0 + 2s is blocked
    assert limiter.is_allowed("client-1", now=t0 + 2.0) is False

    # After window expiry (t0 + 11s), request is allowed again
    assert limiter.is_allowed("client-1", now=t0 + 11.0) is True


def test_rate_limiter_stale_cleanup_and_bounded_storage():
    """RateLimiter stale cleanup removes expired entries and respects max_clients."""
    limiter = RateLimiter(max_requests=5, window_seconds=10.0, max_clients=3)
    t0 = 1000.0

    limiter.is_allowed("c1", now=t0)
    limiter.is_allowed("c2", now=t0)
    limiter.is_allowed("c3", now=t0)
    assert len(limiter._clients) == 3

    # All entries are stale at t0 + 15s
    cleaned = limiter.cleanup_stale_entries(now=t0 + 15.0)
    assert cleaned == 3
    assert len(limiter._clients) == 0


def test_rate_limiter_strictly_bounded_at_max_clients_without_cleanup():
    """RateLimiter._clients length never exceeds max_clients even when new clients arrive before cleanup."""
    limiter = RateLimiter(max_requests=10, window_seconds=60.0, max_clients=5, cleanup_interval_seconds=3600.0)
    t0 = 1000.0
    # Add 50 new clients rapidly within the same second
    for i in range(50):
        client_id = f"client-{i}"
        limiter.is_allowed(client_id, now=t0)
        assert len(limiter._clients) <= 5, f"Limit exceeded: {len(limiter._clients)}"
    assert len(limiter._clients) == 5


# ============================================================================
# 5. Client IP Extraction and Proxy Headers
# ============================================================================

def make_dummy_request(client_ip="192.168.1.50", xff=None):
    """Helper to create a dummy Starlette Request for testing get_client_ip."""
    headers = []
    if xff:
        headers.append((b"x-forwarded-for", xff.encode("utf-8")))
    scope = {
        "type": "http",
        "client": (client_ip, 50000),
        "headers": headers,
    }
    return Request(scope)


def test_client_ip_default_socket_ip_ignores_xff():
    """When trust_proxy_headers=False, socket IP is always used and XFF is ignored."""
    req = make_dummy_request(client_ip="10.0.0.1", xff="203.0.113.195, 70.41.3.18")
    ip = get_client_ip(req, trust_proxy_headers=False, trusted_proxy_hops=1)
    assert ip == "10.0.0.1"


def test_client_ip_trusted_proxy_selects_right_counted_entry():
    """When trust_proxy_headers=True, trusted_proxy_hops selects entry counted from the right."""
    req = make_dummy_request(
        client_ip="10.0.0.1",
        xff="203.0.113.195, 70.41.3.18, 150.172.238.178",
    )
    # hops=1 -> rightmost entry
    ip1 = get_client_ip(req, trust_proxy_headers=True, trusted_proxy_hops=1)
    assert ip1 == "150.172.238.178"

    # hops=2 -> second from the right
    ip2 = get_client_ip(req, trust_proxy_headers=True, trusted_proxy_hops=2)
    assert ip2 == "70.41.3.18"


def test_client_ip_spoofed_left_side_xff_does_not_affect_key():
    """Spoofed entries on the left side of XFF do not affect the right-counted trusted IP."""
    req_clean = make_dummy_request(
        client_ip="10.0.0.1",
        xff="198.51.100.1, 150.172.238.178",
    )
    req_spoofed = make_dummy_request(
        client_ip="10.0.0.1",
        xff="1.2.3.4, 5.6.7.8, 198.51.100.1, 150.172.238.178",
    )
    ip_clean = get_client_ip(req_clean, trust_proxy_headers=True, trusted_proxy_hops=1)
    ip_spoofed = get_client_ip(req_spoofed, trust_proxy_headers=True, trusted_proxy_hops=1)
    assert ip_clean == "150.172.238.178"
    assert ip_spoofed == "150.172.238.178"


def test_client_ip_fallback_to_socket_ip_when_xff_missing_or_too_short():
    """Fallback to socket IP if XFF header is missing or contains fewer entries than hops."""
    req_no_xff = make_dummy_request(client_ip="10.0.0.1", xff=None)
    assert get_client_ip(req_no_xff, trust_proxy_headers=True, trusted_proxy_hops=1) == "10.0.0.1"

    req_short = make_dummy_request(client_ip="10.0.0.1", xff="203.0.113.1")
    # Configured hops=3 but only 1 entry present -> fallback to socket IP
    assert get_client_ip(req_short, trust_proxy_headers=True, trusted_proxy_hops=3) == "10.0.0.1"


def test_invalid_trusted_proxy_hops_configuration_rejected():
    """Invalid HUSH_TRUSTED_PROXY_HOPS (< 1 or non-numeric) must be rejected even when proxy trust is disabled."""
    # Verified: HUSH_TRUST_PROXY_HEADERS=False
    with pytest.raises(ValueError, match="HUSH_TRUSTED_PROXY_HOPS"):
        Settings(HUSH_ENV="development", HUSH_TRUST_PROXY_HEADERS=False, HUSH_TRUSTED_PROXY_HOPS=0)

    with pytest.raises(ValueError, match="HUSH_TRUSTED_PROXY_HOPS"):
        Settings(HUSH_ENV="development", HUSH_TRUST_PROXY_HEADERS=False, HUSH_TRUSTED_PROXY_HOPS=-1)

    with pytest.raises(ValueError, match="HUSH_TRUSTED_PROXY_HOPS"):
        Settings(HUSH_ENV="development", HUSH_TRUST_PROXY_HEADERS=False, HUSH_TRUSTED_PROXY_HOPS="invalid")

    # Also verified when HUSH_TRUST_PROXY_HEADERS=True
    with pytest.raises(ValueError, match="HUSH_TRUSTED_PROXY_HOPS"):
        Settings(HUSH_ENV="development", HUSH_TRUST_PROXY_HEADERS=True, HUSH_TRUSTED_PROXY_HOPS=0)


# ============================================================================
# 6. CORS Configuration
# ============================================================================

def test_cors_production_fails_when_origins_unset_or_empty():
    """In production mode, startup must fail if CORS_ORIGINS is unset or empty."""
    with pytest.raises(ValueError, match="CORS_ORIGINS must be set"):
        Settings(HUSH_ENV="production", CORS_ORIGINS="")

    with pytest.raises(ValueError, match="CORS_ORIGINS must be set"):
        Settings(HUSH_ENV="production", CORS_ORIGINS=[])


def test_cors_production_rejects_wildcard_origin():
    """In production mode, wildcard '*' in CORS_ORIGINS must be rejected with ValueError."""
    with pytest.raises(ValueError, match="Wildcard"):
        Settings(HUSH_ENV="production", CORS_ORIGINS="*")

    with pytest.raises(ValueError, match="Wildcard"):
        Settings(HUSH_ENV="production", CORS_ORIGINS=["*"])


def test_cors_development_mode_permits_default_configuration():
    """In development mode, CORS_ORIGINS defaults to local development origins."""
    s = Settings(HUSH_ENV="development", CORS_ORIGINS="")
    assert "http://localhost:5173" in s.CORS_ORIGINS
    assert "http://127.0.0.1:5173" in s.CORS_ORIGINS


def test_cors_disallowed_origin_rejected_by_policy():
    """Requests from disallowed origins must be rejected by CORS policy."""
    resp = client.options(
        "/api/chat/message",
        headers={
            "Origin": "https://unauthorized-attacker.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers


# ============================================================================
# 7. Health Endpoint Allowlist & Information Hiding
# ============================================================================

def test_health_endpoint_keys_allowlist():
    """GET /health must expose exactly the allowlisted non-sensitive keys."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()

    # Exactly three allowlisted keys: status, generation_mode, decider_mode
    expected_allowlist = {"status", "generation_mode", "decider_mode"}
    assert set(data.keys()) == expected_allowlist

    # Must NOT expose sensitive configuration or environment details
    sensitive_keys = {
        "environment",
        "api_key",
        "openai_api_key",
        "secret",
        "path",
        "internal",
        "credentials",
        "host",
    }
    for key in sensitive_keys:
        assert key not in data


def test_health_endpoint_reports_active_generation_mode(monkeypatch):
    """GET /health reflects the configured generation mode."""
    monkeypatch.setattr(settings, "HUSH_GENERATION_MODE", "stub")
    resp_stub = client.get("/health")
    assert resp_stub.json()["generation_mode"] == "stub"

    monkeypatch.setattr(settings, "HUSH_GENERATION_MODE", "openai")
    resp_openai = client.get("/health")
    assert resp_openai.json()["generation_mode"] == "openai"


# ============================================================================
# 8. Stub Mode Honesty and No Implicit Stub Fallback
# ============================================================================

def test_stub_generator_prefixes_with_simulated_demo_marker():
    """DevelopmentStubGenerator must prefix replies and questions with [SIMULATED DEMO]."""
    stub = DevelopmentStubGenerator()
    context = [Message(speaker="u1", text="Need help with database migration")]

    reply = stub.generate_reply(context)
    assert reply.startswith("[SIMULATED DEMO]")
    assert "Need help with database migration" in reply

    clarification = stub.generate_clarification(context)
    assert clarification.startswith("[SIMULATED DEMO]")
    assert "Need help with database migration" in clarification


def test_openai_mode_missing_key_returns_clear_error_never_stub(monkeypatch):
    """In OpenAI mode, a missing key returns HTTP 500 error, never a stub reply."""
    monkeypatch.setattr(settings, "HUSH_DECIDER_MODE", "stub")
    monkeypatch.setattr(settings, "HUSH_GENERATION_MODE", "openai")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)

    payload = {
        "conversation_id": "room-openai-err",
        "user_id": "alice",
        "message": "Hush please help with deployment",
    }
    resp = client.post("/api/chat/message", json=payload)
    assert resp.status_code == 500
    assert "OpenAI API key is not configured" in resp.json()["detail"]


def test_silent_action_never_invokes_generator_in_stub_and_openai_modes(monkeypatch):
    """SILENT decisions strictly bypass generation in both stub and openai modes."""
    monkeypatch.setattr(settings, "HUSH_DECIDER_MODE", "stub")

    # 1. Stub mode
    monkeypatch.setattr(settings, "HUSH_GENERATION_MODE", "stub")
    payload = {
        "conversation_id": "room-silent-check",
        "user_id": "alice",
        "message": "Just chatting with bob about coffee.",
    }
    resp_stub = client.post("/api/chat/message", json=payload)
    assert resp_stub.status_code == 200
    assert resp_stub.json()["action"] == "SILENT"
    assert resp_stub.json()["reply"] is None
    assert resp_stub.json()["generation_used"] is False

    # 2. OpenAI mode (even with missing key, SILENT succeeds without error because generator is never called)
    monkeypatch.setattr(settings, "HUSH_GENERATION_MODE", "openai")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    resp_openai = client.post("/api/chat/message", json=payload)
    assert resp_openai.status_code == 200
    assert resp_openai.json()["action"] == "SILENT"
    assert resp_openai.json()["reply"] is None
    assert resp_openai.json()["generation_used"] is False
