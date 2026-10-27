"""Pytest configuration and test isolation fixtures for Hush backend."""

import os
import socket
import pytest
from pydantic_settings.sources.providers.dotenv import DotEnvSettingsSource

from app.config import settings
from app.services.decider import clear_decider_cache

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1", "testserver"}


@pytest.fixture(autouse=True, scope="session")
def block_external_sockets():
    """Autouse session fixture strictly blocking non-local outbound socket connections."""
    orig_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = None
        if isinstance(address, tuple) and len(address) >= 1:
            host = address[0]
        elif isinstance(address, str):
            host = address

        if host and (host in ALLOWED_HOSTS or str(host).startswith("127.")):
            return orig_connect(self, address)

        raise RuntimeError(
            f"Offline test suite network isolation violation: "
            f"outbound network connection blocked to {address}."
        )

    socket.socket.connect = guarded_connect
    yield
    socket.socket.connect = orig_connect


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch):
    """Autouse fixture preventing reading real .env files or ambient provider secrets."""
    # Prevent pydantic-settings from reading real .env files during tests
    monkeypatch.setattr(DotEnvSettingsSource, "_read_env_files", lambda self: {})

    # Strip any ambient provider secrets from os.environ for test safety
    for key in (
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SESSION_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def manage_test_decider_mode(request, monkeypatch):
    """Ensure legacy Phase 1 test suites specifically written to test development stub

    heuristics continue to test stub mode, while production and integration tests
    exercise the production restraint classifier.
    """
    module_name = request.module.__name__ if request.module else ""
    stub_modules = {
        "test_api",
        "test_telemetry",
        "test_addressing",
        "test_legacy_compatibility",
        "test_conversation_aware",
        "test_metrics",
    }
    if any(m in module_name for m in stub_modules):
        # Explicit development/testing stub mode for Phase 1 vertical slice tests
        monkeypatch.setattr(settings, "HUSH_DECIDER_MODE", "stub")
        monkeypatch.setattr(settings, "HUSH_GENERATION_MODE", "stub")

    clear_decider_cache()
    yield
    clear_decider_cache()
