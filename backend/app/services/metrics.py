"""In-memory metrics tracking service for Hush.

Maintains process-level telemetry counters for messages evaluated,
action distributions, generation attempts, failures, calls avoided,
and measurable latencies.
"""

import threading
from typing import Optional
from app.config import settings
from app.schemas import (
    Action,
    ActionCounts,
    AddressingResult,
    AddressMethodCounts,
    ConfidenceKind,
    ConfidenceKindCounts,
    LatencySummary,
    MetricsResponse,
    ResetMetricsResponse,
)


class MetricsTracker:
    """Thread-safe in-memory metrics tracker with clear reset boundary."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Reset all counters and latency buffers to initial zero state."""
        with getattr(self, "_lock", threading.Lock()):
            self._messages_evaluated: int = 0
            self._decisions: dict[Action, int] = {
                Action.SPEAK: 0,
                Action.SILENT: 0,
                Action.ASK: 0,
            }
            self._fallback_decisions: int = 0
            self._low_confidence_decisions: int = 0
            self._generation_calls_attempted: int = 0
            self._generation_calls_succeeded: int = 0
            self._generation_calls_failed: int = 0
            self._decision_latencies: list[float] = []
            self._generation_latencies: list[float] = []
            self._address_methods: dict[str, int] = {
                "mention": 0,
                "reply": 0,
                "name": 0,
                "request": 0,
                "none": 0,
            }
            self._addressed_messages: int = 0
            self._unaddressed_messages: int = 0
            self._confidence_kinds: dict[str, int] = {
                "calibrated_probability": 0,
                "uncalibrated_score": 0,
                "none": 0,
            }

    def record_decision(
        self,
        action: Action,
        confidence: Optional[float] = None,
        confidence_kind: ConfidenceKind = "none",
        fallback_used: bool = False,
        decision_latency_ms: Optional[float] = None,
        low_confidence_threshold: float = 0.60,
        addressing: Optional[AddressingResult] = None,
    ) -> None:
        """Record decision evaluation.

        Counting rules:
        - Exactly one action counter (SPEAK, SILENT, or ASK) is incremented.
        - messages_evaluated is incremented by exactly 1.
        - fallback_decisions is incremented if fallback_used is True.
        - low_confidence_decisions is incremented if confidence is measurable and < threshold.
        - decision_latencies records measurable latency in milliseconds.
        - confidence_kinds records distribution across calibrated_probability, uncalibrated_score, none.
        - address_methods records distribution across mention, reply, name, request, none.
        - addressed_messages / unaddressed_messages partition message addressing.
        """
        with self._lock:
            self._messages_evaluated += 1
            self._decisions[action] += 1

            if fallback_used:
                self._fallback_decisions += 1

            if confidence is not None and confidence < low_confidence_threshold:
                self._low_confidence_decisions += 1

            if decision_latency_ms is not None and decision_latency_ms >= 0:
                self._decision_latencies.append(decision_latency_ms)

            # Record confidence kind distribution
            if confidence_kind in self._confidence_kinds:
                self._confidence_kinds[confidence_kind] += 1
            else:
                self._confidence_kinds["none"] += 1

            # Record addressing distribution
            if addressing is not None:
                if addressing.addressed_to_hush:
                    self._addressed_messages += 1
                else:
                    self._unaddressed_messages += 1

                if addressing.address_method in self._address_methods:
                    self._address_methods[addressing.address_method] += 1
                else:
                    self._address_methods["none"] += 1
            else:
                self._unaddressed_messages += 1
                self._address_methods["none"] += 1

    def record_generation_attempt(self) -> None:
        """Record an attempt to call text generation (for SPEAK or ASK)."""
        with self._lock:
            self._generation_calls_attempted += 1

    def record_generation_success(self, latency_ms: Optional[float] = None) -> None:
        """Record a successful text generation invocation."""
        with self._lock:
            self._generation_calls_succeeded += 1
            if latency_ms is not None and latency_ms >= 0:
                self._generation_latencies.append(latency_ms)

    def record_generation_failure(self) -> None:
        """Record a failed text generation invocation."""
        with self._lock:
            self._generation_calls_failed += 1

    def _summarize_latencies(self, samples: list[float]) -> LatencySummary:
        """Compute statistical summary for measurable latencies."""
        if not samples:
            return LatencySummary(count=0, avg_ms=None, min_ms=None, max_ms=None)

        count = len(samples)
        avg_ms = round(sum(samples) / count, 2)
        min_ms = round(min(samples), 2)
        max_ms = round(max(samples), 2)

        return LatencySummary(
            count=count,
            avg_ms=avg_ms,
            min_ms=min_ms,
            max_ms=max_ms,
        )

    def get_metrics(self) -> MetricsResponse:
        """Compute aggregated metrics snapshot."""
        with self._lock:
            calls_avoided = self._decisions[Action.SILENT]
            reduction_rate = (
                round((calls_avoided / self._messages_evaluated) * 100, 2)
                if self._messages_evaluated > 0
                else 0.0
            )

            return MetricsResponse(
                messages_evaluated=self._messages_evaluated,
                decisions=ActionCounts(
                    SPEAK=self._decisions[Action.SPEAK],
                    SILENT=self._decisions[Action.SILENT],
                    ASK=self._decisions[Action.ASK],
                ),
                fallback_decisions=self._fallback_decisions,
                low_confidence_decisions=self._low_confidence_decisions,
                generation_calls_attempted=self._generation_calls_attempted,
                generation_calls_succeeded=self._generation_calls_succeeded,
                generation_calls_failed=self._generation_calls_failed,
                generation_calls_avoided=calls_avoided,
                call_reduction_rate_pct=reduction_rate,
                address_methods=AddressMethodCounts(
                    mention=self._address_methods["mention"],
                    reply=self._address_methods["reply"],
                    name=self._address_methods["name"],
                    request=self._address_methods["request"],
                    none=self._address_methods["none"],
                ),
                addressed_messages=self._addressed_messages,
                unaddressed_messages=self._unaddressed_messages,
                confidence_kinds=ConfidenceKindCounts(
                    calibrated_probability=self._confidence_kinds["calibrated_probability"],
                    uncalibrated_score=self._confidence_kinds["uncalibrated_score"],
                    none=self._confidence_kinds["none"],
                ),
                decision_latency=self._summarize_latencies(self._decision_latencies),
                generation_latency=self._summarize_latencies(self._generation_latencies),
                decider_mode=settings.HUSH_DECIDER_MODE,
                generation_mode=settings.HUSH_GENERATION_MODE,
            )


metrics_tracker = MetricsTracker()
