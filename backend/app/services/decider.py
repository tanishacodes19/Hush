"""Decision engine adapter for Hush.

Keeps decision integration behind a strict interface contract.
Supports an explicitly labelled development stub mode and fails clearly
when real mode is configured but hush_gate is unavailable.
"""

from typing import Optional, Protocol
from app.config import settings
from app.schemas import Action, AddressingResult, DecisionResult, Message
from app.services.addressing import resolve_addressing


class DeciderAdapter(Protocol):
    """Protocol defining the decision adapter interface."""
    def decide(self, messages: list[Message]) -> DecisionResult:
        ...


class DevelopmentStubDecider:
    """Explicitly labelled development stub for offline/local testing.

    Does not fabricate model confidence or latency scores.
    Confidence is set to None and decision_latency_ms is None to avoid
    presenting stub heuristics as real model measurements.
    """

    def decide(
        self,
        messages: list[Message],
        addressing: Optional[AddressingResult] = None,
    ) -> DecisionResult:
        if not messages:
            return DecisionResult(
                action=Action.SILENT,
                confidence=None,
                confidence_kind="none",
                method="stub",
                fallback_used=False,
                decision_latency_ms=None,
            )

        # Resolve addressing if not explicitly provided
        if addressing is None:
            latest = messages[-1]
            addressing = resolve_addressing(
                message_text=latest.text,
                mentions=latest.mentions,
                reply_to=latest.reply_to,
                conversation_history=messages[:-1],
            )

        latest_text = messages[-1].text.strip().lower()

        # 1. Message is directly addressed to Hush (Tier 1 or Tier 2) -> SPEAK
        if addressing.addressed_to_hush:
            return DecisionResult(
                action=Action.SPEAK,
                confidence=None,
                confidence_kind="none",
                method="rules",
                fallback_used=False,
                decision_latency_ms=None,
            )

        # 2. Metadata directs message elsewhere (Tier 1) or addressed to another person -> SILENT
        if (addressing.tier == 1 and not addressing.addressed_to_hush) or (
            "addressed to another" in addressing.reason.lower()
        ):
            return DecisionResult(
                action=Action.SILENT,
                confidence=None,
                confidence_kind="none",
                method="rules",
                fallback_used=False,
                decision_latency_ms=None,
            )

        # 3. Unaddressed group question or error needing clarification -> ASK
        if "?" in latest_text or "help" in latest_text or "error" in latest_text:
            return DecisionResult(
                action=Action.ASK,
                confidence=None,
                confidence_kind="none",
                method="rules",
                fallback_used=False,
                decision_latency_ms=None,
            )

        # 4. Default to silence -> SILENT
        return DecisionResult(
            action=Action.SILENT,
            confidence=None,
            confidence_kind="none",
            method="rules",
            fallback_used=False,
            decision_latency_ms=None,
        )


class RealHushGateDecider:
    """Real decision engine adapter targeting teammate-owned hush_gate.

    Fails explicitly if hush_gate is not yet available or if its interface
    has not been finalized, rather than silently falling back to a stub.
    """

    def decide(self, messages: list[Message]) -> DecisionResult:
        try:
            import hush_gate  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Real decision mode requested (HUSH_DECIDER_MODE='real'), "
                "but the teammate-owned 'hush_gate' package is not installed or available on PYTHONPATH. "
                "Failing explicitly to prevent stub data from being mistaken for real Decider 2B inference."
            ) from exc

        # Once teammate confirms contract function (e.g., hush_gate.decide_action):
        if hasattr(hush_gate, "decide_action"):
            raw_input = [{"speaker": m.speaker, "text": m.text} for m in messages]
            raw_result = hush_gate.decide_action(raw_input)
            return DecisionResult(**raw_result)

        raise NotImplementedError(
            "The 'hush_gate' module is present, but the integration function 'decide_action' "
            "has not been confirmed or implemented by the Decision Engine & Evaluation Developer."
        )


def get_decider(mode: str | None = None) -> DeciderAdapter:
    """Factory returning the active decider adapter based on configuration."""
    active_mode = (mode or settings.HUSH_DECIDER_MODE).lower()
    if active_mode == "real":
        return RealHushGateDecider()
    return DevelopmentStubDecider()
