"""Decision engine adapter for Hush.

Keeps decision integration behind a strict interface contract.
Provides:
- TrainedRestraintClassifierDecider: Production decider loading and evaluating
  the trained 22-feature restraint classifier (MultinomialLogisticRegression).
- DevelopmentStubDecider: Explicitly labelled development stub for offline/local testing.
- RealHushGateDecider: Explicit failure handler when teammate hush_gate is requested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Protocol, Union

from app.config import settings
from app.schemas import Action, AddressingResult, DecisionResult, Message
from app.services.addressing import resolve_addressing
from app.services.baselines import InferenceContext
from app.services.classifier import MultinomialLogisticRegression

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "restraint_classifier_v1.json"
)


def resolve_model_path(model_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve model path independent of the current working directory."""
    if model_path:
        p = Path(model_path)
        if not p.is_absolute():
            repo_root = Path(__file__).resolve().parents[3]
            p = repo_root / p
        return p.resolve()
    if settings.HUSH_MODEL_PATH:
        p = Path(settings.HUSH_MODEL_PATH)
        if not p.is_absolute():
            repo_root = Path(__file__).resolve().parents[3]
            p = repo_root / p
        return p.resolve()
    return DEFAULT_MODEL_PATH.resolve()


class DeciderAdapter(Protocol):
    """Protocol defining the decision adapter interface."""

    def decide(
        self,
        messages: list[Message],
        addressing: Optional[AddressingResult] = None,
    ) -> DecisionResult:
        ...


class TrainedRestraintClassifierDecider:
    """Production decider powered by the trained restraint classifier.

    Loads the serialized multinomial logistic regression model artifact
    independently of working directory, extracts the 22 conversational context
    features, enforces addressing invariants, and returns structured DecisionResult
    with honest uncalibrated confidence scores and decision latency.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        self.model_path = resolve_model_path(model_path)
        self.model = self._load_and_validate_model(self.model_path)

    @staticmethod
    def _load_and_validate_model(path: Path) -> MultinomialLogisticRegression:
        """Load and strictly validate the model artifact.

        Raises FileNotFoundError if the file does not exist.
        Raises ValueError if the artifact is corrupted, invalid JSON, or not fitted.
        Never silently falls back to DevelopmentStubDecider.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Trained restraint classifier artifact not found at: {path}. "
                "Ensure data/models/restraint_classifier_v1.json exists or configure HUSH_MODEL_PATH."
            )
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise ValueError(f"Corrupted or invalid JSON in model artifact at {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Invalid model artifact at {path}: expected JSON object, got {type(data).__name__}")

        required_keys = {"weights", "biases", "is_fitted", "class_labels", "feature_names"}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise ValueError(f"Invalid model artifact at {path}: missing required keys {sorted(missing_keys)}")

        if not data.get("is_fitted", False):
            raise ValueError(f"Invalid model artifact at {path}: model is not fitted (is_fitted=False)")

        return MultinomialLogisticRegression.load(path)

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
                method="ml_restraint_classifier",
                fallback_used=False,
                decision_latency_ms=0.0,
            )

        # Resolve addressing if not explicitly passed
        if addressing is None:
            latest = messages[-1]
            addressing = resolve_addressing(
                message_text=latest.text,
                mentions=latest.mentions,
                reply_to=latest.reply_to,
                conversation_history=messages[:-1],
            )

        # Convert Message list into InferenceContext for features extraction
        msg_dicts: list[dict[str, Any]] = []
        unique_speakers: set[str] = set()
        for idx, m in enumerate(messages):
            spk = m.speaker_id or m.speaker
            unique_speakers.add(spk)
            role = m.role or ("assistant" if spk.lower() in ("hush", "assistant") else "human")
            msg_dicts.append({
                "message_id": m.message_id or f"m{idx + 1}",
                "speaker": m.speaker,
                "speaker_id": spk,
                "role": role,
                "text": m.text,
                "mentions": m.mentions,
                "reply_to": m.reply_to,
                "ts": m.ts,
            })

        participants: list[dict[str, Any]] = [
            {"speaker_id": s, "role": "assistant" if s.lower() in ("hush", "assistant") else "human"}
            for s in unique_speakers
        ]
        if not any(p["speaker_id"].lower() == "hush" for p in participants):
            participants.append({"speaker_id": "hush", "role": "assistant"})

        scenario_id = messages[-1].message_id or "turn-latest"
        ctx = InferenceContext(
            scenario_id=scenario_id,
            conversation_id="conv-live",
            messages=msg_dicts,
            participants=participants,
            addressed_to_hush=addressing.addressed_to_hush,
            address_method=addressing.address_method,
        )

        telemetry = self.model.decide_with_telemetry(ctx)

        # Addressing Invariant Safeguard:
        # When addressed to Hush, NEVER choose SILENT.
        chosen_action_str = telemetry["action"]
        if addressing.addressed_to_hush and chosen_action_str == Action.SILENT.value:
            probs = telemetry.get("probabilities", {})
            speak_p = probs.get(Action.SPEAK.value, 0.0)
            ask_p = probs.get(Action.ASK.value, 0.0)
            chosen_action_str = Action.SPEAK.value if speak_p >= ask_p else Action.ASK.value

        chosen_action = Action(chosen_action_str)
        confidence = telemetry.get("confidence")
        confidence_kind = telemetry.get("confidence_kind", "uncalibrated_score")
        latency_ms = telemetry.get("latency_ms")

        return DecisionResult(
            action=chosen_action,
            confidence=confidence,
            confidence_kind=confidence_kind,
            method="ml_restraint_classifier",
            fallback_used=False,
            decision_latency_ms=latency_ms,
        )


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

    def decide(
        self,
        messages: list[Message],
        addressing: Optional[AddressingResult] = None,
    ) -> DecisionResult:
        try:
            import hush_gate  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Real decision mode requested (HUSH_DECIDER_MODE='real'), "
                "but the teammate-owned 'hush_gate' package is not installed or available on PYTHONPATH. "
                "Failing explicitly to prevent stub data from being mistaken for real Decider 2B inference."
            ) from exc

        if hasattr(hush_gate, "decide_action"):
            raw_input = [{"speaker": m.speaker, "text": m.text} for m in messages]
            raw_result = hush_gate.decide_action(raw_input)
            return DecisionResult(**raw_result)

        raise NotImplementedError(
            "The 'hush_gate' module is present, but the integration function 'decide_action' "
            "has not been confirmed or implemented by the Decision Engine & Evaluation Developer."
        )


_cached_classifier_decider: Optional[TrainedRestraintClassifierDecider] = None


def get_production_decider(
    model_path: Optional[Union[str, Path]] = None,
) -> TrainedRestraintClassifierDecider:
    """Get or instantiate singleton production decider."""
    global _cached_classifier_decider
    if _cached_classifier_decider is None or model_path is not None:
        decider = TrainedRestraintClassifierDecider(model_path=model_path)
        if model_path is None:
            _cached_classifier_decider = decider
        return decider
    return _cached_classifier_decider


def clear_decider_cache() -> None:
    """Clear cached decider instance (for testing)."""
    global _cached_classifier_decider
    _cached_classifier_decider = None


def get_decider(mode: str | None = None) -> DeciderAdapter:
    """Factory returning active decider adapter based on configuration.

    Defaults to the trained restraint classifier in production (TrainedRestraintClassifierDecider).
    DevelopmentStubDecider is retained only for explicit development/testing (HUSH_DECIDER_MODE='stub').
    RealHushGateDecider is retained for testing teammate integration failure handling (HUSH_DECIDER_MODE='real').
    """
    active_mode = (mode or settings.HUSH_DECIDER_MODE).lower()
    if active_mode in ("stub", "development"):
        return DevelopmentStubDecider()
    if active_mode == "real":
        return RealHushGateDecider()
    return get_production_decider()
