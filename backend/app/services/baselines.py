"""Baseline deciders for Hush Conversational Restraint Engine.

Implements baseline decision strategies for reproducible evaluation (docs/dataset_spec.md Section 8):
1. MajorityClassBaseline: Predicts the majority action fit on the development split (deterministic tie-breaking).
2. AlwaysReplyBaseline: Always outputs Action.SPEAK.
3. RuleBasedBaseline: Deterministic rule engine using inference-time features only; strictly respects addressing invariant.
4. LLM-as-Decider: Provider-neutral protocol, structured JSON parser with validation, and MockLLMDecider for offline tests.

Strict feature isolation: None of these baseline predictors access ground-truth labels
or annotation metadata (expected_action, rationale, category, source, annotator_confidence, review_status).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Union

from app.schemas import Action


@dataclass(frozen=True)
class InferenceContext:
    """Isolated inference-time context. Stripped of all annotation/gold metadata."""

    scenario_id: str
    conversation_id: str
    messages: list[dict[str, Any]]
    participants: list[dict[str, Any]]
    addressed_to_hush: bool
    address_method: str

    @property
    def latest_message(self) -> Optional[dict[str, Any]]:
        """Return the most recent message in the conversation thread."""
        if not self.messages:
            return None
        return self.messages[-1]

    @property
    def latest_text(self) -> str:
        """Return the text of the most recent message, or empty string."""
        msg = self.latest_message
        if not msg:
            return ""
        return str(msg.get("text", ""))


def extract_inference_context(record: dict[str, Any]) -> InferenceContext:
    """Extract strictly inference-time features from a scenario record.

    Ensures complete feature isolation: explicitly excludes expected_action, rationale,
    category, source, annotator_confidence, review_status, and split.
    """
    scenario_id = record.get("id") or record.get("scenario_id") or ""
    conversation_id = record.get("conversation_id", "")

    # Clean messages to inference-only keys
    allowed_msg_keys = {"message_id", "speaker_id", "role", "text", "mentions", "reply_to", "ts"}
    clean_messages: list[dict[str, Any]] = []
    for m in record.get("messages", []):
        clean_messages.append({k: v for k, v in m.items() if k in allowed_msg_keys})

    # Clean participants
    allowed_part_keys = {"speaker_id", "participant_id", "display_name", "role"}
    clean_participants: list[dict[str, Any]] = []
    for p in record.get("participants", []):
        clean_participants.append({k: v for k, v in p.items() if k in allowed_part_keys})

    addressed_to_hush = bool(record.get("addressed_to_hush", False))
    address_method = str(record.get("address_method", "none"))

    return InferenceContext(
        scenario_id=str(scenario_id),
        conversation_id=str(conversation_id),
        messages=clean_messages,
        participants=clean_participants,
        addressed_to_hush=addressed_to_hush,
        address_method=address_method,
    )


class BaselinePredictor(Protocol):
    """Protocol for baseline decision strategies."""

    def predict(self, context: InferenceContext) -> str:
        """Predict an action for the given inference context."""
        ...


class MajorityClassBaseline:
    """Predicts the most frequent class observed during fitting on dev split.

    Deterministic tie-breaking order: SILENT > SPEAK > ASK.
    """

    TIE_BREAKING_ORDER = [Action.SILENT.value, Action.SPEAK.value, Action.ASK.value]

    def __init__(self, majority_class: str = Action.SILENT.value):
        if majority_class not in {a.value for a in Action}:
            raise ValueError(f"Invalid action: {majority_class}")
        self.majority_class = majority_class
        self.is_fitted = False

    def fit(self, labels: list[str]) -> "MajorityClassBaseline":
        """Fit on a list of ground truth labels (e.g. from dev set only)."""
        if not labels:
            raise ValueError("Cannot fit on empty labels list")
        counts = Counter(labels)
        max_count = max(counts.values())
        candidates = [lbl for lbl, cnt in counts.items() if cnt == max_count]

        # Deterministic tie-breaking
        for preferred in self.TIE_BREAKING_ORDER:
            if preferred in candidates:
                self.majority_class = preferred
                break
        else:
            self.majority_class = sorted(candidates)[0]

        self.is_fitted = True
        return self

    @classmethod
    def fit_from_file(cls, dev_file_path: Union[str, Path]) -> "MajorityClassBaseline":
        """Load dev records and fit the majority class on dev labels only."""
        labels: list[str] = []
        path = Path(dev_file_path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    action = record.get("expected_action")
                    if action:
                        labels.append(action)
        return cls().fit(labels)

    def predict(self, context: InferenceContext) -> str:
        return self.majority_class


class AlwaysReplyBaseline:
    """Always outputs SPEAK to demonstrate the unrestrained bot baseline."""

    def __init__(self, action: str = Action.SPEAK.value):
        if action not in {a.value for a in Action}:
            raise ValueError(f"Invalid action: {action}")
        self.action = action

    def predict(self, context: InferenceContext) -> str:
        return self.action


class RuleBasedBaseline:
    """Deterministic heuristic baseline using only inference-time features.

    Rules:
    1. Addressing Invariant: When addressed_to_hush is True, NEVER predict SILENT.
       - If addressed_to_hush is True:
         - If text matches ambiguity signals (vague troubleshooting, missing error logs/details) -> ASK.
         - Otherwise -> SPEAK.
    2. If addressed_to_hush is False:
       - If targeted to a human colleague (mentions, reply_to, or directed text) -> SILENT.
       - If text contains '?' (unaddressed question in group chat) -> SPEAK (models naive chatbot jumping in).
       - Otherwise -> SILENT.
    """

    AMBIGUITY_PATTERNS = [
        re.compile(r"\b(it\s+doesn'?t\s+work|not\s+working|broken|fails?|failing)\b", re.IGNORECASE),
        re.compile(r"\b(what\s+should\s+i\s+do|can\s+you\s+help|need\s+help)\b", re.IGNORECASE),
        re.compile(r"\b(clarif(y|ication)|specify|more\s+details?)\b", re.IGNORECASE),
        re.compile(r"\b(fix\s+this|what\s+went\s+wrong)\b", re.IGNORECASE),
    ]

    def __init__(self, allow_ask: bool = True):
        self.allow_ask = allow_ask

    def predict(self, context: InferenceContext) -> str:
        latest = context.latest_message
        text = context.latest_text.strip()

        if context.addressed_to_hush:
            # Addressing invariant: NEVER predict SILENT when addressed
            if self.allow_ask:
                words = text.split()
                has_ambiguity_phrase = any(p.search(text) for p in self.AMBIGUITY_PATTERNS)
                # Short or vague query lacking details
                is_short_vague_query = len(words) <= 7 and has_ambiguity_phrase
                is_missing_details = "it's not working" in text.lower() or "doesn't work" in text.lower() or "fix this" in text.lower()

                if is_short_vague_query or is_missing_details:
                    return Action.ASK.value

            return Action.SPEAK.value

        # Unaddressed
        mentions = latest.get("mentions", []) if latest else []
        reply_to = latest.get("reply_to") if latest else None

        # If explicitly targeted to another human, always SILENT
        has_human_mention = any(m.lower() not in {"hush", "assistant"} for m in mentions)
        if has_human_mention or reply_to:
            return Action.SILENT.value

        # Naive chatbot behavior: responds whenever someone asks a question in channel
        if "?" in text:
            return Action.SPEAK.value

        return Action.SILENT.value


@dataclass
class LLMDecisionResult:
    """Structured decision output from an LLM decider."""

    action: str
    rationale: str
    raw_response: str
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class LLMOutputParser:
    """Strict parser and validator for LLM decision JSON outputs."""

    @staticmethod
    def parse(raw_output: str) -> LLMDecisionResult:
        """Parse and strictly validate raw LLM output text."""
        cleaned = raw_output.strip()

        # Handle markdown fenced code blocks: ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        else:
            # Fallback: search for outermost JSON object { ... }
            brace_match = re.search(r"\{[\s\S]*\}", cleaned)
            if brace_match:
                cleaned = brace_match.group(0).strip()

        try:
            data = json.loads(cleaned)
        except Exception as e:
            return LLMDecisionResult(
                action=Action.SILENT.value,
                rationale="Fallback SILENT due to JSON parse failure",
                raw_response=raw_output,
                success=False,
                error=f"JSON parse error: {e}",
            )

        if not isinstance(data, dict):
            return LLMDecisionResult(
                action=Action.SILENT.value,
                rationale="Fallback SILENT due to invalid JSON root type",
                raw_response=raw_output,
                success=False,
                error="Root JSON element must be an object",
            )

        action = str(data.get("action", "")).upper().strip()
        rationale = str(data.get("rationale", "")).strip()

        if action not in {a.value for a in Action}:
            return LLMDecisionResult(
                action=Action.SILENT.value,
                rationale="Fallback SILENT due to invalid action value",
                raw_response=raw_output,
                success=False,
                error=f"Invalid action '{action}'. Must be one of SPEAK, ASK, SILENT",
            )

        if not rationale:
            return LLMDecisionResult(
                action=action,
                rationale="No rationale provided",
                raw_response=raw_output,
                success=False,
                error="Field 'rationale' must be a non-empty string",
            )

        return LLMDecisionResult(
            action=action,
            rationale=rationale,
            raw_response=raw_output,
            success=True,
            error=None,
        )


class MockLLMDecider:
    """Mock LLM decider for offline testing and deterministic verification.

    Zero external API calls, zero credentials, zero network dependencies.
    """

    def __init__(
        self,
        preset_responses: Optional[dict[str, Union[str, dict[str, Any]]]] = None,
        default_action: str = Action.SILENT.value,
        simulated_latency_ms: float = 15.0,
    ):
        self.preset_responses = preset_responses or {}
        self.default_action = default_action
        self.simulated_latency_ms = simulated_latency_ms

    def decide(self, context: InferenceContext) -> LLMDecisionResult:
        scenario_id = context.scenario_id
        if scenario_id in self.preset_responses:
            preset = self.preset_responses[scenario_id]
            if isinstance(preset, str):
                res = LLMOutputParser.parse(preset)
                res.latency_ms = self.simulated_latency_ms
                return res
            elif isinstance(preset, dict):
                raw = json.dumps(preset)
                res = LLMOutputParser.parse(raw)
                res.latency_ms = self.simulated_latency_ms
                return res

        # Default fallback response
        raw = json.dumps({
            "action": self.default_action,
            "rationale": f"Mock default decision for {scenario_id}",
        })
        res = LLMOutputParser.parse(raw)
        res.latency_ms = self.simulated_latency_ms
        return res

    def predict(self, context: InferenceContext) -> str:
        return self.decide(context).action
