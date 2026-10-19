"""Pydantic v2 schemas for Hush.

All fields and structures are derived strictly from PLAN.md (Sections 1, 5, and 7).
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Action(str, Enum):
    """The three possible actions Hush can select per message (PLAN.md Section 1)."""
    SPEAK = "SPEAK"
    SILENT = "SILENT"
    ASK = "ASK"


class Message(BaseModel):
    """A message in conversation history (PLAN.md Section 4.1)."""
    model_config = ConfigDict(extra="forbid")

    speaker: str = Field(..., min_length=1, description="Identifier of the message author")
    text: str = Field(..., min_length=1, description="Text content of the message")


class ChatMessageRequest(BaseModel):
    """Incoming chat message request (PLAN.md Section 7)."""
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(..., min_length=1, description="Identifier for the conversation room")
    user_id: str = Field(..., min_length=1, description="Identifier for the sending user")
    message: str = Field(..., min_length=1, description="New message text to evaluate")


class DecisionResult(BaseModel):
    """Decision output structure returned by the decision gate (PLAN.md Section 5)."""
    model_config = ConfigDict(extra="forbid")

    action: Action = Field(..., description="Selected action: SPEAK, SILENT, or ASK")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Calibrated decision confidence score, or None if unavailable")
    fallback_used: bool = Field(default=False, description="Whether fallback policy was triggered")
    decision_latency_ms: Optional[float] = Field(default=None, ge=0.0, description="Decision latency in milliseconds")


class ChatMessageResponse(BaseModel):
    """Chat message response returned to the client (PLAN.md Section 7)."""
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(..., min_length=1, description="Identifier for the conversation room")
    action: Action = Field(..., description="Decision action: SPEAK, SILENT, or ASK")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Calibrated confidence from the decision model, or None if unavailable")
    reply: Optional[str] = Field(default=None, description="Generated reply text, or null if SILENT")
    generation_used: bool = Field(..., description="Whether a text-generation model was invoked")
    fallback_used: bool = Field(default=False, description="Whether a fallback policy was applied")
    decision_latency_ms: Optional[float] = Field(default=None, ge=0.0, description="Decision latency in milliseconds")

    @model_validator(mode="after")
    def validate_silent_contract(self) -> "ChatMessageResponse":
        """Enforce the contract from PLAN.md Section 7: For SILENT, reply is null and generation_used is false."""
        if self.action == Action.SILENT:
            if self.generation_used:
                raise ValueError("generation_used must be False when action is SILENT")
            if self.reply is not None:
                raise ValueError("reply must be None when action is SILENT")
        return self


class ActionCounts(BaseModel):
    """Mutually exclusive counts per action."""
    model_config = ConfigDict(extra="forbid")

    SPEAK: int = Field(default=0, ge=0, description="Total SPEAK decisions")
    SILENT: int = Field(default=0, ge=0, description="Total SILENT decisions")
    ASK: int = Field(default=0, ge=0, description="Total ASK decisions")


class LatencySummary(BaseModel):
    """Statistical summary for measurable latencies in milliseconds."""
    model_config = ConfigDict(extra="forbid")

    count: int = Field(default=0, ge=0, description="Number of measured latency samples")
    avg_ms: Optional[float] = Field(default=None, ge=0.0, description="Average latency in ms, or None if unmeasured")
    min_ms: Optional[float] = Field(default=None, ge=0.0, description="Minimum latency in ms, or None if unmeasured")
    max_ms: Optional[float] = Field(default=None, ge=0.0, description="Maximum latency in ms, or None if unmeasured")


class MetricsResponse(BaseModel):
    """Aggregated telemetry metrics response (PLAN.md Section 7)."""
    model_config = ConfigDict(extra="forbid")

    messages_evaluated: int = Field(..., ge=0, description="Total conversation messages evaluated")
    decisions: ActionCounts = Field(..., description="Breakdown of decisions by action")
    fallback_decisions: int = Field(..., ge=0, description="Decisions where fallback logic was triggered")
    low_confidence_decisions: int = Field(..., ge=0, description="Decisions with low calibrated confidence")
    generation_calls_attempted: int = Field(..., ge=0, description="Total text generation invocations attempted")
    generation_calls_succeeded: int = Field(..., ge=0, description="Total text generation invocations succeeded")
    generation_calls_failed: int = Field(..., ge=0, description="Total text generation invocations that failed")
    generation_calls_avoided: int = Field(..., ge=0, description="Calls avoided by SILENT restraint")
    call_reduction_rate_pct: float = Field(..., ge=0.0, le=100.0, description="Percentage of calls avoided vs always-reply")
    decision_latency: LatencySummary = Field(..., description="Decision model latency summary")
    generation_latency: LatencySummary = Field(..., description="Generation model latency summary")
    decider_mode: str = Field(..., description="Active decider operating mode (stub or real)")
    generation_mode: str = Field(..., description="Active generation operating mode (stub or bedrock)")


class ResetMetricsResponse(BaseModel):
    """Response returned upon resetting metrics counters."""
    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="ok", description="Status confirmation")
    message: str = Field(default="Metrics reset successfully", description="Informational message")