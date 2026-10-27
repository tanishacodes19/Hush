"""Pydantic v2 schemas for Hush.

All fields and structures are derived strictly from PLAN.md (Sections 1, 5, and 7).
"""

from enum import Enum
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Action(str, Enum):
    """The three possible actions Hush can select per message (PLAN.md Section 1)."""
    SPEAK = "SPEAK"
    SILENT = "SILENT"
    ASK = "ASK"


AddressMethod = Literal["mention", "reply", "name", "request", "none"]
ConfidenceKind = Literal["calibrated_probability", "uncalibrated_score", "none"]


class AddressingResult(BaseModel):
    """Structured addressing result produced by AddressingService (PLAN.md Section 1)."""
    model_config = ConfigDict(extra="forbid")

    addressed_to_hush: bool = Field(..., description="Whether the message is addressed to Hush")
    address_method: AddressMethod = Field(..., description="Addressing method: mention, reply, name, request, none")
    tier: Optional[int] = Field(default=None, description="Addressing tier (1 for metadata, 2 for text fallback)")
    reason: str = Field(..., description="Explanation of the addressing determination")


class Message(BaseModel):
    """A message in conversation history (PLAN.md Section 4.1)."""
    model_config = ConfigDict(extra="forbid")

    speaker: str = Field(..., min_length=1, description="Identifier of the message author")
    text: str = Field(..., min_length=1, description="Text content of the message")
    message_id: Optional[str] = Field(default=None, description="Unique message ID")
    speaker_id: Optional[str] = Field(default=None, description="Author participant ID")
    role: Optional[Literal["human", "assistant"]] = Field(default=None, description="Author role")
    mentions: list[str] = Field(default_factory=list, description="Mentioned participant IDs")
    reply_to: Optional[str] = Field(default=None, description="Target message ID being replied to")
    ts: Optional[float] = Field(default=None, description="Message timestamp")


class ChatMessageRequest(BaseModel):
    """Incoming chat message request (PLAN.md Section 7)."""
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(..., min_length=1, description="Identifier for the conversation room")
    user_id: str = Field(..., min_length=1, description="Identifier for the sending user")
    message: str = Field(..., min_length=1, description="New message text to evaluate")


class ConversationMessagePayload(BaseModel):
    """Message payload for conversation-aware requests (PLAN.md Section 9)."""
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(..., min_length=1, description="Unique identifier for the incoming message")
    speaker_id: str = Field(..., min_length=1, description="Identifier of the message author")
    text: str = Field(..., min_length=1, description="Text content of the message")
    display_name: Optional[str] = Field(default=None, min_length=1, description="Optional human-readable name of author")
    mentions: list[str] = Field(default_factory=list, description="List of participant IDs explicitly mentioned")
    reply_to: Optional[str] = Field(default=None, min_length=1, description="Optional message_id of the message being replied to")
    ts: Optional[float] = Field(default=None, description="Optional message timestamp")

    @model_validator(mode="before")
    @classmethod
    def handle_field_aliases(cls, data: Any) -> Any:
        """Allow 'message' as an alias for 'text' and 'timestamp' for 'ts'."""
        if isinstance(data, dict):
            data = dict(data)
            if "text" not in data and "message" in data:
                data["text"] = data.pop("message")
            if "ts" not in data and "timestamp" in data:
                data["ts"] = data.pop("timestamp")
        return data

    @field_validator("message_id", "speaker_id", "text", mode="after")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty or whitespace-only")
        return v

    @field_validator("display_name", "reply_to", mode="after")
    @classmethod
    def validate_optional_non_whitespace(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Optional string field must not be empty or whitespace-only when provided")
        return v

    @field_validator("mentions", mode="after")
    @classmethod
    def validate_mentions(cls, v: list[str]) -> list[str]:
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Mentions must contain non-empty string IDs")
        return v

    @field_validator("ts", mode="after")
    @classmethod
    def validate_timestamp(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or not (float("-inf") < v < float("inf"))):
            raise ValueError("Timestamp must be a finite, non-negative number")
        return v


class ConversationAwareChatRequest(BaseModel):
    """Conversation-aware chat request matching PLAN.md Section 9."""
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(..., min_length=1, description="Identifier for the conversation room")
    message: ConversationMessagePayload = Field(..., description="The message payload with speaker and addressing info")

    @field_validator("conversation_id", mode="after")
    @classmethod
    def validate_conversation_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("conversation_id must not be empty or whitespace-only")
        return v


class FlatConversationAwareChatRequest(BaseModel):
    """Flat conversation-aware chat request variant."""
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(..., min_length=1, description="Identifier for the conversation room")
    message_id: str = Field(..., min_length=1, description="Unique identifier for the incoming message")
    speaker_id: str = Field(..., min_length=1, description="Identifier of the message author")
    message: str = Field(..., min_length=1, description="Text content of the message")
    display_name: Optional[str] = Field(default=None, min_length=1, description="Optional human-readable name of author")
    mentions: list[str] = Field(default_factory=list, description="List of participant IDs explicitly mentioned")
    reply_to: Optional[str] = Field(default=None, min_length=1, description="Optional message_id of the message being replied to")
    timestamp: Optional[float] = Field(default=None, description="Optional message timestamp")
    ts: Optional[float] = Field(default=None, description="Optional message timestamp alias")

    @model_validator(mode="before")
    @classmethod
    def handle_field_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "message" not in data and "text" in data:
                data["message"] = data.pop("text")
        return data

    @field_validator("conversation_id", "message_id", "speaker_id", "message", mode="after")
    @classmethod
    def validate_flat_non_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty or whitespace-only")
        return v

    @field_validator("display_name", "reply_to", mode="after")
    @classmethod
    def validate_flat_optional_non_whitespace(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Optional string field must not be empty or whitespace-only when provided")
        return v

    @field_validator("mentions", mode="after")
    @classmethod
    def validate_flat_mentions(cls, v: list[str]) -> list[str]:
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Mentions must contain non-empty string IDs")
        return v

    @field_validator("timestamp", "ts", mode="after")
    @classmethod
    def validate_flat_timestamp(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or not (float("-inf") < v < float("inf"))):
            raise ValueError("Timestamp must be a finite, non-negative number")
        return v


ChatRequestUnion = Union[
    ConversationAwareChatRequest,
    FlatConversationAwareChatRequest,
    ChatMessageRequest,
]


class NormalizedChatRequest(BaseModel):
    """Internal normalized representation of a chat request."""
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    speaker_id: str
    text: str
    format: Literal["legacy", "conversation_aware"]
    message_id: Optional[str] = None
    display_name: Optional[str] = None
    mentions: list[str] = Field(default_factory=list)
    reply_to: Optional[str] = None
    ts: Optional[float] = None


class DecisionResult(BaseModel):
    """Decision output structure returned by the decision gate (PLAN.md Section 4 and Section 5)."""
    model_config = ConfigDict(extra="forbid")

    action: Action = Field(..., description="Selected action: SPEAK, SILENT, or ASK")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Calibrated confidence, uncalibrated score, or None")
    confidence_kind: ConfidenceKind = Field(default="none", description="Semantics of confidence: calibrated_probability, uncalibrated_score, or none")
    method: str = Field(default="rules", description="Decision method: rules, stub, embedding_clf, llm, etc.")
    fallback_used: bool = Field(default=False, description="Whether fallback policy was triggered")
    decision_latency_ms: Optional[float] = Field(default=None, ge=0.0, description="Decision latency in milliseconds")

    @model_validator(mode="before")
    @classmethod
    def set_default_confidence_kind(cls, data: Any) -> Any:
        """Infer confidence_kind if not explicitly provided."""
        if isinstance(data, dict):
            if "confidence_kind" not in data:
                if data.get("confidence") is not None:
                    data["confidence_kind"] = "uncalibrated_score"
                else:
                    data["confidence_kind"] = "none"
        return data

    @model_validator(mode="after")
    def validate_confidence_consistency(self) -> "DecisionResult":
        """Enforce semantic consistency between confidence and confidence_kind."""
        if self.confidence_kind == "none" and self.confidence is not None:
            raise ValueError("confidence must be None when confidence_kind is 'none'")
        if self.confidence_kind != "none" and self.confidence is None:
            raise ValueError(f"confidence must not be None when confidence_kind is '{self.confidence_kind}'")
        return self


class DecisionMetadata(BaseModel):
    """Decision execution details matching PLAN.md Section 4 and Section 9."""
    model_config = ConfigDict(extra="forbid")

    method: str = Field(..., description="Decision method: rules, stub, embedding_clf, llm, etc.")
    fallback_used: bool = Field(default=False, description="Whether fallback policy was applied")
    latency_ms: Optional[float] = Field(default=None, ge=0.0, description="Decision latency in milliseconds")


class ConversationAwareChatMessageResponse(BaseModel):
    """Conversation-aware chat response matching PLAN.md Section 9."""
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(..., min_length=1, description="Identifier for the conversation room")
    message_id: Optional[str] = Field(default=None, description="Unique identifier for the incoming message")
    addressed_to_hush: bool = Field(..., description="Whether the message was addressed to Hush")
    address_method: AddressMethod = Field(..., description="Addressing method: mention, reply, name, request, none")
    action: Action = Field(..., description="Decision action: SPEAK, SILENT, or ASK")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Calibrated confidence or uncalibrated score, or None")
    confidence_kind: ConfidenceKind = Field(default="none", description="Semantics of confidence: calibrated_probability, uncalibrated_score, or none")
    reply: Optional[str] = Field(default=None, description="Generated reply text, or null if SILENT")
    generation_used: bool = Field(..., description="Whether a text-generation model was invoked")
    decision: DecisionMetadata = Field(..., description="Decision execution details")

    @model_validator(mode="before")
    @classmethod
    def set_default_confidence_kind(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "confidence_kind" not in data:
                if data.get("confidence") is not None:
                    data["confidence_kind"] = "uncalibrated_score"
                else:
                    data["confidence_kind"] = "none"
        return data

    @model_validator(mode="after")
    def validate_silent_contract(self) -> "ConversationAwareChatMessageResponse":
        """Enforce the contract from PLAN.md Section 7: For SILENT, reply is null and generation_used is false."""
        if self.action == Action.SILENT:
            if self.generation_used:
                raise ValueError("generation_used must be False when action is SILENT")
            if self.reply is not None:
                raise ValueError("reply must be None when action is SILENT")
        return self

    @model_validator(mode="after")
    def validate_confidence_consistency(self) -> "ConversationAwareChatMessageResponse":
        if self.confidence_kind == "none" and self.confidence is not None:
            raise ValueError("confidence must be None when confidence_kind is 'none'")
        if self.confidence_kind != "none" and self.confidence is None:
            raise ValueError(f"confidence must not be None when confidence_kind is '{self.confidence_kind}'")
        return self


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


class AddressMethodCounts(BaseModel):
    """Mutually exclusive counts per addressing method (PLAN.md Section 1 and Section 9)."""
    model_config = ConfigDict(extra="forbid")

    mention: int = Field(default=0, ge=0, description="Explicit mention (@hush or metadata)")
    reply: int = Field(default=0, ge=0, description="Direct reply to Hush message")
    name: int = Field(default=0, ge=0, description="Name invocation in text")
    request: int = Field(default=0, ge=0, description="Direct request directed to Hush")
    none: int = Field(default=0, ge=0, description="No addressing signal detected")


class ConfidenceKindCounts(BaseModel):
    """Mutually exclusive counts per confidence kind (PLAN.md Section 7)."""
    model_config = ConfigDict(extra="forbid")

    calibrated_probability: int = Field(default=0, ge=0, description="Decisions with calibrated probability")
    uncalibrated_score: int = Field(default=0, ge=0, description="Decisions with uncalibrated score")
    none: int = Field(default=0, ge=0, description="Decisions without numeric confidence")


class MetricsResponse(BaseModel):
    """Aggregated telemetry metrics response (PLAN.md Section 7 and Section 9)."""
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
    address_methods: AddressMethodCounts = Field(
        default_factory=AddressMethodCounts,
        description="Breakdown of evaluated messages by addressing method",
    )
    addressed_messages: int = Field(default=0, ge=0, description="Total messages determined to be addressed to Hush")
    unaddressed_messages: int = Field(default=0, ge=0, description="Total messages determined NOT to be addressed to Hush")
    confidence_kinds: ConfidenceKindCounts = Field(
        default_factory=ConfidenceKindCounts,
        description="Breakdown of decisions by confidence kind",
    )
    decision_latency: LatencySummary = Field(..., description="Decision model latency summary")
    generation_latency: LatencySummary = Field(..., description="Generation model latency summary")
    decider_mode: str = Field(..., description="Active decider operating mode (stub or real)")
    generation_mode: str = Field(..., description="Active generation operating mode (stub or bedrock)")


class ResetMetricsResponse(BaseModel):
    """Response returned upon resetting metrics counters."""
    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="ok", description="Status confirmation")
    message: str = Field(default="Metrics reset successfully", description="Informational message")