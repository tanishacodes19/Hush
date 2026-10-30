"""Explicit request adapter for Hush chat endpoints.

Identifies the incoming request format (legacy vs conversation-aware) and
normalizes it into a consistent internal NormalizedChatRequest representation
while preserving all metadata and keeping the legacy API contract 100% intact.
"""

from typing import Any, Literal, Union
from pydantic import TypeAdapter, ValidationError
from app.schemas import (
    ChatMessageRequest,
    ConversationAwareChatRequest,
    FlatConversationAwareChatRequest,
    NormalizedChatRequest,
)

RequestFormat = Literal["legacy", "conversation_aware"]


def detect_request_format(
    payload: Union[
        dict[str, Any],
        ChatMessageRequest,
        ConversationAwareChatRequest,
        FlatConversationAwareChatRequest,
    ],
) -> RequestFormat:
    """Identify the request format explicitly without message-text heuristics.

    Returns:
        'legacy' or 'conversation_aware'.

    Raises:
        ValueError / ValidationError if payload is mixed, unexpected, or malformed.
    """
    if isinstance(payload, ChatMessageRequest):
        return "legacy"
    if isinstance(payload, (ConversationAwareChatRequest, FlatConversationAwareChatRequest)):
        return "conversation_aware"

    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dictionary or a recognized request model.")

    has_user_id = "user_id" in payload
    has_speaker_id = "speaker_id" in payload
    has_message_id = "message_id" in payload
    has_nested_message = "message" in payload and isinstance(payload["message"], dict)
    has_mentions = "mentions" in payload
    has_reply_to = "reply_to" in payload
    has_ts = "ts" in payload or "timestamp" in payload

    # 1. Reject ambiguous mixtures of legacy and new-format fields
    if has_user_id and (has_speaker_id or has_message_id or has_nested_message or has_mentions or has_reply_to or has_ts):
        raise ValueError(
            "Ambiguous request: cannot mix legacy field 'user_id' with conversation-aware metadata fields."
        )

    # 2. Legacy format
    if has_user_id:
        ChatMessageRequest.model_validate(payload)
        return "legacy"

    # 3. Nested conversation-aware format
    if has_nested_message:
        ConversationAwareChatRequest.model_validate(payload)
        return "conversation_aware"

    # 4. Flat conversation-aware format
    if has_speaker_id or has_message_id:
        FlatConversationAwareChatRequest.model_validate(payload)
        return "conversation_aware"

    # 5. Missing identifiers -> validate against Union to raise standard ValidationError
    TypeAdapter(Union[ConversationAwareChatRequest, FlatConversationAwareChatRequest, ChatMessageRequest]).validate_python(payload)
    return "legacy"


def normalize_request(
    request: Union[
        dict[str, Any],
        ChatMessageRequest,
        ConversationAwareChatRequest,
        FlatConversationAwareChatRequest,
    ],
) -> NormalizedChatRequest:
    """Normalize a legacy or conversation-aware chat request.

    - Legacy requests map `user_id` to `speaker_id`, leave metadata as None/empty,
      and strictly do NOT fabricate metadata.
    - Conversation-aware requests preserve all provided metadata.
    """
    if isinstance(request, dict):
        fmt = detect_request_format(request)
        if fmt == "legacy":
            model = ChatMessageRequest.model_validate(request)
            return NormalizedChatRequest(
                conversation_id=model.conversation_id,
                speaker_id=model.user_id,
                text=model.message,
                format="legacy",
                message_id=None,
                display_name=None,
                mentions=[],
                reply_to=None,
                ts=None,
            )
        else:
            if "message" in request and isinstance(request["message"], dict):
                model_nested = ConversationAwareChatRequest.model_validate(request)
                return NormalizedChatRequest(
                    conversation_id=model_nested.conversation_id,
                    speaker_id=model_nested.message.speaker_id,
                    text=model_nested.message.text,
                    format="conversation_aware",
                    message_id=model_nested.message.message_id,
                    display_name=model_nested.message.display_name,
                    mentions=list(model_nested.message.mentions),
                    reply_to=model_nested.message.reply_to,
                    ts=model_nested.message.ts,
                )
            else:
                model_flat = FlatConversationAwareChatRequest.model_validate(request)
                ts_val = model_flat.timestamp if model_flat.timestamp is not None else model_flat.ts
                return NormalizedChatRequest(
                    conversation_id=model_flat.conversation_id,
                    speaker_id=model_flat.speaker_id,
                    text=model_flat.message,
                    format="conversation_aware",
                    message_id=model_flat.message_id,
                    display_name=model_flat.display_name,
                    mentions=list(model_flat.mentions),
                    reply_to=model_flat.reply_to,
                    ts=ts_val,
                )

    if isinstance(request, ChatMessageRequest):
        return NormalizedChatRequest(
            conversation_id=request.conversation_id,
            speaker_id=request.user_id,
            text=request.message,
            format="legacy",
            message_id=None,
            display_name=None,
            mentions=[],
            reply_to=None,
            ts=None,
        )

    if isinstance(request, ConversationAwareChatRequest):
        return NormalizedChatRequest(
            conversation_id=request.conversation_id,
            speaker_id=request.message.speaker_id,
            text=request.message.text,
            format="conversation_aware",
            message_id=request.message.message_id,
            display_name=request.message.display_name,
            mentions=list(request.message.mentions),
            reply_to=request.message.reply_to,
            ts=request.message.ts,
        )

    if isinstance(request, FlatConversationAwareChatRequest):
        ts_val = request.timestamp if request.timestamp is not None else request.ts
        return NormalizedChatRequest(
            conversation_id=request.conversation_id,
            speaker_id=request.speaker_id,
            text=request.message,
            format="conversation_aware",
            message_id=request.message_id,
            display_name=request.display_name,
            mentions=list(request.mentions),
            reply_to=request.reply_to,
            ts=ts_val,
        )

    raise ValueError(f"Unsupported request type: {type(request)}")
