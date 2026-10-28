"""Chat endpoints for Hush backend."""

import inspect
import time
from typing import Union
from fastapi import APIRouter, HTTPException, Request, status
from app.config import settings
from app.schemas import (
    Action,
    ChatMessageResponse,
    ChatRequestUnion,
    ConversationAwareChatMessageResponse,
    DecisionMetadata,
)
from app.services.addressing import resolve_addressing
from app.services.bedrock import BedrockConfigError, BedrockServiceError
from app.services.conversation import conversation_manager
from app.services.decider import get_decider
from app.services.generation import GenerationConfigError, GenerationServiceError, get_generator
from app.services.metrics import metrics_tracker
from app.services.rate_limiter import get_client_ip, rate_limiter
from app.services.request_adapter import normalize_request

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post(
    "/message",
    response_model=Union[ConversationAwareChatMessageResponse, ChatMessageResponse],
    summary="Process incoming chat message and decide whether to SPEAK, ASK, or remain SILENT",
)
def handle_chat_message(
    request: ChatRequestUnion,
    raw_request: Request,
) -> Union[ConversationAwareChatMessageResponse, ChatMessageResponse]:
    # Check rate limit
    if settings.HUSH_RATE_LIMIT_ENABLED:
        client_ip = get_client_ip(
            raw_request,
            trust_proxy_headers=settings.HUSH_TRUST_PROXY_HEADERS,
            trusted_proxy_hops=settings.HUSH_TRUSTED_PROXY_HOPS,
        )
        if not rate_limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later.",
            )

    # 0. Normalize incoming request (supports legacy and conversation-aware schemas)
    normalized = normalize_request(request)

    # 1. Retrieve recent history before appending to resolve reply_to metadata
    prior_history = conversation_manager.get_history(normalized.conversation_id)

    # 2. Evaluate addressing using Two-Tier addressing policy (PLAN.md Section 1)
    addressing = resolve_addressing(
        message_text=normalized.text,
        mentions=normalized.mentions,
        reply_to=normalized.reply_to,
        conversation_history=prior_history,
    )

    # 3. Update in-memory conversation state with incoming message and metadata
    role = "assistant" if normalized.speaker_id.lower() in ("hush", "assistant") else "human"
    conversation_manager.add_message(
        conversation_id=normalized.conversation_id,
        speaker=normalized.speaker_id,
        text=normalized.text,
        message_id=normalized.message_id,
        speaker_id=normalized.speaker_id,
        role=role,
        mentions=normalized.mentions,
        reply_to=normalized.reply_to,
        ts=normalized.ts,
    )

    # 4. Retrieve bounded conversation context (now includes incoming message)
    context = conversation_manager.get_history(normalized.conversation_id)

    # 5. Decision evaluation via adapter (passing addressing result)
    decider = get_decider()
    try:
        if "addressing" in inspect.signature(decider.decide).parameters:
            decision = decider.decide(context, addressing=addressing)
        else:
            decision = decider.decide(context)
    except (RuntimeError, NotImplementedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    # 6. Telemetry: Record decision
    metrics_tracker.record_decision(
        action=decision.action,
        confidence=decision.confidence,
        confidence_kind=decision.confidence_kind,
        fallback_used=decision.fallback_used,
        decision_latency_ms=decision.decision_latency_ms,
        addressing=addressing,
    )

    # 7. Action routing
    reply: str | None = None
    generation_used: bool = False
    generator = get_generator()

    if decision.action in (Action.SPEAK, Action.ASK):
        metrics_tracker.record_generation_attempt()
        t_start = time.perf_counter()
        try:
            if decision.action == Action.SPEAK:
                reply = generator.generate_reply(context)
            else:
                reply = generator.generate_clarification(context)

            gen_latency_ms = (time.perf_counter() - t_start) * 1000.0
            metrics_tracker.record_generation_success(gen_latency_ms)
            generation_used = True

            assistant_msg_id = (
                f"hush-{normalized.message_id}"
                if normalized.message_id
                else f"hush-{len(context)}"
            )
            conversation_manager.add_message(
                conversation_id=normalized.conversation_id,
                speaker="Hush",
                text=reply,
                message_id=assistant_msg_id,
                speaker_id="hush",
                role="assistant",
            )

        except BedrockConfigError as exc:
            metrics_tracker.record_generation_failure()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Bedrock configuration error: {exc}",
            ) from exc
        except GenerationConfigError as exc:
            metrics_tracker.record_generation_failure()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation configuration error: {exc}",
            ) from exc
        except BedrockServiceError as exc:
            metrics_tracker.record_generation_failure()
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Bedrock generation error: {exc}",
            ) from exc
        except GenerationServiceError as exc:
            metrics_tracker.record_generation_failure()
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Generation provider error: {exc}",
            ) from exc

    elif decision.action == Action.SILENT:
        # INVARIANT: When SILENT, reply is strictly None and generation is never called.
        reply = None
        generation_used = False

    if normalized.format == "legacy":
        return ChatMessageResponse(
            conversation_id=normalized.conversation_id,
            action=decision.action,
            confidence=decision.confidence,
            reply=reply,
            generation_used=generation_used,
            fallback_used=decision.fallback_used,
            decision_latency_ms=decision.decision_latency_ms,
        )

    return ConversationAwareChatMessageResponse(
        conversation_id=normalized.conversation_id,
        message_id=normalized.message_id,
        addressed_to_hush=addressing.addressed_to_hush,
        address_method=addressing.address_method,
        action=decision.action,
        confidence=decision.confidence,
        confidence_kind=decision.confidence_kind,
        reply=reply,
        generation_used=generation_used,
        decision=DecisionMetadata(
            method=decision.method,
            fallback_used=decision.fallback_used,
            latency_ms=decision.decision_latency_ms,
        ),
    )
