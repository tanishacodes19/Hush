"""Chat endpoints for Hush backend."""

import time
from fastapi import APIRouter, HTTPException, status
from app.config import settings
from app.schemas import Action, ChatMessageRequest, ChatMessageResponse
from app.services.bedrock import BedrockConfigError, BedrockServiceError
from app.services.conversation import conversation_manager
from app.services.decider import get_decider
from app.services.generation import get_generator
from app.services.metrics import metrics_tracker

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post(
    "/message",
    response_model=ChatMessageResponse,
    summary="Process incoming chat message and decide whether to SPEAK, ASK, or remain SILENT",
)
def handle_chat_message(request: ChatMessageRequest) -> ChatMessageResponse:
    # 1. Update in-memory conversation state with incoming message
    conversation_manager.add_message(
        conversation_id=request.conversation_id,
        speaker=request.user_id,
        text=request.message,
    )

    # 2. Retrieve bounded conversation context
    context = conversation_manager.get_history(request.conversation_id)

    # 3. Decision evaluation via adapter
    decider = get_decider()
    try:
        decision = decider.decide(context)
    except (RuntimeError, NotImplementedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    # 4. Telemetry: Record decision
    metrics_tracker.record_decision(
        action=decision.action,
        confidence=decision.confidence,
        fallback_used=decision.fallback_used,
        decision_latency_ms=decision.decision_latency_ms,
    )

    # 5. Action routing
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

            conversation_manager.add_message(
                conversation_id=request.conversation_id,
                speaker="Hush",
                text=reply,
            )

        except BedrockConfigError as exc:
            metrics_tracker.record_generation_failure()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Bedrock configuration error: {exc}",
            ) from exc
        except BedrockServiceError as exc:
            metrics_tracker.record_generation_failure()
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Bedrock generation error: {exc}",
            ) from exc

    elif decision.action == Action.SILENT:
        # INVARIANT: When SILENT, reply is strictly None and generation is never called.
        reply = None
        generation_used = False

    return ChatMessageResponse(
        conversation_id=request.conversation_id,
        action=decision.action,
        confidence=decision.confidence,
        reply=reply,
        generation_used=generation_used,
        fallback_used=decision.fallback_used,
        decision_latency_ms=decision.decision_latency_ms,
    )
