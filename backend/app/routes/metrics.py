"""Metrics telemetry endpoints for Hush."""

from fastapi import APIRouter, HTTPException, status
from app.config import settings
from app.schemas import MetricsResponse, ResetMetricsResponse
from app.services.metrics import metrics_tracker

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get(
    "",
    response_model=MetricsResponse,
    summary="Get aggregated telemetry metrics for Hush",
)
def get_metrics() -> MetricsResponse:
    """Return aggregated counts, decision distribution, generation statistics, and latencies."""
    return metrics_tracker.get_metrics()


@router.post(
    "/reset",
    response_model=ResetMetricsResponse,
    summary="Reset telemetry metrics counters",
)
def reset_metrics() -> ResetMetricsResponse:
    """Reset all in-memory telemetry counters and latency buffers to zero."""
    if (settings.HUSH_ENV or "").lower() != "development":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Metrics reset is only permitted in development mode (HUSH_ENV=development).",
        )
    metrics_tracker.reset()
    return ResetMetricsResponse(
        status="ok",
        message="Metrics reset successfully",
    )
