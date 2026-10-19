"""FastAPI application entrypoint for Hush backend."""

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes.chat import router as chat_router
from app.routes.metrics import router as metrics_router

app = FastAPI(
    title="Hush API",
    description="Conversational AI restraint agent using Strands Decider 2B",
    version="0.1.0",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat_router)
app.include_router(metrics_router)


@app.middleware("http")
async def add_custom_headers(request, call_next):
    """Add telemetry and mode headers to response."""
    response: Response = await call_next(request)
    response.headers["X-Hush-Decider-Mode"] = settings.HUSH_DECIDER_MODE
    response.headers["X-Hush-Generation-Mode"] = settings.HUSH_GENERATION_MODE
    return response


@app.get("/health", tags=["health"], summary="Health check endpoint")
def health_check() -> dict:
    """Return health status and active operating modes."""
    return {
        "status": "ok",
        "mode": settings.HUSH_DECIDER_MODE,
        "decider_mode": settings.HUSH_DECIDER_MODE,
        "generation_mode": settings.HUSH_GENERATION_MODE,
        "environment": settings.ENVIRONMENT,
    }
