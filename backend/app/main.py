"""FastAPI application entrypoint for Hush backend."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.routes.chat import router as chat_router
from app.routes.metrics import router as metrics_router
from app.services.decider import get_decider


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate decider and configuration on application startup."""
    # Validate CORS in production
    if settings.HUSH_ENV != "development":
        if not settings.CORS_ORIGINS or all(not str(o).strip() for o in settings.CORS_ORIGINS):
            raise RuntimeError(
                "CORS_ORIGINS must be set to explicit allowed origins in production mode (HUSH_ENV != 'development')."
            )
        if "*" in settings.CORS_ORIGINS or any(str(o).strip() == "*" for o in settings.CORS_ORIGINS):
            raise RuntimeError(
                "Wildcard CORS_ORIGINS='*' is not permitted in production mode. Specify explicit origin domains."
            )
    # Validate trusted proxy hops
    if settings.HUSH_TRUSTED_PROXY_HOPS < 1:
        raise RuntimeError("HUSH_TRUSTED_PROXY_HOPS must be a positive integer (>= 1).")

    # Ensure active decider loads reliably without falling back to stub
    get_decider()
    yield


app = FastAPI(
    title="Hush API",
    description="Conversational Restraint Engine for group chats",
    version="0.1.0",
    lifespan=lifespan,
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
async def limit_request_body_size(request: Request, call_next):
    """Enforce maximum request body size limit."""
    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.HUSH_MAX_REQUEST_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"Request body exceeds maximum size of {settings.HUSH_MAX_REQUEST_BODY_BYTES} bytes."
                        },
                    )
            except ValueError:
                pass

        body = await request.body()
        if len(body) > settings.HUSH_MAX_REQUEST_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": f"Request body exceeds maximum size of {settings.HUSH_MAX_REQUEST_BODY_BYTES} bytes."
                },
            )
    return await call_next(request)


@app.middleware("http")
async def add_custom_headers(request: Request, call_next):
    """Add telemetry and mode headers to response."""
    response: Response = await call_next(request)
    response.headers["X-Hush-Decider-Mode"] = settings.HUSH_DECIDER_MODE
    response.headers["X-Hush-Generation-Mode"] = settings.HUSH_GENERATION_MODE
    return response


@app.get("/health", tags=["health"], summary="Health check endpoint")
def health_check() -> dict:
    """Return health status and active operating modes."""
    dec_mode = (settings.HUSH_DECIDER_MODE or "classifier").lower()
    gen_mode = (settings.HUSH_GENERATION_MODE or "stub").lower()
    return {
        "status": "ok",
        "decider_mode": dec_mode,
        "generation_mode": gen_mode,
    }
