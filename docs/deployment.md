# Hush — Deployment & Operational Hardening Guide

This document describes the operational configuration, environment variables, security safeguards, and deployment guidelines for Hush.

---

## A. Environment-Variable Reference Table

| Variable Name | Required? | Default / Example Value | Secret? | Target Location | Description & Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `HUSH_ENV` | Optional | `production` (in production) / `development` | No | Backend Host | Operating environment. In production (`!= "development"`), metrics reset is disabled (403) and `CORS_ORIGINS` must be explicitly configured. |
| `HUSH_GENERATION_MODE` | Optional | `openai` (or `stub`, `bedrock`) | No | Backend Host | Text generation provider. Options: `openai` (hosted LLM), `stub` (simulated demo marker), or `bedrock`. |
| `HUSH_DECIDER_MODE` | Optional | `classifier` (or `stub`, `real`) | No | Backend Host | Restraint decider implementation. Default `classifier` loads the trained 22-feature restraint classifier. |
| `CORS_ORIGINS` | **Required in Prod** | `https://hush.example.com` | No | Backend Host | Comma-separated or JSON list of allowed origins. In production, startup fails if unset or empty. In development, defaults to `http://localhost:5173,http://127.0.0.1:5173`. |
| `VITE_API_BASE_URL` | Optional | `https://api.hush.example.com` | No | Frontend Build | Optional backend URL for frontend client. If unset, frontend uses same-origin relative URLs (`/api/...`). |
| `HUSH_TRUST_PROXY_HEADERS` | Optional | `false` | No | Backend Host | Whether to inspect `X-Forwarded-For` for client rate limiting. Must remain `false` unless behind a trusted, known proxy. |
| `HUSH_TRUSTED_PROXY_HOPS` | Optional | `1` | No | Backend Host | Number of trusted proxy hops to count from the right of `X-Forwarded-For`. Must be a positive integer >= 1. |
| `HUSH_RATE_LIMIT_ENABLED` | Optional | `true` | No | Backend Host | Master toggle for in-memory sliding-window rate limiting on `POST /api/chat/message`. |
| `HUSH_RATE_LIMIT_REQUESTS` | Optional | `60` | No | Backend Host | Maximum requests allowed per client within the rate limit window. |
| `HUSH_RATE_LIMIT_WINDOW_SECONDS` | Optional | `60.0` | No | Backend Host | Duration of the sliding rate-limit window in seconds. |
| `HUSH_RATE_LIMIT_MAX_CLIENTS` | Optional | `10000` | No | Backend Host | Upper bound on in-memory rate-limiter client storage to prevent memory exhaustion. |
| `HUSH_MAX_MESSAGE_LENGTH` | Optional | `4096` | No | Backend Host | Maximum allowed character length for message text. Oversized messages are rejected with HTTP 422. |
| `HUSH_MAX_REQUEST_BODY_BYTES` | Optional | `65536` | No | Backend Host | Maximum allowed incoming request body size in bytes (64 KB). Requests exceeding this size receive HTTP 413. |
| `HUSH_MAX_CONVERSATIONS` | Optional | `1000` | No | Backend Host | Maximum number of concurrent conversation histories kept in memory. When reached, oldest room is evicted first. |
| `OPENAI_API_KEY` | Optional / Mode-dependent | `sk-proj-EXAMPLE_PLACEHOLDER_NOT_A_REAL_KEY` | **YES** | Backend Host | API key for OpenAI or OpenAI-compatible inference. Required only when `HUSH_GENERATION_MODE=openai`. |
| `OPENAI_MODEL_ID` | Optional | `gpt-4o-mini` | No | Backend Host | Model identifier passed to chat completions endpoint (e.g., `gpt-4o-mini`, `llama-3.3-70b-versatile`). |
| `OPENAI_BASE_URL` | Optional | `https://api.openai.com/v1` | No | Backend Host | Endpoint URL for chat completions (allows OpenRouter, Groq, Mistral, Ollama). Remote hosts require HTTPS. |
| `OPENAI_TIMEOUT_SECONDS` | Optional | `15.0` | No | Backend Host | Network request timeout for hosted LLM generation calls. |
**Supported generation modes:** Use `stub` for simulated demonstrations or `openai` for live generation in the initial deployment. The `bedrock` mode has not been cleared for public deployment and reflects AWS error text in HTTP responses. Do not enable it publicly until provider errors are mapped to fixed, sanitized messages.

---

## B. API-Key Security
**Security note:** Never include credentials, API keys, or other secrets in `OPENAI_BASE_URL`, because an invalid value can appear in startup error messages or logs.

> [!CAUTION]
> **Secret Boundary Invariant:**
> - `OPENAI_API_KEY` belongs exclusively on the backend host environment.
> - **Never** set `OPENAI_API_KEY` in frontend build settings or prefix it with `VITE_` (e.g., `VITE_OPENAI_API_KEY`).
> - Any variable prefixed with `VITE_` is statically baked into client JavaScript bundles and permanently visible to all browser users.

---

## C. Recommended First Deployment

For initial portfolio demonstrations and smoke verification:

1. **Generation Mode:** Set `HUSH_GENERATION_MODE=stub`.
2. **Worker Count:** Run a single backend process (`--workers 1`).
3. **Rate Limiting:** Leave enabled (`HUSH_RATE_LIMIT_ENABLED=true`, default 60 req/min).
4. **CORS:** Set explicit `CORS_ORIGINS` to the exact frontend hosting domain (e.g. `https://my-hush-demo.vercel.app`).
5. **No Provider Secrets:** Do **not** configure a live `OPENAI_API_KEY` for the initial public demo.

> [!WARNING]
> **Spending Safeguard:**
> The backend in-memory rate limiter protects the application from local request bursts, but **it is not a financial spending cap**. Before enabling live OpenAI generation (`HUSH_GENERATION_MODE=openai`), always configure and verify hard monthly spending limits directly in your LLM provider's billing dashboard.

---

## D. Model Artifact Requirements

The backend requires the trained model artifact to be packaged with the deployment:

- Path: `data/models/restraint_classifier_v1.json` (6.2 KB)
- The production decider (`TrainedRestraintClassifierDecider`) loads this artifact at startup.
- **Fail-Fast Invariant:** If this artifact is missing, corrupt, or inaccessible, backend startup or message handling raises a clear error. The system does **not** silently fall back to stub heuristics or keyword rules.

---

## E. Start Command & Worker Model

### Start Command
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
```

### Single-Worker Concurrency Model
- Hush maintains ephemeral conversation state (`ConversationManager`) and rate-limit tracking (`RateLimiter`) in process memory.
- If multiple Uvicorn workers or multi-instance clusters are deployed without an external cache (e.g. Redis), clients hitting different worker processes will not share conversation context or rate-limit quotas.
- Free-tier hosting providers (e.g. Render, Koyeb, Railway) often shut down or restart idle containers, which resets in-memory conversation histories and rate-limit counters.

---

## F. Health Check Endpoint

Health checks should poll `GET /health`.

### Response Allowlist
The endpoint returns HTTP 200 with strictly allow-listed, non-sensitive operational fields:
```json
{
  "status": "ok",
  "decider_mode": "classifier",
  "generation_mode": "openai"
}
```
- **Information Hiding:** No environment variable dumps, credentials, database paths, server hostnames, or internal stack traces are ever emitted by `/health`.
- The reported `generation_mode` directly reflects the active configured generation factory setting.

---

## G. Python Runtime Version

- **Validated Environment:** Python `3.12.7` (tested: `3.12.7` on Windows 64-bit).
- Configure your deployment platform to use Python 3.12 (e.g. via your host's runtime configuration file or build settings).

---

## H. Proxy Header Configuration & Security

- **Default Behavior:** When `HUSH_TRUST_PROXY_HEADERS=false` (the default), Hush identifies clients strictly by the physical TCP socket IP address (`request.client.host`). Incoming `X-Forwarded-For` headers are completely ignored.
- **When Behind a Reverse Proxy / Load Balancer:**
  - If trusting proxy headers, set `HUSH_TRUST_PROXY_HEADERS=true` and configure `HUSH_TRUSTED_PROXY_HOPS` (default `1`).
  - Hush uses **right-counted hop selection**: `entries[-HUSH_TRUSTED_PROXY_HOPS]`. The rightmost IP was appended by the immediately upstream proxy. Entries farther to the left can be trivially spoofed by the client and are disregarded.
  - If `X-Forwarded-For` is missing or contains fewer entries than the configured hop count, Hush falls back safely to the socket IP.
  - Check your hosting provider's reverse proxy documentation (e.g., Cloudflare, AWS ALB, Nginx) to determine exactly how many proxy hops exist between the public internet and your container.

---

## I. Smoke-Test Checklist (Offline / No Live Provider Calls)

The following checklist verifies deployment readiness without making live LLM provider calls or incurring billing charges:

- [x] **SILENT Bypasses Generation:** When incoming conversation evaluates to `SILENT`, text generation is completely bypassed in both `stub` and `openai` modes with 0 provider calls.
- [x] **OpenAI Key Missing Returns Clear Error:** When `HUSH_GENERATION_MODE=openai` and `OPENAI_API_KEY` is unset or invalid, chat requests return HTTP 500 with a clear configuration message; they never fall back to stub replies.
- [x] **CORS Rejection of Disallowed Origins:** Options preflight requests from unlisted origins return HTTP 400 and omit `access-control-allow-origin`.
- [x] **Oversized Message Rejected:** Messages longer than `HUSH_MAX_MESSAGE_LENGTH` (4096 characters) are rejected with HTTP 422.
- [x] **Legacy Seven-Field Response Preserved:** Requests using the legacy `{conversation_id, user_id, message}` schema return exactly the seven canonical response fields.
- [x] **Rate Limit Burst Enforcement:** Bursts exceeding `HUSH_RATE_LIMIT_REQUESTS` return HTTP 429 (`"Too many requests. Please try again later."`).
- [x] **Demo Notice Controlled by Health:** Frontend displays `"Demo mode: replies are simulated; restraint decisions come from the trained classifier"` only when `/health` returns `generation_mode: "stub"`.
- [x] **Health Allowlist Verified:** `/health` exposes exactly three allow-listed fields: `status`, `generation_mode`, and `decider_mode`.
