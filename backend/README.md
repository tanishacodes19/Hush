# Hush Backend

FastAPI service for Hush, the conversational AI restraint agent.

## Architecture & Responsibilities

- **Chat Processing**: Receives conversation messages and maintains context windows.
- **Decision Gating**: Queries the `hush_gate` adapter to evaluate action (`SPEAK`, `SILENT`, `ASK`) and calibrated confidence.
- **SILENT Invariant**: When the action is `SILENT`, Bedrock response generation is strictly bypassed.
- **Bedrock Generation**: Invokes Amazon Bedrock only when the decision is `SPEAK` (response) or `ASK` (clarifying question).
- **Metrics**: Tracks evaluations, call reductions, action distribution, and latencies.

## Directory Structure

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # Application entrypoint (Step 2/3)
│   ├── schemas.py           # Pydantic v2 schemas (Step 2)
│   ├── routes/              # API endpoints (Step 4)
│   └── services/            # Business logic & integrations (Step 2-4)
├── tests/                   # Pytest test suite (Step 5)
├── requirements.txt         # Core dependencies
├── requirements-dev.txt     # Test and lint dependencies
├── .env.example             # Configuration template
└── README.md
```

## Setup & Running

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your specific AWS_REGION, BEDROCK_MODEL_ID, etc.
   ```
