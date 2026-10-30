# Hush — Conversational Restraint Engine

**Hush** is a standalone, decision-first Conversational Restraint Engine for group chats (2+ human participants and an assistant). Instead of reflexively responding to every message in a channel, Hush evaluates the multi-turn conversational context and determines *whether* and *how* to intervene before any text generation model is invoked.

> **Core Principle:** Decide first. Generate only when justified. Restrain by default.

---

## 1. The Problem & Core Concept

Most conversational assistants in team messaging platforms (e.g., Slack, Teams, Discord) interject too frequently. In multi-participant collaborative environments, reflexive generation causes significant disruption:
- **Interruption of Human Collaboration:** Assistants interrupt natural problem-solving dialogues between human team members.
- **Unnecessary Latency & Token Waste:** Costly LLM generation calls are burned on casual chatter, acknowledgments, or questions directed at other colleagues.
- **Unclear Clarification:** Assistants often guess answers to underspecified prompts instead of asking a single focused clarifying question.

Hush decouples conversational turn-taking into two distinct phases:

```text
Incoming Message + History
           │
           ▼
┌────────────────────────────────────────┐
│      Hush Decision Service             │
│  Evaluates addressing & context features│
└──────────────────┬─────────────────────┘
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
     [ SILENT ] [ SPEAK ] [ ASK ]
         │         │         │
         │         └────┬────┘
         │              │
         │              ▼
         │      ┌───────────────┐
         │      │ Text Model /  │
         │      │ LLM Generator │
         │      └───────┬───────┘
         │              │
         ▼              ▼
   Bypass Model    Send Response
  (0 Gen Calls)   (Reply / Question)
```

### Action Taxonomy

| Action | Meaning | Generation Behavior |
|---|---|---|
| **`SILENT`** | No assistant response is needed (e.g., human-to-human discussion, casual chatter, or third-person mentions). | **Strictly bypassed.** Zero LLM or generation calls are made (`reply = None`, `generation_used = False`). |
| **`SPEAK`** | The assistant is addressed and has sufficient contextual information to contribute a direct, helpful response. | Text generation is invoked to produce a relevant answer. |
| **`ASK`** | The assistant is addressed, but the request is ambiguous, underspecified, or missing key parameters. | Text generation is invoked to ask **one** concise clarifying question. |

---

## 2. Addressing Architecture (Two-Tier Invariant)

A core requirement in group chat restraint is determining whether a message is genuinely intended for the assistant. Hush enforces a strict, two-tier addressing protocol across the API, feature extraction, and annotation rubrics:

### Tier 1 — Authoritative Metadata (Evaluated First)
1. **Explicit Mention:** The message payload includes Hush's ID in its `mentions` list.
2. **Reply to Assistant:** The message `reply_to` pointer references a prior message authored by Hush.

*Rule:* If metadata points to a human colleague (e.g., replying to user `u-alex`, or mentioning only human peers), the message is **not** addressed to Hush, even if the user text contains the word "hush" (e.g., *"Hush told me about that bug"* or *"Hush, the baby is asleep"*).

### Tier 2 — Text Fallback (Only when Tier 1 Metadata is Absent)
3. **Name Invocation:** Hush's name or handle appears as a standalone token at the start of a turn.
4. **Role Request:** Direct natural language requests explicitly naming the assistant role (e.g., *"assistant, what does git rebase do?"*).

---

## 3. Current Production Status & Architecture Guarantees

> [!NOTE]
> **Production Decider:** The live production API endpoint (`POST /api/chat/message`) defaults to [`TrainedRestraintClassifierDecider`](backend/app/services/decider.py) (`HUSH_DECIDER_MODE=classifier`). It automatically resolves and evaluates the trained 22-feature restraint classifier artifact from `data/models/restraint_classifier_v1.json` independently of the process's working directory.

### Configuration & Modes
The decision engine is controlled via environment variables:
- `HUSH_DECIDER_MODE=classifier` (default): Production decider loading the 22-feature regularized multinomial logistic regression model artifact.
- `HUSH_DECIDER_MODE=stub`: Development stub decider with rule-based heuristics for testing.
- `HUSH_DECIDER_MODE=real`: Adapter for teammate-owned `hush_gate` integration, failing explicitly if unavailable.
- `HUSH_MODEL_PATH`: Optional custom model artifact path (defaults to repo-root `data/models/restraint_classifier_v1.json`).

### Architectural Invariants & Governance
- **Zero-Generation SILENT Invariant:** When `SILENT` is chosen, text generation is strictly bypassed. Exactly zero generation or LLM calls are made (`reply = None`, `generation_used = False`). Schema validation strictly rejects any response pairing `SILENT` with non-null text or generation flags.
- **Addressing Invariant Safeguards:** Directly addressed messages (explicit mentions, reply-to Hush pointers, or unambiguous direct invocations) are safeguarded against ever returning `SILENT`.
- **Decoupled Decision & Generation:** Decision and text generation are strictly isolated. When `SPEAK` or `ASK` is chosen, generation connects to a hosted LLM provider via `HUSH_GENERATION_MODE=openai` (default, supporting OpenAI, Groq, OpenRouter, Mistral, and local endpoints like Ollama) or Amazon Bedrock Converse API (`HUSH_GENERATION_MODE=bedrock`). For offline development and testing, explicit stub mode is available (`HUSH_GENERATION_MODE=stub`). Missing provider credentials produce clear HTTP 500 configuration errors rather than silently falling back to placeholders.
- **Provider API Usage & Charges:** Using a live hosted LLM provider (`OPENAI_API_KEY` or AWS Bedrock credentials) sends prompts to external API endpoints and may incur billing charges from the respective provider. All automated tests run strictly offline with mocked providers and zero external network calls.
- **Honest Confidence Telemetry:** The classifier emits honest uncalibrated confidence scores with `confidence_kind = "uncalibrated_score"`. Rule stubs emit `confidence = None` with `confidence_kind = "none"`. Calibrated probabilities are only reported when backed by empirical calibration models.
- **Provisional Status of Model Metrics:** The classifier is an ML baseline trained on synthetic data with documented template leakage across historical splits ([`docs/leakage_and_calibration_audit.md`](docs/leakage_and_calibration_audit.md)); evaluation metrics are strictly provisional pending uncompromised fresh-holdout collection.

---

## 4. Dataset, Dual Human Review & Provenance Audit

### Dataset Overview
- **Master Dataset (`data/dataset_v1.jsonl`):** 300 curated multi-turn conversational scenarios across technical and workplace domains (DevOps, incident management, code reviews, system architecture, and team banter).
- **Fixed Splits:**
  - **Development Split (`data/splits/dev.jsonl`):** 180 scenarios (grouped by conversation ID).
  - **Historical Test Split (`data/splits/test.jsonl`):** 120 scenarios (grouped by conversation ID).

### Independent Dual Human Review (Phase 6)
To validate annotation reliability, an independent dual-reviewer protocol was executed on 60 stratified scenarios across dev and test:
- **Reviewers:** Two independent human annotators evaluating blinded packets with zero access to gold labels or model predictions.
- **Raw Agreement Rate:** **98.33%** (59 out of 60 scenarios in complete agreement).
- **Inter-Annotator Agreement:** **Cohen's $\kappa = 0.9740$**, substantially exceeding the project quality gate threshold ($\kappa \ge 0.75$).
- **Dispute Adjudication:** Exactly one dispute occurred on scenario `s-192` (Reviewer 1 selected `SPEAK`; Reviewer 2 selected `ASK`). The project owner adjudicated `s-192` as `SPEAK` (`is_ambiguous = false`) based on canonical PEP 530 async comprehension syntax guidance.
- **Audit Documentation:** Full methodology and governance records are documented in [`docs/phase5_evaluation_integrity.md`](docs/phase5_evaluation_integrity.md) and [`data/review/review_status_summary.md`](data/review/review_status_summary.md).

### Leakage Audit & Provisional Metric Limitations
A comprehensive dataset leakage audit ([`docs/leakage_and_calibration_audit.md`](docs/leakage_and_calibration_audit.md)) identified:
1. **51 cross-split pairs** sharing word Jaccard similarity $\ge 0.70$.
2. **14 shared multi-turn opening templates** between the development and historical test sets.

Because the historical 120-scenario test set was evaluated during early development and shares structural stems with development data, **historical test set metrics are provisional and subject to template leakage bias**. They must not be cited as uncompromised out-of-distribution generalization results.

---

## 5. Fresh Holdout Benchmark Status (Phase 7)

To establish genuine generalization performance, Phase 7 established the governance and quarantine tooling for a brand-new **50-scenario holdout benchmark** (`holdout_v1`):

- **Current Status:** **PENDING INDEPENDENT COLLECTION.**
- **Quarantine Guarantee:** `data/holdout/holdout_v1.jsonl` **does not yet exist** on disk and has not been fabricated with synthetic LLM text.
- **Three-Tier Overlap Filter:** Tooling in [`scripts/prepare_fresh_holdout.py`](scripts/prepare_fresh_holdout.py) screens candidate submissions against all 300 existing scenarios:
  1. Latest message word Jaccard similarity ($< 0.70$).
  2. Full dialogue word Jaccard similarity ($< 0.70$).
  3. Conversational prefix stem similarity ($< 0.70$).
- **Mandatory Provenance:** Every candidate scenario must record `author_id`, `created_at`, `source_type`, and an explicit `derived_from_existing: false` flag.
- **Single-Use Protocol:** The fresh holdout will be evaluated exactly **once** upon project completion. Details are documented in [`docs/phase7_holdout_readiness.md`](docs/phase7_holdout_readiness.md).

---

## 6. Offline Research Restraint Classifier

In offline research mode, Hush includes a pure-Python, zero-dependency 3-class regularized logistic regression classifier:
- **Feature Extraction ([`backend/app/services/features.py`](backend/app/services/features.py)):** Extracts 22 interpretable conversational context features across addressing metadata, discourse structure, question semantics, ambiguity markers, and multi-turn speaker dynamics.
- **Pure-Python Model ([`backend/app/services/classifier.py`](backend/app/services/classifier.py)):** Multinomial softmax logistic regression trained with L2 regularization ($C = 1.0$) and mini-batch gradient descent.
- **Calibration ([`scripts/analyze_calibration.py`](scripts/analyze_calibration.py)):** Evaluates temperature scaling ($T \approx 0.20$), Brier scores, and reliability diagrams to prevent overconfident abstention.
- **Baseline Implementations ([`backend/app/services/baselines.py`](backend/app/services/baselines.py)):** Majority Class, Always Reply, Rule-Based Addressing Baseline, and Mock LLM-as-Decider.

> [!WARNING]
> **Known Limitations:**
> - **Synthetic Training Data:** The model is trained on synthetic multi-turn conversational scenarios rather than real production dialogue logs.
> - **Historical Template Overlap:** The historical test split has documented template overlap (65.6% prefix stem sharing), rendering past test split metrics provisional rather than clean generalization evidence.
> - **Uncalibrated Confidence:** Model confidence scores are currently uncalibrated in production inference.
> - **Unevaluated Holdout:** No fresh holdout benchmark has been evaluated yet; collection status remains pending.
> - **Holdout Finalization Workflow:** --finalize-holdout currently trusts a quality_gate_passed flag in the review manifest instead of recomputing inter-annotator agreement directly from completed reviewer CSV forms; this must be updated before a real holdout dataset is finalized.

---

## 7. Architecture & Repository Structure

```text
Hush/
|-- .gitignore                      # Excludes secrets, caches, and private reviewer submissions
|-- LICENSE                         # MIT License
|-- README.md                       # System documentation and portfolio summary
|-- backend/
|   |-- .env.example                # Configuration template
|   |-- requirements.txt            # Core dependencies (FastAPI, Pydantic, Boto3)
|   |-- requirements-dev.txt        # Test dependencies (pytest, pytest-asyncio, httpx)
|   |-- app/
|   |   |-- config.py               # Pydantic Settings
|   |   |-- main.py                 # FastAPI application, CORS, and health endpoint
|   |   |-- schemas.py              # Pydantic models and SILENT invariant validators
|   |   |-- routes/
|   |   |   |-- chat.py             # POST /api/chat/message (decision & generation routing)
|   |   |   `-- metrics.py          # GET /api/metrics & POST /api/metrics/reset
|   |   `-- services/
|   |       |-- addressing.py       # Two-tier addressing detector & metadata resolver
|   |       |-- baselines.py        # Majority, AlwaysReply, RuleBased & MockLLM deciders
|   |       |-- bedrock.py          # Amazon Bedrock Runtime Converse API client
|   |       |-- classifier.py       # Pure-Python regularized multinomial logistic regression
|   |       |-- conversation.py     # In-memory sliding window history (max 20 turns)
|   |       |-- decider.py          # DeciderAdapter (stub vs real routing)
|   |       |-- evaluation.py       # Statistical metrics, bootstrap CIs, McNemar tests
|   |       |-- features.py         # 22 conversational context feature extractors
|   |       |-- generation.py       # GenerationService router (stub vs Bedrock)
|   |       |-- metrics.py          # In-memory MetricsTracker & telemetry
|   |       `-- request_adapter.py  # Backward-compatible request shape adapter
|   `-- tests/                      # 227 comprehensive unit, integration, and audit tests
|-- data/
|   |-- dataset_v1.jsonl            # 300 curated master scenarios
|   |-- holdout/                    # Holdout collection templates and specifications
|   |-- models/                     # Serialized restraint classifier artifacts
|   |-- results/                    # Baseline, error analysis, and calibration results
|   |-- review/                     # Review guidelines and public agreement summaries
|   `-- splits/                     # Fixed dev (180) and historical test (120) splits
|-- docs/                           # Epistemic audits, rubrics, and engineering specs
|   |-- annotation_rubric_v2.md
|   |-- dataset_provenance_audit.md
|   |-- leakage_and_calibration_audit.md
|   |-- phase5_evaluation_integrity.md
|   |-- phase7_holdout_readiness.md
|   `-- PLAN.md                     # Authoritative engineering specification
|-- frontend/                       # React 18 + TypeScript + Vite demo UI
`-- scripts/                        # Dataset curation, validation, training, and review tooling
```

---

## 8. Installation & Setup

### Prerequisites
- **Python 3.12+** (tested on Python 3.12.7)
- **Node.js 18+** & **npm 9+** (tested on Node v24.21.0, npm 11.19.0)
- **PowerShell 5.1+ / 7+** (Windows) or standard bash / zsh (macOS / Linux)

### 1. Set Up the Backend
From the repository root:

```powershell
# Create and activate virtual environment inside backend/
python -m venv backend\.venv
.\backend\.venv\Scripts\Activate.ps1

# Install runtime and development dependencies
pip install -r backend\requirements-dev.txt

# Copy environment template
Copy-Item backend\.env.example backend\.env
```

### 2. Set Up the Frontend

```powershell
cd frontend
npm install
cd ..
```

---

## 9. Running the Application

Run the backend API and frontend dashboard in separate terminal sessions:

### Terminal 1: Backend Service

```powershell
$env:PYTHONPATH="backend"
.\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Health Endpoint: `http://127.0.0.1:8000/health`
- Swagger Documentation: `http://127.0.0.1:8000/docs`

### Terminal 2: Frontend Dashboard

```powershell
cd frontend
npm run dev
```
- Web UI: `http://localhost:5173`

---

## 10. Automated Verification & Testing

### Backend Test Suite (227 Tests Passing)
Run the complete backend test suite across all 17 test modules:

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
$env:PYTHONPATH="backend"
& .\backend\.venv\Scripts\python.exe -m pytest backend/tests -p no:cacheprovider -v
```

**Verification Results:**
- **227 passed**, 0 failures in ~2.5 seconds.
- Test suites cover:
  - `test_addressing.py`: Two-tier addressing rules, metadata conflicts, text fallbacks, false-positive rejection.
  - `test_api.py`: Health endpoint, chat message routing, Bedrock error translation (500/502/503), zero calls on `SILENT`.
  - `test_baselines_and_eval.py`: Majority class, AlwaysReply, RuleBased baseline, bootstrap CIs, McNemar paired tests.
  - `test_bedrock.py`: Mocked AWS Bedrock Converse API payload formatting, client timeouts, and error handling.
  - `test_conversation_aware.py`: Multi-turn sliding window history and participant tracking.
  - `test_dataset_tooling.py`: Dataset validation, category distributions, split integrity.
  - `test_features_and_classifier.py`: 22 context feature extractors, pure-Python logistic regression training, serialization.
  - `test_legacy_compatibility.py`: Backward compatibility with legacy chat request schemas.
  - `test_metrics.py`: Telemetry counters, generation call savings, and metric resets.
  - `test_openai_generation.py`: Hosted OpenAI-compatible generation, model routing, prompt formatting, SILENT zero-call invariant, missing credential handling (500), sanitized error mapping (502).
  - `test_phase4_analysis.py`: Error analysis, threshold trade-off sweeps, calibration curve calculations.
  - `test_phase5_integrity.py`: Overlap detection, reviewer packet blinding, Cohen's kappa calculations.
  - `test_phase6_review_execution.py`: Human review CSV parsing, dispute logging, quality gate enforcement.
  - `test_phase7_holdout.py`: Candidate schema and provenance validation, three-level overlap filter, quarantine invariants.
  - `test_production_decider.py`: Model loading, working-directory independence, corruption and missing artifact handling, default routing, SILENT zero-generation invariant, addressing safeguard, legacy compatibility.
  - `test_schemas.py`: Pydantic models, action enums, confidence bounds, strict `SILENT` invariant checks.
  - `test_telemetry.py`: Address method logging and honest confidence type semantics (`none`, `uncalibrated_score`, `calibrated_probability`).

### Frontend Build Verification
```powershell
cd frontend
npm run build
cd ..
```
- **Result:** TypeScript compilation (`tsc`) and Vite bundling succeed with zero errors.

---

## 11. Security & Epistemic Governance Practices

- **Zero Secret Exposure:** Verified zero hardcoded credentials, API keys, or active `.env` files committed. All configuration is managed via environment variables.
- **Private Reviewer Isolation:** Individual human-reviewer submission CSVs, private rationales, and dispute ledgers are excluded from public git tracking via `.gitignore`.
- **Honest Telemetry:** Mock stubs and heuristics never fabricate numeric confidence scores; unmeasured fields explicitly return `null`.
- **Epistemic Humility:** When data leakage was discovered between splits, rather than reporting inflated metrics, the project halted promotion, conducted a rigorous audit, and designed a fresh holdout quarantine protocol.

---

## 12. License

This project is licensed under the [MIT License](LICENSE).
