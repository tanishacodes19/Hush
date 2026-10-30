# Hush — Conversational Restraint Engine

**Project type:** Personal flagship project (Software Engineering + Applied AI/ML)
**Status:** Independent development; not tied to a hackathon
**One-line pitch:** A group-chat assistant that decides *whether* to speak before it generates anything, and is evaluated honestly against simple and LLM-based alternatives.

---

## 1. Overview

Most conversational systems focus on generating a good reply. Hush focuses on the decision that comes first: **should the assistant respond at all, and what kind of action is appropriate?**

| Action | Meaning |
|---|---|
| **SPEAK** | A reply is appropriate; the generation component produces one. |
| **ASK** | The request is unclear or missing key information; ask one focused clarifying question. |
| **SILENT** | No response is needed; the generation component is **not** called. |

### Setting (fixed for Version 1)

Hush is a participant in a **multi-person group chat** (2+ humans plus the assistant). This matters: in a 1:1 chat, SILENT is almost never correct because the user just spoke to the assistant. In a group chat, deciding *not* to speak is the interesting problem.

Every message carries a **speaker identity**, and every message is classified as *addressed to Hush* or not by a defined method (see "Addressing" below). The same definition is used in the API, the features, and the labeling guideline, so code and labels cannot drift apart.

### Addressing (single definition used everywhere)

Addressing is resolved in **two tiers**. Metadata is authoritative; text analysis is only a fallback.

**Tier 1 — Authoritative metadata (checked first)**

1. **Explicit mention:** the client supplies Hush's ID in the message's `mentions` list.
2. **Reply to Hush:** the message's `reply_to` points at a message authored by Hush.

If either fires, the message is addressed to Hush, and no text analysis is run.

If metadata is present and points elsewhere (e.g., `reply_to` targets a human, or `mentions` lists only other people), the message is **not addressed to Hush**, even if the text contains Hush's name. A human-directed message that happens to include the word "hush" does not trigger addressing.

**Tier 2 — Text fallback (only when there is no mention and no `reply_to` metadata at all)**

3. **Name detection:** Hush's name or handle appears as a standalone token, detected server-side.
4. **Direct request for Hush's help:** an explicit request that names the assistant's role (e.g., "can the bot / assistant ..."). Rule-based and conservative.

Record which method fired (`address_method`: `mention` | `reply` | `name` | `request` | `none`) and which tier. Tier 2 is the weakest signal, so it is logged separately and tested for false matches (see below).

A message that is a question between two humans, with no Tier 1 or Tier 2 signal, is **not** addressed to Hush and remains eligible for SILENT.

**Required tests for addressing**
- Mention present → addressed (Tier 1), regardless of text.
- Reply to a Hush message → addressed (Tier 1).
- Reply to a human, text contains "hush" → **not** addressed (metadata wins).
- No metadata, "hush" used as an ordinary word ("hush, the baby is asleep") → name-detection false match; document the chosen behavior and test it.
- A participant whose display name is "Hush" or a similar string → must not be confused with the assistant (match on ID, not display name, in Tier 1).
- No metadata, "hey Hush, can you check this" → addressed (Tier 2).
- Human-to-human question with no signals → not addressed.

## 2. Problem Statement

Responding to every message causes unnecessary interruptions, irrelevant replies, extra latency, and wasted generation calls. Hush adds a decision layer before generation.

Two error types must be balanced, and they are **not equally costly**:

- **Unnecessary response (false SPEAK/ASK):** Hush speaks when it should not. Annoying, noisy.
- **Missed contribution (false SILENT):** Hush stays quiet when a response was needed. Usually worse, especially when someone asked it directly.

The system should also recognize when asking is better than guessing.

---

## 3. Goals and Non-Goals

### Goals
1. Build a reliable, tested backend for a conversational AI system.
2. Integrate and evaluate pretrained models rather than training from scratch.
3. Compare three families of decision approaches on identical held-out data: rules, embeddings + classifier, and an LLM-as-decider.
4. Measure decision quality, latency, and generation calls avoided.
5. Ship a reproducible project with tests, CI, documentation, and a working demo.

### Non-Goals (Version 1)
- Training a foundation model from scratch.
- A general-purpose autonomous agent.
- Fine-tuning before a baseline is established.
- Claiming accuracy, speed, or cost wins without measurements.
- Voice, multi-agent behavior, or a complex frontend before the decision pipeline is solid.

---

## 4. Architecture

```text
Client (chat UI)
      |
      v
FastAPI endpoint
      |
      v
Input validation + addressing detection + bounded conversation context (speaker-labeled)
      |
      v
Decision service  (stable interface, swappable implementations)
      |-- RuleBasedDecider        (baseline)
      |-- EmbeddingClassifier     (main model)
      |-- LLMDecider              (baseline / optional cascade)
      |-- Decision policy: thresholds, addressed-by-name rule, fallbacks
      |
      +---- SPEAK ---> Generation service ---> Reply
      +---- ASK ----> Clarification service -> One question
      +---- SILENT -------------------------> No generation call
      |
      v
Response schema + structured logs + metrics
```

**Key rule:** decision and generation are separate components. A decision never triggers generation unless the chosen action requires it.

### Decider interface

```python
class Decider(Protocol):
    def decide(self, context: ConversationContext) -> Decision: ...

@dataclass
class Decision:
    action: Literal["SPEAK", "ASK", "SILENT"]
    confidence: float | None      # None if the method has no meaningful score
    confidence_kind: Literal["calibrated_probability", "uncalibrated_score", "none"]
    method: str                   # "rules" | "embedding_clf" | "llm"
    fallback_used: bool
    latency_ms: float
```

Any implementation can be swapped in without touching the API. A stub decider stays available for tests and local dev.

---

## 5. Technology Stack

| Area | Choice |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI + Pydantic |
| Embeddings | Sentence Transformers (start with `all-MiniLM-L6-v2`; provisional) |
| Classifier | scikit-learn (logistic regression first, then gradient boosting if justified) |
| LLM baseline | Hosted LLM API with structured/JSON output |
| Data | pandas, NumPy |
| Testing | pytest, HTTPX |
| Quality | ruff (lint/format), mypy (optional), GitHub Actions CI |
| Config | Environment variables, `.env.example` |
| Packaging | Pinned dependencies, Docker once locally stable |
| Storage | In-memory for prototype; SQLite/Postgres only if persistence is needed |

---

## 6. Model Strategy

Start with pretrained components. Add complexity only when evidence demands it.

### Stage 1 — Small, regularized classifier (main model)

A single-message embedding compared to labeled examples will struggle, because the right action depends on *context*: who is talking to whom, whether a question was already answered, and how recent it is. But with about 180 development scenarios out of an initial ~300, a model with hundreds of embedding dimensions will overfit easily. So the model starts small and grows only if validation supports it.

**Step 1a — Handcrafted features only (start here)**
- `addressed_to_hush` and `address_method`
- `last_assistant_message_distance` (how recently Hush spoke; whether this follows up on it)
- `has_question_mark` / starts with a question word
- `messages_since_last_question`
- `last_question_already_answered` (a different speaker replied after it)
- `same_speaker_as_previous`
- `speaker_count_in_window`
- `time_gap_since_previous_message` (if timestamps are available)
- `message_length`, `is_repeat_of_recent_message`

Model: **logistic regression with strong L2 regularization** (tune `C` by cross-validation).

**Step 1b — Add embeddings as an ablation, not a default**
- Latest-message embedding only (not the full context window), optionally reduced with PCA to a small number of components.
- Keep embeddings only if they improve **grouped cross-validated** performance beyond the handcrafted-only model by more than the noise level.

**Training discipline**
- **`GroupKFold` by `conversation_id`** so scenarios from the same conversation never appear in both train and validation folds.
- Strong regularization; compare against the majority-class baseline at every step.
- Report feature-ablation results; do not keep features that do not help.
- Treat first results as **preliminary** and label them so in the README. Revisit when the dataset grows.

### Stage 2 — Baselines (all three compared on the same held-out data)

| System | Description |
|---|---|
| **Rules** | Speak if mentioned or a question mark is present; otherwise silent. |
| **Embedding + classifier** | Stage 1 model. |
| **LLM-as-decider** | Prompted to return SPEAK/ASK/SILENT as validated JSON. |

Also report a **majority-class** baseline and an **always-reply** baseline for context.

The central research-style question: *Is a small, fast classifier competitive with an LLM decider, and at what latency and cost?* The LLM baseline is mandatory, not optional.

**Fairness rules for the comparison**
- **Equivalent inputs:** every system receives the same last K messages, the same speaker identities, and the same addressing metadata (including the rules baseline). No system sees information another is denied.
- **Same held-out conversations** for every system, with the same preprocessing.
- **Tune on dev only,** including the LLM prompt. Iterate on the prompt using dev scenarios, freeze it, then run the test set once. Do not under-tune the LLM baseline to make the classifier look better, and do not tune it on test.
- **Cache LLM results** keyed by (scenario ID, prompt version, model version, parameters). Pin the model version and use temperature 0. This controls API cost and makes the evaluation reproducible; log cache hit rates and any non-determinism you observe.
- Record the prompt text and version in the repo alongside results.

### Stage 3 — Improve only with evidence

If errors cluster in a clear weakness, consider better embeddings, richer features, a different classifier, or fine-tuning. Fine-tuning is optional and must be justified by evaluation results.

### Cascade (v1.1, not part of Version 1)

Once the classifier and LLM decider are both evaluated, a cascade can use the classifier when confident and escalate borderline cases to the LLM decider. Report how often escalation happens and the resulting accuracy, latency, and cost trade-off. Do not start this before the core pipeline and evaluation are complete.

---

## 7. Decision Policy and Reliability

- Return only validated actions: SPEAK, ASK, or SILENT.
- **Do not fabricate confidence.** Use it only when the method provides a meaningful score. Raw cosine similarity is not a calibrated probability.
- Classifier probabilities must be checked for calibration before being treated as confidence (reliability diagram, optional Platt/isotonic calibration fitted on dev data only). With a small test set the calibration bins are noisy, so a classifier score may still not earn the label "probability."
- **Never silently ignore a message that is explicitly addressed to Hush, or a direct request for Hush's help** (the addressing rule in Section 1 overrides SILENT). Human-to-human questions are **not** covered by this rule and stay eligible for SILENT.
- Explicit, documented behavior for: invalid model output, timeouts, unavailable dependencies, low confidence.
- Suggested defaults: low confidence → SILENT unless addressed; addressed + low confidence → ASK; service failure → SILENT unless addressed, logged.
- Log fallbacks and failures **without logging message content by default.**
- Bound conversation history; handle missing or malformed context.
- Tests cover edge cases and ambiguous inputs, not just a few hand-picked examples, including human-to-human questions and each addressing method.

### Confidence display

The API and UI must not imply more certainty than the method supports. Each decision carries a `confidence_kind`:

| `confidence_kind` | Meaning | UI treatment |
|---|---|---|
| `calibrated_probability` | Probability that passed a reliability check (e.g., calibrated classifier) | May show as a percentage, with a note about the evaluation set size |
| `uncalibrated_score` | Model score without a calibration check (raw classifier probability, LLM self-reported confidence) | Show as "score," not a percentage; never as "X% sure" |
| `none` | No meaningful score (rules, similarity thresholds, fallbacks) | Show the action and method only; no confidence meter |

Raw similarity values are never displayed as probabilities. Thresholds on uncalibrated scores are tuned on dev data and described as thresholds, not probabilities.

### Cost matrix

Not all errors are equal. Define an explicit cost matrix (e.g., missed direct request > unnecessary reply > wrong ASK/SPEAK choice) and use it to pick thresholds. Document the chosen costs and justify them. Report a **threshold sweep** showing the unnecessary-vs-missed trade-off curve.

---

## 8. Dataset, Labeling, and Evaluation

### 8.1 Dataset

- **Starting target: ~300 labeled scenarios.** Treat this as enough for an initial experiment, not for strong claims. Grow it if the headline comparison turns out to be within noise (see 8.5).
- Sources: hand-written scenarios, LLM-generated scenarios that are **reviewed and relabeled by hand**, and optionally public dialogue/clarification corpora (check licenses; cite sources).
- **Split:** ~60% development (training, feature design, threshold tuning, cross-validation) / ~40% held-out test. The test set is touched **once** for reported numbers. Split by conversation, not by message, to avoid leakage.
- Balance the three classes reasonably; record class counts overall and **per split** and **per category** before running any model.
- Expect ~120 test scenarios at a 60/40 split. That is small: per-category test counts will be around 10, which is too few for category-level conclusions (see 8.5).

### 8.2 Scenario categories

Direct questions and explicit requests; statements needing no response; venting/emotional messages; casual banter; human-to-human questions; ambiguous messages; follow-ups dependent on history; repeated messages; messages addressed to the assistant; clarification-worthy requests; already-answered questions; empty, malformed, or very long inputs.

### 8.3 Labeling guideline (write this first)

Write a one-page guideline before labeling anything. Draft rules:

| Label | Use when |
|---|---|
| **SPEAK** | The assistant is addressed (per the Section 1 rule), OR there is an unanswered request for help that is open to the group (not aimed at a specific human) and the assistant can answer it with the information available. |
| **ASK** | The assistant is (or should be) involved, but a good answer is impossible without a missing detail (which item, which error, which version, etc.). |
| **SILENT** | Human-to-human exchange (including questions one person asks another), banter, venting without a request, already-answered questions, or statements needing no input. |

Tie-break rules to decide in the guideline: apply the Section 1 addressing definition exactly; how to treat a question aimed at a specific human versus one open to the group; what to do with emotional venting that is *directed at* the assistant; how stale an unanswered question can be before it no longer counts.

Labeling is subjective, so:
- Re-label a random ~20% subset a few days later yourself and report self-agreement, or have a friend label a subset and report **Cohen's kappa**.
- Record disagreements and how the guideline was revised.

### 8.4 Scenario schema (`scenarios.jsonl`)

The schema mirrors the API: speaker identities, message IDs, roles, mentions, and replies.

```json
{
  "id": "s-017",
  "conversation_id": "c-009",
  "split": "dev",
  "category": "ambiguous_request",
  "participants": [
    {"speaker_id": "u-alex", "display_name": "alex", "role": "human"},
    {"speaker_id": "u-sam",  "display_name": "sam",  "role": "human"},
    {"speaker_id": "hush",   "display_name": "Hush", "role": "assistant"}
  ],
  "messages": [
    {"message_id": "m1", "speaker_id": "u-alex", "text": "@Hush the build is broken again.", "mentions": ["hush"], "reply_to": null, "ts": 0},
    {"message_id": "m2", "speaker_id": "u-sam",  "text": "Which one?", "mentions": [], "reply_to": "m1", "ts": 14},
    {"message_id": "m3", "speaker_id": "u-alex", "text": "The one from this morning, I guess.", "mentions": [], "reply_to": "m2", "ts": 31}
  ],
  "address_method": "mention",
  "addressed_to_hush": true,
  "expected_action": "ASK",
  "rationale": "Hush was mentioned, but it cannot help without knowing which build and the error."
}
```

`addressed_to_hush` and `address_method` are derived from the Section 1 rule; validate them with a script rather than labeling them by hand, and hand-label only `expected_action`.

### 8.5 Metrics

Report on the held-out test set:

- Overall accuracy; per-action precision, recall, F1; macro-F1
- Confusion matrix
- **Unnecessary response rate** and **missed contribution rate** (always reported together)
- Clarification appropriateness (rubric-based; see below)
- Decision latency (median and p95) per method, with hardware and runtime documented
- Generation calls avoided (count and %) vs. always-reply, **reported next to missed contributions**
- Fallback and failure rates
- Calibration (reliability diagram, Brier score or ECE) for methods that emit probabilities
- Estimated cost per 1,000 decisions, **only** under documented assumptions with current pricing

**Statistical honesty**
- Report **bootstrap 95% confidence intervals** for headline metrics.
- Compare systems with a **paired test** (paired bootstrap or McNemar), since every system is scored on the same examples. This is tighter than comparing two separate intervals.
- At ~120 test examples and ~80% accuracy, the interval is roughly ±7 points. Differences inside that range are not claims; say so explicitly.
- Report **per-category results as descriptive only** (about 10 examples each). No category-level claims.
- Check class balance and per-category counts before interpreting any metric. Macro-F1 can swing heavily on a rare class.
- If the key comparison (classifier vs. LLM decider) is within noise, expand the dataset before concluding anything.
- Use cross-validation on the dev split for model selection.

**Clarification rubric:** an ASK is appropriate if a good answer was impossible without the missing detail and the question targets that detail. Score a sample yourself against the rubric and state the sample size.

### 8.6 Ablations

- Context window size K (e.g., 1, 3, 5, 10)
- Handcrafted features only vs. features + latest-message embedding vs. features + context embedding (grouped cross-validation)
- Embedding model choice (one or two alternatives)
- Classifier type
- Threshold sweep with the cost matrix

### 8.7 Reporting rules

- Same held-out examples, same input context, and same preprocessing for every system.
- Label first results as preliminary and say so in the README.
- No tuning on the test set; no repeated peeking.
- Include dataset size, class balance, limitations, and failure cases.
- Do not claim Hush is more accurate, cheaper, or faster unless the numbers (with intervals) support it.
- No fabricated or rounded-up numbers anywhere.

---

## 9. Backend API

The API is **conversation-aware**: the server owns the history, and each message carries a speaker identity, a message ID, and addressing information.

**Compatibility with the existing prototype:** the current backend accepts a simpler request (`conversation_id`, `user_id`, `message`). Do not replace it in one step. The legacy shape stays supported through an adapter that maps `user_id` to `speaker_id`, generates a `message_id`, and treats `mentions` and `reply_to` as empty (so addressing falls back to Tier 2). Add compatibility tests for the legacy request and response *before* changing any schema. The conversation-aware request is added alongside it (either as a new field set on the same route or a versioned route), and the legacy shape is deprecated explicitly once the demo and tests no longer need it.

### `POST /api/chat/message`

Request:
```json
{
  "conversation_id": "demo-room",
  "message": {
    "message_id": "m42",
    "speaker_id": "u-alex",
    "display_name": "alex",
    "text": "Does anyone know why the deployment failed?",
    "mentions": [],
    "reply_to": null,
    "ts": 1730000000
  }
}
```

- `speaker_id` is required and stable per participant; `display_name` is optional and used for context formatting.
- `mentions` is a list of participant IDs the client detected. This is the reliable addressing path.
- `reply_to` is an optional `message_id`. The server resolves whether it points at one of Hush's own messages.
- If `mentions` and `reply_to` are empty, the server runs name detection as a fallback. The method used is returned as `address_method`.
- The server stores history per `conversation_id` (bounded), including Hush's own replies with `role: "assistant"`. Clients may also pass `history` to seed a conversation (useful for evaluation and tests).

Response:
```json
{
  "conversation_id": "demo-room",
  "message_id": "m42",
  "addressed_to_hush": false,
  "address_method": "none",
  "action": "SILENT",
  "confidence": 0.91,
  "confidence_kind": "uncalibrated_score",
  "reply": null,
  "generation_used": false,
  "decision": {
    "method": "embedding_clf",
    "fallback_used": false,
    "latency_ms": 12.4
  }
}
```

For SPEAK/ASK, `reply` holds the generated text and `generation_used` is `true`. `confidence` is `null` and `confidence_kind` is `"none"` for methods without a meaningful score; clients must follow the display rules in Section 7. Validation errors (missing `speaker_id`, unknown `reply_to`, empty text, oversized text) return structured 4xx responses, not a decision.

### Other endpoints
- `GET /health` — status and active decision mode
- `GET /api/metrics` — counters: messages, decisions by action, generation calls, calls avoided, fallbacks, address-method counts, latency summaries
- `POST /api/metrics/reset` — dev environments only

Keep the contract stable while swapping decision implementations.

---

## 10. Scope and Roadmap

### Version 1 scope (build this first)

- Existing backend audited and extended incrementally (legacy request shape still supported)
- Conversation-aware API with the two-tier addressing rule
- Labeled dataset (~300 starting target) and labeling guideline
- Rules, majority, and always-reply baselines
- Small regularized classifier (handcrafted features first; embeddings as an ablation)
- LLM-as-decider baseline
- Evaluation harness with paired comparisons, bootstrap CIs, confusion matrices
- Tests and CI
- README with honest results

### Deferred until Version 1 is complete

- Demo frontend, side-by-side view
- Docker and deployment
- Cascade (classifier + LLM escalation)
- Fine-tuning, extra embedding models beyond the ablation
- Persistent storage

**Gate rule:** do not start a deferred item until the Version 1 Definition of Done is met. CI and the bootstrap/paired-test code are *not* deferred; both are cheap and go in early.

### Phases

**Phase 0 — Foundations**
- Fix the setting (group chat) and the addressing rule.
- Write `LABELING_GUIDELINE.md` and the scenario schema.
- Repo, CI (tests + ruff), pinned dependencies, `.env.example`.
- *Deliverable:* repo skeleton with passing CI and an agreed schema.

**Phase 1 — Audit and incrementally extend the existing backend**
- **Audit first:** read the current FastAPI app, schemas, services, config, and tests. List what already works. Keep working components; do not rewrite.
- Write **compatibility tests** for the existing request/response behavior and confirm they pass.
- Add the conversation-aware schema, addressing detection, bounded history, and metrics **incrementally**, keeping the existing tests green at each step.
- Keep the stub decider; verify SPEAK/ASK/SILENT routing; **test that SILENT never calls generation.**
- Test addressing with the required cases from Section 1 (mention, reply-to-Hush, reply-to-human with "hush" in text, name false match, display-name confusion, human-to-human question).
- *Deliverable:* extended backend with the legacy tests still passing and the new addressing tests passing.

**Phase 2 — Dataset and baselines**
- Build and label scenarios; split by conversation; check class and category counts.
- Implement rules, majority, and always-reply baselines.
- Write the evaluation script (all metrics, bootstrap CIs, paired tests).
- *Deliverable:* reproducible baseline report.

**Phase 3 — Models**
- Feature extraction + embedding classifier; cross-validate on dev.
- LLM-as-decider with validated structured output.
- Wire both behind the `Decider` interface with timeouts and fallbacks.
- *Deliverable:* three systems running in the same evaluation harness.

**Phase 4 — Analysis**
- Confusion matrices, error analysis, ablations, calibration, threshold sweep.
- Evaluate once on the held-out test set; decide whether the dataset is large enough to support the comparison.
- *Deliverable:* `docs/evaluation.md` with honest results and trade-offs.

**Phase 5 — Portfolio polish (after Version 1 is done)**
- Demo UI with live decision + confidence display and a side-by-side "always-reply vs. Hush" view.
- Docker; optional deployment.
- README headline, architecture diagram, demo GIF, limitations.
- Optional: cascade (v1.1).
- *Deliverable:* public, understandable, reproducible project.

---

## 11. Repository Structure

```text
hush/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── routes/chat.py
│   │   └── services/
│   │       ├── decider.py          # interface + factory
│   │       ├── generator.py
│   │       ├── conversation.py
│   │       └── metrics.py
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
├── ml/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── LABELING_GUIDELINE.md
│   ├── scripts/
│   │   ├── prepare_data.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── src/hush_ml/
│   │   ├── features.py
│   │   ├── embeddings.py
│   │   ├── deciders/               # rules.py, embedding_clf.py, llm.py
│   │   └── evaluation.py
│   └── tests/
├── frontend/                       # demo UI (Phase 5, after Version 1)
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── .github/workflows/ci.yml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

Adapt to the existing repository rather than moving files without reason. Where the existing prototype already has equivalents, keep them and extend; this tree is a target, not a rewrite instruction.

---

## 12. Portfolio Value and Novelty

Hush demonstrates the ability to:
- Build and test a maintainable Python API with CI
- Integrate pretrained models behind clean interfaces
- Separate inference from business logic
- Design a fair evaluation with baselines, ablations, and confidence intervals
- Analyze errors, calibration, and trade-offs
- Handle latency, failures, and resource usage
- Explain limitations and reproduce results

The core concept overlaps with existing work on dialogue turn-taking, clarification, and model abstention. **Do not claim novelty.** The value is in implementation quality, the comparison of a small classifier against an LLM decider, and honest reporting.

### README headline (fill in only with measured numbers)

> On a held-out set of *N* group-chat scenarios, the embedding classifier achieved *X* macro-F1 (95% CI *a–b*) vs. *Y* for the LLM decider, at *Z×* lower median decision latency, avoiding *P%* of generation calls with a missed-contribution rate of *Q%*.

Leave this blank until the numbers exist.

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Dataset too small or noisy for conclusions | ~300 starting target, split by conversation, bootstrap CIs, paired tests, cross-validation, expand if key comparison is within noise, state limits plainly |
| Classifier overfits on ~180 dev scenarios | Start with handcrafted features, strong L2, `GroupKFold` by conversation, embeddings only if they beat the baseline, label results preliminary |
| Unfair baseline comparison | Same inputs and context for all systems, LLM prompt tuned on dev only, frozen before test, cached and versioned LLM calls |
| UI overstates confidence | `confidence_kind` field; percentages shown only for calibrated probabilities; none shown for rules/similarity |
| Schema change breaks the existing prototype | Audit first, legacy adapter, compatibility tests before changes, incremental commits with tests green |
| Ambiguous "addressed to Hush" definition | One rule in Section 1 used by API, features, and guideline; address method logged; dedicated tests |
| Direct-address rule silences or over-triggers on human-to-human questions | Rule limited to messages explicitly addressed to Hush; test both cases |
| Subjective labels | Written guideline, agreement measurement, documented revisions |
| Embeddings miss conversational structure | Hand-crafted context features, window-size ablation |
| LLM-generated data leaks style artifacts | Hand-review and relabel; include hand-written scenarios; note it as a limitation |
| Overfitting to the test set | Test set touched once; all tuning on dev only |
| Uncalibrated confidence misused | Calibration check; use `None` where no meaningful score exists |
| Scope creep (UI, deployment, fine-tuning) | Gate each phase on the previous deliverable |
| API cost or rate limits for the LLM baseline | Cache results per scenario, set a budget cap, document pricing assumptions |

---

## 14. Definition of Done (Version 1)

- [ ] Group-chat setting and labeling guideline documented
- [ ] API accepts a message with speaker identity, message ID, mentions, and reply_to, keeps bounded history, and returns a validated action
- [ ] Existing backend audited; legacy request shape still works and has compatibility tests
- [ ] Addressing rule implemented once with Tier 1 (metadata) authoritative and Tier 2 (text) as fallback, and tested including false matches and display-name confusion
- [ ] SILENT never calls generation (tested)
- [ ] SPEAK and ASK use the correct response paths
- [ ] Decider interface with rules, embedding classifier, and LLM implementations
- [ ] Invalid output, timeout, low-confidence, and service-failure behavior tested
- [ ] ~300 labeled scenarios (starting target) with dev/held-out split by conversation, class and category counts, and agreement measurement
- [ ] Baselines (majority, always-reply, rules) plus classifier and LLM decider evaluated on the same held-out conversations with equivalent input context
- [ ] LLM results cached with pinned model/prompt versions; prompt tuned on dev only and frozen
- [ ] Classifier trained with grouped cross-validation and regularization; first results labeled preliminary
- [ ] Confidence shown in the UI/API only according to `confidence_kind`; no raw similarity displayed as a probability
- [ ] Metrics reported with bootstrap confidence intervals, paired comparisons, confusion matrices, calibration, and the unnecessary-vs-missed trade-off
- [ ] Per-category results labeled descriptive; no claims where differences are within noise
- [ ] Ablations and error analysis documented, including failure cases
- [ ] Results reproducible with a documented command
- [ ] CI passing; secrets not committed; `.env.example` provided
- [ ] README covers setup, architecture, model choice, metrics, limitations, and how to run the evaluation
- [ ] (After the above) demo UI, Docker, and cascade are tracked as v1.1 items, not blockers

---

## 15. Final Direction

Build Hush as a software-first, model-powered system. Fix the problem setting, build a trustworthy dataset, establish baselines, compare a small classifier against an LLM decider, and report what the evidence actually shows, including where it fails. The goal is not to train a model for its own sake. The goal is a system that works, can be evaluated, and can be explained confidently in a technical interview.
