# Hush — The AI That Earns Its Right to Speak

**Hackathon:** Agent Experience Hackathon: Strands Decider (AWS Builder Center)
**Window:** Oct 16, 9:00 AM PT – Oct 23, 2026, **11:59 PM PT** (per the official Terms and Conditions, last updated Oct 9, 2026; the About page shows 12:00 AM PDT for the start, so treat 9:00 AM PT as the official start)
**Team:** 2 members
**Decision pattern tag:** `#policy` (plus the required `#strands-decider-2b`)
**Status:** Final plan, checked against the official Terms and Conditions. Items marked **VERIFY** still need confirming against the real SDK.

---

## 1. Project Overview

**Hush** is a conversational AI participant in a group chat. Instead of replying to every message, it first asks a fast decision model whether it should contribute. **Strands Decider 2B** makes that decision. **Amazon Bedrock** writes text only when the decision calls for it.

> **Core principle:** Decide first. Generate only when needed.

### Problem
Most chat assistants respond too often. That interrupts people, adds noise, increases latency, and burns model calls.

### Solution
For each new message, Hush evaluates the recent conversation and selects one action:

| Action | Meaning |
|---|---|
| **SPEAK** | Generate a helpful reply with Bedrock. |
| **SILENT** | Do nothing. No generation call is made. |
| **ASK** | Generate one short, focused clarifying question with Bedrock. |

Each decision comes with a confidence value. The confidence value controls what Hush does when it is unsure.

### Why this fits Decider 2B
- Decider 2B does not generate text. It reads a state and a typed question and returns a choice, yes/no, or score with a calibrated confidence. That matches "should I speak now?" exactly.
- The decision is high-frequency (every message), so speed and call reduction are visible and measurable.
- Public reports put Decider 2B at 72.3% on 231 public JevBench tasks, with 100% on the easy tier and about 50.5% on the hard tier. So Hush is designed to use Decider for bounded decisions and to use confidence to fall back safely when it is unsure. **VERIFY** current numbers on the model card.

---

## 2. Hackathon Requirements Checklist

Source: the official Terms and Conditions (Rules tab) and the About page.

**Qualification is pass/fail.** A submission missing the article, the repository (with a reusable pattern), or the video does not advance to judging, however good the idea is. The agent must also genuinely use Decider 2B.

**Every submission must include:**
- [ ] A working agent on AWS that uses Decider 2B for at least one meaningful decision, paired with Amazon Bedrock for generation
- [ ] A written article **published on AWS Builder Center** describing the agent, the decision handled by Decider 2B, and what the agent does. Measured latency/cost vs. an LLM, decision accuracy, and low-confidence handling are encouraged (and match our evaluation plan)
- [ ] A **public repository with an OSI-approved open source license** containing the source code and at least one reusable, **clearly labeled** decision pattern: the typed question(s) and option set, the wiring that places the decision in the agent's path, and a short note on what it does and how to use it
- [ ] A demo video **under 3 minutes** (judges need not watch beyond 3:00; aim for 2:50 or less), uploaded and **publicly visible on YouTube, Vimeo, Facebook Video, or Youku**, with the link in the submission. **The decision must be shown firing with its confidence score.** No third-party trademarks or copyrighted music/material without permission
- [ ] One declared decision pattern (ours: **Policy**)
- [ ] Both tags: `#strands-decider-2b` and `#policy`
- [ ] An original agent that has not been previously published

**Judging:** Gate 1 is AI plus human scoring, and the **top 100** submissions advance. Gate 2 is a panel of AWS experts and community leaders who pick **3 winners**. The announcement may also include a crowd favorite and recognition for the most-reused or best decision pattern (no cash value), which is another reason to polish `hush_gate`.

**Rubric → how we address it**

| Criterion | Weight | Our approach |
|---|---|---|
| Creativity and cool factor | 30% | Live confidence meter, side-by-side "always-reply bot vs. Hush", restraint counter |
| Effective use of Decider 2B | 30% | Real decision gate, reusable pattern package, ablation, comparison with Bedrock-as-decider, measured latency/call reduction |
| Communication quality | 25% | Clear article + tight video, honest results including failures |
| Originality | 15% | "An AI that knows when to stay quiet" as the decision, not another classifier or chatbot |

**Eligibility (from the Terms)**
- [ ] Both members are 18+ and have a builder.aws.com profile
- [ ] Neither member resides in an excluded country or region: Argentina, Australia, Brazil, Hong Kong, Indonesia, Italy, Malaysia, the Philippines, Vietnam, Singapore, Russia, Thailand, Cuba, Iran, North Korea, Syria, Belarus, Crimea, the so-called DNR and LNR regions, and the United Arab Emirates
- [ ] Neither member is an Amazon/AWS employee or an immediate family/household member of one
- [ ] Finalists may be asked to verify eligibility (e.g., via LinkedIn); misrepresentation can lead to disqualification even after winners are announced

**Open questions to resolve with the organizers (Discussion tab)**
- The Terms say "limit one entry per person" but do not describe team submissions or how a prize is split between two members. Ask early how a 2-person team should enter, and decide which account submits.
- The Terms say to build during the Submission Period (starts Oct 16, 9:00 AM PT). Keep pre-Oct-16 work to setup and research (see Section 11), and ask if you want to do more.
- "On AWS" is not defined beyond Bedrock being required; confirm what is expected for where Decider runs.

**Conduct:** no unlawful or infringing content, no malicious code, nothing offensive or harmful to others or AWS. By submitting, you license the repo and decision pattern for community use under the chosen OSI license. Winners also owe tax documentation.

---

## 3. Decision Design (the core of the project)

### 3.1 The gate
Hush runs one decision per trigger. The recommended trigger for the MVP is **a new message**. A timer-based trigger (an unanswered question after N seconds) is a stretch goal.

### 3.2 Two designs to compare (ablation)

**Design A — single 3-way question**

> State: last K messages (with speaker labels) + whether the latest message mentions Hush.
> Question: "Should the assistant respond now?"
> Options: `SPEAK`, `SILENT`, `ASK`

**Design B — decomposed yes/no questions**

1. "Is the assistant being addressed directly?" → yes/no
2. "Is there an unanswered question or request for help in the recent messages?" → yes/no
3. "Is the available information sufficient to answer?" → yes/no (distinguishes SPEAK from ASK)

Then a small rule table maps the answers to SPEAK / SILENT / ASK.

Decider is strongest on easy, bounded questions, so Design B may beat Design A. Run both on the held-out test set and report the result either way. The winner becomes the reusable pattern; the other becomes an article section.

> The exact wording above is a **draft**. Confirm the real typed-question and option-set format from the official Decider example before writing code.

### 3.3 Confidence policy
Use calibrated confidence from the model, not invented scores. Thresholds are tuned on the **development set only**.

| Condition | Behavior |
|---|---|
| Directly addressed by name | Never choose SILENT. Prefer SPEAK, or ASK if confidence is low. |
| Confidence ≥ τ_high | Follow the decided action. |
| τ_low ≤ confidence < τ_high | Follow SILENT/ASK; for SPEAK, optionally ask a clarifying question instead. |
| Confidence < τ_low | Default to SILENT (and log as "low confidence"). |
| Model error or invalid output | Default to SILENT, log the error, show a status in the UI. |

Optional cascade (if time permits): for borderline confidence, ask Bedrock to make the decision and record that a fallback was used. This directly demonstrates "Decider handles the easy cases, the big model handles the hard ones."

### 3.4 What Decider does not give us
- It does **not** generate explanations. A free-text `reason` field must not be faked. If we want a reason, it should come from a separate typed question with its own option set (e.g., `ADDRESSED`, `UNANSWERED_QUESTION`, `NOT_RELEVANT`, `SOCIAL_BANTER`), or we omit it.
- Confidence must come from the supported model output. Never fabricate it. Document exactly how it is obtained.

---

## 4. Evaluation Plan

### 4.1 Dataset
- Target: **100–150 labeled scenarios** (more is better, quality matters most). Each scenario is a short conversation snippet ending at the message to be decided on.
- Split: **~40% development** (used to design questions and tune thresholds) and **~60% held-out test** (used only for reported numbers). Never tune on the test set.
- Cover: direct questions to Hush, questions between humans, venting, banter, repeated messages, ambiguous requests, requests for help, long silences, messages where interrupting would be rude, and edge cases.
- **Both members label** a shared subset independently to measure labeling agreement, then resolve disagreements.

Suggested `scenarios.jsonl` row:

```json
{
  "id": "s-017",
  "split": "dev",
  "category": "ambiguous_request",
  "messages": [
    {"speaker": "alex", "text": "The build is broken again."},
    {"speaker": "sam", "text": "Which one?"},
    {"speaker": "alex", "text": "The one from this morning, I guess."}
  ],
  "addressed_to_hush": false,
  "expected_action": "ASK",
  "annotator_notes": "Hush can help only if it knows which build and the error."
}
```

### 4.2 Systems compared on the same scenarios

| System | Description |
|---|---|
| **Always-reply** | Bedrock generates a reply to every message. |
| **Heuristic** | Reply only if the message contains a question mark or mentions Hush. |
| **Bedrock-as-decider** | Bedrock is prompted to choose SPEAK/SILENT/ASK before any reply. This is the key comparison for "effective use of Decider". |
| **Hush (Decider 2B)** | Decider makes the decision; Bedrock generates only for SPEAK/ASK. |

Run Hush in both Design A and Design B.

### 4.3 Metrics

- **Decision accuracy** vs. labeled expected action (overall and per category)
- **Unnecessary response rate** (spoke when silence was expected)
- **Missed contribution rate** (silent when a response was expected)
- **ASK appropriateness** — judged by both members using a written rubric (e.g., "ASK is appropriate if a good answer was impossible without the missing detail"). Report agreement.
- **Generation-call reduction** vs. always-reply baseline
- **Decision latency** (median and p95) for Decider vs. Bedrock-as-decider
- **End-to-end response latency**
- **Estimated cost** under documented assumptions, only if reliable pricing data is available (check current Bedrock pricing)
- **Calibration check** — plot accuracy against confidence bins on the test set, to show whether confidence is meaningful in this task

### 4.4 Reporting rules
- Do not claim fewer calls means better performance. Always report missed contributions next to call reduction.
- Document hardware, runtime, model version, and test setup for every latency figure.
- Include failure cases and what they teach us.
- No fabricated or rounded-up numbers anywhere (article, video, README).

---

## 5. Team Roles

Both members must understand the full architecture and be able to explain exactly how Decider 2B is used.

### Member 1 — Backend, Agent, and Chat UI
- FastAPI app, schemas, validation, chat endpoints
- Conversation state and context preparation
- Integrate the decision module through the agreed interface
- Route SPEAK / SILENT / ASK; **SILENT must never call generation**
- Bedrock integration (reply and clarifying-question generation)
- Metrics logging and endpoints
- Chat UI with the **live confidence meter**, restraint counter, and the side-by-side comparison view
- Backend tests, error handling, `.env.example`, setup docs

### Member 2 — Decision Engineering and Evaluation
- Run the official Decider 2B example; confirm real API, output format, and confidence handling
- Implement the Decider wrapper and the **reusable `hush_gate` package** (see Section 8)
- Design A vs. Design B; threshold tuning on the dev set
- Build the dataset, evaluation script, baselines, and calibration analysis
- Measure latency and document hardware/runtime
- Failure analysis

### Shared
- Both label the dataset (with an overlapping subset for agreement)
- Both review each other's code and test the integrated app
- Write and review the README and the Builder Center article together
- **Video:** one person writes the script, the other records and edits; both rehearse
- Keep credentials out of Git; verify final submission against live rules

### Interface contract (proposed)

```python
def decide_action(messages: list[dict]) -> dict:
    """
    Input: recent conversation messages.
    Example output:
    {
        "action": "SILENT",              # SPEAK | SILENT | ASK
        "confidence": 0.91,              # from the model output; do not invent
        "fallback_used": False,
        "decision_latency_ms": 118
    }
    """
```

This is an application-level contract. Adapt the wrapper to the real Decider 2B API and document how each field is obtained.

---

## 6. Architecture

```text
Message arrives
      |
      v
Chat UI  ->  FastAPI backend
               - validate input
               - update conversation state
               - build decision context (last K messages)
      |
      v
Strands Decider 2B  (typed question + options -> choice + confidence)
      |
      v
Confidence policy (thresholds from dev set; addressed-by-name rule; error fallback)
      |
      +-----------------+------------------+
      |                 |                  |
      v                 v                  v
    SPEAK             SILENT              ASK
      |                 |                  |
      v                 v                  v
Bedrock writes     No generation      Bedrock writes one
a reply            Log decision       clarifying question
      |                                    |
      +------------------+-----------------+
                         |
                         v
        Chat UI (live confidence meter) + metrics dashboard
```

Where Decider runs (laptop vs. cloud GPU instance) must be decided early. The ~115 ms figure was reported on an RTX 3090; test actual latency on your hardware in the first days. **VERIFY** what counts as "on AWS" in the rules.

---

## 7. API Contract (proposed)

### `POST /api/chat/message`

Request:
```json
{
  "conversation_id": "demo-room",
  "user_id": "user-1",
  "message": "Does anyone know why the deployment failed?"
}
```

Response (ASK example):
```json
{
  "conversation_id": "demo-room",
  "action": "ASK",
  "confidence": 0.84,
  "reply": "Do you have the deployment log or error message?",
  "generation_used": true,
  "fallback_used": false,
  "decision_latency_ms": 121
}
```

For SILENT: `reply` is `null` and `generation_used` is `false`.

### `GET /api/metrics`
Aggregate counts: messages evaluated, decisions by action, Bedrock generation calls, calls avoided vs. always-reply baseline, low-confidence and fallback counts, decision and generation latency summaries.

### `GET /health`
Simple health status.

---

## 8. Reusable Decision Pattern: `hush_gate`

This is a required deliverable and part of the "effective use" score. Build it as a standalone package that works **without FastAPI**.

**Contents**
- The typed question(s) and option set(s) (Design A and/or B)
- The confidence policy (thresholds, addressed-by-name rule, fallback behavior)
- The wiring that puts the gate in the agent's path (the rules mention a Strands `InterventionHandler` as an example — **VERIFY** the current interface in the Strands docs)
- `README` usage note: install, minimal example, how to change options and thresholds, known limitations
- A tiny example agent (≤ 30 lines) showing the gate in front of a Bedrock reply

Goal: another builder can drop "should my agent respond now?" into any chat agent in minutes.

---

## 9. Repository Structure

```text
hush/
├── hush_gate/                  # reusable decision pattern (standalone)
│   ├── __init__.py
│   ├── questions.py            # typed questions + option sets
│   ├── policy.py               # confidence policy
│   ├── handler.py              # Strands wiring (InterventionHandler or equivalent)
│   └── README.md               # how to reuse
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── routes/chat.py
│   │   └── services/
│   │       ├── agent.py
│   │       ├── decider.py      # calls hush_gate
│   │       ├── conversation.py
│   │       ├── bedrock.py
│   │       └── metrics.py
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── frontend/                   # chat UI + live meter + comparison view
├── evaluation/
│   ├── scenarios.jsonl
│   ├── evaluate.py
│   ├── baselines.py
│   └── results/
├── docs/
│   ├── architecture.md
│   ├── demo-script.md
│   └── article-draft.md
├── .gitignore
├── LICENSE                     # required: OSI-approved license
└── README.md
```

Keep scope tight. No voice, no multi-agent behavior, no large simulation until the core loop and the evaluation work.

---

## 10. Demo and Video (under 3 minutes)

**Creative hook:** a live confidence meter beside the chat, so judges *see* Hush deciding before it speaks, plus a side-by-side view of an always-reply bot and Hush on the same conversation.

| Time | Segment |
|---|---|
| 0:00–0:20 | The problem: an always-reply bot talking over a human conversation |
| 0:20–1:10 | Hush on the same conversation: stays silent, then speaks or asks at the right moment, meter visible |
| 1:10–1:50 | Measured results: calls avoided, latency vs. Bedrock-as-decider, unnecessary/missed response rates |
| 1:50–2:20 | The reusable `hush_gate` pattern in a few lines of code |
| 2:20–2:50 | A real failure case and how confidence/fallback handled it; what we learned |

Rules for the demo:
- Use real model decisions and measured numbers only; no mock decisions presented as Decider output.
- **The confidence score must be visible on screen each time the decision fires** (the live meter covers this).
- No third-party logos/trademarks or copyrighted music. Use no music or royalty-free/own audio.
- Upload publicly to YouTube, Vimeo, Facebook Video, or Youku and test the link while logged out.

---

## 11. Timeline

Today: Oct 13. The Submission Period opens Oct 16, 9:00 AM PT and closes Oct 23, 11:59 PM PT — **aim to submit by Oct 22.** Submitting after the deadline is grounds for disqualification.

| Dates | Goals |
|---|---|
| **Oct 13–15** (prep only) | Terms read. Check eligibility for both members. Run the official Decider example, confirm Bedrock access and costs, test Decider latency on your hardware, agree on the interface, and ask the organizers the open questions. The Terms say to build during the Submission Period, so keep this to setup and research. |
| **Oct 16–17** | End-to-end skeleton: message → Decider → SPEAK/SILENT/ASK → Bedrock → UI. Start labeling (both members). Backend tests for routing. |
| **Oct 18–19** | Run baselines and Design A vs. B. Tune thresholds on dev set. Failure analysis. Build `hush_gate` package cleanly. |
| **Oct 20–21** | Final test-set run (once). Polish UI and meter. Write video script, record, edit. |
| **Oct 22** | Article published on Builder Center, README, video uploaded publicly (link tested logged out), both tags, final checks, **submit.** |
| **Oct 23** | Buffer only. |

---

## 12. Important Engineering Decisions

- Trigger decisions on new messages (or meaningful events); do not re-run inference on unchanged state.
- Keep the decision module behind a stable interface so backend and ML work proceed in parallel.
- **Never present a mock decision as a real Decider 2B result.**
- Do not invent confidence values; document how they are derived.
- Explicit fallback for model failure or invalid output (default SILENT + logged).
- Low-confidence behavior is part of the tests.
- Credentials in environment variables or an AWS profile; commit only a safe `.env.example`.
- Do not log credentials, tokens, or unnecessary personal data.
- Check current AWS pricing; set a budget alert and usage limits.
- No cost/speed claims until measured under a documented setup.

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Decider is weak on ambiguous cases (about 50% on hard tier in public reports) | Use bounded questions, Design B, confidence thresholds, default-SILENT fallback, optional Bedrock cascade |
| No GPU / slow local inference | Test latency on day 1; use a cloud GPU instance if needed; document hardware |
| Labeled dataset too small or biased | Both members label; overlap subset for agreement; dev/test split |
| Scope creep (voice, simulations) | Stretch goals only after the Definition of Done items for the core loop are met |
| Last-day submission problems | Target Oct 22; buffer on Oct 23 |
| Team entry / prize split unclear under "one entry per person" | Ask organizers in the Discussion tab early; decide the submitting account |
| Terms change before the deadline | Recheck the Rules tab on Oct 22 (current version: last updated Oct 9, 2026) |
| Missing a pass/fail part (article, repo + pattern, video) | Treat all three as blocking items on the Oct 22 checklist |

---

## 14. Definition of Done

- [ ] Terms read; eligibility confirmed for both members; team-entry question resolved
- [ ] Official Decider 2B integration working with real outputs
- [ ] Amazon Bedrock generation working
- [ ] SPEAK, SILENT, and ASK paths tested
- [ ] SILENT never calls the generation model (tested)
- [ ] Conversation context passed correctly
- [ ] Low-confidence, invalid-output, and service-failure behavior tested
- [ ] Dataset documented with dev/test split and labeling agreement
- [ ] All four systems evaluated; Design A vs. B reported
- [ ] Results include failures and calibration check
- [ ] `hush_gate` works standalone with README and example
- [ ] Setup instructions work on a clean environment
- [ ] No secrets committed
- [ ] Public repo with an OSI-approved license, article published on Builder Center, video (<3 min) public on an allowed platform with the confidence score visible and no third-party trademarks/music, and both tags (`#strands-decider-2b`, `#policy`) all verified before the deadline

---

## 15. Resources

- Hackathon page: https://builder.aws.com/build/hackathons/4f50957c-1d08-37ab-b11c-97516b9adeb4/agent-experience-hackathon-strands-decider
- Official Terms and Conditions (Rules tab): https://builder.aws.com/build/hackathons/4f50957c-1d08-37ab-b11c-97516b9adeb4/agent-experience-hackathon?tab=rules
- Decider announcement: https://strandsagents.com/blog/introducing-strands-decider/
- Strands Agents docs: https://strandsagents.com/
- Amazon Bedrock docs: https://docs.aws.amazon.com/bedrock/
- Install reference reported publicly: `pip install strands-decider`, then serve the model with `strands-decider serve` — **VERIFY** the current command and model version on the official model card.

---

*All endpoints, schemas, thresholds, and typed-question wording in this document are proposals. Confirm them against the live rules and the real SDK before implementation.*
