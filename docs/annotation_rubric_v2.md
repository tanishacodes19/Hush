# Hush — Conversational Restraint Annotation Rubric & Review Guidelines (v2.0)

**Project:** Hush — Conversational Restraint Engine  
**Protocol:** Independent Dual Annotation & Adjudication ($\kappa \ge 0.75$)

---

## 1. Core Actions & Concrete Decision Rules

Review each multi-turn group-chat scenario chronologically. Examine the conversation history and the final incoming message to select the single most appropriate action for **Hush** (the assistant):

### Action 1: `SPEAK`
- **Definition:** Generate a direct, helpful answer immediately.
- **When to choose:**
  1. **Direct Request:** The incoming message directly addresses Hush (`@hush`, name invocation, or replying to Hush) with sufficient technical details, parameters, or logs to answer.
  2. **Group Technical Question:** An open technical request directed to the channel without naming or replying to a specific human, where an assistant is expected to provide documentation or solution.
  3. **Critical Rule:** When `addressed_to_hush == True`, choose `SPEAK` if the query can be answered without needing missing user parameters.

### Action 2: `ASK`
- **Definition:** Generate a single, clarifying question; do NOT guess or hallucinate.
- **When to choose:**
  1. **Underspecified Problem:** Hush is addressed, but critical parameters are missing (e.g. *"@hush the build failed, can you fix it?"* without build ID, branch, or error log).
  2. **Ambiguous Referent / Stale Follow-up:** A user asks *"Can you deploy that one?"* or *"Same error as before"* where prior conversation history leaves multiple possible targets.
  3. **Choice Clarification:** A request where multiple conflicting options exist (e.g. *"Deploy staging"* when both US and EU staging environments exist).
  4. **Addressing Rule:** When `addressed_to_hush == True`, if you cannot provide a precise solution without guessing, choose `ASK`.

### Action 3: `SILENT`
- **Definition:** Exercise conversational restraint; make ZERO generation calls.
- **When to choose:**
  1. **Human-to-Human Discussion:** Two or more human colleagues conversing, debating, or questioning each other.
  2. **Targeted Human Questions:** Questions explicitly naming or mentioning a colleague (e.g. *"Sam, did you push the commit?"*, *"@alex where is the key?"*), even if technical keywords appear.
  3. **Casual Banter & Status:** Social chat, greetings (*"good morning"*, *"thanks"*), signing off, emojis, or memes.
  4. **Emotional Venting Without Request:** Frustration expressed about code or tools (e.g. *"I hate legacy code!"*) without directly summoning the assistant.
  5. **Already-Answered Questions:** A question was asked earlier, but a human colleague already answered it satisfactorily in preceding turns.
  6. **False-Positive Name Usage:** The word "hush" used as an ordinary verb or noun (*"a hush fell over the room"*, *"keep it hush-hush"*).

---

## 2. Inviolable Project Invariant
> **THE ADDRESSING INVARIANT:**  
> **A message addressed to Hush or directly requesting Hush's help must NEVER be silently ignored.**  
> If `addressed_to_hush == True` (via mention, reply, name, or role request), `SILENT` is strictly forbidden. You must choose either `SPEAK` (if answerable) or `ASK` (if ambiguous/missing info).

---

## 3. Concrete Edge-Case Disambiguation Matrix

| Scenario Type | Key Indicators | Prescribed Gold Action | Decision Justification |
|---|---|---|---|
| **Direct Request to Hush** | Mentions `@hush` or *"Hush, ... "* with complete query | `SPEAK` | Addressed and actionable. |
| **Ambiguous / Incomplete Request** | Mentions `@hush` or *"Hush, ... "* but lacks error log/parameters | `ASK` | Addressed but requires clarification before answer. |
| **Human Question in Group** | *"Sam, did you check the staging logs?"* or `@alex where is the PR?` | `SILENT` | Targeted to a human colleague. Bot must stay silent. |
| **Unaddressed Open Question** | *"Does anyone know why port 8080 is blocked?"* | `SPEAK` (or `SILENT` if answered) | Open question open to the group. |
| **Already-Answered Question** | Question asked at m1, but colleague answered at m2 | `SILENT` | Problem already solved; bot intrusion creates noise. |
| **Emotional Venting at Hush** | *"Hush, your previous answer was completely wrong."* | `SPEAK` | Direct critique of assistant behavior; requires acknowledgment. |
| **Casual Social Banter** | *"Thanks Hush!"* or *"Good morning everyone"* | `SILENT` | Zero-call silence is preferred for routine acknowledgments. |
| **Name False Positive** | *"Let's keep this hush-hush until the release."* | `SILENT` | Ordinary English usage; not an invocation of the assistant. |
| **Metadata Conflict** | Text mentions "hush", but `@alex` is explicitly mentioned | `SILENT` | Authoritative metadata overrides display text. Targeted to human. |

---

## 4. Submission Instructions
1. Review the 60 scenarios in your assigned form (`reviewer_1_form.csv` or `reviewer_2_form.csv`).
2. Fill out `reviewer_action`: `"SPEAK"`, `"ASK"`, or `"SILENT"`.
3. Fill out `is_ambiguous`: `"YES"` or `"NO"`.
4. Fill out `reviewer_confidence`: `"high"`, `"medium"`, or `"borderline"`.
5. Provide a 1-sentence `reviewer_rationale`.
6. Submit completed form to the evaluation coordinator. Do NOT discuss decisions with the other reviewer before submission.
