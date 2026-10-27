# Hush — Fresh Holdout Collection Instructions (`holdout_v1`)

## 1. Purpose & Core Objective
The fresh holdout dataset (`holdout_v1`) provides an uncompromised, uncontaminated evaluation benchmark for Hush.
Historical Phase 2 evaluations and leakage audits established that development and historical test splits share conversational stems.
To establish genuine generalization performance:
- Scenarios must be **independently authored from scratch** or collected from real anonymized communication.
- No scenario may be derived, modified, or paraphrased from existing dataset examples in `data/dataset_v1.jsonl`.
- No LLM may be used to fabricate supposedly independent human responses or candidate dialogues.

---

## 2. Target Dataset Composition
The holdout set comprises exactly **50 multi-turn scenarios**:

| Action | Target Count | Target Pct | Scenario Types |
|---|---|---|---|
| **`SILENT`** | 23 | 46.0% | Human-to-human technical questions, lunch/casual chats, metadata conflicts (mentions human with "hush" in text), third-person discussions of Hush |
| **`SPEAK`**  | 17 | 34.0% | Direct mentions `@hush`, replies to Hush messages, explicit role requests with complete context/parameters |
| **`ASK`**    | 10 | 20.0% | Addressed requests that are underspecified, ambiguous, or missing critical operational parameters |
| **Total**    | **50** | **100.0%** | Multi-turn dialogues (2 to 6 turns each) across realistic domains |

---

## 3. Strict Rules for Scenario Authors

### Rule 1: Zero Overlap with Existing Scenarios
Your submissions will be automatically screened against all 300 existing scenarios across three levels:
1. **Latest turn text:** Alphanumeric word Jaccard similarity must be $< 0.70$.
2. **Full dialogue text:** Alphanumeric word Jaccard similarity must be $< 0.70$.
3. **Dialogue prefix stems:** Zero reuse of synthetic opening stems (e.g., "Starting morning code review session", "Reviewing pull request updates", "Investigating service metrics").

### Rule 2: Mandatory Provenance Tracking
Every scenario must explicitly record:
- `author_id`: Real author or collection agent ID (e.g., `author_human_1`).
- `created_at`: ISO 8601 timestamp.
- `source_type`: Must be one of `fresh_human_authored`, `anonymized_chat_export`, or `production_log`.
- `derived_from_existing`: Must be explicitly set to `false`. Any submission marked `true` is automatically rejected.

### Rule 3: Realistic Addressing Edge Cases
Incorporate genuine group chat edge cases:
- **Metadata Conflict:** Colleague clicks "reply" on a human teammate's message and types: `"Hush told me yesterday to check the index, did you verify?"` -> Must be classified as `SILENT` (Tier 1 metadata points to human).
- **Third-Person Reference:** Colleague says: `"Hush seems down today, can anyone ping devops?"` -> Must be `SILENT` (not addressed to Hush).
- **Ambiguous Request:** Colleague says: `@hush run the tests` without specifying repo, branch, or test suite -> Classified as `ASK`.

---

## 4. Submission & Validation Workflow

1. Populate scenarios into `data/holdout/holdout_collection_template.csv` or JSONL.
2. Run validation check:
   ```powershell
   python scripts/prepare_fresh_holdout.py --validate path/to/my_candidates.csv
   ```
3. When the full 50-scenario collection is ready, run target distribution verification:
   ```powershell
   python scripts/prepare_fresh_holdout.py --validate path/to/my_candidates.csv --check-distribution
   ```
4. Once validation passes, candidates proceed to the blinded dual-reviewer protocol (`data/review/`).
5. Only after inter-annotator agreement reaches $\kappa \ge 0.75$ and all disputes are resolved will `data/holdout/holdout_v1.jsonl` be finalized.
