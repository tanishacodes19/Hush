# Hush — Fresh Holdout Collection & Dual-Review Instructions (`holdout_v1`)

## 1. Purpose & Generalization Integrity
The fresh holdout dataset (`holdout_v1`) provides an uncompromised, single-use final generalization benchmark for the Hush Conversational Restraint Engine.

Evaluation audits established that previous development splits and synthetic scenarios shared conversational stems. To establish genuine generalization:
- Scenarios must be **freshly authored by independent human contributors** or collected from real anonymized communication.
- No scenario may be derived, modified, or paraphrased from existing dataset examples in `data/dataset_v1.jsonl`.
- No LLM may be used to fabricate supposedly independent human responses or candidate dialogues.
- The holdout dataset will be evaluated exactly **once** upon project finalization.

---

## 2. Target Dataset Composition (Prospective Targets)
The target holdout set comprises exactly **50 multi-turn scenarios**.

> [!NOTE]
> The numbers below are **prospective targets** for collection planning. No scenarios are pre-populated or fabricated; collection status remains `PENDING` until human review completes.

| Action | Prospective Target | Target Pct | Scenario Description & Addressing Characteristics |
|---|---|---|---|
| **`SILENT`** | 23 | 46.0% | Human-to-human technical questions, lunch/casual chats, metadata conflicts (mentions human with "hush" in text), third-person discussions of Hush |
| **`SPEAK`**  | 17 | 34.0% | Direct mentions `@hush`, replies to Hush messages, explicit role requests with complete context/parameters |
| **`ASK`**    | 10 | 20.0% | Addressed requests that are underspecified, ambiguous, or missing critical operational parameters |
| **Total**    | **50** | **100.0%** | Multi-turn dialogues (2 to 6 turns each) across realistic engineering & collaboration domains |

---

## 3. Strict Rules for Scenario Authors

### Rule 1: Zero Overlap with Existing Scenarios (Three-Level Screening)
Submissions are automatically screened against all 300 existing scenarios across three levels:
1. **Latest turn text:** Alphanumeric word Jaccard similarity must be $< 0.70$.
2. **Full dialogue text:** Alphanumeric word Jaccard similarity must be $< 0.70$.
3. **Dialogue prefix stems:** Zero reuse of synthetic opening stems (e.g., *"Starting morning code review session"*, *"Reviewing pull request updates"*, *"Investigating service metrics"*).

### Rule 2: Mandatory Provenance Tracking
Every scenario must explicitly record:
- `author_id`: Real author or collection agent ID (e.g., `author_human_1`).
- `created_at`: ISO 8601 timestamp (e.g., `2026-10-28T10:00:00Z`).
- `source_type`: Must be one of `fresh_human_authored`, `anonymized_chat_export`, or `production_log`.
- `derived_from_existing`: Must be explicitly set to `false`. Any submission marked `true` is automatically rejected.

### Rule 3: Realistic Addressing Edge Cases
Incorporate genuine group chat edge cases:
- **Metadata Conflict:** Colleague clicks "reply" on a human teammate's message and types: `"Hush told me yesterday to check the index, did you verify?"` -> Must be classified as `SILENT` (Tier 1 metadata points to human).
- **Third-Person Reference:** Colleague says: `"Hush seems down today, can anyone ping devops?"` -> Must be `SILENT` (not addressed to Hush).
- **Ambiguous Request:** Colleague says: `@hush run the tests` without specifying repo, branch, or test suite -> Classified as `ASK`.

---

## 4. End-to-End Holdout Collection & Dual-Review Protocol

```
+------------------------------------+
| 1. Author 50 Candidate Scenarios   | (CSV or JSONL with full provenance)
+------------------------------------+
                  |
                  v
+------------------------------------+
| 2. Candidate Validation & Overlap  | (python scripts/prepare_fresh_holdout.py --validate ... --check-distribution)
+------------------------------------+
                  |
                  v
+------------------------------------+
| 3. Blinded Review Packaging        | (python scripts/prepare_fresh_holdout.py --package-review ... --output-dir ...)
|    (Labels & Rationales Stripped)  | -> reviewer_1_form.csv, reviewer_2_form.csv, holdout_review_manifest.json
+------------------------------------+
                  |
                  v
+------------------------------------+
| 4. Dual Blind Human Review         | (Reviewer 1 & Reviewer 2 annotate independently)
|    & Agreement Audit (kappa >= 0.75)| (python scripts/validate_review_manifest.py ... --save-manifest)
|    & Adjudicate Disputes           |
+------------------------------------+
                  |
                  v
+------------------------------------+
| 5. Finalize Holdout Dataset        | (python scripts/prepare_fresh_holdout.py --finalize-holdout ...)
|    (Quality Gate Enforced)         | -> data/holdout/holdout_v1.jsonl
+------------------------------------+
```

### Step 1: Scenario Authoring
Populate candidate scenarios using either:
- CSV: [`data/holdout/holdout_collection_template.csv`](holdout_collection_template.csv)
- JSONL: [`data/holdout/holdout_candidate_template.jsonl`](holdout_candidate_template.jsonl)

### Step 2: Candidate Validation & Overlap Screening
Verify candidate schema, provenance, and 3-level overlap against master scenarios:
```powershell
python scripts/prepare_fresh_holdout.py --validate data/holdout/my_candidates.csv --check-distribution
```

### Step 3: Package Blinded Review Forms
Generate blinded forms where candidate actions and candidate rationales are stripped:
```powershell
python scripts/prepare_fresh_holdout.py --package-review data/holdout/my_candidates.csv --output-dir data/holdout/review
```
Outputs:
- `data/holdout/review/reviewer_1_form.csv` & `reviewer_2_form.csv`
- `data/holdout/review/reviewer_1_packet.jsonl` & `reviewer_2_packet.jsonl`
- `data/holdout/review/holdout_review_manifest.json`

### Step 4: Dual Blind Human Review & Agreement Audit
Two independent reviewers label `reviewer_1_completed.csv` and `reviewer_2_completed.csv` following [`docs/annotation_rubric_v2.md`](../../docs/annotation_rubric_v2.md).
Audit agreement and log any disputes:
```powershell
python scripts/validate_review_manifest.py data/holdout/review/holdout_review_manifest.json `
    --r1 data/holdout/review/reviewer_1_completed.csv `
    --r2 data/holdout/review/reviewer_2_completed.csv `
    --disputes-path data/holdout/review/disputes_v1.json `
    --save-manifest
```
- **Agreement Threshold:** Cohen's $\kappa \ge 0.75$.
- **Dispute Adjudication:** Any conflicting labels are reviewed and adjudicated by the project owner.

### Step 5: Finalize Holdout Dataset
Once the review manifest passes the quality gate ($\kappa \ge 0.75$ and 0 unresolved disputes):
```powershell
python scripts/prepare_fresh_holdout.py --finalize-holdout `
    --candidates data/holdout/my_candidates.csv `
    --manifest data/holdout/review/holdout_review_manifest.json `
    --output-file data/holdout/holdout_v1.jsonl
```

---

## 5. Security & Privacy Notice
All private reviewer forms, completed submissions, dispute logs, and the finalized `holdout_v1.jsonl` are strictly excluded from Git tracking via `.gitignore`. Never commit raw reviewer identities, dispute deliberations, or holdout benchmark data to public repositories.
