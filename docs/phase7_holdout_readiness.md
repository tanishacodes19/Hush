# Hush — Phase 7: Fresh Holdout Collection & Validation Readiness Report

**Status:** Tooling and Quarantine Ready; Candidate Collection Pending  
**Date:** October 2026  
**Auditor / Engineer:** Antigravity AI Assistant  
**Repository:** `c:\Users\Admin\Documents\Hush`

---

## 1. Executive Summary & Objective

Phase 7 establishes the infrastructure, collection templates, multi-level leakage prevention, and governance rules required to gather and validate a **brand-new, independent 50-scenario holdout set** (`holdout_v1`).

### Why a Fresh Holdout is Required
In Phase 2 and Phase 3, an initial 120-scenario test split was evaluated historically. Subsequent audits revealed:
1. **Cross-Split Template Leakage:** 51 development-test pairs shared word Jaccard similarity $\ge 0.70$.
2. **Shared Dialogue Prefix Stems:** 14 multi-turn opening templates were shared across splits.
3. **Optimistic Bias:** Re-evaluating models on the historical test set risks test-set overfitting and invalidates generalization claims.

To obtain an uncompromised final evaluation of Hush's Conversational Restraint Engine, a fresh 50-scenario holdout must be collected independently, screened for zero overlap, and dual-reviewed by humans.

### Fundamental Integrity Invariants Enforced
- **Zero Synthetic Fabrication:** No synthetic scenarios, LLM labels, or fake annotator submissions have been generated.
- **Quarantine Invariant:** `data/holdout/holdout_v1.jsonl` does **NOT** exist. It will only be created after genuine independent scenarios are authored, screened, and verified through dual-human review.
- **Production API Protection:** Production routing strictly retains `DevelopmentStubDecider`. The learned classifier remains in offline research mode. `SILENT` actions strictly make zero generation calls.
- **Historical Data Preservation:** All existing datasets, splits, checksums, and Phase 6 reviewer artifacts remain untouched.

---

## 2. Multi-Level Overlap Verification Tooling

The verification engine in [`scripts/prepare_fresh_holdout.py`](file:///c:/Users/Admin/Documents/Hush/scripts/prepare_fresh_holdout.py) implements three complementary layers of similarity screening against all 300 existing scenarios in `data/dataset_v1.jsonl`:

| Level | Scope | Mechanism | Rejection Threshold |
|---|---|---|---|
| **1. Latest Turn** | Incoming message triggering decision | Lowercase alphanumeric word Jaccard similarity | $J \ge 0.70$ |
| **2. Full Dialogue** | Entire conversation history concatenated | Concatenated dialogue word Jaccard similarity | $J \ge 0.70$ |
| **3. Conversational Prefix** | All turns preceding the incoming message | Dialogue stem word Jaccard similarity | $J \ge 0.70$ |

### Verification of Boundary Behavior
The overlap detection implementation was verified with targeted unit tests in [`backend/tests/test_phase7_holdout.py`](file:///c:/Users/Admin/Documents/Hush/backend/tests/test_phase7_holdout.py):
- **Exact & Near Matches:** Identical or paraphrased messages ($J \ge 0.70$) are flagged with the exact matched master scenario ID and text preview.
- **Prefix Leakage Detection:** Scenarios reusing existing multi-turn stems with novel endings are caught and rejected as `dialogue_prefix` leaks.
- **Boundary Precision:** Pairs at exactly $J = 0.70$ trigger rejection; clean domain-specific dialogues ($J < 0.70$) pass cleanly.

---

## 3. Candidate Schema & Mandatory Provenance

To eliminate unverified or synthetic entries, every candidate scenario must include strict provenance metadata:

```json
{
  "id": "holdout-cand-001",
  "domain": "cloud_infrastructure",
  "provenance": {
    "author_id": "author_human_1",
    "created_at": "2026-10-27T10:00:00Z",
    "source_type": "fresh_human_authored",
    "derived_from_existing": false,
    "collection_method": "template_submission"
  },
  "participants": [
    {"speaker_id": "u-lead", "display_name": "Lead", "role": "human"},
    {"speaker_id": "hush", "display_name": "Hush", "role": "assistant"}
  ],
  "messages": [
    {"message_id": "m1", "speaker_id": "u-lead", "text": "Postgres replica lag is 0ms.", "mentions": [], "reply_to": null},
    {"message_id": "m2", "speaker_id": "u-lead", "text": "@hush summarize migration logs", "mentions": ["hush"], "reply_to": null}
  ],
  "addressed_to_hush": true,
  "address_method": "mention",
  "candidate_action": "SPEAK",
  "is_ambiguous": false,
  "candidate_rationale": "Directly addressed with clear diagnostic question."
}
```

### Mandatory Provenance Rules
1. **`author_id`:** Non-empty identifier of the independent human author or data collector.
2. **`created_at`:** ISO 8601 creation timestamp.
3. **`source_type`:** Must be one of `fresh_human_authored`, `anonymized_chat_export`, `production_log`, or `independently_authored`.
4. **`derived_from_existing`:** Must be explicitly `false`. Any submission marked `true` is automatically rejected by the validator.
5. **Multi-Turn Requirement:** Must feature between 2 and 6 turns of realistic conversation. Single-turn queries are rejected.

---

## 4. Collection Templates and Authoring Guidelines

The following artifacts have been created in `data/holdout/` to facilitate authoring:

1. **[`data/holdout/holdout_collection_template.csv`](file:///c:/Users/Admin/Documents/Hush/data/holdout/holdout_collection_template.csv):**  
   Tabular collection template designed for spreadsheets (Excel, Google Sheets) or CSV editors. Includes sample rows illustrating `SILENT`, `SPEAK`, and `ASK` without overlapping master records.
2. **[`data/holdout/holdout_candidate_template.jsonl`](file:///c:/Users/Admin/Documents/Hush/data/holdout/holdout_candidate_template.jsonl):**  
   JSON Lines candidate format matching Hush's internal schemas.
3. **[`data/holdout/collection_instructions.md`](file:///c:/Users/Admin/Documents/Hush/data/holdout/collection_instructions.md):**  
   Comprehensive authoring instructions covering addressing edge cases, metadata conflicts, and ambiguous questions.
4. **[`data/holdout/holdout_spec.md`](file:///c:/Users/Admin/Documents/Hush/data/holdout/holdout_spec.md):**  
   Specification covering inclusion/exclusion criteria, target counts, and quarantine rules.
5. **[`data/holdout/holdout_manifest_template.json`](file:///c:/Users/Admin/Documents/Hush/data/holdout/holdout_manifest_template.json):**  
   Manifest template initialized with status `pending_collection_and_review`.

### Target Scenario Distribution

| Action | Target Count | Percentage | Description & Required Edge Cases |
|---|---|---|---|
| **`SILENT`** | 23 | 46.0% | Human-to-human inquiries, lunch/casual chats, metadata conflicts (`reply_to` pointing to human colleague while text contains "hush"), and third-person references to Hush. |
| **`SPEAK`**  | 17 | 34.0% | Direct mentions (`@hush`), replies to Hush messages, explicit role requests with complete operational parameters. |
| **`ASK`**    | 10 | 20.0% | Addressed queries lacking essential context, environment, or parameters (e.g., `"@hush run the deploy script"` without specifying service or cluster). |
| **Total**    | **50** | **100.0%** | Multi-turn dialogues (2–6 turns) across diverse technical domains. |

---

## 5. Holdout Quarantine & Governance Protocol

1. **Quarantine Directory:** All holdout assets are strictly restricted to `data/holdout/`.
2. **Pipeline Inaccessibility:** Training pipelines ([`train_classifier.py`](file:///c:/Users/Admin/Documents/Hush/scripts/train_classifier.py)), feature extraction ([`features.py`](file:///c:/Users/Admin/Documents/Hush/backend/app/services/features.py)), calibration diagnostics ([`analyze_calibration.py`](file:///c:/Users/Admin/Documents/Hush/scripts/analyze_calibration.py)), and threshold sweeps ([`analyze_thresholds.py`](file:///c:/Users/Admin/Documents/Hush/scripts/analyze_thresholds.py)) are forbidden from importing or reading from `data/holdout/`.
3. **Dual Human Review Gate:** Candidate scenarios must undergo the Phase 6 blinded dual-reviewer protocol. The holdout is finalized only after achieving Cohen's kappa $\ge 0.75$ and resolving all disputes.
4. **Single-Use Final Benchmark:** Once finalized, `holdout_v1` is evaluated exactly **once** to report unbiased final performance metrics in the portfolio README.

---

## 6. Dataset Immutability & Test Suite Verification

### Checksum Verification
All existing master datasets, development/test splits, and completed Phase 6 reviewer submissions were verified bit-for-bit unchanged:

| File Path | SHA256 Checksum | Status |
|---|---|---|
| `data/dataset_v1.jsonl` | `64bd1f39e1bc589361b2281537d384f59470ff28309a39d416c0b5f77b5bc0fb` | Verified Unchanged |
| `data/splits/dev.jsonl` | `d2b1928c0fe8db97ba5efab35fd6e0d2ebd07de71a6d6e984e1cb5346f861aa6` | Verified Unchanged |
| `data/splits/test.jsonl` | `f0c7d4284e3c8df32d0d72021b2ce746f2c56fe4e32c1abd9183f56a0a5ff543` | Verified Unchanged |
| `data/review/reviewer_1_completed.csv` | `f832f322394d79a52f5c0d012ce25f31e7485e6da74343802b6f0c86e3e54e52` | Verified Unchanged |
| `data/review/reviewer_2_completed.csv` | `26ad48438f89d1e013c75991fcbcd415e0e939c61ef329f0872e5ccc6ccd95fc` | Verified Unchanged |
| `data/review/disputes_v1.json` | `3caa443514c40b1141058e12e1cbdb9c1c81797e24b7e7c58df4a244acc94fb1` | Verified Unchanged |

### Full Test Suite Execution
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
$env:PYTHONPATH="backend"
& .\backend\.venv\Scripts\python.exe -m pytest backend/tests -p no:cacheprovider -v
```

**Result:**
```text
======================= 182 passed, 1 warning in 4.01s =======================
```
All 182 tests across 15 test suites pass cleanly.

---

## 7. Blockers & Required Next Steps

### Current Blocker
- **Awaiting Independent Human Authoring:** The holdout cannot be completed until human authors independently write 50 multi-turn candidate scenarios.

### Action Items for Human Contributors
1. **Scenario Authoring:** Populate 50 scenarios in [`data/holdout/holdout_collection_template.csv`](file:///c:/Users/Admin/Documents/Hush/data/holdout/holdout_collection_template.csv) or JSONL following [`data/holdout/collection_instructions.md`](file:///c:/Users/Admin/Documents/Hush/data/holdout/collection_instructions.md).
2. **Execute Validation Gate:**
   ```powershell
   python scripts/prepare_fresh_holdout.py --validate data/holdout/my_candidates.csv --check-distribution
   ```
3. **Execute Dual Blind Review:** Package validated scenarios into blinded packets and collect independent reviews.
4. **Finalize Manifest & Holdout:** After achieving $\kappa \ge 0.75$ and resolving disputes, generate `data/holdout/holdout_v1.jsonl`.
5. **Run Single-Use Benchmark:** Benchmark baseline systems and the restraint classifier once against `holdout_v1`.
