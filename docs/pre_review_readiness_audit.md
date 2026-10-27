# Hush — Pre-Review Readiness Audit and Evaluation Integrity Report

**Date:** October 27, 2026  
**Auditor:** Senior Applied AI/ML & Software Backend Pair  
**Review Status:** `PENDING_HUMAN_REVIEW` (Labels Remain Unverified Ground Truth)  
**Holdout Status:** `PENDING_COLLECTION_AND_REVIEW` (Specification Defined, Data Not Fabricated)  
**Test Suite Baseline:** 175 passed in 2.50s  

---

## 1. Executive Summary

This pre-review readiness audit comprehensively evaluates the state of the Hush Conversational Restraint Engine codebase, review artifacts, evaluation methodology, and production invariants prior to human reviewer engagement.

**Core Findings:**
1. **Human-Review Readiness Confirmed:** Dual reviewer packets and CSV forms are completely blinded. Zero gold labels, original rationales, scenario categories, or model predictions are exposed. The 60 scenarios are identical and in matching order.
2. **Defensive Validation Operational:** Submission ingestion scripts defensively detect missing answers, invalid actions, duplicate scenario IDs, mismatched scenario sets, and manifest discrepancies. Cohen's kappa and 3x3 confusion matrices are computed strictly from submitted data. Disagreements are logged to [`data/review/disputes_v1.json`](../data/review/disputes_v1.json).
3. **Evidence Integrity Strictly Preserved:** Review status remains `PENDING_HUMAN_REVIEW` (0 / 60 scenarios completed). No reviewer responses, timestamps, agreement figures, or synthetic decisions were fabricated.
4. **Critical Epistemic Risks Documented:**
   - **Template & Prefix Leakage:** 51 scenario pairs across `dev` and `test` exhibit word Jaccard similarity $\ge 0.70$ due to synthetic prompt templates and shared multi-turn conversation prefixes.
   - **Degenerate Grouping:** `conversation_id` is 1-to-1 with `scenario_id` (300 unique IDs for 300 scenarios); therefore `GroupKFold` is mathematically identical to standard $K$-Fold and provides zero protection against template leakage.
   - **Quarantine of Historical Test Set:** The 120-scenario test set was evaluated in Phase 2 and cannot serve as an untouched holdout.
5. **Production Invariants Intact:** `DevelopmentStubDecider` is active in API routing. SILENT strictly guarantees 0 generator calls and `generation_used = false`.

---

## 2. Review Packet & CSV Form Readiness

All reviewer artifacts in [`data/review/`](../data/review) were audited programmatically and verified:

| Artifact | Format | Item Count | Blinding Verification | Annotation Fields Present | Status |
|---|---|---|---|---|---|
| [`reviewer_1_packet.jsonl`](../data/review/reviewer_1_packet.jsonl) | JSONL | 60 | Zero gold labels, rationales, categories, or predictions. | `reviewer_action`, `is_ambiguous`, `reviewer_confidence`, `reviewer_rationale`, `reviewer_comments` (all blank/null) | Ready |
| [`reviewer_2_packet.jsonl`](../data/review/reviewer_2_packet.jsonl) | JSONL | 60 | Identical 60 scenarios in identical sequence as Reviewer 1. | Same 5 annotation fields (all blank/null) | Ready |
| [`reviewer_1_form.csv`](../data/review/reviewer_1_form.csv) | CSV | 60 | Only transcript, domain context, and addressing signals shown. | Blank columns for action, ambiguity, confidence, rationale, comments | Ready |
| [`reviewer_2_form.csv`](../data/review/reviewer_2_form.csv) | CSV | 60 | Identical 60 rows in identical order as Form 1. | Same blank columns | Ready |
| [`review_guidelines.md`](../data/review/review_guidelines.md) | Markdown | — | Clear SPEAK, ASK, SILENT criteria and edge-case matrix. | Mandatory instruction to work independently without prior discussion | Ready |
| [`disputes_v1.json`](../data/review/disputes_v1.json) | JSON | 0 disputes | Dedicated ledger for disagreements requiring adjudication. | Schema version 1.0.0; status: `pending_human_review` | Ready |

### Review Guidelines & Rubric Alignment
The guidelines in [`data/review/review_guidelines.md`](../data/review/review_guidelines.md) and [`docs/annotation_rubric_v2.md`](annotation_rubric_v2.md) unambiguously define:
- `SPEAK`: When addressed with complete information or open group technical inquiry.
- `ASK`: When addressed but underspecified, ambiguous, or missing crucial parameters.
- `SILENT`: Human-to-human discussion, targeted human colleague questions, banter, emotional venting without direct request, already-answered inquiries, or false-positive name usage.
- **Inviolable Addressing Invariant:** A message addressed to Hush must **never** be resolved to `SILENT`.
- **Independence Instruction:** Explicitly instructs annotators: *"Do NOT discuss decisions with the other reviewer before submission."*

---

## 3. Human-Review Status & Quality-Gate State

- **Current Audit Status:** `PENDING_HUMAN_REVIEW`
- **Reviewer 1 Progress:** 0 / 60 scenarios completed (0.0%)
- **Reviewer 2 Progress:** 0 / 60 scenarios completed (0.0%)
- **Disputes Ledger:** 0 disputes logged (awaiting submissions)
- **Quality Gate Threshold:**
  $$\kappa \ge 0.75 \quad \text{AND} \quad \text{Unresolved Disputes} = 0$$
- **Current Quality Gate Determination:** `PENDING` (Provisional research results only; models cannot be benchmarked or promoted until human review passes).

---

## 4. Evaluation-Integrity Audit: Issues & Evidence

During in-depth inspection of the evaluation harness, feature pipeline, split files, and calibration scripts, four notable issues were identified:

### Issue 1: Synthetic Prompt Template & Multi-Turn Prefix Leakage across Splits
- **Severity:** **HIGH**
- **Evidence:** Analysis of word Jaccard similarity between all 180 development scenarios and all 120 test scenarios revealed **51 cross-split pairs** with Jaccard similarity $\ge 0.70$.
  - *Example 1:* `s-036` (dev) vs `s-030` (test) have identical conversational opening:
    - Turn 1: `u-dev1: Working on the implementation draft.` (identical)
    - Turn 2: `u-dev2: assistant, what is the difference between __str__ and __repr__ in Python?` vs `assistant, what is the difference between mutable and immutable types in Python?`
  - *Example 2:* `s-197` (dev) vs `s-193` (test) share identical multi-turn prefixes:
    - Turn 1: `u-alex: @hush how do I filter a dictionary in Python?` (identical)
    - Turn 2: `hush: You can use a dictionary comprehension: {k: v for k, v in d.items() if condition(v)}.` (identical)
    - Turn 3: `What happens if the input dictionary is empty in that case?` vs `What is the time complexity of that solution?`
- **Impact:** High vocabulary and structural overlap artificially inflates generalization performance when evaluating models trained on `dev` against `test`.
- **Remediation:** Evaluators must recognize `data/splits/test.jsonl` as compromised by template overlap and post-hoc exposure. The true benchmark must rely solely on the fresh holdout (`holdout_v1`), which enforces a hard Jaccard threshold $< 0.70$ against all 300 existing scenarios.

### Issue 2: Degenerate Conversation ID Grouping
- **Severity:** **HIGH**
- **Evidence:** Inspection of `data/dataset_v1.jsonl` showed that all 300 scenarios have a unique `conversation_id` (`c-001` through `c-300`).
- **Impact:** Scenarios that are logically continuation turns of the same dialog (e.g. `s-197` and `s-193`) were assigned distinct conversation IDs. Consequently, `GroupKFold(groups=conversation_id)` degenerates to standard unstratified $K$-Fold, failing to keep related conversation threads in the same fold.
- **Remediation:** In Version 2 dataset curation, thread lineage must be captured with true thread IDs rather than 1-to-1 scenario identifiers.

### Issue 3: In-Sample Calibration Parameter Selection (Now Corrected)
- **Severity:** **MEDIUM**
- **Evidence:** In [`scripts/analyze_calibration.py`](../scripts/analyze_calibration.py), temperature scaling parameter $T^*$ was optimized using the entire set of 180 out-of-fold logits, and then evaluated on those exact same 180 logits, reporting an in-sample ECE drop from $0.4076$ to $0.1104$ at $T=0.20$.
- **Impact:** Represented an empirical bound rather than a strictly validated out-of-sample calibration.
- **Correction Made:** Added nested 5-fold cross-validation inside `scripts/analyze_calibration.py`. In each fold, $T$ is selected strictly on the remaining 4 training folds and evaluated on the held-out validation fold. All 5 inner folds converged to $T^* = 0.20$, verifying that true nested out-of-sample ECE is **0.1104** and nested NLL is **0.3773**. Both metrics are now explicitly saved in `data/results/phase4_calibration.json` with clear leakage-free labels.

### Issue 4: Historical Test Split Exposure
- **Severity:** **MEDIUM**
- **Evidence:** The 120 scenarios in `data/splits/test.jsonl` were evaluated during Phase 2 baseline runs.
- **Impact:** Treating `data/splits/test.jsonl` as an untouched holdout would violate evaluation hygiene.
- **Mitigation:** The historical test set is permanently quarantined as a baseline reference. `data/holdout/holdout_spec.md` specifies a brand-new 50-scenario holdout that remains uncollected.

---

## 5. Changes Made & Regression Tests Added

### Code Improvements
1. **Submission Ingestion & Mismatched Set Detection:**
   Updated [`scripts/validate_review_manifest.py`](../scripts/validate_review_manifest.py):
   - Added verification that Reviewer 1 and Reviewer 2 evaluate identical scenario ID sets (`error_type: "mismatched_scenario_sets"`).
   - Added verification that submitted scenario sets match manifest scenario IDs (`error_type: "manifest_mismatch"`).
   - Added explicit detection of missing answers / blank action fields in submissions (`error_type: "missing_answers"`).
   - Added automated dispute ledger output to [`data/review/disputes_v1.json`](../data/review/disputes_v1.json).
2. **Nested Out-Of-Sample Temperature Scaling:**
   Updated [`scripts/analyze_calibration.py`](../scripts/analyze_calibration.py) to perform nested 5-fold CV for temperature estimation, ensuring zero parameter leakage.
3. **Disputes Ledger Initialization:**
   Created [`data/review/disputes_v1.json`](../data/review/disputes_v1.json) initialized with `pending_human_review` status.

### Regression Tests Added ([`backend/tests/test_phase6_review_execution.py`](../backend/tests/test_phase6_review_execution.py))
- `test_dispute_ledger_file_writing`: Verifies disagreements between reviewers are properly formatted and written to the external JSON ledger.
- `test_mismatched_scenario_sets_detection`: Verifies submissions with diverging scenario sets are rejected with `mismatched_scenario_sets`.
- `test_missing_answers_detection`: Verifies CSV forms with empty/blank reviewer actions are identified and flagged with `missing_answers`.

---

## 6. Production Invariants Verification

The production API was verified to adhere to all architectural constraints:
- **Active Decider:** [`backend/app/services/decider.py`](../backend/app/services/decider.py#L130-L135) strictly loads `DevelopmentStubDecider` in local/offline mode.
- **Learned Classifier Isolation:** The logistic regression classifier is **not** imported or invoked in API routing.
- **Silence Invariant:** In [`backend/app/routes/chat.py`](../backend/app/routes/chat.py#L133-L137):
  - When `action == Action.SILENT`, `reply = None` and `generation_used = False`.
  - Generator functions `generate_reply` and `generate_clarification` are **never** invoked.
- **Contract & Legacy Compatibility:** Full backwards compatibility with legacy requests and responses is verified by passing tests in `test_legacy_compatibility.py` and `test_conversation_aware.py`.

---

## 7. Full Test Suite Verification

Executed using the project virtual environment:
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
$env:PYTHONPATH="backend"
& .\backend\.venv\Scripts\python.exe -m pytest backend/tests -p no:cacheprovider -v
```

**Result:**
```text
======================= 175 passed, 1 warning in 2.50s =======================
```
- Total test modules: 14
- Total test cases: 175
- Pass rate: **100% (175/175)**

---

## 8. Remaining Blockers & Safe Next Steps

### Blockers Before Benchmark Finalization
1. **Independent Human Review Execution:** Real human reviewers must independently complete `reviewer_1_form.csv` and `reviewer_2_form.csv`.
2. **Quality Gate Requirement:** Cohen's kappa must achieve $\kappa \ge 0.75$, and all logged disputes must be resolved.
3. **Fresh Holdout Collection:** The 50-scenario fresh holdout (`holdout_v1`) must be collected, verified against Jaccard template similarity, and dual-annotated.

### Exact Next Steps Once Real Submissions Arrive
1. Place completed reviewer CSV files in `data/review/` (or specified paths).
2. Execute the ingestion and validation command:
   ```powershell
   python scripts/validate_review_manifest.py --r1 data/review/reviewer_1_form.csv --r2 data/review/reviewer_2_form.csv
   ```
3. Check [`data/review/disputes_v1.json`](../data/review/disputes_v1.json) for any identified disagreements ($R_1 \ne R_2$).
4. Have an arbitrator adjudicate disputed scenarios and record consensus gold actions in `review_manifest_v1.json`.
5. Run [`scripts/summarize_review_status.py`](../scripts/summarize_review_status.py) to generate the finalized review status report.
6. Only after the quality gate passes ($\kappa \ge 0.75$ and 0 unresolved disputes) may the verified dataset be used for model benchmarking.
