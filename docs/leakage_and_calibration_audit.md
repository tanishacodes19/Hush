# Hush — Leakage and Calibration Methodology Audit (Verified)

**Date:** October 27, 2026  
**Auditor:** Senior Applied AI/ML & Software Backend Pair  
**Review Status:** `PENDING_HUMAN_REVIEW` (Labels Remain Unverified Ground Truth)  
**Holdout Status:** `PENDING_COLLECTION_AND_REVIEW` (Specification Defined; Data Uncollected)  
**Test Suite Baseline:** 176 passed in 2.63s  

---

## 1. Verified Findings

### 1.1 Dataset Leakage & Prefix Overlap Verification
- **Cross-Split Pair Reproduction:**
  - Under whitespace-delimited word tokenization (`text.lower().split()`), exactly **51 cross-split scenario pairs** exhibit word Jaccard similarity $\ge 0.70$.
  - Under alphanumeric word tokenization (`re.findall(r"\w+", text.lower())`), exactly **48 cross-split scenario pairs** exhibit word Jaccard similarity $\ge 0.70$.
  - Maximum cross-split Jaccard similarity reaches **0.818**.
- **Dialogue Prefix Accounting & Normalization:**
  - *Definition:* For any scenario with $\ge 2$ messages, the dialogue prefix is defined as the ordered tuple of all messages prior to the final message:
    $$\text{Prefix} = \big( (s_1, m_1), (s_2, m_2), \dots, (s_{k-1}, m_{k-1}) \big)$$
    where $s_j$ is the `speaker_id` and $m_j$ is the raw message text.
  - *Multi-Turn Scenarios ($\ge 2$ turns):*
    - Exactly **14 distinct multi-turn prefixes** are shared between `dev.jsonl` and `test.jsonl`.
    - Exactly **118 out of 180 development scenarios (65.6%)** share an exact multi-turn prefix with test scenarios.
    - Exactly **83 out of 120 test scenarios (69.2%)** share an exact multi-turn prefix with development scenarios.
  - *Total Scenarios Accounting:*
    - When single-turn scenarios (which possess no preceding conversational history) are grouped under a shared single-turn bucket, the shared count rises to **146 / 180 dev (81.1%)** and **105 / 120 test (87.5%)**.
- **Structural Origin of Overlap:**
  - The 14 shared prefixes represent synthetic conversational stems (e.g., `Debugging the afternoon release`, `Incident triage in progress`, `@hush how do I filter a dictionary in Python?`).
  - Variations branching from each stem were authored to test distinct actions (`SPEAK`, `ASK`, `SILENT`).
  - Because each scenario was assigned an independent `conversation_id` (`c-001` through `c-300`), the split generator partitioned them independently, scattering variants of identical conversational stems across both splits.

---

### 1.2 Calibration Integrity & Logit Data Flow Verification
- **Data Flow Separation in Nested Cross-Validation:**
  - **Outer Fold Isolation:** In each outer fold $k \in \{0, 1, 2, 3, 4\}$, model weights ($W^{(k)}, b^{(k)}$) are fitted strictly on the 144 training instances. The 36 validation instances are never observed during model training.
  - **Temperature Selection Isolation:** In inner nested cross-validation, the temperature parameter $T^{(k)}$ is selected strictly using the 144 out-of-fold logits of the training folds. The outer validation fold's logits are withheld.
  - **Evaluation Ordering:** Outer-fold calibrated predictions are generated only *after* $T^{(k)}$ selection is complete.
  - **No In-Sample Contamination in Nested Metrics:** Held-out predictions from each validation fold are evaluated out-of-sample and aggregated without mixing in-sample selection scores.
- **Reproduced Metrics Across Temperatures:**
  - **$T = 1.00$ (Uncalibrated Baseline):**
    - Evaluated on all 180 outer-fold logits.
    - $\text{NLL} = \mathbf{0.8018}$, $\text{ECE} = \mathbf{0.4076}$, Brier Score = $\mathbf{0.4615}$.
    - Genuine out-of-sample estimate of uncalibrated performance.
  - **$T = 0.20$ (Boundary-Constrained Grid $[0.20, 3.00]$):**
    - *Historical In-Sample:* Single-pass fit on all 180 logits $\implies \text{NLL} = 0.3773$, $\text{ECE} = 0.1104$.
    - *Nested Out-of-Sample:* All 5 inner folds select $T^{(k)} = 0.20$ (hitting the lower boundary of the grid) $\implies \text{Nested NLL} = \mathbf{0.3773}$, $\text{Nested ECE} = \mathbf{0.1104}$.
  - **$T = 0.10$ (Unconstrained Grid $[0.05, 3.00]$):**
    - *Full-Dev Empirical Fit:* $\text{NLL} = 0.3301$, $\text{ECE} = 0.0574$.
    - *Nested Out-of-Sample:* All 5 inner folds independently select $T^{(k)} = 0.10$ $\implies \text{Nested NLL} = \mathbf{0.3301}$, $\text{Nested ECE} = \mathbf{0.0574}$.
- **Calibration vs. Accuracy Distinction:**
  - Temperature scaling scales logits by a positive scalar $1/T$. It preserves the argmax order:
    $$\frac{z_c}{T} > \frac{z_{c'}}{T} \iff z_c > z_{c'} \quad (\forall T > 0)$$
  - Classification accuracy is **strictly invariant** to temperature scaling: **86.7%** (156 / 180) across $T=1.00$, $T=0.20$, and $T=0.10$.
  - The improvement from $T=0.10$ is strictly in probability calibration (reliability) and sharpness (NLL), driven by counteracting the logit shrinkage caused by $L_2$ regularization ($\lambda = 0.5$).

---

### 1.3 Holdout Overlap Checker Verification
- The overlap tool [`scripts/prepare_fresh_holdout.py`](../scripts/prepare_fresh_holdout.py) was audited and verified:
  - Compares every candidate scenario against all 300 master scenarios.
  - Evaluates latest incoming turn, full concatenated dialogue, and conversational prefix turns.
  - Accurately catches exact prefix reuse even when final turns differ.
  - Threshold boundary behavior: Exactly $J = 0.70$ triggers rejection; $J = 0.60$ passes. Empty messages are handled safely without errors.

---

## 2. Provisional Historical Results

The following metrics are documented as **provisional research artifacts** subject to unverified gold labels and template leakage limitations:

| Metric / Result | Historical Reported Value | Methodological Status | Provenance Limitation |
|---|---|---|---|
| **Development Accuracy** | 86.7% (156 / 180) | Evaluated on `dev.jsonl` | Labels assigned programmatically without independent dual annotation. |
| **Development Macro-F1** | 0.833 | Evaluated on `dev.jsonl` | Provisional pending human review. |
| **Test Set Accuracy** | 72.5% (Rule-Based Baseline) | Evaluated on `test.jsonl` | Historical test set shares 14 multi-turn prefixes with dev split (69.2% of test scenarios). |
| **In-Sample ECE at $T=0.20$** | 0.1104 | Evaluated on 180 dev logits | Fit in-sample with respect to $T$ on all 180 logits; search grid artificially truncated at $T \ge 0.20$. |

---

## 3. Unresolved Methodological Concerns

1. **Test Split Compromised for Generalization Claims:**
   Because 69.2% of multi-turn test scenarios reuse synthetic stems from the development set, performance on `data/splits/test.jsonl` reflects in-distribution interpolation across prompt variations rather than true out-of-distribution conversational restraint.
2. **Degenerate Grouping in Dataset Schema:**
   Every scenario in `dataset_v1.jsonl` has a unique `conversation_id`. Future dataset versions must introduce a true `thread_id` or `stem_id` to prevent grouped splitting tools from scattering related scenarios across folds.
3. **Unverified Master Labels:**
   All master labels in `dataset_v1.jsonl` remain provisional until the independent dual-review process is completed and arbitrated.

---

## 4. Code Changes Made & Regression Tests Added

1. **Enhanced Holdout Overlap Tool ([`scripts/prepare_fresh_holdout.py`](../scripts/prepare_fresh_holdout.py)):**
   - Upgraded `check_holdout_overlap` to calculate latest message similarity, full dialogue similarity, and conversational prefix similarity.
   - Accurately tags `overlap_type` (`"latest_message"`, `"full_dialogue"`, `"dialogue_prefix"`).
   - Synchronized exclusion criteria in [`data/holdout/holdout_spec.md`](../data/holdout/holdout_spec.md).
2. **Regression Tests Added ([`backend/tests/test_phase5_integrity.py`](../backend/tests/test_phase5_integrity.py)):**
   - `test_holdout_overlap_detection`: Extended to verify multi-turn prefix leakage detection.
   - `test_holdout_overlap_threshold_boundaries`: Verifies exact boundary behavior ($J=0.70$ triggers; $J=0.60$ passes; empty candidate handled cleanly).

---

## 5. Full Test Suite Verification

Executed using the project virtual environment:
```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
$env:PYTHONPATH="backend"
& .\backend\.venv\Scripts\python.exe -m pytest backend/tests -p no:cacheprovider -v
```

**Result:**
```text
======================= 176 passed, 1 warning in 2.63s =======================
```
All 176 tests across 14 test modules pass cleanly.

---

## 6. Next Actions

1. **Deliver Blinded Forms:** Provide [`data/review/reviewer_1_form.csv`](../data/review/reviewer_1_form.csv) and [`data/review/reviewer_2_form.csv`](../data/review/reviewer_2_form.csv) to real human reviewers.
2. **Maintain Status:** Keep manifest review status as `PENDING_HUMAN_REVIEW` until both completed forms are returned.
3. **Execute Fresh Holdout Collection:** Collect 50 genuinely novel multi-turn scenarios adhering to the multi-level overlap filter and independent dual-review gate specified in Section 4.
4. **Single-Use Final Benchmark:** Benchmark baseline deciders and ML classifier against `holdout_v1` strictly once after review arbitration has finalized ground truth.
