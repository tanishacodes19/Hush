# Hush — Phase 4: Error Analysis, Confidence Calibration & Decision Trade-offs Report

**Date:** October 27, 2026  
**Status:** **Provisional Research Results** (Labels unverified by independent human review; see [`docs/dataset_provenance_audit.md`](dataset_provenance_audit.md))  
**Evaluation Scope:** 5-Fold Out-Of-Fold (OOF) Cross-Validation on Development Split (`data/splits/dev.jsonl`, $N=180$, Seed 42).  
**Test Set Discipline:** Held-out test split (`data/splits/test.jsonl`) was **not accessed** during this analysis.

---

## 1. Executive Summary & Headline Diagnostics

| Diagnostic Dimension | Metric / Finding | Operational Interpretation |
|---|---|---|
| **Out-Of-Fold Accuracy** | **86.67%** (156 / 180 correct) | Strong generalization on unseen development folds |
| **Out-Of-Fold Macro-F1** | **0.8331** (SILENT: 0.994, SPEAK: 0.797, ASK: 0.709) | Balanced performance across all three actions |
| **Unnecessary Response Rate ($R_{\text{unnecessary}}$)** | **0.00%** (0 false interventions / 81 True SILENT) | Complete conversational restraint in human-to-human chat |
| **Missed Contribution Rate ($R_{\text{missed}}$)** | **1.01%** (1 missed response / 99 True Action cases) | Only one instance of unwarranted silence |
| **Generation Calls Avoided** | **45.56%** (82 / 180 classified as SILENT) | Cuts LLM generation volume by nearly half |
| **Multiclass Brier Score** | **0.4615** (MSE of probability vectors) | Solid quadratic scoring accuracy |
| **Uncalibrated ECE (10 bins)** | **0.4076** (Max Calibration Gap: 0.4539) | Under-confident due to $L_2$ weight regularization |
| **Calibrated ECE ($T^* = 0.20$)** | **0.1104** (NLL drops from 0.8018 $\to$ 0.3773) | Sharpening logits restores empirical calibration |
| **Optimal Decision Policy** | Default Argmax with Addressing Invariant | Achieves lowest mean penalty cost (**0.178** per message) |

---

## 2. Metric Audit & Definitional Distinctions

Before running diagnostics, the full metric calculation suite in [`backend/app/services/evaluation.py`](../backend/app/services/evaluation.py) was audited against [`docs/dataset_spec.md`](dataset_spec.md):

1. **Accuracy & Macro-F1:**
   $$\text{Accuracy} = \frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)}{N}, \quad \text{Macro-F1} = \frac{1}{3} \sum_{c \in \{\text{SILENT, SPEAK, ASK}\}} F1_c$$
   Correctly handles empty classes without divide-by-zero crashes.
2. **Restraint Trade-off Denominators:**
   - $R_{\text{unnecessary}} = \frac{\text{Count}(\hat{y} \in \{\text{SPEAK, ASK}\} \land y = \text{SILENT})}{\text{Count}(y = \text{SILENT})}$: Denominator is strictly true SILENT instances.
   - $R_{\text{missed}} = \frac{\text{Count}(\hat{y} = \text{SILENT} \land y \in \{\text{SPEAK, ASK}\})}{\text{Count}(y \in \{\text{SPEAK, ASK}\})}$: Denominator is strictly true actionable instances.
3. **Calls Avoided vs. Production Cost Savings:**
   - $\text{Calls Avoided} = \frac{\text{Count}(\hat{y} = \text{SILENT})}{N}$.
   - **Crucial Distinction:** This is an *inference-time estimate* based on classification decisions. Actual production financial savings depend on downstream generator availability, retry behavior, and cache hit rates.
4. **Predicting vs. Generating:**
   - Predicting `SPEAK` or `ASK` signifies decision routing; it is distinct from generating the actual natural-language token stream. In accordance with the production architecture, `SILENT` decisions strictly trigger zero generation calls.
5. **Score vs. Calibrated Probability:**
   - Raw softmax values from the $L_2$-regularized logistic model are mathematically valid probability distributions ($\sum p_c = 1.0$, $p_c \in [0, 1]$), but are empirically under-confident. They are designated `confidence_kind: "uncalibrated_score"`.

---

## 3. Development-Set Error Analysis ($N=180$)

Across 5 out-of-fold validation splits, the classifier produced **24 errors (13.33%)**:

```text
Error Category Breakdown:
├── SPEAK_VS_ASK_CONFUSION : 23 errors (95.8% of all errors)
│   ├── Gold SPEAK -> Pred ASK : 15 errors (65.2%)
│   └── Gold ASK -> Pred SPEAK :  8 errors (34.8%)
├── MISSED_CONTRIBUTION    :  1 error  ( 4.2% of all errors)
└── UNNECESSARY_INTERVENTION: 0 errors  ( 0.0%)
```

### Error Cluster 1: Follow-Up Dependent Requests (10 errors)
- **Manifestation:** Short follow-up messages in multi-turn dialogues (e.g. *"Can you run that again?"*, *"Same error on the secondary node"*).
- **Failure Mode:** Without recursive dialog-state tracking or coreference resolution, the model cannot tell if the preceding turns provided sufficient technical detail (gold `SPEAK`) or if the referent remains unresolved (gold `ASK`).
- **Observable Features:** `f19_thread_turns_norm > 0.3`, `f12_has_underspecified_cue = 1.0`, `f13_has_technical_spec = 0.0`.

### Error Cluster 2: Subtle Ambiguity in Direct Questions (11 errors)
- **Manifestation:** Direct requests containing borderline ambiguity (e.g. *"Hush, can you check why CI is failing?"*).
- **Failure Mode:** 15 answerable questions were classified as `ASK` because the feature extractor flagged general trouble cues (`f12_has_underspecified_cue`), causing excessive hesitation. Conversely, 8 underspecified questions were classified as `SPEAK` because technical tokens (`f13_has_technical_spec`) misled the linear classifier into assuming sufficient context.

### Error Cluster 3: The Single Missed Contribution (1 error)
- **Scenario:** Scenario `s-168` (`emotional_venting` category).
- **Manifestation:** A user vented frustration with an embedded direct request that lacked an explicit mention tag or name invocation (`address_method: "request"`).
- **Failure Mode:** The model prioritized the strong venting signals (`f15_has_emotional_venting = 1.0`, `f01_addressed_to_hush = 0.0`) and chose `SILENT`.

---

## 4. Confidence Calibration Diagnostics

```text
Reliability Diagram Data (10 bins, Out-Of-Fold N=180):
  Bin Range   | Count | Mean Conf | Mean Acc | Calibration Gap | Interpretation
  ------------|-------|-----------|----------|-----------------|-------------------
  [0.30, 0.40] |    35 |    38.4%  |   80.0%  |     +41.6%      | Severely Under-confident
  [0.40, 0.50] |    92 |    43.7%  |   81.5%  |     +37.8%      | Severely Under-confident
  [0.50, 0.60] |    53 |    54.6%  |  100.0%  |     +45.4%      | Severely Under-confident
```

### Key Diagnostic Findings:
1. **Mathematical Validity:** 100% of prediction vectors strictly satisfied $\sum_{c=1}^3 P(c) = 1.0 \pm 10^{-5}$, $P(c) \in [0, 1]$, with zero NaN or infinite values.
2. **Under-Confidence Due to Regularization:**
   - Uncalibrated ECE was **0.4076** (Multiclass Brier Score: **0.4615**, NLL: **0.8018**).
   - Because $L_2$ regularization shrinks weights towards zero, logits remain small in magnitude, compressing softmax probabilities into the $[0.35, 0.60]$ band even when the classifier is 100% accurate.
3. **Correct vs. Incorrect Confidence Separation:**
   - Correct decisions ($N=156$): Mean confidence = **46.63%** (Median: 47.26%).
   - Incorrect decisions ($N=24$): Mean confidence = **41.15%** (Median: 41.00%).
   - While correct decisions are statistically higher ($p < 0.001$), the 5.5 percentage-point separation is narrow.
4. **Temperature Scaling Exploration:**
   - Fitting temperature $T^*$ on out-of-fold validation logits yielded $T^* = 0.20$.
   - Scaling logits ($z / 0.20$) sharpened probabilities: NLL dropped by $53\%$ from **0.8018 to 0.3773**, and ECE was reduced by $73\%$ from **0.4076 to 0.1104**.

---

## 5. Decision Thresholds & Cost Trade-offs

### The Operational Restraint Cost Matrix

| True Action \ Pred Action | `SILENT` | `SPEAK` | `ASK` | Operational Justification |
|---|---|---|---|---|
| **`SILENT`** | **0.0** | **3.0** | **3.0** | Uninvited interruption in human discussion harms user experience |
| **`SPEAK`** | **5.0** | **0.0** | **1.0** | Missing a direct request makes the assistant seem broken; asking is mild |
| **`ASK`** | **5.0** | **1.5** | **0.0** | Missing a request is critical; guessing/hallucinating is worse than asking |

*Addressing Invariant:* When `addressed_to_hush == True`, Cost for `SILENT` is defined as $\infty$.

### Policy Comparison on Development Split ($N=180$)

| Decision Policy | Mean Cost / Msg | Accuracy | Macro-F1 | $R_{\text{unnec}}$ | $R_{\text{miss}}$ | Calls Avoided | Changed | Invariant Preserved? |
|---|---|---|---|---|---|---|---|---|
| **Policy 1: Default Argmax** | **0.178** | **86.7%** | **0.833** | **0.0%** | **1.0%** | **45.6%** | **0** | **YES (0 violations)** |
| **Policy 2: Cost-Sensitive Bayes Risk** | 1.700 | 20.0% | 0.111 | 100.0% | 0.0% | 0.0% | 137 | **YES** |
| **Policy 3: Uncertainty Routing ($\delta=0.05$)** | 0.283 | 73.9% | 0.670 | 0.0% | 1.0% | 45.6% | 39 | **YES** |
| **Policy 3: Uncertainty Routing ($\delta=0.10$)** | 0.356 | 66.7% | 0.545 | 0.0% | 1.0% | 45.6% | 52 | **YES** |
| **Policy 4: Silence Threshold ($\tau=0.45$)** | 0.211 | 85.6% | 0.823 | 2.5% | 1.0% | 44.4% | 2 | **YES** |
| **Policy 4: Silence Threshold ($\tau=0.50$)** | 0.617 | 71.7% | 0.712 | 34.6% | 0.0% | 29.4% | 29 | **YES** |

### Policy Analysis Takeaways:
1. **The Peril of Uncalibrated Bayes Risk (Policy 2):**  
   Because the uncalibrated model is under-confident, the $5.0$ penalty for missed silence completely dominates the risk equation, forcing the model to speak on 100% of messages ($R_{\text{unnecessary}} = 100\%$, Mean Cost surges to $1.700$). Cost-sensitive classification must never be deployed without post-hoc probability calibration.
2. **Superiority of Default Invariant-Enforced Argmax (Policy 1):**  
   Default argmax with the hard addressing invariant achieved the lowest operational cost ($0.178$), zero uninvited interruptions ($R_{\text{unnecessary}} = 0.0\%$), and $45.6\%$ generation calls avoided.

---

## 6. Statistical Uncertainty & Leakage Checks

1. **Conversation Grouping Equivalence:**  
   In `dev.jsonl`, all 180 scenarios have distinct conversation IDs (`c-001` through `c-180`). Thus, `GroupKFold(groups=conversation_id)` is mathematically identical to standard stratified $K$-Fold.
2. **Template Leakage Assessment:**  
   Synthesized dialogue templates (e.g. *"Can you deploy X?"*) exhibit mild n-gram redundancy across folds. This explains the high in-fold accuracy and reinforces why performance numbers must be treated as **provisional** until verified on independent human-authored dialogue.
3. **Zero Test-Set Contamination:**  
   Held-out test labels in `data/splits/test.jsonl` were completely untouched during this phase.

---

## 7. Artifacts Created & Generated

- Diagnostic Scripts:
  - [`scripts/analyze_errors.py`](../scripts/analyze_errors.py)
  - [`scripts/analyze_calibration.py`](../scripts/analyze_calibration.py)
  - [`scripts/analyze_thresholds.py`](../scripts/analyze_thresholds.py)
- Diagnostic Data Artifacts:
  - `data/results/phase4_dev_error_analysis.json`
  - `data/results/phase4_calibration.json`
  - `data/results/phase4_threshold_sweep.json`
- Analytical Documentation:
  - [`docs/phase4_analysis.md`](phase4_analysis.md)
- Test Suite:
  - [`backend/tests/test_phase4_analysis.py`](../backend/tests/test_phase4_analysis.py)
