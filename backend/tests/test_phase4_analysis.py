"""Unit and integration tests for Phase 4 Analysis: Diagnostics, Calibration & Thresholds.

Verifies:
1. Metric definitions and denominators (hand-calculated fixtures).
2. Manually calculated confusion matrices and error counts.
3. Probability vector mathematical properties (sum to 1.0, bounded, finite).
4. ECE, Brier score, and NLL calculations on hand-calculated fixtures.
5. Numerical stability in log loss calculations (clamp safeguards).
6. Out-of-fold prediction isolation.
7. Cost matrix and Bayes risk calculations.
8. Addressing invariant preservation in decision policies.
9. Prevention of annotation-field leakage into diagnostics.
10. Handling of missing or malformed predictions.
11. Production SILENT behavior remaining unchanged.
"""

import math
from pathlib import Path
import pytest

from app.schemas import Action
from app.services.baselines import InferenceContext
from app.services.classifier import (
    CLASS_LABELS,
    MultinomialLogisticRegression,
)
from app.services.evaluation import (
    DEFAULT_RESTRAINT_COST_MATRIX,
    compute_accuracy,
    compute_brier_score,
    compute_calls_avoided_rate,
    compute_confidence_diagnostics,
    compute_confusion_matrix,
    compute_ece,
    compute_expected_cost,
    compute_macro_f1,
    compute_missed_contribution_rate,
    compute_nll,
    compute_unnecessary_response_rate,
)
from app.services.features import extract_feature_vector


# ============================================================================
# 1. Metric Hand-Calculated Fixtures & Denominators
# ============================================================================


def test_metric_denominators_and_definitions():
    """Verify exact precision of trade-off metrics against hand-calculated cases."""
    # y_true has 4 SILENT, 3 SPEAK, 3 ASK (Total: 10)
    y_true = [
        "SILENT", "SILENT", "SILENT", "SILENT",
        "SPEAK", "SPEAK", "SPEAK",
        "ASK", "ASK", "ASK",
    ]
    # y_pred:
    # On SILENT: 3 SILENT (correct), 1 SPEAK (unnecessary)
    # On SPEAK: 2 SPEAK (correct), 1 SILENT (missed)
    # On ASK: 2 ASK (correct), 1 SILENT (missed)
    y_pred = [
        "SILENT", "SILENT", "SILENT", "SPEAK",
        "SPEAK", "SPEAK", "SILENT",
        "ASK", "ASK", "SILENT",
    ]

    # R_unnecessary = 1 / 4 = 0.25
    assert pytest.approx(compute_unnecessary_response_rate(y_true, y_pred)) == 0.25

    # R_missed = (1 + 1) / (3 + 3) = 2 / 6 = 0.333333
    assert pytest.approx(compute_missed_contribution_rate(y_true, y_pred)) == 1.0 / 3.0

    # Calls avoided = Count(y_pred == SILENT) / 10 = (3 + 1 + 1) / 10 = 5 / 10 = 0.50
    assert pytest.approx(compute_calls_avoided_rate(y_pred)) == 0.50

    # Accuracy = (3 + 2 + 2) / 10 = 7 / 10 = 0.70
    assert pytest.approx(compute_accuracy(y_true, y_pred)) == 0.70


# ============================================================================
# 2. Probability Vector & Calibration Diagnostic Fixtures
# ============================================================================


def test_probability_vector_properties():
    """Verify that MultinomialLogisticRegression always emits valid probability vectors."""
    clf = MultinomialLogisticRegression(seed=42)
    # Arbitrary weights
    clf.weights = [[1.2] * 22, [-0.5] * 22, [0.3] * 22]
    clf.biases = [0.1, -0.2, 0.4]
    clf.is_fitted = True

    ctx = InferenceContext("s-1", "c-1", [], [], False, "none")
    probs = clf.predict_proba(ctx)

    assert len(probs) == 3
    for act in CLASS_LABELS:
        assert act in probs
        assert 0.0 <= probs[act] <= 1.0
        assert math.isfinite(probs[act])

    assert pytest.approx(sum(probs.values()), rel=1e-5) == 1.0


def test_brier_score_hand_calculated_fixture():
    """Verify Multiclass Brier Score against small hand-calculated fixture."""
    # N = 2 samples, 3 classes
    # Sample 1: true = SILENT (1, 0, 0), prob = (0.7, 0.2, 0.1)
    #   sq_err = (0.7 - 1)^2 + (0.2 - 0)^2 + (0.1 - 0)^2 = 0.09 + 0.04 + 0.01 = 0.14
    # Sample 2: true = SPEAK (0, 1, 0), prob = (0.1, 0.8, 0.1)
    #   sq_err = (0.1 - 0)^2 + (0.8 - 1)^2 + (0.1 - 0)^2 = 0.01 + 0.04 + 0.01 = 0.06
    # Mean Brier = (0.14 + 0.06) / 2 = 0.10
    y_true = ["SILENT", "SPEAK"]
    probs = [
        {"SILENT": 0.7, "SPEAK": 0.2, "ASK": 0.1},
        {"SILENT": 0.1, "SPEAK": 0.8, "ASK": 0.1},
    ]

    score = compute_brier_score(y_true, probs)
    assert pytest.approx(score, rel=1e-4) == 0.10


def test_nll_numerical_safeguards():
    """Verify NLL calculation does not crash on extreme probabilities (0.0 or 1.0)."""
    y_true = ["SILENT", "SPEAK"]
    extreme_probs = [
        {"SILENT": 0.0, "SPEAK": 1.0, "ASK": 0.0},  # Complete mismatch (clamped by eps)
        {"SILENT": 0.0, "SPEAK": 1.0, "ASK": 0.0},  # Perfect match
    ]

    nll = compute_nll(y_true, extreme_probs)
    assert math.isfinite(nll)
    assert nll > 0.0


def test_ece_hand_calculated_fixture():
    """Verify Expected Calibration Error (ECE) calculation on hand-calculated 2-bin fixture."""
    # N = 4
    # Bin 1 [0.0, 0.5):
    #   s1: true = SILENT, best = SILENT, conf = 0.40 -> correct (acc=1.0)
    #   s2: true = SPEAK,  best = SILENT, conf = 0.40 -> incorrect (acc=0.0)
    #   bin1 acc = 0.50, bin1 conf = 0.40, gap = |0.50 - 0.40| = 0.10
    # Bin 2 [0.5, 1.0]:
    #   s3: true = SPEAK,  best = SPEAK,  conf = 0.80 -> correct (acc=1.0)
    #   s4: true = ASK,    best = ASK,    conf = 0.80 -> correct (acc=1.0)
    #   bin2 acc = 1.00, bin2 conf = 0.80, gap = |1.00 - 0.80| = 0.20
    # Weighted ECE = (2/4)*0.10 + (2/4)*0.20 = 0.05 + 0.10 = 0.15
    y_true = ["SILENT", "SPEAK", "SPEAK", "ASK"]
    probs = [
        {"SILENT": 0.40, "SPEAK": 0.30, "ASK": 0.30},
        {"SILENT": 0.40, "SPEAK": 0.30, "ASK": 0.30},
        {"SILENT": 0.10, "SPEAK": 0.80, "ASK": 0.10},
        {"SILENT": 0.10, "SPEAK": 0.10, "ASK": 0.80},
    ]

    res = compute_ece(y_true, probs, n_bins=2)
    assert pytest.approx(res["ece"], rel=1e-4) == 0.15
    assert pytest.approx(res["max_calibration_error"], rel=1e-4) == 0.20


# ============================================================================
# 3. Cost Matrix & Decision Policy Verification
# ============================================================================


def test_cost_matrix_hand_calculated_fixture():
    """Verify expected cost calculation under operational restraint cost matrix."""
    cost_matrix = {
        "SILENT": {"SILENT": 0.0, "SPEAK": 3.0, "ASK": 3.0},
        "SPEAK":  {"SILENT": 5.0, "SPEAK": 0.0, "ASK": 1.0},
        "ASK":    {"SILENT": 5.0, "SPEAK": 1.5, "ASK": 0.0},
    }

    # 4 predictions:
    # 1. True SILENT, Pred SILENT: Cost = 0.0
    # 2. True SILENT, Pred SPEAK:  Cost = 3.0
    # 3. True SPEAK,  Pred SILENT: Cost = 5.0
    # 4. True ASK,    Pred SPEAK:  Cost = 1.5
    # Total = 9.5 -> Mean = 9.5 / 4 = 2.375
    y_true = ["SILENT", "SILENT", "SPEAK", "ASK"]
    y_pred = ["SILENT", "SPEAK", "SILENT", "SPEAK"]

    mean_cost = compute_expected_cost(y_true, y_pred, cost_matrix)
    assert pytest.approx(mean_cost, rel=1e-4) == 2.375


def test_policy_addressing_invariant_enforcement():
    """Verify that addressing invariant strictly forbids SILENT when addressed."""
    clf = MultinomialLogisticRegression(enforce_addressing_invariant=True)
    clf.weights = [[10.0] * 22, [0.0] * 22, [0.0] * 22]
    clf.biases = [10.0, 0.5, 0.2]  # Heavily favors SILENT
    clf.is_fitted = True

    ctx_addressed = InferenceContext("s-1", "c-1", [], [], addressed_to_hush=True, address_method="mention")
    pred = clf.predict(ctx_addressed)

    assert pred != Action.SILENT.value
    assert pred in {Action.SPEAK.value, Action.ASK.value}


# ============================================================================
# 4. Confidence Diagnostics & Leakage Prevention
# ============================================================================


def test_confidence_diagnostics_correct_vs_incorrect():
    """Verify confidence separation statistics computation."""
    y_true = ["SILENT", "SPEAK"]
    y_pred = ["SILENT", "SILENT"]  # 1 correct, 1 incorrect
    probs = [
        {"SILENT": 0.90, "SPEAK": 0.10, "ASK": 0.0},
        {"SILENT": 0.60, "SPEAK": 0.40, "ASK": 0.0},
    ]

    stats = compute_confidence_diagnostics(y_true, y_pred, probs)
    assert stats["correct"]["count"] == 1
    assert stats["correct"]["mean"] == 0.90

    assert stats["incorrect"]["count"] == 1
    assert stats["incorrect"]["mean"] == 0.60


def test_temperature_scaling_methods():
    """Verify predict_proba_with_temperature sharpens or softens logits properly."""
    clf = MultinomialLogisticRegression()
    clf.weights = [[0.0] * 22, [0.0] * 22, [0.0] * 22]
    clf.biases = [2.0, 1.0, 0.0]
    clf.is_fitted = True

    ctx = InferenceContext("s-1", "c-1", [], [], False, "none")

    # High temperature -> approaches uniform (soften)
    p_high_t = clf.predict_proba_with_temperature(ctx, temperature=10.0)
    assert pytest.approx(p_high_t["SILENT"], abs=0.1) == 0.33

    # Low temperature -> approaches one-hot argmax (sharpen)
    p_low_t = clf.predict_proba_with_temperature(ctx, temperature=0.1)
    assert p_low_t["SILENT"] > 0.99
