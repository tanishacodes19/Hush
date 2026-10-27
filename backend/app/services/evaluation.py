"""Unified evaluation harness for Hush Conversational Restraint Engine.

Implements all metrics, trade-off rates, confidence intervals, and paired significance
tests defined in docs/dataset_spec.md (Section 9):
1. Accuracy and Macro-F1 (unweighted mean across SILENT, SPEAK, ASK).
2. Per-class Precision, Recall, and F1.
3. Confusion matrix with stable canonical ordering (SILENT, SPEAK, ASK).
4. Restraint trade-off metrics:
   - Unnecessary Response Rate: R_unnecessary = False(SPEAK or ASK) / True(SILENT)
   - Missed Contribution Rate: R_missed = False(SILENT) / True(SPEAK or ASK)
   - Generation Calls Avoided: Count(pred == SILENT) / Total
5. Bootstrap 95% Confidence Intervals (1,000 resamples, percentile method, seed=42).
6. Paired significance comparisons:
   - McNemar's test for accuracy differences.
   - Paired bootstrap test for Macro-F1 delta.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from app.schemas import Action
from app.services.baselines import BaselinePredictor, extract_inference_context

CANONICAL_LABELS: list[str] = [Action.SILENT.value, Action.SPEAK.value, Action.ASK.value]


# ============================================================================
# Core Metric Calculations
# ============================================================================


def compute_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute overall classification accuracy."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)


def compute_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = CANONICAL_LABELS,
) -> dict[str, dict[str, int]]:
    """Compute 2D confusion matrix with fixed canonical label order.

    matrix[true_label][predicted_label] = count
    """
    matrix: dict[str, dict[str, int]] = {lbl: {p_lbl: 0 for p_lbl in labels} for lbl in labels}
    for yt, yp in zip(y_true, y_pred):
        if yt in matrix and yp in matrix[yt]:
            matrix[yt][yp] += 1
    return matrix


def compute_per_class_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = CANONICAL_LABELS,
) -> dict[str, dict[str, float]]:
    """Compute per-class Precision, Recall, F1, and Support."""
    cm = compute_confusion_matrix(y_true, y_pred, labels)
    results: dict[str, dict[str, float]] = {}

    for c in labels:
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in labels if r != c)
        fn = sum(cm[c][col] for col in labels if col != c)
        support = sum(cm[c][col] for col in labels)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }

    return results


def compute_macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = CANONICAL_LABELS,
) -> float:
    """Compute Macro-F1 as the unweighted mean of per-class F1 scores."""
    per_class = compute_per_class_metrics(y_true, y_pred, labels)
    f1_scores = [per_class[c]["f1"] for c in labels]
    return sum(f1_scores) / len(labels) if labels else 0.0


# ============================================================================
# Restraint Trade-off Metrics (docs/dataset_spec.md Section 9.2)
# ============================================================================


def compute_unnecessary_response_rate(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute Unnecessary Response Rate (R_unnecessary).

    Fraction of true SILENT instances where the model incorrectly spoke or asked:
    R_unnecessary = Count(y_pred in {SPEAK, ASK} and y_true == SILENT) / Count(y_true == SILENT)
    """
    true_silent_count = sum(1 for yt in y_true if yt == Action.SILENT.value)
    if true_silent_count == 0:
        return 0.0
    false_responses = sum(
        1
        for yt, yp in zip(y_true, y_pred)
        if yt == Action.SILENT.value and yp in {Action.SPEAK.value, Action.ASK.value}
    )
    return false_responses / true_silent_count


def compute_missed_contribution_rate(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute Missed Contribution Rate (R_missed).

    Fraction of true non-SILENT instances where the model incorrectly remained silent:
    R_missed = Count(y_pred == SILENT and y_true in {SPEAK, ASK}) / Count(y_true in {SPEAK, ASK})
    """
    true_action_count = sum(
        1 for yt in y_true if yt in {Action.SPEAK.value, Action.ASK.value}
    )
    if true_action_count == 0:
        return 0.0
    missed_actions = sum(
        1
        for yt, yp in zip(y_true, y_pred)
        if yt in {Action.SPEAK.value, Action.ASK.value} and yp == Action.SILENT.value
    )
    return missed_actions / true_action_count


def compute_calls_avoided_rate(y_pred: Sequence[str], total: Optional[int] = None) -> float:
    """Compute percentage of total instances where generation was avoided:

    Calls Avoided = Count(y_pred == SILENT) / Total
    """
    n = total if total is not None else len(y_pred)
    if n == 0:
        return 0.0
    silent_count = sum(1 for yp in y_pred if yp == Action.SILENT.value)
    return silent_count / n


# ============================================================================
# Statistical Confidence & Significance Testing (docs/dataset_spec.md Section 9.4)
# ============================================================================


def _percentile(values: list[float], p: float) -> float:
    """Compute empirical percentile with linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    n = len(values)
    rank = p * (n - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric_fn: Callable[[list[str], list[str]], float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap 95% confidence interval using percentile method.

    Returns:
        (nominal_metric, lower_bound, upper_bound)
    """
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0

    nominal = metric_fn(list(y_true), list(y_pred))
    rng = random.Random(seed)

    indices = list(range(n))
    replicates: list[float] = []

    for _ in range(n_resamples):
        sample_idx = rng.choices(indices, k=n)
        sample_true = [y_true[i] for i in sample_idx]
        sample_pred = [y_pred[i] for i in sample_idx]
        replicates.append(metric_fn(sample_true, sample_pred))

    replicates.sort()
    lower = _percentile(replicates, alpha / 2.0)
    upper = _percentile(replicates, 1.0 - (alpha / 2.0))

    return nominal, lower, upper


def paired_mcnemar_test(
    y_true: Sequence[str],
    y_pred_a: Sequence[str],
    y_pred_b: Sequence[str],
) -> dict[str, Any]:
    """Perform McNemar's test on paired classification results.

    Tests whether Model A and Model B have statistically significant difference
    in accuracy on the identical set of instances.

    Contingency table:
        b (n01): A correct, B incorrect
        c (n10): A incorrect, B correct
    """
    if len(y_true) != len(y_pred_a) or len(y_true) != len(y_pred_b):
        raise ValueError("Length mismatch between ground truth and predictions")

    b = 0  # A correct, B wrong
    c = 0  # A wrong, B correct
    n00 = 0  # Both correct
    n11 = 0  # Both wrong

    for yt, ya, yb in zip(y_true, y_pred_a, y_pred_b):
        a_correct = (ya == yt)
        b_correct = (yb == yt)

        if a_correct and b_correct:
            n00 += 1
        elif a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1
        else:
            n11 += 1

    discordant = b + c

    if discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "b": b,
            "c": c,
            "both_correct": n00,
            "both_incorrect": n11,
            "method": "identical",
            "significant_at_05": False,
        }

    # If small number of discordant pairs, use exact two-tailed binomial test
    if discordant < 25:
        # Sum binomial probabilities for k <= min(b, c) with p=0.5
        k_max = min(b, c)
        p_tail = sum(math.comb(discordant, k) * (0.5 ** discordant) for k in range(k_max + 1))
        p_value = min(1.0, 2.0 * p_tail)
        stat = float(b - c)
        method = "exact_binomial"
    else:
        # McNemar statistic with Edward's continuity correction
        stat = ((abs(b - c) - 1.0) ** 2) / discordant
        # Survival function of chi2 with df=1: erfc(sqrt(stat / 2))
        p_value = math.erfc(math.sqrt(stat / 2.0))
        method = "mcnemar_chi2"

    return {
        "statistic": round(stat, 4),
        "p_value": round(p_value, 6),
        "b": b,
        "c": c,
        "both_correct": n00,
        "both_incorrect": n11,
        "method": method,
        "significant_at_05": p_value < 0.05,
    }


def paired_bootstrap_macro_f1(
    y_true: Sequence[str],
    y_pred_a: Sequence[str],
    y_pred_b: Sequence[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute paired bootstrap difference in Macro-F1 between two models."""
    n = len(y_true)
    f1_a = compute_macro_f1(y_true, y_pred_a)
    f1_b = compute_macro_f1(y_true, y_pred_b)
    nominal_delta = f1_a - f1_b

    rng = random.Random(seed)
    deltas: list[float] = []
    indices = list(range(n))

    for _ in range(n_resamples):
        sample_idx = rng.choices(indices, k=n)
        sample_true = [y_true[i] for i in sample_idx]
        sample_a = [y_pred_a[i] for i in sample_idx]
        sample_b = [y_pred_b[i] for i in sample_idx]

        m_a = compute_macro_f1(sample_true, sample_a)
        m_b = compute_macro_f1(sample_true, sample_b)
        deltas.append(m_a - m_b)

    deltas.sort()
    ci_lower = _percentile(deltas, 0.025)
    ci_upper = _percentile(deltas, 0.975)

    # Empirical two-tailed p-value: fraction of deltas <= 0 or >= 0 depending on sign
    if nominal_delta >= 0:
        tail = sum(1 for d in deltas if d <= 0)
    else:
        tail = sum(1 for d in deltas if d >= 0)
    p_value = min(1.0, 2.0 * (tail / n_resamples))

    return {
        "f1_model_a": round(f1_a, 4),
        "f1_model_b": round(f1_b, 4),
        "delta": round(nominal_delta, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 6),
        "significant_at_05": p_value < 0.05,
    }


# ============================================================================
# Structured Evaluation Report
# ============================================================================


@dataclass
class MetricWithCI:
    """Scalar metric accompanied by its 95% bootstrap confidence interval."""
    value: float
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> dict[str, float]:
        return {
            "value": round(self.value, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
        }

    def format(self, as_pct: bool = False) -> str:
        multiplier = 100.0 if as_pct else 1.0
        unit = "%" if as_pct else ""
        return f"{self.value * multiplier:.2f}{unit} [{self.ci_lower * multiplier:.2f}{unit}, {self.ci_upper * multiplier:.2f}{unit}]"


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for a decision strategy."""

    model_name: str
    split: str
    total_samples: int
    accuracy: MetricWithCI
    macro_f1: MetricWithCI
    unnecessary_response_rate: MetricWithCI
    missed_contribution_rate: MetricWithCI
    calls_avoided_rate: MetricWithCI
    per_class: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]
    predictions: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "split": self.split,
            "total_samples": self.total_samples,
            "accuracy": self.accuracy.to_dict(),
            "macro_f1": self.macro_f1.to_dict(),
            "unnecessary_response_rate": self.unnecessary_response_rate.to_dict(),
            "missed_contribution_rate": self.missed_contribution_rate.to_dict(),
            "calls_avoided_rate": self.calls_avoided_rate.to_dict(),
            "per_class": {
                c: {k: round(v, 4) for k, v in metrics.items()}
                for c, metrics in self.per_class.items()
            },
            "confusion_matrix": self.confusion_matrix,
        }

    def to_markdown(self) -> str:
        """Render a formatted markdown summary."""
        lines = [
            f"### Evaluation Report: {self.model_name} (Split: `{self.split}`, N={self.total_samples})",
            "",
            "| Metric | Nominal Value | 95% Bootstrap CI |",
            "|---|---|---|",
            f"| **Accuracy** | {self.accuracy.value * 100:.2f}% | [{self.accuracy.ci_lower * 100:.2f}%, {self.accuracy.ci_upper * 100:.2f}%] |",
            f"| **Macro-F1** | {self.macro_f1.value:.4f} | [{self.macro_f1.ci_lower:.4f}, {self.macro_f1.ci_upper:.4f}] |",
            f"| **Unnecessary Response Rate ($R_{{unnecessary}}$)** | {self.unnecessary_response_rate.value * 100:.2f}% | [{self.unnecessary_response_rate.ci_lower * 100:.2f}%, {self.unnecessary_response_rate.ci_upper * 100:.2f}%] |",
            f"| **Missed Contribution Rate ($R_{{missed}}$)** | {self.missed_contribution_rate.value * 100:.2f}% | [{self.missed_contribution_rate.ci_lower * 100:.2f}%, {self.missed_contribution_rate.ci_upper * 100:.2f}%] |",
            f"| **Generation Calls Avoided** | {self.calls_avoided_rate.value * 100:.2f}% | [{self.calls_avoided_rate.ci_lower * 100:.2f}%, {self.calls_avoided_rate.ci_upper * 100:.2f}%] |",
            "",
            "#### Per-Class Performance",
            "| Class | Precision | Recall | F1-Score | Support |",
            "|---|---|---|---|---|",
        ]
        for c in CANONICAL_LABELS:
            metrics = self.per_class.get(c, {})
            p = metrics.get("precision", 0.0)
            r = metrics.get("recall", 0.0)
            f = metrics.get("f1", 0.0)
            s = int(metrics.get("support", 0))
            lines.append(f"| `{c}` | {p:.4f} | {r:.4f} | {f:.4f} | {s} |")

        lines.extend([
            "",
            "#### Confusion Matrix (Rows: Ground Truth, Columns: Predicted)",
            "| True \\ Pred | SILENT | SPEAK | ASK |",
            "|---|---|---|---|",
        ])
        for r in CANONICAL_LABELS:
            row_vals = [str(self.confusion_matrix.get(r, {}).get(col, 0)) for col in CANONICAL_LABELS]
            lines.append(f"| **{r}** | " + " | ".join(row_vals) + " |")

        return "\n".join(lines)


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    model_name: str = "model",
    split: str = "dev",
    scenario_ids: Optional[Sequence[str]] = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> EvaluationReport:
    """Evaluate predictions against ground truth labels."""
    total = len(y_true)

    # Core metrics + CIs
    acc_val, acc_low, acc_high = bootstrap_ci(
        y_true, y_pred, compute_accuracy, n_resamples=n_resamples, seed=seed
    )
    f1_val, f1_low, f1_high = bootstrap_ci(
        y_true, y_pred, compute_macro_f1, n_resamples=n_resamples, seed=seed
    )
    unrec_val, unrec_low, unrec_high = bootstrap_ci(
        y_true, y_pred, compute_unnecessary_response_rate, n_resamples=n_resamples, seed=seed
    )
    miss_val, miss_low, miss_high = bootstrap_ci(
        y_true, y_pred, compute_missed_contribution_rate, n_resamples=n_resamples, seed=seed
    )

    # Calls avoided CI
    def _calls_avoided_helper(yt: list[str], yp: list[str]) -> float:
        return compute_calls_avoided_rate(yp, total=len(yp))

    ca_val, ca_low, ca_high = bootstrap_ci(
        y_true, y_pred, _calls_avoided_helper, n_resamples=n_resamples, seed=seed
    )

    per_class = compute_per_class_metrics(y_true, y_pred)
    confusion_matrix = compute_confusion_matrix(y_true, y_pred)

    predictions: list[dict[str, str]] = []
    if scenario_ids:
        for sid, yt, yp in zip(scenario_ids, y_true, y_pred):
            predictions.append({"id": sid, "expected_action": yt, "predicted_action": yp})

    return EvaluationReport(
        model_name=model_name,
        split=split,
        total_samples=total,
        accuracy=MetricWithCI(acc_val, acc_low, acc_high),
        macro_f1=MetricWithCI(f1_val, f1_low, f1_high),
        unnecessary_response_rate=MetricWithCI(unrec_val, unrec_low, unrec_high),
        missed_contribution_rate=MetricWithCI(miss_val, miss_low, miss_high),
        calls_avoided_rate=MetricWithCI(ca_val, ca_low, ca_high),
        per_class=per_class,
        confusion_matrix=confusion_matrix,
        predictions=predictions,
    )


def evaluate_baseline_on_file(
    predictor: BaselinePredictor,
    file_path: Union[str, Path],
    model_name: Optional[str] = None,
    split_name: Optional[str] = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> EvaluationReport:
    """Load a split JSONL file, generate predictions via isolated context, and evaluate."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    y_true: list[str] = []
    y_pred: list[str] = []
    scenario_ids: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Feature isolation: extract only inference features
            context = extract_inference_context(record)
            pred = predictor.predict(context)
            expected = record.get("expected_action", "")

            y_true.append(expected)
            y_pred.append(pred)
            scenario_ids.append(context.scenario_id)

    name = model_name or predictor.__class__.__name__
    split = split_name or path.stem

    return evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        model_name=name,
        split=split,
        scenario_ids=scenario_ids,
        n_resamples=n_resamples,
        seed=seed,
    )


# ============================================================================
# Calibration Diagnostics & Decision Cost (Phase 4)
# ============================================================================


DEFAULT_RESTRAINT_COST_MATRIX: dict[str, dict[str, float]] = {
    Action.SILENT.value: {
        Action.SILENT.value: 0.0,
        Action.SPEAK.value: 3.0,
        Action.ASK.value: 3.0,
    },
    Action.SPEAK.value: {
        Action.SILENT.value: 5.0,
        Action.SPEAK.value: 0.0,
        Action.ASK.value: 1.0,
    },
    Action.ASK.value: {
        Action.SILENT.value: 5.0,
        Action.SPEAK.value: 1.5,
        Action.ASK.value: 0.0,
    },
}


def compute_brier_score(
    y_true: Sequence[str],
    probabilities: Sequence[dict[str, float]],
    labels: Sequence[str] = CANONICAL_LABELS,
) -> float:
    """Compute multiclass Brier score: mean squared error of predicted probabilities."""
    if not y_true or len(y_true) != len(probabilities):
        return 0.0
    total_sq_err = 0.0
    for yt, probs in zip(y_true, probabilities):
        for lbl in labels:
            y_val = 1.0 if yt == lbl else 0.0
            p_val = probs.get(lbl, 0.0)
            total_sq_err += (p_val - y_val) ** 2
    return total_sq_err / len(y_true)


def compute_nll(
    y_true: Sequence[str],
    probabilities: Sequence[dict[str, float]],
    eps: float = 1e-15,
) -> float:
    """Compute Negative Log-Likelihood (cross-entropy log loss) with numerical clamp."""
    if not y_true or len(y_true) != len(probabilities):
        return 0.0
    total_nll = 0.0
    for yt, probs in zip(y_true, probabilities):
        p_val = max(probs.get(yt, 0.0), eps)
        total_nll -= math.log(p_val)
    return total_nll / len(y_true)


def compute_ece(
    y_true: Sequence[str],
    probabilities: Sequence[dict[str, float]],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Expected Calibration Error (ECE) and binned reliability stats."""
    n = len(y_true)
    if n == 0 or len(probabilities) != n:
        return {"ece": 0.0, "max_error": 0.0, "bins": []}

    # Extract predicted class and confidence
    confidences: list[float] = []
    accuracies: list[float] = []

    for yt, probs in zip(y_true, probabilities):
        best_cls = max(probs.items(), key=lambda kv: kv[1])[0]
        conf = probs[best_cls]
        confidences.append(conf)
        accuracies.append(1.0 if best_cls == yt else 0.0)

    bin_width = 1.0 / n_bins
    bins_data: list[dict[str, Any]] = []
    total_ece = 0.0
    max_error = 0.0

    for b in range(n_bins):
        bin_low = b * bin_width
        bin_high = (b + 1) * bin_width

        # In last bin include 1.0
        if b == n_bins - 1:
            indices = [i for i, c in enumerate(confidences) if bin_low <= c <= bin_high]
        else:
            indices = [i for i, c in enumerate(confidences) if bin_low <= c < bin_high]

        bin_count = len(indices)
        if bin_count > 0:
            bin_acc = sum(accuracies[i] for i in indices) / bin_count
            bin_conf = sum(confidences[i] for i in indices) / bin_count
            bin_gap = abs(bin_acc - bin_conf)
            weighted_gap = (bin_count / n) * bin_gap
            total_ece += weighted_gap
            max_error = max(max_error, bin_gap)
        else:
            bin_acc = 0.0
            bin_conf = 0.0
            bin_gap = 0.0

        bins_data.append({
            "bin_idx": b,
            "range": [round(bin_low, 2), round(bin_high, 2)],
            "count": bin_count,
            "accuracy": round(bin_acc, 4),
            "confidence": round(bin_conf, 4),
            "gap": round(bin_gap, 4),
        })

    return {
        "ece": round(total_ece, 4),
        "max_calibration_error": round(max_error, 4),
        "n_bins": n_bins,
        "bins": bins_data,
    }


def compute_confidence_diagnostics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: Sequence[dict[str, float]],
) -> dict[str, Any]:
    """Compute confidence distribution statistics for correct versus incorrect predictions."""
    correct_confs: list[float] = []
    incorrect_confs: list[float] = []

    for yt, yp, probs in zip(y_true, y_pred, probabilities):
        conf = probs.get(yp, max(probs.values()) if probs else 0.0)
        if yt == yp:
            correct_confs.append(conf)
        else:
            incorrect_confs.append(conf)

    def _stats(arr: list[float]) -> dict[str, float]:
        if not arr:
            return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        s = sorted(arr)
        mid = len(s) // 2
        med = s[mid] if len(s) % 2 != 0 else (s[mid - 1] + s[mid]) / 2.0
        return {
            "count": len(arr),
            "mean": round(sum(arr) / len(arr), 4),
            "median": round(med, 4),
            "min": round(min(arr), 4),
            "max": round(max(arr), 4),
        }

    return {
        "correct": _stats(correct_confs),
        "incorrect": _stats(incorrect_confs),
    }


def compute_expected_cost(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    cost_matrix: Optional[dict[str, dict[str, float]]] = None,
) -> float:
    """Compute average penalty cost per message under given CostMatrix."""
    costs = cost_matrix or DEFAULT_RESTRAINT_COST_MATRIX
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    total_cost = sum(costs.get(yt, {}).get(yp, 0.0) for yt, yp in zip(y_true, y_pred))
    return total_cost / len(y_true)

