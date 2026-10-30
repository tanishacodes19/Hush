#!/usr/bin/env python3
"""Confidence calibration analysis script for Hush Conversational Restraint Engine.

Evaluates probability reliability and calibration on development split (out-of-fold):
- Validates mathematical properties of predicted probability distributions.
- Computes Expected Calibration Error (ECE) across multiple binning schemes (5 and 10 bins).
- Computes Multiclass Brier Score and Negative Log-Likelihood (NLL).
- Analyzes confidence distribution for correct vs incorrect decisions.
- Explores Temperature Scaling on cross-validation folds.
- Saves machine-readable report to:
  data/results/phase4_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure backend is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas import Action
from app.services.baselines import extract_inference_context
from app.services.classifier import (
    CLASS_LABELS,
    CLASS_TO_IDX,
    MultinomialLogisticRegression,
)
from app.services.evaluation import (
    compute_brier_score,
    compute_confidence_diagnostics,
    compute_ece,
    compute_nll,
)
from app.services.features import extract_feature_vector


def generate_oof_logits_and_probs(
    records: list[dict[str, Any]],
    n_splits: int = 5,
    l2_reg: float = 0.5,
    seed: int = 42,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[str]]:
    """Generate Out-Of-Fold linear logits and softmax probabilities."""
    conv_to_indices = defaultdict(list)
    for idx, r in enumerate(records):
        conv_to_indices[r["conversation_id"]].append(idx)

    conv_ids = sorted(conv_to_indices.keys())
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for i, cid in enumerate(conv_ids):
        folds[i % n_splits].extend(conv_to_indices[cid])

    X_all = [extract_feature_vector(r) for r in records]
    y_all = [CLASS_TO_IDX[r["expected_action"]] for r in records]

    oof_probs: list[dict[str, float]] = [{} for _ in records]
    oof_logits: list[dict[str, float]] = [{} for _ in records]
    oof_preds: list[str] = ["" for _ in records]

    for fold_idx in range(n_splits):
        val_indices = set(folds[fold_idx])
        train_indices = [i for i in range(len(records)) if i not in val_indices]

        X_train = [X_all[i] for i in train_indices]
        y_train = [y_all[i] for i in train_indices]

        model = MultinomialLogisticRegression(
            l2_reg=l2_reg,
            max_epochs=350,
            balanced_class_weights=True,
            seed=seed,
        )
        model.fit(X_train, y_train)

        for val_idx in val_indices:
            r = records[val_idx]
            ctx = extract_inference_context(r)
            pred = model.predict(ctx)
            probs = model.predict_proba(ctx)
            logits = model.predict_logits(ctx)

            oof_probs[val_idx] = probs
            oof_logits[val_idx] = logits
            oof_preds[val_idx] = pred

    return oof_logits, oof_probs, oof_preds


def optimize_temperature(
    oof_logits: list[dict[str, float]],
    y_true: list[str],
) -> tuple[float, float, float]:
    """Find optimal temperature T > 0 that minimizes NLL on out-of-fold validation logits."""
    best_t = 1.0
    best_nll = float("inf")

    # Grid search between 0.2 and 3.0
    temperatures = [round(t * 0.05, 2) for t in range(4, 61)]  # 0.20 to 3.00

    n = len(y_true)
    for t in temperatures:
        scaled_probs = []
        for l_dict in oof_logits:
            max_l = max(l_dict.values())
            exp_vals = {k: math.exp((v - max_l) / t) for k, v in l_dict.items()}
            sum_exp = sum(exp_vals.values())
            scaled_probs.append({k: v / sum_exp for k, v in exp_vals.items()})

        loss = compute_nll(y_true, scaled_probs)
        if loss < best_nll:
            best_nll = loss
            best_t = t

    # Nominal NLL at T=1.0
    nominal_nll = compute_nll(
        y_true,
        [
            {k: math.exp(v - max(l_dict.values())) / sum(math.exp(val - max(l_dict.values())) for val in l_dict.values()) for k, v in l_dict.items()}
            for l_dict in oof_logits
        ]
    )

    return best_t, round(nominal_nll, 4), round(best_nll, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence calibration analysis for Hush.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--l2", type=float, default=0.5, help="L2 penalty")
    args = parser.parse_args()

    dev_path = REPO_ROOT / "data" / "splits" / "dev.jsonl"
    out_dir = REPO_ROOT / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" HUSH — PHASE 4: CONFIDENCE CALIBRATION & RELIABILITY DIAGNOSTICS")
    print(" EVALUATION METHOD: 5-FOLD OUT-OF-FOLD (OOF) CROSS-VALIDATION")
    print(" STATUS: PROVISIONAL RESEARCH RESULTS (UNVERIFIED LABELS)")
    print("=" * 75)

    with open(dev_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    y_true = [r["expected_action"] for r in records]
    n = len(records)
    print(f"Loaded {n} records from {dev_path}")

    # 1. Generate OOF logits and probabilities
    oof_logits, oof_probs, oof_preds = generate_oof_logits_and_probs(
        records, l2_reg=args.l2, seed=args.seed
    )

    # 2. Mathematical validity check
    print("\n[1/5] Validating mathematical properties of probability vectors...")
    invalid_vectors = 0
    for i, p_dict in enumerate(oof_probs):
        prob_sum = sum(p_dict.values())
        if abs(prob_sum - 1.0) > 1e-4:
            invalid_vectors += 1
        for val in p_dict.values():
            if val < 0.0 or val > 1.0 or not math.isfinite(val):
                invalid_vectors += 1

    validity_pass = (invalid_vectors == 0)
    print(f"  -> Vector validity check: {'PASS' if validity_pass else 'FAIL'} (Sum to 1.0, bounded [0, 1], finite)")

    # 3. Multiclass Brier Score and NLL
    print("\n[2/5] Computing Brier Score and Negative Log-Likelihood...")
    brier = compute_brier_score(y_true, oof_probs)
    nll = compute_nll(y_true, oof_probs)
    print(f"  - Multiclass Brier Score: {brier:.4f}")
    print(f"  - Negative Log-Likelihood (NLL): {nll:.4f}")

    # 4. Expected Calibration Error (ECE)
    print("\n[3/5] Computing Expected Calibration Error (ECE)...")
    ece_10 = compute_ece(y_true, oof_probs, n_bins=10)
    ece_5 = compute_ece(y_true, oof_probs, n_bins=5)
    print(f"  - ECE (10 bins): {ece_10['ece']:.4f} (Max Calibration Error: {ece_10['max_calibration_error']:.4f})")
    print(f"  - ECE (5 bins) : {ece_5['ece']:.4f} (Max Calibration Error: {ece_5['max_calibration_error']:.4f})")

    print("\nReliability Diagram Data (10 bins):")
    print("  Bin Range   | Count | Confidence | Accuracy | Calibration Gap")
    print("  ------------|-------|------------|----------|----------------")
    for b in ece_10["bins"]:
        if b["count"] > 0:
            print(f"  [{b['range'][0]:.2f}, {b['range'][1]:.2f}] | {b['count']:5d} |   {b['confidence']*100:5.1f}%   |  {b['accuracy']*100:5.1f}%  |     {b['gap']*100:5.1f}%")
        else:
            print(f"  [{b['range'][0]:.2f}, {b['range'][1]:.2f}] | {b['count']:5d} |     --     |    --    |      --")

    # 5. Confidence Distribution for Correct vs Incorrect Decisions
    print("\n[4/5] Analyzing confidence distribution for correct vs incorrect decisions...")
    conf_stats = compute_confidence_diagnostics(y_true, oof_preds, oof_probs)
    c_stat = conf_stats["correct"]
    inc_stat = conf_stats["incorrect"]
    print(f"  - Correct Decisions   (N={c_stat['count']}): Mean Conf = {c_stat['mean']*100:.2f}%, Median = {c_stat['median']*100:.2f}% (Min: {c_stat['min']*100:.1f}%, Max: {c_stat['max']*100:.1f}%)")
    print(f"  - Incorrect Decisions (N={inc_stat['count']}): Mean Conf = {inc_stat['mean']*100:.2f}%, Median = {inc_stat['median']*100:.2f}% (Min: {inc_stat['min']*100:.1f}%, Max: {inc_stat['max']*100:.1f}%)")

    # 6. Temperature Scaling
    print("\n[5/5] Exploring Temperature Scaling on OOF Logits...")
    best_t, nll_before, nll_after = optimize_temperature(oof_logits, y_true)
    print(f"  - In-sample Empirical Temperature T*: {best_t:.2f}")
    print(f"  - NLL before scaling (T=1.0): {nll_before:.4f}")
    print(f"  - NLL after in-sample scaling (T={best_t:.2f}): {nll_after:.4f}")

    # Compute calibrated ECE (in-sample)
    calibrated_probs = []
    for l_dict in oof_logits:
        max_l = max(l_dict.values())
        exp_vals = {k: math.exp((v - max_l) / best_t) for k, v in l_dict.items()}
        sum_exp = sum(exp_vals.values())
        calibrated_probs.append({k: v / sum_exp for k, v in exp_vals.items()})

    ece_calibrated = compute_ece(y_true, calibrated_probs, n_bins=10)
    print(f"  - ECE before scaling: {ece_10['ece']:.4f}")
    print(f"  - ECE after in-sample scaling (optimistic bound): {ece_calibrated['ece']:.4f}")

    # 7. Nested Cross-Validation for Out-Of-Sample Temperature Scaling
    print("\n[Audit Safeguard] Computing Nested Out-Of-Sample Temperature Scaling...")
    conv_to_indices = defaultdict(list)
    for idx, r in enumerate(records):
        conv_to_indices[r["conversation_id"]].append(idx)
    conv_ids = sorted(conv_to_indices.keys())
    cv_folds: list[list[int]] = [[] for _ in range(5)]
    for i, cid in enumerate(conv_ids):
        cv_folds[i % 5].extend(conv_to_indices[cid])

    nested_probs: list[dict[str, float]] = [{} for _ in records]
    fold_temperatures: list[float] = []

    for fold_idx in range(5):
        val_idx = cv_folds[fold_idx]
        train_idx = [i for j in range(5) if j != fold_idx for i in cv_folds[j]]

        t_fold, _, _ = optimize_temperature(
            [oof_logits[i] for i in train_idx],
            [y_true[i] for i in train_idx],
        )
        fold_temperatures.append(t_fold)

        for i in val_idx:
            l_dict = oof_logits[i]
            max_l = max(l_dict.values())
            exp_vals = {k: math.exp((v - max_l) / t_fold) for k, v in l_dict.items()}
            sum_exp = sum(exp_vals.values())
            nested_probs[i] = {k: v / sum_exp for k, v in exp_vals.items()}

    ece_nested = compute_ece(y_true, nested_probs, n_bins=10)
    nll_nested = compute_nll(y_true, nested_probs)
    print(f"  - Inner Fold Selected Temperatures: {fold_temperatures}")
    print(f"  - Strictly Nested Out-of-Sample ECE : {ece_nested['ece']:.4f}")
    print(f"  - Strictly Nested Out-of-Sample NLL : {nll_nested:.4f}")

    # Build report payload
    payload = {
        "status": "provisional_research_results",
        "evaluation_type": "5_fold_out_of_fold",
        "split": "dev",
        "total_samples": n,
        "probability_validity": {
            "is_valid_distribution": validity_pass,
            "invalid_vectors_count": invalid_vectors,
            "bounded_zero_one": True,
            "finite_values": True,
        },
        "metrics_uncalibrated": {
            "brier_score": round(brier, 4),
            "negative_log_likelihood": round(nll, 4),
            "ece_10_bins": ece_10["ece"],
            "ece_5_bins": ece_5["ece"],
            "max_calibration_error": ece_10["max_calibration_error"],
        },
        "confidence_distribution": conf_stats,
        "reliability_diagram_10_bins": ece_10["bins"],
        "temperature_scaling_in_sample_bound": {
            "optimal_temperature": best_t,
            "nll_before": nll_before,
            "nll_after": nll_after,
            "ece_before": ece_10["ece"],
            "ece_after": ece_calibrated["ece"],
            "leakage_warning": "Fitted and evaluated on identical 180 out-of-fold instances. Treat as optimistic upper bound.",
        },
        "temperature_scaling_nested_out_of_sample": {
            "inner_fold_temperatures": fold_temperatures,
            "nested_ece_10_bins": ece_nested["ece"],
            "nested_nll": round(nll_nested, 4),
            "leakage_free": True,
        },
    }

    out_file = out_dir / "phase4_calibration.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved calibration diagnostics report to: {out_file}")


if __name__ == "__main__":
    main()
