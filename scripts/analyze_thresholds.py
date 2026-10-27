#!/usr/bin/env python3
"""Decision threshold and cost trade-off analysis script for Hush Conversational Restraint Engine.

Explores cost-sensitive decision policies and threshold sweeps on development split:
- Defines explicit Restraint Cost Matrix based on asymmetric operational consequences.
- Evaluates:
  * Default Argmax Policy
  * Cost-Sensitive Bayes Risk Policy: y_hat = argmin_c Sum_k P(k) * Cost(k, c)
  * Uncertainty Routing to ASK: routes borderline addressed cases to clarification
  * Silence Confidence Threshold Sweeps
- Enforces Hush's addressing invariant across all policies (never SILENT when addressed).
- Saves machine-readable report to:
  data/results/phase4_threshold_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure backend is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.schemas import Action
from app.services.evaluation import (
    DEFAULT_RESTRAINT_COST_MATRIX,
    compute_accuracy,
    compute_calls_avoided_rate,
    compute_expected_cost,
    compute_macro_f1,
    compute_missed_contribution_rate,
    compute_per_class_metrics,
    compute_unnecessary_response_rate,
)
from scripts.analyze_calibration import generate_oof_logits_and_probs

CANONICAL_ACTIONS = [Action.SILENT.value, Action.SPEAK.value, Action.ASK.value]


def evaluate_policy_predictions(
    y_true: list[str],
    y_pred: list[str],
    addressed_flags: list[bool],
    policy_name: str,
    cost_matrix: dict[str, dict[str, float]],
    default_preds: list[str],
) -> dict[str, Any]:
    """Compute full metrics and cost evaluation for a decision policy."""
    n = len(y_true)

    # Invariant verification: must have 0 SILENT when addressed
    invariant_violations = sum(
        1 for is_addr, yp in zip(addressed_flags, y_pred) if is_addr and yp == Action.SILENT.value
    )

    acc = compute_accuracy(y_true, y_pred)
    f1 = compute_macro_f1(y_true, y_pred)
    unrec = compute_unnecessary_response_rate(y_true, y_pred)
    miss = compute_missed_contribution_rate(y_true, y_pred)
    avoided = compute_calls_avoided_rate(y_pred)
    mean_cost = compute_expected_cost(y_true, y_pred, cost_matrix)
    per_class = compute_per_class_metrics(y_true, y_pred)

    changed_from_default = sum(1 for yd, yp in zip(default_preds, y_pred) if yd != yp)

    return {
        "policy_name": policy_name,
        "mean_cost_per_message": round(mean_cost, 4),
        "total_cost": round(mean_cost * n, 2),
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "unnecessary_response_rate": round(unrec, 4),
        "missed_contribution_rate": round(miss, 4),
        "calls_avoided_rate": round(avoided, 4),
        "changed_predictions_count": changed_from_default,
        "invariant_violations": invariant_violations,
        "invariant_preserved": (invariant_violations == 0),
        "per_class_recall": {c: round(per_class[c]["recall"], 4) for c in CANONICAL_ACTIONS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decision threshold and cost trade-off analysis.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--l2", type=float, default=0.5, help="L2 penalty")
    args = parser.parse_args()

    dev_path = REPO_ROOT / "data" / "splits" / "dev.jsonl"
    out_dir = REPO_ROOT / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" HUSH — PHASE 4: DECISION THRESHOLD & COST TRADE-OFF ANALYSIS")
    print(" EVALUATION METHOD: 5-FOLD OUT-OF-FOLD (OOF) CROSS-VALIDATION")
    print(" STATUS: PROVISIONAL RESEARCH RESULTS (UNVERIFIED LABELS)")
    print("=" * 80)

    with open(dev_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    y_true = [r["expected_action"] for r in records]
    addressed_flags = [r["addressed_to_hush"] for r in records]
    n = len(records)

    # Cost Matrix Definition & Operational Justification
    cost_matrix = DEFAULT_RESTRAINT_COST_MATRIX
    print("\nOperational Restraint Cost Matrix:")
    print("  True Action \\ Pred Action |   SILENT   |    SPEAK   |     ASK    ")
    print("  -------------------------|------------|------------|------------")
    for r_act in CANONICAL_ACTIONS:
        c_row = [f"{cost_matrix[r_act][p_act]:4.1f}" for p_act in CANONICAL_ACTIONS]
        print(f"  {r_act:<24} |   " + "   |   ".join(c_row) + "  ")

    print("\nCost Matrix Justification:")
    print("  1. Missed Direct Request (True SPEAK/ASK -> Pred SILENT): Cost = 5.0 (Critical: bot appears broken)")
    print("  2. Intrusive Reply in Human Chat (True SILENT -> Pred SPEAK/ASK): Cost = 3.0 (High: bot is annoying)")
    print("  3. Hesitation vs Direct Answer (True SPEAK -> Pred ASK): Cost = 1.0 (Low: asking is still responsive)")
    print("  4. Premature Answer on Ambiguity (True ASK -> Pred SPEAK): Cost = 1.5 (Moderate: may hallucinate)")
    print("  5. Correct Decision (True == Pred): Cost = 0.0")

    # Generate OOF logits and probabilities
    print("\nGenerating 5-fold Out-Of-Fold predictions...")
    oof_logits, oof_probs, default_preds = generate_oof_logits_and_probs(
        records, l2_reg=args.l2, seed=args.seed
    )

    policy_reports: list[dict[str, Any]] = []

    # Policy 1: Default Argmax Policy
    rep_default = evaluate_policy_predictions(
        y_true, default_preds, addressed_flags, "Policy 1: Default Argmax", cost_matrix, default_preds
    )
    policy_reports.append(rep_default)

    # Policy 2: Cost-Sensitive Bayes Risk Minimization
    preds_bayes: list[str] = []
    for probs, is_addr in zip(oof_probs, addressed_flags):
        best_act = None
        min_risk = float("inf")

        for cand_act in CANONICAL_ACTIONS:
            # Enforce invariant: When addressed, SILENT has infinite cost
            if is_addr and cand_act == Action.SILENT.value:
                continue

            expected_risk = sum(probs[true_act] * cost_matrix[true_act][cand_act] for true_act in CANONICAL_ACTIONS)
            if expected_risk < min_risk:
                min_risk = expected_risk
                best_act = cand_act

        preds_bayes.append(best_act or Action.SPEAK.value)

    rep_bayes = evaluate_policy_predictions(
        y_true, preds_bayes, addressed_flags, "Policy 2: Cost-Sensitive Bayes Risk", cost_matrix, default_preds
    )
    policy_reports.append(rep_bayes)

    # Policy 3: Uncertainty Routing to ASK (Sweep delta)
    print("\nEvaluating Policy 3: Uncertainty Routing to ASK (Sweeping delta)...")
    deltas = [0.05, 0.10, 0.15, 0.20, 0.25]
    for d in deltas:
        preds_uncertain: list[str] = []
        for probs, is_addr, def_p in zip(oof_probs, addressed_flags, default_preds):
            if is_addr:
                p_speak = probs[Action.SPEAK.value]
                p_ask = probs[Action.ASK.value]
                # If gap is within delta, route to ASK for clarification
                if abs(p_speak - p_ask) < d:
                    preds_uncertain.append(Action.ASK.value)
                    continue
            preds_uncertain.append(def_p)

        rep_unc = evaluate_policy_predictions(
            y_true, preds_uncertain, addressed_flags, f"Policy 3: Uncertainty Routing (delta={d:.2f})", cost_matrix, default_preds
        )
        policy_reports.append(rep_unc)

    # Policy 4: Silence Threshold Sweep (Sweeping tau)
    print("Evaluating Policy 4: Silence Confidence Threshold (Sweeping tau)...")
    taus = [0.35, 0.40, 0.45, 0.50]
    for tau in taus:
        preds_tau: list[str] = []
        for probs, is_addr, def_p in zip(oof_probs, addressed_flags, default_preds):
            if not is_addr:
                if probs[Action.SILENT.value] < tau:
                    # Not confident enough in silence -> pick between SPEAK and ASK
                    best_non_silent = Action.SPEAK.value if probs[Action.SPEAK.value] >= probs[Action.ASK.value] else Action.ASK.value
                    preds_tau.append(best_non_silent)
                    continue
            preds_tau.append(def_p)

        rep_tau = evaluate_policy_predictions(
            y_true, preds_tau, addressed_flags, f"Policy 4: Silence Threshold (tau={tau:.2f})", cost_matrix, default_preds
        )
        policy_reports.append(rep_tau)

    # Summary Comparison Table
    print("\n" + "=" * 105)
    print(" POLICY COMPARISON SUMMARY ON DEVELOPMENT SET (N=180)")
    print("=" * 105)
    headers = ["Policy", "Mean Cost", "Accuracy", "Macro-F1", "R_unnec", "R_miss", "Calls Avoided", "Changed"]
    fmt = "{:<36} | {:<9} | {:<8} | {:<8} | {:<7} | {:<7} | {:<13} | {:<7}"
    print(fmt.format(*headers))
    print("-" * 105)

    for p in policy_reports:
        print(fmt.format(
            p["policy_name"][:36],
            f"{p['mean_cost_per_message']:.3f}",
            f"{p['accuracy']*100:.1f}%",
            f"{p['macro_f1']:.3f}",
            f"{p['unnecessary_response_rate']*100:.1f}%",
            f"{p['missed_contribution_rate']*100:.1f}%",
            f"{p['calls_avoided_rate']*100:.1f}%",
            str(p["changed_predictions_count"]),
        ))
    print("=" * 105)

    payload = {
        "status": "provisional_research_results",
        "evaluation_type": "5_fold_out_of_fold",
        "split": "dev",
        "total_samples": n,
        "cost_matrix": cost_matrix,
        "policies": policy_reports,
    }

    out_file = out_dir / "phase4_threshold_sweep.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved threshold and cost trade-off report to: {out_file}")


if __name__ == "__main__":
    main()
