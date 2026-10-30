#!/usr/bin/env python3
"""Training and validation script for Hush Conversational Restraint Classifier.

Implements Phase 3 model training:
1. Feature extraction from development split (data/splits/dev.jsonl).
2. Grouped K-Fold cross-validation on dev split (grouped by conversation_id) for lambda selection.
3. Model training using pure-Python MultinomialLogisticRegression with balanced class weighting.
4. Preserves model weights in data/models/restraint_classifier_v1.json.
5. Evaluates on dev split using unified evaluation harness.
6. Clearly labels all outputs as PROVISIONAL RESEARCH RESULTS per dataset provenance audit.
"""

from __future__ import annotations

import argparse
import json
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

from app.services.classifier import (
    CLASS_LABELS,
    CLASS_TO_IDX,
    MultinomialLogisticRegression,
    train_classifier_on_dev,
)
from app.services.evaluation import (
    evaluate_baseline_on_file,
    paired_bootstrap_macro_f1,
    paired_mcnemar_test,
)
from app.services.features import extract_feature_vector


def grouped_k_fold_cv(
    records: list[dict[str, Any]],
    n_splits: int = 5,
    candidates_l2: list[float] = [0.05, 0.1, 0.5, 1.0, 2.0],
    balanced: bool = True,
    seed: int = 42,
) -> tuple[float, dict[float, float]]:
    """Grouped K-Fold Cross Validation by conversation_id on development split."""
    conv_to_indices = defaultdict(list)
    for idx, r in enumerate(records):
        conv_to_indices[r["conversation_id"]].append(idx)

    conv_ids = sorted(conv_to_indices.keys())
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for i, cid in enumerate(conv_ids):
        folds[i % n_splits].extend(conv_to_indices[cid])

    X_all = [extract_feature_vector(r) for r in records]
    y_all = [CLASS_TO_IDX[r["expected_action"]] for r in records]

    l2_scores: dict[float, float] = {}

    for l2 in candidates_l2:
        fold_accuracies: list[float] = []
        for fold_idx in range(n_splits):
            val_indices = set(folds[fold_idx])
            train_indices = [i for i in range(len(records)) if i not in val_indices]

            X_train = [X_all[i] for i in train_indices]
            y_train = [y_all[i] for i in train_indices]

            X_val = [X_all[i] for i in val_indices]
            y_val = [y_all[i] for i in val_indices]

            model = MultinomialLogisticRegression(
                l2_reg=l2,
                max_epochs=350,
                balanced_class_weights=balanced,
                seed=seed,
            )
            model.fit(X_train, y_train)

            correct = 0
            for x, y_true in zip(X_val, y_val):
                probs = model.predict_proba(x)
                pred_c = max(probs.items(), key=lambda kv: kv[1])[0]
                if CLASS_TO_IDX[pred_c] == y_true:
                    correct += 1
            fold_accuracies.append(correct / len(X_val) if X_val else 0.0)

        mean_acc = sum(fold_accuracies) / len(fold_accuracies)
        l2_scores[l2] = mean_acc

    best_l2 = max(l2_scores.items(), key=lambda kv: kv[1])[0]
    return best_l2, l2_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Hush Restraint Classifier on Dev Split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--l2", type=float, default=None, help="L2 regularization penalty (auto-tuned if None)")
    parser.add_argument("--balanced", action="store_true", default=True, help="Use balanced class weights")
    args = parser.parse_args()

    dev_file = REPO_ROOT / "data" / "splits" / "dev.jsonl"
    models_dir = REPO_ROOT / "data" / "models"
    results_dir = REPO_ROOT / "data" / "results"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" HUSH — CONVERSATIONAL RESTRAINT CLASSIFIER TRAINING")
    print(" STATUS: PROVISIONAL RESEARCH PIPELINE (PER PROVENANCE AUDIT)")
    print("=" * 70)

    # 1. Load dev records
    with open(dev_file, "r", encoding="utf-8") as f:
        dev_records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(dev_records)} records from development split: {dev_file}")

    # 2. Hyperparameter selection on dev via Grouped-CV
    if args.l2 is None:
        print("\nRunning 5-fold Grouped-CV (by conversation_id) on dev split...")
        best_l2, l2_scores = grouped_k_fold_cv(dev_records, balanced=args.balanced, seed=args.seed)
        for l2_val, score in l2_scores.items():
            print(f"  - L2 penalty {l2_val:4.2f}: Mean CV Accuracy = {score * 100:.2f}%")
        print(f"Selected optimal L2 penalty: {best_l2}")
    else:
        best_l2 = args.l2
        print(f"\nUsing manual L2 penalty: {best_l2}")

    # 3. Train final model on full dev split
    print(f"\nFitting final MultinomialLogisticRegression (L2={best_l2}, balanced={args.balanced}, seed={args.seed})...")
    model = train_classifier_on_dev(dev_file, l2_reg=best_l2, balanced_class_weights=args.balanced, seed=args.seed)

    # 4. Save model weights artifact
    model_path = models_dir / "restraint_classifier_v1.json"
    model.save(model_path)
    print(f"Saved trained model artifact to: {model_path}")

    # 5. Evaluate on dev split using unified evaluation harness
    print("\nEvaluating trained classifier on dev split...")
    report_clf = evaluate_baseline_on_file(
        model,
        dev_file,
        model_name="ML Restraint Classifier (L2 Reg, Balanced)",
        split_name="dev",
        n_resamples=1000,
        seed=args.seed,
    )

    print("\n" + report_clf.to_markdown())

    # 6. Compare vs Rule-Based Baseline on dev
    dev_results_file = results_dir / "dev_baseline_results.json"
    paired_summary = {}
    if dev_results_file.exists():
        with open(dev_results_file, "r", encoding="utf-8") as f:
            dev_baseline_data = json.load(f)
        rules_report = dev_baseline_data.get("reports", {}).get("Rule-Based Heuristic")
        if rules_report:
            y_true = [p["expected_action"] for p in report_clf.predictions]
            preds_clf = [p["predicted_action"] for p in report_clf.predictions]
            # Match predictions from baseline
            report_rules = evaluate_baseline_on_file(
                from_service := __import__("app.services.baselines", fromlist=["RuleBasedBaseline"]).RuleBasedBaseline(),
                dev_file,
                split_name="dev",
                n_resamples=100,
            )
            preds_rules = [p["predicted_action"] for p in report_rules.predictions]

            mcnemar = paired_mcnemar_test(y_true, preds_clf, preds_rules)
            f1_diff = paired_bootstrap_macro_f1(y_true, preds_clf, preds_rules, n_resamples=1000, seed=args.seed)

            print("\n### Paired Significance Tests: Classifier vs Rule-Based Baseline (Dev)")
            print(f"- McNemar's Test: statistic={mcnemar['statistic']}, p-value={mcnemar['p_value']} ({mcnemar['method']})")
            print(f"- Delta Macro-F1: {f1_diff['delta']:+.4f} (95% CI [{f1_diff['ci_lower']:+.4f}, {f1_diff['ci_upper']:+.4f}], p={f1_diff['p_value']})")

            paired_summary = {
                "vs_rule_based": {
                    "mcnemar": mcnemar,
                    "delta_macro_f1": f1_diff,
                }
            }

    # 7. Save results payload
    payload = {
        "status": "provisional_research_results",
        "provenance_gate": "unverified_human_labels",
        "split": "dev",
        "n_samples": len(dev_records),
        "model_artifact": str(model_path.relative_to(REPO_ROOT)),
        "hyperparameters": {
            "l2_reg": best_l2,
            "balanced_class_weights": args.balanced,
            "max_epochs": model.max_epochs,
            "learning_rate": model.learning_rate,
            "seed": args.seed,
        },
        "evaluation_report": report_clf.to_dict(),
        "paired_comparisons": paired_summary,
    }

    out_file = results_dir / "dev_classifier_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved dev classifier results to: {out_file}")


if __name__ == "__main__":
    main()
