#!/usr/bin/env python3
"""Development error analysis script for Hush Conversational Restraint Engine.

Performs fine-grained error categorization and diagnostic reporting on dev split:
- Generates Out-Of-Fold (OOF) predictions via 5-fold cross-validation.
- Compares Classifier errors vs Rule-Based Baseline errors on identical scenarios.
- Categorizes errors:
  * MISSED_CONTRIBUTION (silence when user needed response)
  * UNNECESSARY_INTERVENTION (spoke uninvited in human chat)
  * SPEAK_VS_ASK_CONFUSION (direct vs clarification confusion)
  * METADATA_OR_CONTEXT_MISMATCH
- Produces summary statistics and saves machine-readable report to:
  data/results/phase4_dev_error_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Ensure backend is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas import Action
from app.services.baselines import (
    RuleBasedBaseline,
    extract_inference_context,
)
from app.services.classifier import (
    CLASS_LABELS,
    CLASS_TO_IDX,
    MultinomialLogisticRegression,
)
from app.services.evaluation import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_macro_f1,
    compute_missed_contribution_rate,
    compute_per_class_metrics,
    compute_unnecessary_response_rate,
)
from app.services.features import extract_feature_vector, extract_features


def generate_oof_predictions(
    records: list[dict[str, Any]],
    n_splits: int = 5,
    l2_reg: float = 0.5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate strictly out-of-fold predictions to prevent evaluating on training data."""
    conv_to_indices = defaultdict(list)
    for idx, r in enumerate(records):
        conv_to_indices[r["conversation_id"]].append(idx)

    conv_ids = sorted(conv_to_indices.keys())
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for i, cid in enumerate(conv_ids):
        folds[i % n_splits].extend(conv_to_indices[cid])

    X_all = [extract_feature_vector(r) for r in records]
    y_all = [CLASS_TO_IDX[r["expected_action"]] for r in records]

    oof_results: list[dict[str, Any]] = [{} for _ in records]

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
            pred_action = model.predict(ctx)
            probs = model.predict_proba(ctx)
            logits = model.predict_logits(ctx)

            oof_results[val_idx] = {
                "scenario_id": r["id"],
                "conversation_id": r["conversation_id"],
                "category": r["category"],
                "address_method": r["address_method"],
                "addressed_to_hush": r["addressed_to_hush"],
                "expected_action": r["expected_action"],
                "predicted_action": pred_action,
                "confidence": probs[pred_action],
                "probabilities": probs,
                "logits": logits,
                "is_correct": (pred_action == r["expected_action"]),
                "fold": fold_idx,
            }

    return oof_results


def categorize_error(
    gold: str,
    pred: str,
    addressed: bool,
    category: str,
    features: dict[str, float],
) -> tuple[str, str]:
    """Assign an interpretable error category and observable feature explanation."""
    if gold == pred:
        return "CORRECT", "Prediction matches ground truth label."

    if gold == Action.SILENT.value and pred in {Action.SPEAK.value, Action.ASK.value}:
        if features.get("f07_has_question_mark", 0.0) == 1.0 and not addressed:
            return (
                "UNNECESSARY_INTERVENTION",
                "Spoke uninvited on unaddressed human question mark in channel.",
            )
        return (
            "UNNECESSARY_INTERVENTION",
            "Model predicted response on message where silence was expected.",
        )

    if gold in {Action.SPEAK.value, Action.ASK.value} and pred == Action.SILENT.value:
        return (
            "MISSED_CONTRIBUTION",
            "Model incorrectly remained silent when response or clarification was needed.",
        )

    if gold == Action.SPEAK.value and pred == Action.ASK.value:
        if features.get("f12_has_underspecified_cue", 0.0) == 1.0:
            return (
                "SPEAK_VS_ASK_CONFUSION",
                "Model detected ambiguity cues and requested clarification on answerable query.",
            )
        return (
            "SPEAK_VS_ASK_CONFUSION",
            "Model hesitated and chose ASK when direct answer was expected.",
        )

    if gold == Action.ASK.value and pred == Action.SPEAK.value:
        return (
            "SPEAK_VS_ASK_CONFUSION",
            "Model attempted direct answer on an underspecified or incomplete request.",
        )

    return "OTHER_ERROR", "Discrepancy between predicted and expected action."


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze development set errors for Hush.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--l2", type=float, default=0.5, help="L2 penalty")
    args = parser.parse_args()

    dev_path = REPO_ROOT / "data" / "splits" / "dev.jsonl"
    out_dir = REPO_ROOT / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" HUSH — PHASE 4: DEVELOPMENT ERROR ANALYSIS")
    print(" EVALUATION METHOD: 5-FOLD OUT-OF-FOLD (OOF) CROSS-VALIDATION")
    print(" STATUS: PROVISIONAL RESEARCH RESULTS (UNVERIFIED LABELS)")
    print("=" * 75)

    with open(dev_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} records from {dev_path}")

    # 1. Out-of-fold predictions for classifier
    print("Generating 5-fold Out-Of-Fold predictions for ML Restraint Classifier...")
    oof_results = generate_oof_predictions(records, l2_reg=args.l2, seed=args.seed)

    # 2. Rule-based baseline predictions
    rule_baseline = RuleBasedBaseline()
    rule_predictions = []
    for r in records:
        ctx = extract_inference_context(r)
        rule_predictions.append(rule_baseline.predict(ctx))

    y_true = [r["expected_action"] for r in records]
    y_pred_clf = [o["predicted_action"] for o in oof_results]
    y_pred_rul = rule_predictions

    # Core performance metrics
    acc_clf = compute_accuracy(y_true, y_pred_clf)
    f1_clf = compute_macro_f1(y_true, y_pred_clf)
    unrec_clf = compute_unnecessary_response_rate(y_true, y_pred_clf)
    miss_clf = compute_missed_contribution_rate(y_true, y_pred_clf)

    acc_rul = compute_accuracy(y_true, y_pred_rul)
    f1_rul = compute_macro_f1(y_true, y_pred_rul)
    unrec_rul = compute_unnecessary_response_rate(y_true, y_pred_rul)
    miss_rul = compute_missed_contribution_rate(y_true, y_pred_rul)

    print(f"\nOOF Classifier Performance: Accuracy={acc_clf*100:.2f}%, Macro-F1={f1_clf:.4f}")
    print(f"  - R_unnecessary: {unrec_clf*100:.2f}%, R_missed: {miss_clf*100:.2f}%")
    print(f"Rule Baseline Performance: Accuracy={acc_rul*100:.2f}%, Macro-F1={f1_rul:.4f}")
    print(f"  - R_unnecessary: {unrec_rul*100:.2f}%, R_missed: {miss_rul*100:.2f}%")

    # 3. Compile per-scenario error entries
    error_entries: list[dict[str, Any]] = []
    error_categories_counter: Counter[str] = Counter()
    gold_pred_counter: Counter[str] = Counter()
    category_error_counter: Counter[str] = Counter()

    for idx, (r, oof, pred_r) in enumerate(zip(records, oof_results, y_pred_rul)):
        gold = r["expected_action"]
        pred_c = oof["predicted_action"]
        feats = extract_features(r)

        err_cat, explanation = categorize_error(
            gold=gold,
            pred=pred_c,
            addressed=r["addressed_to_hush"],
            category=r["category"],
            features=feats,
        )

        is_err = not oof["is_correct"]
        if is_err:
            error_categories_counter[err_cat] += 1
            gold_pred_counter[f"{gold} -> {pred_c}"] += 1
            category_error_counter[r["category"]] += 1

            error_entries.append({
                "scenario_id": r["id"],
                "conversation_id": r["conversation_id"],
                "category": r["category"],
                "address_method": r["address_method"],
                "addressed_to_hush": r["addressed_to_hush"],
                "gold_action": gold,
                "predicted_action": pred_c,
                "rule_predicted_action": pred_r,
                "confidence": round(oof["confidence"], 4),
                "probabilities": oof["probabilities"],
                "error_category": err_cat,
                "explanation": explanation,
                "key_observable_features": {
                    "has_question_mark": feats["f07_has_question_mark"],
                    "has_action_request": feats["f11_has_action_request"],
                    "has_underspecified_cue": feats["f12_has_underspecified_cue"],
                    "has_technical_spec": feats["f13_has_technical_spec"],
                    "has_social_banter": feats["f14_has_social_banter"],
                    "is_reply_to_human": feats["f17_is_reply_to_human"],
                    "is_reply_to_assistant": feats["f18_is_reply_to_assistant"],
                },
            })

    total_errors = len(error_entries)
    print(f"\nTotal Out-Of-Fold Errors: {total_errors} / {len(records)} ({total_errors/len(records)*100:.2f}%)")
    print("\nError Breakdown by Error Category:")
    for cat, cnt in error_categories_counter.most_common():
        print(f"  - {cat:<30}: {cnt:2d} ({cnt/total_errors*100:.1f}%)")

    print("\nError Breakdown by (Gold -> Predicted):")
    for trans, cnt in gold_pred_counter.most_common():
        print(f"  - {trans:<20}: {cnt:2d}")

    print("\nError Breakdown by Scenario Category:")
    for sc_cat, cnt in category_error_counter.most_common():
        print(f"  - {sc_cat:<32}: {cnt:2d}")

    cm_oof = compute_confusion_matrix(y_true, y_pred_clf)
    per_class_oof = compute_per_class_metrics(y_true, y_pred_clf)

    report_payload = {
        "status": "provisional_research_results",
        "evaluation_type": "5_fold_out_of_fold",
        "split": "dev",
        "total_samples": len(records),
        "total_errors": total_errors,
        "overall_accuracy": round(acc_clf, 4),
        "macro_f1": round(f1_clf, 4),
        "unnecessary_response_rate": round(unrec_clf, 4),
        "missed_contribution_rate": round(miss_clf, 4),
        "confusion_matrix": cm_oof,
        "per_class_metrics": per_class_oof,
        "error_categories_summary": dict(error_categories_counter),
        "gold_to_pred_summary": dict(gold_pred_counter),
        "scenario_category_errors": dict(category_error_counter),
        "error_details": error_entries,
    }

    out_file = out_dir / "phase4_dev_error_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\nSaved detailed error analysis report to: {out_file}")


if __name__ == "__main__":
    main()
