#!/usr/bin/env python3
"""Unified baseline evaluation script for Hush Conversational Restraint Engine.

Evaluates baseline decision strategies on identical split data:
1. Majority-Class Baseline (fit strictly on dev split)
2. Always-Reply Baseline (always SPEAK)
3. Rule-Based Baseline (heuristics with addressing invariant)
4. Mock LLM Decider (provider-neutral interface, offline test)

Computes:
- Accuracy, Macro-F1, per-class metrics
- Restraint trade-off metrics (R_unnecessary, R_missed, Calls Avoided)
- 95% bootstrap confidence intervals (1,000 resamples, seed 42)
- Paired statistical tests (McNemar test for accuracy, paired bootstrap for Macro-F1)
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

from app.services.baselines import (
    AlwaysReplyBaseline,
    MajorityClassBaseline,
    MockLLMDecider,
    RuleBasedBaseline,
)
from app.services.evaluation import (
    CANONICAL_LABELS,
    EvaluationReport,
    evaluate_baseline_on_file,
    paired_bootstrap_macro_f1,
    paired_mcnemar_test,
)


def run_evaluation_suite(
    split_file: Path,
    dev_file: Path,
    split_name: str = "dev",
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run full evaluation suite across all baselines on the specified split file."""
    print(f"\n=======================================================")
    print(f" Running Evaluation Suite on split: {split_name.upper()}")
    print(f" File: {split_file}")
    print(f" Seed: {seed}, Bootstrap Resamples: {n_resamples}")
    print(f"=======================================================\n")

    # 1. Majority Class Baseline (must fit strictly on dev split only!)
    print("[1/4] Fitting and evaluating Majority-Class Baseline...")
    majority = MajorityClassBaseline.fit_from_file(dev_file)
    print(f"  -> Majority class fitted from dev: {majority.majority_class}")
    report_majority = evaluate_baseline_on_file(
        majority,
        split_file,
        model_name="Majority Class (SILENT)",
        split_name=split_name,
        n_resamples=n_resamples,
        seed=seed,
    )

    # 2. Always-Reply Baseline
    print("[2/4] Evaluating Always-Reply Baseline (SPEAK)...")
    always_reply = AlwaysReplyBaseline()
    report_always = evaluate_baseline_on_file(
        always_reply,
        split_file,
        model_name="Always-Reply (SPEAK)",
        split_name=split_name,
        n_resamples=n_resamples,
        seed=seed,
    )

    # 3. Rule-Based Baseline
    print("[3/4] Evaluating Rule-Based Baseline...")
    rule_baseline = RuleBasedBaseline(allow_ask=True)
    report_rules = evaluate_baseline_on_file(
        rule_baseline,
        split_file,
        model_name="Rule-Based Heuristic",
        split_name=split_name,
        n_resamples=n_resamples,
        seed=seed,
    )

    # 4. Mock LLM Decider
    print("[4/4] Evaluating Mock LLM-as-Decider Baseline (offline)...")
    mock_llm = MockLLMDecider(default_action="SILENT")
    report_mock_llm = evaluate_baseline_on_file(
        mock_llm,
        split_file,
        model_name="Mock LLM Decider",
        split_name=split_name,
        n_resamples=n_resamples,
        seed=seed,
    )

    reports = [report_majority, report_always, report_rules, report_mock_llm]

    # Paired comparisons against Majority and Always-Reply
    y_true = [p["expected_action"] for p in report_majority.predictions]
    preds_maj = [p["predicted_action"] for p in report_majority.predictions]
    preds_alw = [p["predicted_action"] for p in report_always.predictions]
    preds_rul = [p["predicted_action"] for p in report_rules.predictions]

    print("\nComputing paired significance tests for Rule-Based vs Baselines...")
    mcnemar_vs_maj = paired_mcnemar_test(y_true, preds_rul, preds_maj)
    f1_diff_vs_maj = paired_bootstrap_macro_f1(y_true, preds_rul, preds_maj, n_resamples=n_resamples, seed=seed)

    mcnemar_vs_alw = paired_mcnemar_test(y_true, preds_rul, preds_alw)
    f1_diff_vs_alw = paired_bootstrap_macro_f1(y_true, preds_rul, preds_alw, n_resamples=n_resamples, seed=seed)

    # Display comparison table
    print("\n" + "=" * 95)
    print(f" SUMMARY BASELINE COMPARISON — SPLIT: {split_name.upper()} (N={len(y_true)})")
    print("=" * 95)
    headers = ["Model", "Accuracy (95% CI)", "Macro-F1 (95% CI)", "R_unnecessary", "R_missed", "Calls Avoided"]
    row_fmt = "{:<24} | {:<22} | {:<22} | {:<13} | {:<13} | {:<12}"
    print(row_fmt.format(*headers))
    print("-" * 95)

    for r in reports:
        acc_str = f"{r.accuracy.value * 100:.1f}% [{r.accuracy.ci_lower * 100:.1f}%, {r.accuracy.ci_upper * 100:.1f}%]"
        f1_str = f"{r.macro_f1.value:.3f} [{r.macro_f1.ci_lower:.3f}, {r.macro_f1.ci_upper:.3f}]"
        unrec_str = f"{r.unnecessary_response_rate.value * 100:.1f}%"
        miss_str = f"{r.missed_contribution_rate.value * 100:.1f}%"
        ca_str = f"{r.calls_avoided_rate.value * 100:.1f}%"
        print(row_fmt.format(r.model_name[:24], acc_str, f1_str, unrec_str, miss_str, ca_str))
    print("=" * 95)

    # Print markdown reports
    for r in reports:
        print("\n" + r.to_markdown())

    # Print paired tests
    print("\n### Paired Statistical Significance Tests")
    print(f"- **Rule-Based vs Majority Class:**")
    print(f"  - McNemar's Test: statistic={mcnemar_vs_maj['statistic']}, p-value={mcnemar_vs_maj['p_value']} ({mcnemar_vs_maj['method']})")
    print(f"  - Delta Macro-F1: {f1_diff_vs_maj['delta']:+.4f} (95% CI [{f1_diff_vs_maj['ci_lower']:+.4f}, {f1_diff_vs_maj['ci_upper']:+.4f}], p={f1_diff_vs_maj['p_value']})")
    print(f"- **Rule-Based vs Always-Reply:**")
    print(f"  - McNemar's Test: statistic={mcnemar_vs_alw['statistic']}, p-value={mcnemar_vs_alw['p_value']} ({mcnemar_vs_alw['method']})")
    print(f"  - Delta Macro-F1: {f1_diff_vs_alw['delta']:+.4f} (95% CI [{f1_diff_vs_alw['ci_lower']:+.4f}, {f1_diff_vs_alw['ci_upper']:+.4f}], p={f1_diff_vs_alw['p_value']})")

    results_payload = {
        "split": split_name,
        "n_samples": len(y_true),
        "seed": seed,
        "n_resamples": n_resamples,
        "reports": {r.model_name: r.to_dict() for r in reports},
        "paired_comparisons": {
            "rule_vs_majority": {
                "mcnemar": mcnemar_vs_maj,
                "delta_macro_f1": f1_diff_vs_maj,
            },
            "rule_vs_always_reply": {
                "mcnemar": mcnemar_vs_alw,
                "delta_macro_f1": f1_diff_vs_alw,
            },
        },
    }

    return results_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline deciders on Hush dataset splits.")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test", "all"], help="Dataset split to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap resampling")
    parser.add_argument("--resamples", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument("--output-dir", type=str, default="data/results", help="Directory to save evaluation results")
    args = parser.parse_args()

    data_dir = REPO_ROOT / "data"
    splits_dir = data_dir / "splits"
    dev_file = splits_dir / "dev.jsonl"
    test_file = splits_dir / "test.jsonl"

    if not dev_file.exists():
        print(f"Error: Dev split not found at {dev_file}", file=sys.stderr)
        sys.exit(1)

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_to_run = []
    if args.split in ("dev", "all"):
        splits_to_run.append(("dev", dev_file))
    if args.split in ("test", "all"):
        if not test_file.exists():
            print(f"Error: Test split not found at {test_file}", file=sys.stderr)
            sys.exit(1)
        splits_to_run.append(("test", test_file))

    for split_name, file_path in splits_to_run:
        res = run_evaluation_suite(
            split_file=file_path,
            dev_file=dev_file,
            split_name=split_name,
            n_resamples=args.resamples,
            seed=args.seed,
        )
        out_file = out_dir / f"{split_name}_baseline_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved {split_name} baseline results to: {out_file}")


if __name__ == "__main__":
    main()
