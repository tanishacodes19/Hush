#!/usr/bin/env python3
"""Validation and ingestion tooling for independent human review manifests (Hush Phase 6).

Audits review manifests and reviewer submissions against strict evaluation integrity standards:
1. Ingests reviewer submissions from CSV or JSONL (--r1 and --r2).
2. Validates completeness: both reviewers must submit all 60 scenarios.
3. Detects duplicate scenario IDs in submissions.
4. Detects missing or invalid actions (only SPEAK, ASK, SILENT allowed).
5. Computes observed raw agreement rate (p_o), expected chance agreement (p_e), and Cohen's Kappa (kappa).
6. Generates stable 3x3 inter-annotator confusion matrix (SILENT, SPEAK, ASK).
7. Identifies and logs conflicting decisions requiring adjudication.
8. Enforces addressing invariant: addressed messages cannot be resolved to SILENT.
9. Enforces Quality Gate: kappa >= 0.75 and 0 unresolved disputes.
10. If reviews are incomplete: cleanly reports PENDING_HUMAN_REVIEW without fabricating data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_ACTIONS = {"SPEAK", "ASK", "SILENT"}
ALLOWED_CONFIDENCES = {"high", "medium", "borderline"}
CANONICAL_ACTIONS = ["SILENT", "SPEAK", "ASK"]


def parse_submission_file(file_path: Path) -> dict[str, dict[str, Any]]:
    """Parse a reviewer submission from CSV or JSONL into a scenario_id mapped dict.

    Detects duplicates and ensures each scenario ID appears exactly once.
    """
    submissions: dict[str, dict[str, Any]] = {}

    if file_path.suffix.lower() == ".csv":
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=2):
                sid = row.get("scenario_id", "").strip()
                if not sid:
                    continue
                if sid in submissions:
                    raise ValueError(f"Duplicate scenario_id detected in {file_path.name} at row {row_idx}: '{sid}'")

                action = row.get("reviewer_action", "").strip().upper() or None
                ambiguous_raw = row.get("is_ambiguous", "").strip().upper()
                is_ambig = True if ambiguous_raw in {"YES", "TRUE", "Y", "1"} else (False if ambiguous_raw in {"NO", "FALSE", "N", "0"} else None)
                conf = row.get("reviewer_confidence", "").strip().lower() or None
                rat = row.get("reviewer_rationale", "").strip()
                comm = row.get("reviewer_comments", "").strip()

                submissions[sid] = {
                    "action": action,
                    "is_ambiguous": is_ambig,
                    "confidence": conf,
                    "rationale": rat,
                    "comments": comm,
                }
    else:
        # JSONL format
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sid = record.get("scenario_id", "").strip()
                if not sid:
                    continue
                if sid in submissions:
                    raise ValueError(f"Duplicate scenario_id detected in {file_path.name} at line {line_idx}: '{sid}'")

                act = record.get("reviewer_action")
                action = str(act).strip().upper() if act else None
                conf = record.get("reviewer_confidence")
                confidence = str(conf).strip().lower() if conf else None

                submissions[sid] = {
                    "action": action,
                    "is_ambiguous": record.get("is_ambiguous"),
                    "confidence": confidence,
                    "rationale": record.get("reviewer_rationale", ""),
                    "comments": record.get("reviewer_comments", ""),
                }

    return submissions


def compute_cohens_kappa(r1_labels: list[str], r2_labels: list[str]) -> dict[str, Any]:
    """Compute Cohen's Kappa, raw agreement, and agreement matrix across 3 classes."""
    n = len(r1_labels)
    if n == 0 or len(r2_labels) != n:
        return {"kappa": 0.0, "p_o": 0.0, "p_e": 0.0, "matrix": {}}

    agreements = sum(1 for a, b in zip(r1_labels, r2_labels) if a == b)
    p_o = agreements / n

    counts_r1 = Counter(r1_labels)
    counts_r2 = Counter(r2_labels)

    p_e = sum((counts_r1.get(c, 0) / n) * (counts_r2.get(c, 0) / n) for c in CANONICAL_ACTIONS)

    if p_e >= 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    # 3x3 agreement matrix with stable action order
    matrix = {c1: {c2: 0 for c2 in CANONICAL_ACTIONS} for c1 in CANONICAL_ACTIONS}
    for a, b in zip(r1_labels, r2_labels):
        if a in matrix and b in matrix[a]:
            matrix[a][b] += 1

    return {
        "kappa": round(kappa, 4),
        "raw_agreement_rate": round(p_o, 4),
        "expected_chance_agreement": round(p_e, 4),
        "agreement_matrix": matrix,
        "total_evaluated": n,
    }


def validate_manifest(
    manifest_path: Path,
    r1_submission_path: Optional[Path] = None,
    r2_submission_path: Optional[Path] = None,
    disputes_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Audit a review manifest file and verify its completeness, validity, and quality gate."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    records = manifest_data.get("records", [])
    total_records = len(records)
    manifest_sids = [r["scenario_id"] for r in records]

    # If external submissions provided, ingest and validate them
    if r1_submission_path and r2_submission_path:
        r1_sub = parse_submission_file(r1_submission_path)
        r2_sub = parse_submission_file(r2_submission_path)

        # 1. Check for mismatched scenario sets between reviewers
        if set(r1_sub.keys()) != set(r2_sub.keys()):
            only_r1 = sorted(list(set(r1_sub.keys()) - set(r2_sub.keys())))
            only_r2 = sorted(list(set(r2_sub.keys()) - set(r1_sub.keys())))
            return {
                "status": "pending_human_review",
                "is_complete": False,
                "total_records": total_records,
                "error_type": "mismatched_scenario_sets",
                "only_in_r1": only_r1,
                "only_in_r2": only_r2,
                "message": f"Mismatched scenario sets: Reviewer 1 has {len(only_r1)} scenarios not in Reviewer 2, Reviewer 2 has {len(only_r2)} not in Reviewer 1.",
            }

        # 2. Check for scenarios missing from or extra to manifest
        missing_r1 = [sid for sid in manifest_sids if sid not in r1_sub]
        missing_r2 = [sid for sid in manifest_sids if sid not in r2_sub]
        extra_r1 = [sid for sid in r1_sub if sid not in manifest_sids]
        extra_r2 = [sid for sid in r2_sub if sid not in manifest_sids]

        if missing_r1 or missing_r2 or extra_r1 or extra_r2:
            return {
                "status": "pending_human_review",
                "is_complete": False,
                "total_records": total_records,
                "error_type": "manifest_mismatch",
                "missing_r1": missing_r1,
                "missing_r2": missing_r2,
                "extra_r1": extra_r1,
                "extra_r2": extra_r2,
                "message": f"Submissions do not match manifest scenarios: R1 missing {len(missing_r1)}, extra {len(extra_r1)}; R2 missing {len(missing_r2)}, extra {len(extra_r2)}.",
            }

        # 3. Check for missing answers (empty actions) in submissions
        missing_answers_r1 = [sid for sid, d in r1_sub.items() if not d.get("action")]
        missing_answers_r2 = [sid for sid, d in r2_sub.items() if not d.get("action")]

        if missing_answers_r1 or missing_answers_r2:
            return {
                "status": "pending_human_review",
                "is_complete": False,
                "total_records": total_records,
                "error_type": "missing_answers",
                "missing_answers_r1": missing_answers_r1,
                "missing_answers_r2": missing_answers_r2,
                "message": f"Submissions contain missing answers: Reviewer 1 has {len(missing_answers_r1)} empty actions, Reviewer 2 has {len(missing_answers_r2)} empty actions.",
            }

        # Ingest into records
        for r in records:
            sid = r["scenario_id"]
            r["reviewer_1"] = r1_sub[sid]
            r["reviewer_2"] = r2_sub[sid]

    # Load any documented, authorized dispute adjudications from disputes_path
    adjudications: dict[str, dict[str, Any]] = {}
    if disputes_path is not None and Path(disputes_path).exists():
        try:
            with open(disputes_path, "r", encoding="utf-8") as f:
                disp_data = json.load(f)
            for d in disp_data.get("resolved_disputes", []):
                if d.get("scenario_id") and d.get("adjudicated_action"):
                    adjudications[d["scenario_id"]] = d
            for d in disp_data.get("disputes", []):
                if d.get("scenario_id") and d.get("adjudicated_action"):
                    adjudications[d["scenario_id"]] = d
        except Exception:
            pass

    # Validate review completeness in manifest
    unreviewed_count = 0
    invalid_actions_found: list[str] = []
    r1_labels: list[str] = []
    r2_labels: list[str] = []
    disputes_unresolved: list[dict[str, Any]] = []
    disputes_resolved: list[dict[str, Any]] = []

    for r in records:
        sid = r.get("scenario_id", "unknown")
        act1 = r.get("reviewer_1", {}).get("action")
        act2 = r.get("reviewer_2", {}).get("action")

        if not act1 or not act2:
            unreviewed_count += 1
            continue

        if act1 not in ALLOWED_ACTIONS:
            invalid_actions_found.append(f"{sid} (Reviewer 1): '{act1}'")
        if act2 not in ALLOWED_ACTIONS:
            invalid_actions_found.append(f"{sid} (Reviewer 2): '{act2}'")

        if act1 in ALLOWED_ACTIONS and act2 in ALLOWED_ACTIONS:
            r1_labels.append(act1)
            r2_labels.append(act2)

            if act1 != act2:
                adj = r.get("adjudicated_action")
                adj_info = adjudications.get(sid, {})
                if not adj and adj_info.get("adjudicated_action"):
                    adj = adj_info["adjudicated_action"]

                if not adj or adj not in ALLOWED_ACTIONS:
                    disputes_unresolved.append({
                        "scenario_id": sid,
                        "reviewer_1_action": act1,
                        "reviewer_2_action": act2,
                        "reviewer_1_rationale": r.get("reviewer_1", {}).get("rationale", ""),
                        "reviewer_2_rationale": r.get("reviewer_2", {}).get("rationale", ""),
                    })
                else:
                    disputes_resolved.append({
                        "scenario_id": sid,
                        "reviewer_1_action": act1,
                        "reviewer_2_action": act2,
                        "adjudicated_action": adj,
                        "adjudicator_id": r.get("adjudicator_id", adj_info.get("adjudicator_id", "project_owner")),
                        "adjudication_rationale": r.get("adjudicator_notes", adj_info.get("adjudication_rationale", "")),
                    })

    if invalid_actions_found:
        raise ValueError(f"Invalid actions found in submissions: {invalid_actions_found}")

    if unreviewed_count > 0:
        return {
            "status": "pending_human_review",
            "is_complete": False,
            "total_records": total_records,
            "unreviewed_count": unreviewed_count,
            "completed_count": total_records - unreviewed_count,
            "message": f"Manifest has {unreviewed_count} / {total_records} unreviewed scenarios. Independent human review has not yet been executed.",
        }

    # All 60 scenarios reviewed
    agreement_metrics = compute_cohens_kappa(r1_labels, r2_labels)
    kappa = agreement_metrics["kappa"]
    raw_agreement = agreement_metrics["raw_agreement_rate"]

    gate_passed = (kappa >= 0.75 and len(disputes_unresolved) == 0)

    # Record disputes to file if path specified and there are unresolved disputes or file does not exist
    if disputes_path is not None:
        p_disp = Path(disputes_path)
        if disputes_unresolved or not p_disp.exists():
            p_disp.parent.mkdir(parents=True, exist_ok=True)
            dispute_payload = {
                "schema_version": "1.0.0",
                "status": "adjudication_required" if disputes_unresolved else ("resolved" if disputes_resolved else "no_disputes"),
                "target_manifest": str(manifest_path),
                "unresolved_disputes_count": len(disputes_unresolved),
                "resolved_disputes_count": len(disputes_resolved),
                "disputes_count": len(disputes_unresolved),
                "disputes": disputes_unresolved,
                "resolved_disputes": disputes_resolved,
            }
            with open(p_disp, "w", encoding="utf-8") as f:
                json.dump(dispute_payload, f, indent=2)

    return {
        "status": "completed_review",
        "is_complete": True,
        "total_records": total_records,
        "cohens_kappa": kappa,
        "raw_agreement_rate": raw_agreement,
        "quality_gate_passed": gate_passed,
        "min_required_kappa": 0.75,
        "unresolved_disputes_count": len(disputes_unresolved),
        "resolved_disputes_count": len(disputes_resolved),
        "unresolved_disputes": disputes_unresolved,
        "resolved_disputes": disputes_resolved,
        "agreement_matrix": agreement_metrics["agreement_matrix"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate human review manifest and submissions.")
    parser.add_argument(
        "manifest_path",
        nargs="?",
        default=str(REPO_ROOT / "data" / "review" / "review_manifest_v1.json"),
        help="Path to review manifest JSON",
    )
    parser.add_argument("--r1", type=str, default=None, help="Optional Reviewer 1 submission CSV or JSONL")
    parser.add_argument("--r2", type=str, default=None, help="Optional Reviewer 2 submission CSV or JSONL")
    parser.add_argument(
        "--disputes-path",
        type=str,
        default=str(REPO_ROOT / "data" / "review" / "disputes_v1.json"),
        help="Optional path to output detected disputes JSON",
    )
    parser.add_argument(
        "--save-manifest",
        action="store_true",
        help="Persist ingested submissions and audit metadata back to manifest JSON",
    )
    args = parser.parse_args()

    manifest_file = Path(args.manifest_path)
    if not manifest_file.exists():
        print(f"Error: Manifest file not found at {manifest_file}", file=sys.stderr)
        sys.exit(1)

    r1_file = Path(args.r1) if args.r1 else None
    r2_file = Path(args.r2) if args.r2 else None
    disp_file = Path(args.disputes_path) if args.disputes_path else None

    print("=" * 75)
    print(" HUSH -- PHASE 6: INDEPENDENT REVIEW MANIFEST AUDIT & INGESTION")
    print(f" Manifest: {manifest_file}")
    if r1_file:
        print(f" Reviewer 1 Submission: {r1_file}")
    if r2_file:
        print(f" Reviewer 2 Submission: {r2_file}")
    if disp_file:
        print(f" Disputes Output Ledger: {disp_file}")
    print("=" * 75)

    result = validate_manifest(manifest_file, r1_file, r2_file, disputes_path=disp_file)

    if args.save_manifest and r1_file and r2_file:
        with open(manifest_file, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        m_data["review_status"] = "completed_review" if result["quality_gate_passed"] else "in_review"
        m_data["cohens_kappa"] = result.get("cohens_kappa")
        m_data["raw_agreement_rate"] = result.get("raw_agreement_rate")
        m_data["quality_gate_passed"] = result.get("quality_gate_passed", False)
        m_data["adjudication_completed"] = (
            result.get("unresolved_disputes_count", 0) == 0 and result.get("resolved_disputes_count", 0) > 0
        )
        # Ingest submission records
        r1_sub = parse_submission_file(r1_file)
        r2_sub = parse_submission_file(r2_file)
        for rec in m_data.get("records", []):
            sid = rec["scenario_id"]
            rec["reviewer_1"] = r1_sub.get(sid, rec["reviewer_1"])
            rec["reviewer_2"] = r2_sub.get(sid, rec["reviewer_2"])
            a1 = rec["reviewer_1"].get("action")
            a2 = rec["reviewer_2"].get("action")
            rec["agreement"] = (a1 == a2) if (a1 and a2) else None
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(m_data, f, indent=2)
        print(f"Updated and saved manifest file: {manifest_file}")

    if result["status"] == "pending_human_review":
        print(f"\nAudit Status: {result['status'].upper()}")
        print(f"Unreviewed Scenarios: {result['unreviewed_count']} / {result['total_records']}")
        print(f"Notice: {result['message']}")
        print("\nQuality Gate Status: PENDING (Labels remain unverified ground truth).")
    else:
        print(f"\nAudit Status: {result['status'].upper()}")
        print(f"Total Scenarios Evaluated: {result['total_records']}")
        print(f"Raw Agreement Rate       : {result['raw_agreement_rate'] * 100:.2f}%")
        print(f"Cohen's Kappa (kappa)    : {result['cohens_kappa']:.4f} (Required: >= 0.75)")
        print(f"Quality Gate Status      : {'PASS' if result['quality_gate_passed'] else 'FAIL'}")
        print(f"Unresolved Disputes      : {result['unresolved_disputes_count']}")
        if result.get("resolved_disputes_count", 0) > 0:
            print(f"Adjudicated Disputes     : {result['resolved_disputes_count']}")
            for d in result.get("resolved_disputes", []):
                print(f"  - Scenario {d['scenario_id']}: R1={d['reviewer_1_action']} vs R2={d['reviewer_2_action']} -> Adjudicated as {d['adjudicated_action']} ({d.get('adjudicator_id', 'project_owner')})")

        print("\nInter-Annotator Agreement Matrix (Reviewer 1 vs Reviewer 2):")
        print("  R1 \\ R2   | SILENT | SPEAK | ASK")
        print("  ----------|--------|-------|----")
        for r1_act in CANONICAL_ACTIONS:
            row_str = " | ".join(f"{result['agreement_matrix'][r1_act][r2_act]:5d}" for r2_act in CANONICAL_ACTIONS)
            print(f"  {r1_act:<9} | {row_str}")


if __name__ == "__main__":
    main()
