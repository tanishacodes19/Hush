#!/usr/bin/env python3
"""Summary reporting script for independent dataset review status (Hush Phase 6).

Generates a concise, executive markdown status report from review manifest:
- Number of scenarios reviewed.
- Completion rate for Reviewer 1 and Reviewer 2.
- Raw agreement rate and Cohen's Kappa.
- 3x3 agreement matrix.
- Conflicting decisions requiring adjudication.
- Quality-gate pass/fail status (kappa >= 0.75).
- Remaining missing or invalid records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure backend and scripts are in sys.path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_review_manifest import CANONICAL_ACTIONS, validate_manifest


def generate_review_summary_markdown(manifest_path: Path) -> str:
    """Generate human-friendly markdown summary of review manifest status."""
    res = validate_manifest(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    total = len(records)

    r1_done = sum(1 for r in records if r.get("reviewer_1", {}).get("action"))
    r2_done = sum(1 for r in records if r.get("reviewer_2", {}).get("action"))

    lines = [
        "# Hush -- Independent Dataset Review Status Report",
        "",
        f"- **Manifest File:** `{manifest_path.name}`",
        f"- **Manifest Version:** `{data.get('manifest_version', 'unknown')}`",
        f"- **Sample Size:** {total} stratified scenarios",
        f"- **Review Status:** `{res['status'].upper()}`",
        "",
        "## 1. Reviewer Completion Status",
        "",
        "| Reviewer | Assigned Scenarios | Completed Scenarios | Completion Rate | Status |",
        "|---|---|---|---|---|",
        f"| **Reviewer 1** | {total} | {r1_done} | {r1_done/total*100:.1f}% | {'COMPLETED' if r1_done == total else 'PENDING'} |",
        f"| **Reviewer 2** | {total} | {r2_done} | {r2_done/total*100:.1f}% | {'COMPLETED' if r2_done == total else 'PENDING'} |",
        "",
    ]

    if res["status"] == "pending_human_review":
        lines.extend([
            "## 2. Quality-Gate Readiness",
            "",
            "> [!IMPORTANT]",
            f"> **Quality Gate Status: PENDING (Labels Remain Unverified)**  ",
            f"> Independent human review has not yet been executed ({res['unreviewed_count']} / {total} scenarios remaining).",
            "> No synthetic decisions or approval records were fabricated.",
            "> Actual human annotators must complete `reviewer_1_form.csv` and `reviewer_2_form.csv`.",
            "",
            "## 3. Next Steps for Execution",
            "1. Deliver `data/review/reviewer_1_form.csv` to Human Reviewer 1.",
            "2. Deliver `data/review/reviewer_2_form.csv` to Human Reviewer 2.",
            "3. Ensure both reviewers read `data/review/review_guidelines.md`.",
            "4. Ingest submissions using `python scripts/validate_review_manifest.py --r1 <r1.csv> --r2 <r2.csv>`.",
            r"5. Verify Cohen's Kappa reaches $\kappa \ge 0.75$.",
        ])
    else:
        matrix = res.get("agreement_matrix", {})
        lines.extend([
            "## 2. Agreement Metrics & Quality Gate",
            "",
            "| Metric | Measured Value | Standard Required | Status |",
            "|---|---|---|---|",
            f"| **Observed Raw Agreement** | {res['raw_agreement_rate']*100:.2f}% | Informative | - |",
            rf"| **Cohen's Kappa ($\kappa$)** | {res['cohens_kappa']:.4f} | $\ge 0.75$ | {'PASS' if res['cohens_kappa'] >= 0.75 else 'FAIL'} |",
            f"| **Unresolved Disputes** | {res['unresolved_disputes_count']} | $0$ | {'PASS' if res['unresolved_disputes_count'] == 0 else 'FAIL'} |",
            f"| **Quality Gate Determination** | - | **ALL PASS** | **{'PASS' if res['quality_gate_passed'] else 'FAIL'}** |",
            "",
            "### Inter-Annotator Agreement Matrix (Reviewer 1 vs Reviewer 2)",
            "",
            "| R1 \\ R2 | SILENT | SPEAK | ASK |",
            "|---|---|---|---|",
        ])
        for r1_act in CANONICAL_ACTIONS:
            r_vals = " | ".join(str(matrix.get(r1_act, {}).get(r2_act, 0)) for r2_act in CANONICAL_ACTIONS)
            lines.append(f"| **{r1_act}** | {r_vals} |")

        if res.get("resolved_disputes"):
            lines.extend([
                "",
                "### Adjudicated Decisions",
                "",
                "| Scenario ID | Reviewer 1 Action | Reviewer 2 Action | Adjudicated Action | Adjudicator | Rationale |",
                "|---|---|---|---|---|---|",
            ])
            for d in res["resolved_disputes"]:
                lines.append(
                    f"| `{d['scenario_id']}` | `{d['reviewer_1_action']}` | `{d['reviewer_2_action']}` | "
                    f"**`{d['adjudicated_action']}`** | `{d.get('adjudicator_id', 'project_owner')}` | "
                    f"{d.get('adjudication_rationale', '')[:80]}... |"
                )

        if res.get("unresolved_disputes"):
            lines.extend([
                "",
                "### Conflicting Decisions Requiring Adjudication",
                "",
                "| Scenario ID | Reviewer 1 Action | Reviewer 2 Action | R1 Rationale | R2 Rationale |",
                "|---|---|---|---|---|",
            ])
            for d in res["unresolved_disputes"][:10]:
                lines.append(f"| `{d['scenario_id']}` | `{d['reviewer_1_action']}` | `{d['reviewer_2_action']}` | {d['reviewer_1_rationale'][:40]} | {d['reviewer_2_rationale'][:40]} |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate human-friendly review status summary.")
    parser.add_argument(
        "manifest_path",
        nargs="?",
        default=str(REPO_ROOT / "data" / "review" / "review_manifest_v1.json"),
        help="Path to review manifest JSON",
    )
    args = parser.parse_args()

    manifest_file = Path(args.manifest_path)
    if not manifest_file.exists():
        print(f"Error: Manifest file not found at {manifest_file}", file=sys.stderr)
        sys.exit(1)

    summary_md = generate_review_summary_markdown(manifest_file)
    print(summary_md)

    out_file = REPO_ROOT / "data" / "review" / "review_status_summary.md"
    out_file.write_text(summary_md, encoding="utf-8")
    print(f"\nSaved review status summary to: {out_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
