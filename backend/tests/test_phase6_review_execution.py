"""Unit and integration tests for Phase 6 Annotation Review Execution & Quality-Gate Readiness.

Verifies:
1. Reviewer packet privacy and strict exclusion of gold labels, rationales, and predictions.
2. Reviewer CSV form structure and field mappings.
3. Incomplete submission handling (reports pending_human_review).
4. Duplicate scenario submission detection (raises ValueError).
5. Invalid action label detection (raises ValueError).
6. Cohen's Kappa calculation on hand-calculated and degenerate fixtures.
7. Disagreement identification and dispute logging.
8. Quality gate enforcement (kappa >= 0.75 and 0 unresolved disputes).
9. Guarantee that missing human responses are never auto-filled from gold labels or models.
"""

import csv
import json
from pathlib import Path
import pytest

from scripts.validate_review_manifest import (
    ALLOWED_ACTIONS,
    compute_cohens_kappa,
    parse_submission_file,
    validate_manifest,
)


# ============================================================================
# 1. Privacy & Label Exclusion Verification
# ============================================================================


def test_reviewer_artifacts_blinded_to_gold_labels():
    """Verify reviewer packets and CSV forms contain zero gold labels, rationales, or predictions."""
    r1_jsonl = Path("data/review/reviewer_1_packet.jsonl")
    r2_jsonl = Path("data/review/reviewer_2_packet.jsonl")
    r1_csv = Path("data/review/reviewer_1_form.csv")
    r2_csv = Path("data/review/reviewer_2_form.csv")

    for p in [r1_jsonl, r2_jsonl]:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                assert "expected_action" not in rec
                assert "rationale" not in rec
                assert "category" not in rec
                assert "predicted_action" not in rec
                assert rec["reviewer_action"] is None

    for p in [r1_csv, r2_csv]:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            assert "expected_action" not in headers
            assert "category" not in headers
            for row in reader:
                assert row["reviewer_action"] == ""


# ============================================================================
# 2. Submission Ingestion & Validation Checks
# ============================================================================


def test_parse_submission_csv(tmp_path: Path):
    """Verify parsing valid completed CSV submission into scenario mapping."""
    csv_file = tmp_path / "submission_test.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_id", "conversation_transcript", "addressing_metadata",
            "reviewer_action", "is_ambiguous", "reviewer_confidence",
            "reviewer_rationale", "reviewer_comments"
        ])
        writer.writerow(["s-001", "Text 1", "Metadata 1", "SPEAK", "NO", "high", "Clear request", ""])
        writer.writerow(["s-002", "Text 2", "Metadata 2", "SILENT", "NO", "medium", "Human banter", ""])

    parsed = parse_submission_file(csv_file)
    assert len(parsed) == 2
    assert parsed["s-001"]["action"] == "SPEAK"
    assert parsed["s-001"]["is_ambiguous"] is False
    assert parsed["s-001"]["confidence"] == "high"
    assert parsed["s-002"]["action"] == "SILENT"


def test_duplicate_submission_detection(tmp_path: Path):
    """Verify duplicate scenario IDs in submission file trigger ValueError."""
    csv_dup = tmp_path / "submission_dup.csv"
    with open(csv_dup, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "reviewer_action"])
        writer.writerow(["s-001", "SPEAK"])
        writer.writerow(["s-001", "ASK"])  # Duplicate ID

    with pytest.raises(ValueError, match="Duplicate scenario_id detected"):
        parse_submission_file(csv_dup)


def test_invalid_action_label_detection(tmp_path: Path):
    """Verify invalid action values trigger validation failure."""
    manifest_data = {
        "manifest_version": "2.0.0",
        "records": [
            {
                "scenario_id": "s-001",
                "reviewer_1": {"action": "IGNORE_EVERYTHING"},  # Invalid!
                "reviewer_2": {"action": "SPEAK"},
            }
        ],
    }
    p = tmp_path / "invalid_manifest.json"
    p.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid actions found"):
        validate_manifest(p)


# ============================================================================
# 3. Agreement & Cohen's Kappa Calculations
# ============================================================================


def test_cohens_kappa_hand_calculated_fixture():
    """Verify Cohen's Kappa against hand-calculated 10-sample 3-class fixture."""
    # N = 10
    # Both agree on: 4 SILENT, 3 SPEAK, 1 ASK -> Agreements = 8, p_o = 0.80
    # Disagreements:
    # 1: R1=SPEAK, R2=SILENT
    # 1: R1=ASK,   R2=SPEAK
    # Marginals R1: 4 SILENT, 4 SPEAK, 2 ASK (0.4, 0.4, 0.2)
    # Marginals R2: 5 SILENT, 4 SPEAK, 1 ASK (0.5, 0.4, 0.1)
    # p_e = (0.4*0.5) + (0.4*0.4) + (0.2*0.1) = 0.20 + 0.16 + 0.02 = 0.38
    # kappa = (0.80 - 0.38) / (1 - 0.38) = 0.42 / 0.62 = 0.6774
    r1 = ["SILENT"] * 4 + ["SPEAK"] * 3 + ["ASK"] * 1 + ["SPEAK", "ASK"]
    r2 = ["SILENT"] * 4 + ["SPEAK"] * 3 + ["ASK"] * 1 + ["SILENT", "SPEAK"]

    res = compute_cohens_kappa(r1, r2)
    assert pytest.approx(res["raw_agreement_rate"], rel=1e-4) == 0.80
    assert pytest.approx(res["expected_chance_agreement"], rel=1e-4) == 0.38
    assert pytest.approx(res["kappa"], rel=1e-3) == 0.6774


def test_cohens_kappa_degenerate_constant_case():
    """Verify Cohen's Kappa handles degenerate case (single constant class) safely."""
    r1 = ["SILENT"] * 10
    r2 = ["SILENT"] * 10

    res = compute_cohens_kappa(r1, r2)
    assert res["kappa"] == 1.0
    assert res["raw_agreement_rate"] == 1.0


# ============================================================================
# 4. Dispute Identification & Quality Gate Enforcement
# ============================================================================


def test_dispute_logging_and_quality_gate(tmp_path: Path):
    """Verify disagreements are logged and quality gate fails if kappa < 0.75."""
    records = []
    # 10 agreements, 10 disagreements -> kappa will be low (< 0.75)
    for i in range(10):
        records.append({
            "scenario_id": f"s-{i:03d}",
            "reviewer_1": {"action": "SILENT", "rationale": "r1 notes"},
            "reviewer_2": {"action": "SILENT", "rationale": "r2 notes"},
            "adjudicated_action": None,
        })
    for i in range(10, 20):
        records.append({
            "scenario_id": f"s-{i:03d}",
            "reviewer_1": {"action": "SPEAK", "rationale": "looks answerable"},
            "reviewer_2": {"action": "ASK", "rationale": "looks ambiguous"},
            "adjudicated_action": None,
        })

    p = tmp_path / "dispute_manifest.json"
    p.write_text(json.dumps({"records": records}), encoding="utf-8")

    res = validate_manifest(p)
    assert res["status"] == "completed_review"
    assert res["is_complete"] is True
    assert res["unresolved_disputes_count"] == 10
    assert len(res["unresolved_disputes"]) == 10
    assert res["quality_gate_passed"] is False  # Fails quality gate


def test_dispute_ledger_file_writing(tmp_path: Path):
    """Verify disputes are written to external JSON ledger file when provided."""
    records = [
        {
            "scenario_id": "s-001",
            "reviewer_1": {"action": "SPEAK", "rationale": "Clear request"},
            "reviewer_2": {"action": "ASK", "rationale": "Missing parameter"},
            "adjudicated_action": None,
        }
    ]
    manifest_p = tmp_path / "manifest.json"
    manifest_p.write_text(json.dumps({"records": records}), encoding="utf-8")
    disputes_p = tmp_path / "disputes_out.json"

    res = validate_manifest(manifest_p, disputes_path=disputes_p)
    assert res["unresolved_disputes_count"] == 1
    assert disputes_p.exists()

    with open(disputes_p, "r", encoding="utf-8") as f:
        d_data = json.load(f)
    assert d_data["status"] == "adjudication_required"
    assert d_data["disputes_count"] == 1
    assert d_data["disputes"][0]["scenario_id"] == "s-001"


def test_mismatched_scenario_sets_detection(tmp_path: Path):
    """Verify submissions with differing scenario sets are detected."""
    manifest_data = {"records": [{"scenario_id": "s-001"}, {"scenario_id": "s-002"}]}
    manifest_p = tmp_path / "manifest.json"
    manifest_p.write_text(json.dumps(manifest_data), encoding="utf-8")

    r1_csv = tmp_path / "r1.csv"
    r2_csv = tmp_path / "r2.csv"
    with open(r1_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "reviewer_action"])
        writer.writerow(["s-001", "SPEAK"])
        writer.writerow(["s-002", "SILENT"])

    with open(r2_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "reviewer_action"])
        writer.writerow(["s-001", "SPEAK"])
        writer.writerow(["s-003", "SILENT"])  # Mismatched set!

    res = validate_manifest(manifest_p, r1_csv, r2_csv)
    assert res["status"] == "pending_human_review"
    assert res["error_type"] == "mismatched_scenario_sets"


def test_missing_answers_detection(tmp_path: Path):
    """Verify submission files with blank/empty reviewer_action fields are detected."""
    manifest_data = {"records": [{"scenario_id": "s-001"}, {"scenario_id": "s-002"}]}
    manifest_p = tmp_path / "manifest.json"
    manifest_p.write_text(json.dumps(manifest_data), encoding="utf-8")

    r1_csv = tmp_path / "r1.csv"
    r2_csv = tmp_path / "r2.csv"
    with open(r1_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "reviewer_action"])
        writer.writerow(["s-001", "SPEAK"])
        writer.writerow(["s-002", ""])  # Missing answer!

    with open(r2_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "reviewer_action"])
        writer.writerow(["s-001", "SPEAK"])
        writer.writerow(["s-002", "SILENT"])

    res = validate_manifest(manifest_p, r1_csv, r2_csv)
    assert res["status"] == "pending_human_review"
    assert res["error_type"] == "missing_answers"
    assert "s-002" in res["missing_answers_r1"]


# ============================================================================
# 5. Non-Fabrication & Pending Status Preservation
# ============================================================================


def test_no_automatic_filling_of_missing_human_responses():
    """Assert review manifest in data/review/ remains pending and contains zero filled actions."""
    manifest_file = Path("data/review/review_manifest_v1.json")
    if not manifest_file.exists():
        pytest.skip("Manifest file not found")

    res = validate_manifest(manifest_file)
    assert res["status"] == "pending_human_review"
    assert res["is_complete"] is False
    assert res["unreviewed_count"] == 60

    # Ensure no auto-filling of reviewer actions from gold labels
    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for rec in data["records"]:
        assert rec["reviewer_1"]["action"] is None
        assert rec["reviewer_2"]["action"] is None
        if rec["scenario_id"] != "s-192":
            assert rec["adjudicated_action"] is None


def test_completed_submissions_with_adjudication_passes_gate():
    """Verify that completed human submissions with s-192 adjudication pass the quality gate."""
    manifest_p = Path("data/review/review_manifest_v1.json")
    r1_p = Path("data/review/reviewer_1_completed.csv")
    r2_p = Path("data/review/reviewer_2_completed.csv")
    disp_p = Path("data/review/disputes_v1.json")

    if not (manifest_p.exists() and r1_p.exists() and r2_p.exists()):
        pytest.skip("Review artifacts not available")

    res = validate_manifest(manifest_p, r1_p, r2_p, disputes_path=disp_p)
    assert res["status"] == "completed_review"
    assert res["is_complete"] is True
    assert res["total_records"] == 60
    assert pytest.approx(res["raw_agreement_rate"], rel=1e-4) == 0.9833
    assert pytest.approx(res["cohens_kappa"], rel=1e-4) == 0.9740
    assert res["cohens_kappa"] >= 0.75
    assert res["unresolved_disputes_count"] == 0
    assert res["resolved_disputes_count"] == 1
    assert res["quality_gate_passed"] is True
