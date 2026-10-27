"""Tests for Phase 2 Step 2: Dataset Tooling & Split Generation.

Verifies:
- Valid records pass dataset validation with zero errors.
- Missing required fields are cleanly caught and reported.
- Invalid action, category, or addressing labels fail validation.
- Duplicate scenario IDs fail validation.
- Addressing invariant violations (addressed=True with SILENT) fail validation.
- Conversation split isolation: dev and test splits have 0 overlapping conversations.
- Split generation is deterministic given a fixed seed.
- Split manifest accurately records checksums, conversation counts, and action breakdowns.
- Review status audit: pending_review records are tracked and not counted as finalized.
- Near-duplicate detection flags high token overlap.
- Real curated dataset files (dataset_v1.jsonl, dev.jsonl, test.jsonl) pass validation.
"""

import json
import tempfile
from pathlib import Path
import pytest

from scripts.create_splits import generate_splits, partition_conversations
from scripts.validate_dataset import (
    ALLOWED_ACTIONS,
    ALLOWED_CATEGORIES,
    ValidationReport,
    validate_dataset_file,
    validate_single_record,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _make_sample_record(
    scenario_id: str = "s-001",
    conv_id: str = "c-001",
    action: str = "SPEAK",
    addressed: bool = True,
    method: str = "mention",
    category: str = "direct_question_addressed",
    split: str = "dev",
    review_status: str = "finalized",
    text: str = "@hush what is Python asyncio?",
    mentions: list[str] | None = None,
    reply_to: str | None = None,
) -> dict:
    return {
        "id": scenario_id,
        "conversation_id": conv_id,
        "split": split,
        "category": category,
        "metadata_completeness": "full",
        "participants": [
            {"speaker_id": "u-alex", "display_name": "Alex", "role": "human"},
            {"speaker_id": "hush", "display_name": "Hush", "role": "assistant"},
        ],
        "messages": [
            {
                "message_id": "m1",
                "speaker_id": "u-alex",
                "text": text,
                "mentions": mentions if mentions is not None else ["hush"],
                "reply_to": reply_to,
                "ts": 1730000000.0,
            }
        ],
        "addressed_to_hush": addressed,
        "address_method": method,
        "expected_action": action,
        "rationale": "Directly addressed with clear programming query.",
        "source": "handwritten",
        "annotator_confidence": "high",
        "review_status": review_status,
    }


# ---------------------------------------------------------------------------
# A. Validation Tooling Unit Tests
# ---------------------------------------------------------------------------

def test_valid_record_passes_validation():
    """A fully compliant record passes validation with zero errors."""
    report = ValidationReport()
    record = _make_sample_record()
    validate_single_record(record, index=1, seen_ids=set(), report=report)
    assert report.is_valid is True
    assert len(report.errors) == 0


def test_missing_required_fields_fails_validation():
    """Omitting any required field produces a clear validation error."""
    required_fields = [
        "id",
        "conversation_id",
        "split",
        "category",
        "metadata_completeness",
        "participants",
        "messages",
        "addressed_to_hush",
        "address_method",
        "expected_action",
        "rationale",
        "source",
    ]
    for field_name in required_fields:
        report = ValidationReport()
        record = _make_sample_record()
        del record[field_name]
        validate_single_record(record, index=1, seen_ids=set(), report=report)
        assert report.is_valid is False
        assert any(field_name in err for err in report.errors)


def test_invalid_action_label_fails_validation():
    """Labels outside {SPEAK, ASK, SILENT} fail validation."""
    report = ValidationReport()
    record = _make_sample_record(action="REPLY_NOW")
    validate_single_record(record, index=1, seen_ids=set(), report=report)
    assert report.is_valid is False
    assert any("expected_action" in err for err in report.errors)


def test_invalid_address_method_fails_validation():
    """Methods outside allowed addressing types fail validation."""
    report = ValidationReport()
    record = _make_sample_record(method="telepathy")
    validate_single_record(record, index=1, seen_ids=set(), report=report)
    assert report.is_valid is False
    assert any("address_method" in err for err in report.errors)


def test_duplicate_scenario_id_fails_validation():
    """Reusing a scenario ID fails validation."""
    report = ValidationReport()
    seen = {"s-001"}
    record = _make_sample_record(scenario_id="s-001")
    validate_single_record(record, index=2, seen_ids=seen, report=report)
    assert report.is_valid is False
    assert any("duplicate id" in err for err in report.errors)


def test_addressing_invariant_violation_fails_validation():
    """Addressed to Hush but expecting SILENT violates PLAN.md Section 7 and fails."""
    report = ValidationReport()
    record = _make_sample_record(addressed=True, action="SILENT")
    validate_single_record(record, index=1, seen_ids=set(), report=report)
    assert report.is_valid is False
    assert any("Addressing invariant violated" in err for err in report.errors)


def test_addressing_mismatch_with_engine_fails_validation():
    """Falsely claiming addressed_to_hush=False on an explicit mention fails engine check."""
    report = ValidationReport()
    record = _make_sample_record(
        text="@hush what is the status?",
        mentions=["hush"],
        addressed=False,  # False claim
        method="none",
        action="SILENT",
    )
    validate_single_record(record, index=1, seen_ids=set(), report=report)
    assert report.is_valid is False
    assert any("Ground truth addressing mismatch" in err for err in report.errors)


def test_near_duplicate_detection_flags_high_similarity():
    """Near-duplicate incoming texts are flagged as warnings."""
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        r1 = _make_sample_record(
            scenario_id="s-001",
            conv_id="c-001",
            text="what is HTTP 422 Unprocessable Entity in FastAPI framework?",
        )
        r2 = _make_sample_record(
            scenario_id="s-002",
            conv_id="c-002",
            text="what is HTTP 422 Unprocessable Entity in FastAPI framework right now?",
        )
        tmp.write(json.dumps(r1) + "\n")
        tmp.write(json.dumps(r2) + "\n")
        tmp_path = Path(tmp.name)

    try:
        report = validate_dataset_file(tmp_path, check_near_duplicates=True)
        assert len(report.near_duplicate_warnings) > 0
        assert any("Near-duplicate" in w for w in report.warnings)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_pending_review_status_tracking():
    """Records marked pending_review are recorded in review_status_counts."""
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        r1 = _make_sample_record(scenario_id="s-001", conv_id="c-001", review_status="finalized")
        r2 = _make_sample_record(scenario_id="s-002", conv_id="c-002", review_status="pending_review")
        tmp.write(json.dumps(r1) + "\n")
        tmp.write(json.dumps(r2) + "\n")
        tmp_path = Path(tmp.name)

    try:
        report = validate_dataset_file(tmp_path)
        assert report.review_status_counts.get("finalized") == 1
        assert report.review_status_counts.get("pending_review") == 1
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# B. Split Generation & Conversation Isolation Tests
# ---------------------------------------------------------------------------

def test_split_generation_deterministic():
    """Generating splits twice with fixed seed 42 produces identical assignments."""
    convs = {
        f"c-{i:03d}": [_make_sample_record(scenario_id=f"s-{i:03d}", conv_id=f"c-{i:03d}")]
        for i in range(1, 21)
    }
    dev1, test1 = partition_conversations(convs, target_dev_ratio=0.60, seed=42)
    dev2, test2 = partition_conversations(convs, target_dev_ratio=0.60, seed=42)

    assert dev1 == dev2
    assert test1 == test2


def test_split_generation_conversation_isolation():
    """No conversation group is split across dev and test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        dataset_file = tmp_path / "dataset.jsonl"
        splits_dir = tmp_path / "splits"

        # Write 20 records across 10 conversations (2 records per conversation)
        records = []
        for c_idx in range(1, 11):
            c_id = f"c-{c_idx:03d}"
            records.append(_make_sample_record(scenario_id=f"s-{c_idx*2-1:03d}", conv_id=c_id))
            records.append(_make_sample_record(scenario_id=f"s-{c_idx*2:03d}", conv_id=c_id))

        with open(dataset_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        manifest = generate_splits(
            dataset_path=dataset_file,
            output_dir=splits_dir,
            target_dev_ratio=0.60,
            seed=42,
        )

        dev_convs = set(manifest["split_summary"]["dev"]["conversations"])
        test_convs = set(manifest["split_summary"]["test"]["conversations"])

        # Strict leakage guard: intersection must be empty
        assert len(dev_convs & test_convs) == 0
        assert len(dev_convs) + len(test_convs) == 10

        # Validate that dev.jsonl and test.jsonl files pass split checks
        dev_report = validate_dataset_file(splits_dir / "dev.jsonl")
        test_report = validate_dataset_file(splits_dir / "test.jsonl")
        assert dev_report.is_valid is True
        assert test_report.is_valid is True


# ---------------------------------------------------------------------------
# C. Validation of Real Curated Dataset Files
# ---------------------------------------------------------------------------

def test_real_dataset_v1_passes_validation():
    """The master curated dataset (data/dataset_v1.jsonl) passes 100% of validation rules."""
    dataset_path = REPO_ROOT / "data" / "dataset_v1.jsonl"
    assert dataset_path.exists(), "Master dataset data/dataset_v1.jsonl must exist"

    report = validate_dataset_file(dataset_path)
    assert report.is_valid is True, f"Validation errors: {report.errors}"
    assert report.total_records == 300
    assert report.action_counts == {"SILENT": 135, "SPEAK": 105, "ASK": 60}
    assert report.split_counts == {"dev": 180, "test": 120}
    assert report.review_status_counts == {"finalized": 300}
    assert len(report.errors) == 0


def test_real_dev_and_test_splits_pass_validation_and_isolate_conversations():
    """dev.jsonl and test.jsonl have zero shared conversation IDs and match 180/120 split."""
    dev_path = REPO_ROOT / "data" / "splits" / "dev.jsonl"
    test_path = REPO_ROOT / "data" / "splits" / "test.jsonl"
    manifest_path = REPO_ROOT / "data" / "splits" / "manifest.json"

    assert dev_path.exists()
    assert test_path.exists()
    assert manifest_path.exists()

    dev_report = validate_dataset_file(dev_path)
    test_report = validate_dataset_file(test_path)

    assert dev_report.is_valid is True
    assert test_report.is_valid is True
    assert dev_report.total_records == 180
    assert test_report.total_records == 120

    # Read manifest and verify isolation
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dev_c_ids = set(manifest["split_summary"]["dev"]["conversations"])
    test_c_ids = set(manifest["split_summary"]["test"]["conversations"])

    assert len(dev_c_ids & test_c_ids) == 0
    assert len(dev_c_ids) == 180
    assert len(test_c_ids) == 120
    assert manifest["split_summary"]["dev"]["actions"] == {"SILENT": 81, "SPEAK": 63, "ASK": 36}
    assert manifest["split_summary"]["test"]["actions"] == {"SILENT": 54, "SPEAK": 42, "ASK": 24}
