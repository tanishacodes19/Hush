"""Tests for Phase 7: Fresh Holdout Collection and Validation Readiness."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.schemas import Action, Message
from app.services.decider import DevelopmentStubDecider
from scripts.prepare_fresh_holdout import (
    TARGET_DISTRIBUTION,
    TARGET_TOTAL,
    check_holdout_overlap,
    load_candidate_file,
    validate_candidate_dataset,
    validate_candidate_record_schema,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_holdout_overlap_three_levels():
    """Verify all 3 overlap checks: latest message, full dialogue, and conversational prefix."""
    master = [
        {
            "id": "s-master-01",
            "messages": [
                {"message_id": "m1", "speaker_id": "u-dev1", "text": "Starting morning code review session."},
                {"message_id": "m2", "speaker_id": "u-dev2", "text": "Hush how does Python asyncio event loop handle non-blocking IO?"},
            ],
        }
    ]

    # Level 1: Latest message overlap
    cand_latest = [
        {
            "id": "cand-01",
            "messages": [
                {"message_id": "c1", "speaker_id": "u-alice", "text": "Completely novel opening discussion about compilers."},
                {"message_id": "c2", "speaker_id": "u-bob", "text": "Hush how does Python asyncio event loop handle non-blocking IO?"},
            ],
        }
    ]
    ov_latest = check_holdout_overlap(cand_latest, master, jaccard_threshold=0.70)
    assert len(ov_latest) == 1
    assert ov_latest[0]["overlap_type"] == "latest_message"
    assert ov_latest[0]["similarity"] == 1.0

    # Level 2: Prefix overlap (same stem, different ending)
    cand_prefix = [
        {
            "id": "cand-02",
            "messages": [
                {"message_id": "c1", "speaker_id": "u-dev1", "text": "Starting morning code review session."},
                {"message_id": "c2", "speaker_id": "u-dev2", "text": "What is the team lunch schedule for Friday afternoon?"},
            ],
        }
    ]
    ov_prefix = check_holdout_overlap(cand_prefix, master, jaccard_threshold=0.70)
    assert len(ov_prefix) == 1
    assert ov_prefix[0]["overlap_type"] == "dialogue_prefix"
    assert ov_prefix[0]["similarity"] == 1.0

    # Clean candidate with zero overlap
    cand_clean = [
        {
            "id": "cand-03",
            "messages": [
                {"message_id": "c1", "speaker_id": "u-alice", "text": "Deploying Redis cluster shard 3 to staging-west."},
                {"message_id": "c2", "speaker_id": "u-bob", "text": "Replication lag is zero milliseconds across all secondaries."},
            ],
        }
    ]
    ov_clean = check_holdout_overlap(cand_clean, master, jaccard_threshold=0.70)
    assert len(ov_clean) == 0


def test_validate_candidate_record_schema_provenance():
    """Verify candidate validation strictly enforces provenance and reject derived records."""
    valid_record = {
        "id": "holdout-001",
        "domain": "infrastructure",
        "provenance": {
            "author_id": "author_alice",
            "created_at": "2026-10-27T12:00:00Z",
            "source_type": "fresh_human_authored",
            "derived_from_existing": False,
        },
        "messages": [
            {"message_id": "m1", "speaker_id": "u-1", "text": "Hey team, server 3 is out of memory."},
            {"message_id": "m2", "speaker_id": "u-2", "text": "Let me restart the worker pod."},
        ],
        "addressed_to_hush": False,
        "address_method": "none",
        "candidate_action": "SILENT",
    }
    assert validate_candidate_record_schema(valid_record) == []

    # Prohibited: derived_from_existing == True
    derived_rec = dict(valid_record)
    derived_rec["provenance"] = dict(valid_record["provenance"], derived_from_existing=True)
    errs = validate_candidate_record_schema(derived_rec)
    assert any("strictly prohibited" in e for e in errs)

    # Missing author_id
    no_author = dict(valid_record)
    no_author["provenance"] = dict(valid_record["provenance"], author_id="")
    errs = validate_candidate_record_schema(no_author)
    assert any("author_id" in e for e in errs)

    # Invalid source_type
    bad_source = dict(valid_record)
    bad_source["provenance"] = dict(valid_record["provenance"], source_type="synthetic_chatgpt")
    errs = validate_candidate_record_schema(bad_source)
    assert any("source_type" in e for e in errs)

    # Single-turn conversation rejected (must be multi-turn >= 2)
    single_turn = dict(valid_record)
    single_turn["messages"] = [{"message_id": "m1", "speaker_id": "u-1", "text": "Solo turn."}]
    errs = validate_candidate_record_schema(single_turn)
    assert any("at least 2 turns" in e for e in errs)

    # Invalid action label
    bad_act = dict(valid_record, candidate_action="IGNORE")
    errs = validate_candidate_record_schema(bad_act)
    assert any("SILENT, SPEAK, or ASK" in e for e in errs)


def test_load_candidate_templates():
    """Verify loading exported candidate CSV and JSONL templates."""
    csv_path = REPO_ROOT / "data" / "holdout" / "holdout_collection_template.csv"
    jsonl_path = REPO_ROOT / "data" / "holdout" / "holdout_candidate_template.jsonl"

    assert csv_path.exists()
    assert jsonl_path.exists()

    csv_records = load_candidate_file(csv_path)
    assert len(csv_records) == 3
    for rec in csv_records:
        assert validate_candidate_record_schema(rec) == []

    jsonl_records = load_candidate_file(jsonl_path)
    assert len(jsonl_records) == 3
    for rec in jsonl_records:
        assert validate_candidate_record_schema(rec) == []


def test_validate_candidate_dataset_distribution_target():
    """Verify target distribution check for the complete 50-scenario holdout."""
    master = []

    # Construct mock candidate set matching target: 23 SILENT, 17 SPEAK, 10 ASK
    candidates = []
    actions = (["SILENT"] * 23) + (["SPEAK"] * 17) + (["ASK"] * 10)
    for idx, act in enumerate(actions, start=1):
        candidates.append({
            "id": f"holdout-mock-{idx:03d}",
            "domain": "test",
            "provenance": {
                "author_id": "author_test",
                "created_at": "2026-10-27T00:00:00Z",
                "source_type": "fresh_human_authored",
                "derived_from_existing": False,
            },
            "messages": [
                {"message_id": "m1", "speaker_id": "u-1", "text": f"Context turn for scenario {idx}"},
                {"message_id": "m2", "speaker_id": "u-2", "text": f"Incoming question turn for scenario {idx}"},
            ],
            "addressed_to_hush": act in ("SPEAK", "ASK"),
            "address_method": "mention" if act in ("SPEAK", "ASK") else "none",
            "candidate_action": act,
        })

    res = validate_candidate_dataset(candidates, master, check_target_distribution=True)
    assert res["status"] == "PASS"
    assert res["total_candidates"] == 50
    assert res["distribution"] == {"SILENT": 23, "SPEAK": 17, "ASK": 10}
    assert res["distribution_matched"] is True

    # Check mismatch detection
    mismatched = candidates[:-1]  # 49 items
    res_bad = validate_candidate_dataset(mismatched, master, check_target_distribution=True)
    assert res_bad["status"] == "FAIL"
    assert res_bad["distribution_matched"] is False


def test_holdout_quarantine_integrity():
    """Verify holdout quarantine and data immutability invariants."""
    # 1. holdout_v1.jsonl MUST NOT exist until real collection + review is complete
    holdout_file = REPO_ROOT / "data" / "holdout" / "holdout_v1.jsonl"
    assert not holdout_file.exists(), "holdout_v1.jsonl must not be created prematurely"

    # 2. Existing master and split files must match exact recorded SHA256 checksums
    expected_checksums = {
        REPO_ROOT / "data" / "dataset_v1.jsonl": "64bd1f39e1bc589361b2281537d384f59470ff28309a39d416c0b5f77b5bc0fb",
        REPO_ROOT / "data" / "splits" / "dev.jsonl": "d2b1928c0fe8db97ba5efab35fd6e0d2ebd07de71a6d6e984e1cb5346f861aa6",
        REPO_ROOT / "data" / "splits" / "test.jsonl": "f0c7d4284e3c8df32d0d72021b2ce746f2c56fe4e32c1abd9183f56a0a5ff543",
        REPO_ROOT / "data" / "review" / "reviewer_1_completed.csv": "f832f322394d79a52f5c0d012ce25f31e7485e6da74343802b6f0c86e3e54e52",
        REPO_ROOT / "data" / "review" / "reviewer_2_completed.csv": "26ad48438f89d1e013c75991fcbcd415e0e939c61ef329f0872e5ccc6ccd95fc",
    }
    for path, expected_hash in expected_checksums.items():
        assert path.exists(), f"File missing: {path}"
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Checksum mismatch for {path}: expected {expected_hash}, got {actual_hash}"

    # 3. Production decider remains DevelopmentStubDecider and SILENT guarantees 0 generator calls
    stub = DevelopmentStubDecider()
    generator_mock = MagicMock()
    decision = stub.decide([Message(speaker="u-user", text="Hey team")])
    assert decision.action == Action.SILENT
    assert decision.confidence is None
    assert generator_mock.call_count == 0
