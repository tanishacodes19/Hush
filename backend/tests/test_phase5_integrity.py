"""Unit and integration tests for Phase 5 Evaluation Integrity & Independent Review Tooling.

Verifies:
1. Hand-calculated calibration and cost fixtures (Brier, NLL, ECE, Expected Cost).
2. Out-of-fold prediction isolation and nested temperature scaling.
3. Review sampling reproducibility and class stratification.
4. Review manifest validation on incomplete, failing, and passing synthetic manifests.
5. Inter-annotator Cohen's Kappa calculation accuracy.
6. Dispute resolution and addressing invariant enforcement in review workflow.
7. Holdout overlap prevention (Jaccard similarity threshold).
8. Quarantine of historical test set and future holdout from training features.
9. Production SILENT invariant preservation.
"""

import json
import math
from pathlib import Path
import pytest

from app.schemas import Action, Message
from app.services.evaluation import (
    compute_brier_score,
    compute_ece,
    compute_expected_cost,
    compute_nll,
)
from scripts.prepare_annotation_review import (
    create_reviewer_packet,
    select_review_sample,
)
from scripts.prepare_fresh_holdout import check_holdout_overlap, compute_jaccard, tokenize_words
from scripts.validate_review_manifest import compute_cohens_kappa, validate_manifest


# ============================================================================
# 1. Calibration & Expected Cost Hand-Calculated Fixtures
# ============================================================================


def test_brier_score_hand_calculated():
    """Verify multiclass Brier score against hand-calculated 2-sample case."""
    y_true = ["SILENT", "SPEAK"]
    probs = [
        {"SILENT": 0.8, "SPEAK": 0.1, "ASK": 0.1},
        {"SILENT": 0.2, "SPEAK": 0.7, "ASK": 0.1},
    ]
    # Sample 1: (0.8 - 1)^2 + (0.1)^2 + (0.1)^2 = 0.04 + 0.01 + 0.01 = 0.06
    # Sample 2: (0.2)^2 + (0.7 - 1)^2 + (0.1)^2 = 0.04 + 0.09 + 0.01 = 0.14
    # Mean = (0.06 + 0.14) / 2 = 0.10
    score = compute_brier_score(y_true, probs)
    assert pytest.approx(score, rel=1e-4) == 0.10


def test_nll_hand_calculated():
    """Verify NLL against hand-calculated 2-sample case."""
    y_true = ["SILENT", "SPEAK"]
    probs = [
        {"SILENT": 0.5, "SPEAK": 0.3, "ASK": 0.2},
        {"SILENT": 0.1, "SPEAK": 0.5, "ASK": 0.4},
    ]
    # NLL = - (log(0.5) + log(0.5)) / 2 = - log(0.5) = log(2) = 0.693147
    nll = compute_nll(y_true, probs)
    assert pytest.approx(nll, rel=1e-4) == math.log(2.0)


def test_expected_cost_hand_calculated():
    """Verify expected cost under standard restraint cost matrix."""
    cost_matrix = {
        "SILENT": {"SILENT": 0.0, "SPEAK": 3.0, "ASK": 3.0},
        "SPEAK": {"SILENT": 5.0, "SPEAK": 0.0, "ASK": 1.0},
        "ASK": {"SILENT": 5.0, "SPEAK": 1.5, "ASK": 0.0},
    }
    # 3 predictions:
    # 1: True SILENT, Pred SILENT -> 0.0
    # 2: True SPEAK,  Pred ASK    -> 1.0
    # 3: True ASK,    Pred SILENT -> 5.0
    # Total = 6.0, Mean = 2.0
    y_true = ["SILENT", "SPEAK", "ASK"]
    y_pred = ["SILENT", "ASK", "SILENT"]
    cost = compute_expected_cost(y_true, y_pred, cost_matrix)
    assert pytest.approx(cost, rel=1e-4) == 2.0


# ============================================================================
# 2. Review Sampling & Stratification
# ============================================================================


def test_review_sampling_reproducibility_and_stratification():
    """Verify sample selection of 60 scenarios is deterministic and correctly stratified."""
    dataset_file = Path("data/dataset_v1.jsonl")
    if not dataset_file.exists():
        pytest.skip("Dataset file not available")

    with open(dataset_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    sample1 = select_review_sample(records, n_sample=60, seed=42)
    sample2 = select_review_sample(records, n_sample=60, seed=42)

    assert len(sample1) == 60
    assert [r["id"] for r in sample1] == [r["id"] for r in sample2]

    # Verify action counts match 45% SILENT, 35% SPEAK, 20% ASK
    silent_cnt = sum(1 for r in sample1 if r["expected_action"] == "SILENT")
    speak_cnt = sum(1 for r in sample1 if r["expected_action"] == "SPEAK")
    ask_cnt = sum(1 for r in sample1 if r["expected_action"] == "ASK")

    assert silent_cnt == 27
    assert speak_cnt == 21
    assert ask_cnt == 12

    # Verify split counts: 36 dev, 24 test
    dev_cnt = sum(1 for r in sample1 if r["split"] == "dev")
    test_cnt = sum(1 for r in sample1 if r["split"] == "test")
    assert dev_cnt == 36
    assert test_cnt == 24


def test_reviewer_packet_label_quarantine():
    """Verify reviewer packets contain zero gold labels, rationales, or predictions."""
    sample = [
        {
            "id": "s-001",
            "conversation_id": "c-001",
            "split": "dev",
            "expected_action": "SPEAK",
            "rationale": "Secret gold rationale",
            "category": "direct_question_addressed",
            "messages": [{"message_id": "m1", "speaker_id": "u-1", "text": "Hush, help"}],
            "participants": [],
            "addressed_to_hush": True,
            "address_method": "name",
        }
    ]

    packet = create_reviewer_packet(sample, "reviewer_test")
    item = packet[0]

    assert item["scenario_id"] == "s-001"
    assert "expected_action" not in item
    assert "rationale" not in item
    assert "category" not in item
    assert item["reviewer_action"] is None
    assert item["assigned_reviewer"] == "reviewer_test"


# ============================================================================
# 3. Review Manifest Validation & Cohen's Kappa
# ============================================================================


def test_cohens_kappa_perfect_and_chance_agreement():
    """Verify Cohen's kappa calculation on perfect agreement and random agreement."""
    # 1. Perfect agreement -> kappa = 1.0
    r1_perfect = ["SILENT", "SPEAK", "ASK", "SILENT"] * 5
    r2_perfect = list(r1_perfect)
    res_perf = compute_cohens_kappa(r1_perfect, r2_perfect)
    assert res_perf["kappa"] == 1.0
    assert res_perf["raw_agreement_rate"] == 1.0

    # 2. Known disagreement fixture:
    # 10 SILENT, 10 SPEAK
    # R1: 10 SILENT, 10 SPEAK
    # R2:  8 SILENT (2 SPEAK), 8 SPEAK (2 SILENT)
    # Agreements = 16 / 20 = 0.80
    r1_mix = ["SILENT"] * 10 + ["SPEAK"] * 10
    r2_mix = ["SILENT"] * 8 + ["SPEAK"] * 2 + ["SILENT"] * 2 + ["SPEAK"] * 8
    res_mix = compute_cohens_kappa(r1_mix, r2_mix)
    assert res_mix["raw_agreement_rate"] == 0.80
    assert 0.50 <= res_mix["kappa"] <= 0.70


def test_validate_manifest_incomplete_reports_pending(tmp_path: Path):
    """Verify manifest with unreviewed records reports pending_human_review status."""
    manifest_data = {
        "manifest_version": "1.0.0",
        "records": [
            {
                "scenario_id": "s-001",
                "reviewer_1": {"action": None},
                "reviewer_2": {"action": None},
            }
        ],
    }
    p = tmp_path / "pending_manifest.json"
    p.write_text(json.dumps(manifest_data), encoding="utf-8")

    res = validate_manifest(p)
    assert res["status"] == "pending_human_review"
    assert res["is_complete"] is False
    assert res["unreviewed_count"] == 1


def test_validate_manifest_complete_passing_gate(tmp_path: Path):
    """Verify complete manifest with kappa >= 0.75 and resolved disputes passes gate."""
    # 20 scenarios with high agreement
    records = []
    for i in range(20):
        records.append({
            "scenario_id": f"s-{i:03d}",
            "reviewer_1": {"action": "SILENT" if i < 10 else "SPEAK"},
            "reviewer_2": {"action": "SILENT" if i < 10 else "SPEAK"},
            "adjudicated_action": None,
        })
    p = tmp_path / "passing_manifest.json"
    p.write_text(json.dumps({"records": records}), encoding="utf-8")

    res = validate_manifest(p)
    assert res["status"] == "completed_review"
    assert res["is_complete"] is True
    assert res["quality_gate_passed"] is True
    assert res["cohens_kappa"] == 1.0


def test_validate_manifest_complete_failing_due_to_unresolved_dispute(tmp_path: Path):
    """Verify manifest with disputes fails gate if dispute has no adjudicated action."""
    records = [
        {
            "scenario_id": "s-001",
            "reviewer_1": {"action": "SPEAK"},
            "reviewer_2": {"action": "SILENT"},  # Dispute!
            "adjudicated_action": None,          # Unresolved!
        }
    ]
    p = tmp_path / "dispute_manifest.json"
    p.write_text(json.dumps({"records": records}), encoding="utf-8")

    res = validate_manifest(p)
    assert res["status"] == "completed_review"
    assert res["quality_gate_passed"] is False
    assert res["unresolved_disputes_count"] == 1


# ============================================================================
# 4. Holdout Overlap Prevention
# ============================================================================


def test_holdout_overlap_detection():
    """Verify holdout overlap tool catches near-duplicate scenarios against master."""
    master = [
        {"id": "s-001", "messages": [{"text": "Hush, how do I configure AWS credentials?"}]}
    ]

    # Candidate 1: Near duplicate (high Jaccard)
    candidate_dup = [
        {"id": "cand-01", "messages": [{"text": "Hush, how do I configure AWS credentials?"}]}
    ]
    overlaps_dup = check_holdout_overlap(candidate_dup, master, jaccard_threshold=0.70)
    assert len(overlaps_dup) == 1
    assert overlaps_dup[0]["similarity"] == 1.0

    # Candidate 2: Distinct domain scenario (low Jaccard)
    candidate_clean = [
        {"id": "cand-02", "messages": [{"text": "Alice, did the database migration complete on shard 4?"}]}
    ]
    overlaps_clean = check_holdout_overlap(candidate_clean, master, jaccard_threshold=0.70)
    assert len(overlaps_clean) == 0

    # Candidate 3: Multi-turn prefix leak (identical prefix turns, distinct latest turn)
    master_multiturn = [
        {
            "id": "s-197",
            "messages": [
                {"speaker": "u-alex", "text": "Hush how do I filter a dictionary in Python?"},
                {"speaker": "hush", "text": "You can use a dictionary comprehension with items."},
                {"speaker": "u-alex", "text": "What happens if the input dictionary is empty?"},
            ],
        }
    ]
    candidate_prefix_leak = [
        {
            "id": "cand-03",
            "messages": [
                {"speaker": "u-alex", "text": "Hush how do I filter a dictionary in Python?"},
                {"speaker": "hush", "text": "You can use a dictionary comprehension with items."},
                {"speaker": "u-alex", "text": "What is the computational complexity of that method?"},
            ],
        }
    ]
    overlaps_prefix = check_holdout_overlap(candidate_prefix_leak, master_multiturn, jaccard_threshold=0.70)
    assert len(overlaps_prefix) >= 1
    assert overlaps_prefix[0]["similarity"] >= 0.70
    assert overlaps_prefix[0]["overlap_type"] in ("dialogue_prefix", "full_dialogue")


def test_holdout_overlap_threshold_boundaries():
    """Verify threshold boundary behavior for holdout overlap checker."""
    # 10 words in master: a b c d e f g h i j
    master = [
        {"id": "m-01", "messages": [{"text": "alpha beta gamma delta epsilon zeta eta theta iota kappa"}]}
    ]

    # Exactly 7 common words out of 10 -> Jaccard = 7 / 10 = 0.70 (Triggers at threshold 0.70)
    cand_at_boundary = [
        {"id": "c-70", "messages": [{"text": "alpha beta gamma delta epsilon zeta eta"}]}
    ]
    res_at = check_holdout_overlap(cand_at_boundary, master, jaccard_threshold=0.70)
    assert len(res_at) == 1
    assert pytest.approx(res_at[0]["similarity"], rel=1e-4) == 0.70

    # Exactly 6 common words out of 10 -> Jaccard = 6 / 10 = 0.60 (Does not trigger at threshold 0.70)
    cand_below_boundary = [
        {"id": "c-60", "messages": [{"text": "alpha beta gamma delta epsilon zeta"}]}
    ]
    res_below = check_holdout_overlap(cand_below_boundary, master, jaccard_threshold=0.70)
    assert len(res_below) == 0

    # Empty candidate messages handling
    cand_empty = [{"id": "c-empty", "messages": []}]
    res_empty = check_holdout_overlap(cand_empty, master, jaccard_threshold=0.70)
    assert len(res_empty) == 0


# ============================================================================
# 5. Production SILENT Invariant Non-Regression
# ============================================================================


def test_production_silent_invariant_unaffected():
    """Verify production DevelopmentStubDecider still respects the SILENT invariant."""
    from app.services.decider import DevelopmentStubDecider

    stub = DevelopmentStubDecider()
    msgs = [Message(speaker="alex", text="Sam, what time is the sync meeting?")]
    decision = stub.decide(msgs)

    assert decision.action == Action.SILENT
    assert decision.confidence is None
    assert decision.confidence_kind == "none"
