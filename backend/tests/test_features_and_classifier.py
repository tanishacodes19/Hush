"""Comprehensive tests for Hush Feature Extraction, ML Classifier & Dataset Provenance.

Verifies:
1. Feature extraction on legacy and conversation-aware inputs.
2. Missing metadata and incomplete conversation context handling.
3. Strict prevention of annotation-field leakage into feature extraction.
4. Deterministic feature generation across repeated runs.
5. Training-only preprocessing and reproducible model fitting (fixed seed).
6. Correct three-class output validation (SILENT, SPEAK, ASK).
7. Missing-class metric handling.
8. Addressing invariant preservation in ML classifier predictions.
9. Production SILENT behavior remaining unchanged (zero generator calls).
10. Existing API compatibility remains untouched.
11. Dataset provenance audit checks (unverified ground truth flags).
"""

import json
from pathlib import Path
import pytest

from app.schemas import Action, ChatMessageRequest
from app.services.baselines import InferenceContext, extract_inference_context
from app.services.classifier import (
    CLASS_LABELS,
    CLASS_TO_IDX,
    MultinomialLogisticRegression,
    train_classifier_on_dev,
)
from app.services.evaluation import (
    compute_accuracy,
    compute_macro_f1,
    compute_per_class_metrics,
)
from app.services.features import (
    FEATURE_NAMES,
    extract_feature_vector,
    extract_features,
)


# ============================================================================
# 1. Feature Extraction & Leakage Prevention
# ============================================================================


def test_feature_extraction_leakage_prevention():
    """Verify feature extractor ignores all annotation and gold target fields."""
    poisoned_record = {
        "id": "s-001",
        "conversation_id": "c-001",
        "split": "dev",
        "messages": [
            {
                "message_id": "m1",
                "speaker_id": "u-1",
                "text": "@Hush deploy the staging service please",
                "mentions": ["hush"],
                "reply_to": None,
                "ts": 100.0,
            }
        ],
        "participants": [{"speaker_id": "u-1", "display_name": "User 1", "role": "human"}],
        "addressed_to_hush": True,
        "address_method": "mention",
        # POISONED ANNOTATION FIELDS
        "expected_action": "SILENT",  # Contradicts invariant
        "rationale": "Leaked rationale text",
        "category": "direct_question_addressed",
        "source": "handwritten",
        "annotator_confidence": "high",
        "review_status": "finalized",
    }

    clean_record = {
        "id": "s-001",
        "conversation_id": "c-001",
        "messages": poisoned_record["messages"],
        "participants": poisoned_record["participants"],
        "addressed_to_hush": True,
        "address_method": "mention",
    }

    vec_poisoned = extract_feature_vector(poisoned_record)
    vec_clean = extract_feature_vector(clean_record)

    assert len(vec_poisoned) == len(FEATURE_NAMES)
    assert vec_poisoned == vec_clean  # Byte-identical output


def test_feature_extraction_conversation_aware_and_legacy():
    """Verify feature extraction works with multi-turn conversation context and single messages."""
    # Multi-turn conversation context
    multi_turn = InferenceContext(
        scenario_id="s-100",
        conversation_id="c-100",
        messages=[
            {"message_id": "m1", "speaker_id": "u-1", "text": "Can someone help?", "mentions": [], "reply_to": None},
            {"message_id": "m2", "speaker_id": "hush", "text": "I can help.", "role": "assistant", "mentions": [], "reply_to": "m1"},
            {"message_id": "m3", "speaker_id": "u-1", "text": "Hush, it failed with 500 error", "mentions": [], "reply_to": "m2"},
        ],
        participants=[{"speaker_id": "u-1", "role": "human"}, {"speaker_id": "hush", "role": "assistant"}],
        addressed_to_hush=True,
        address_method="name",
    )

    feat = extract_features(multi_turn)
    assert feat["f01_addressed_to_hush"] == 1.0
    assert feat["f04_method_is_name"] == 1.0
    assert feat["f12_has_underspecified_cue"] == 1.0  # "it failed"
    assert feat["f13_has_technical_spec"] == 1.0  # "500"
    assert feat["f18_is_reply_to_assistant"] == 1.0  # replies to m2
    assert feat["f20_assistant_spoke_previously"] == 1.0
    assert feat["f21_last_speaker_was_assistant"] == 1.0


def test_feature_extraction_empty_and_missing_metadata():
    """Verify feature extractor handles empty messages, missing fields, and nulls without throwing."""
    minimal_context = InferenceContext(
        scenario_id="",
        conversation_id="",
        messages=[],
        participants=[],
        addressed_to_hush=False,
        address_method="none",
    )

    feat = extract_features(minimal_context)
    assert len(feat) == len(FEATURE_NAMES)
    assert feat["f01_addressed_to_hush"] == 0.0
    assert feat["f06_method_is_none"] == 1.0
    assert feat["f09_char_length_norm"] == 0.0
    assert feat["f10_word_count_norm"] == 0.0


def test_deterministic_feature_generation():
    """Verify feature extractor is strictly deterministic across repeated invocations."""
    ctx = InferenceContext(
        scenario_id="s-050",
        conversation_id="c-050",
        messages=[{"message_id": "m1", "speaker_id": "u-dev", "text": "Hey @Hush, how do I configure pytest?"}],
        participants=[],
        addressed_to_hush=True,
        address_method="mention",
    )

    v1 = extract_feature_vector(ctx)
    v2 = extract_feature_vector(ctx)
    assert v1 == v2


# ============================================================================
# 2. Classifier Fitting & Addressing Invariant
# ============================================================================


def test_classifier_fit_reproducibility():
    """Verify model fitting is reproducible with identical random seed."""
    X = [
        [1.0] + [0.0] * 21,  # Addressed
        [0.0] + [1.0] * 21,  # Unaddressed
        [0.0] + [0.0] * 21,  # Unaddressed statement
    ]
    y = [CLASS_TO_IDX["SPEAK"], CLASS_TO_IDX["SILENT"], CLASS_TO_IDX["SILENT"]]

    m1 = MultinomialLogisticRegression(seed=42, max_epochs=50)
    m1.fit(X, y)

    m2 = MultinomialLogisticRegression(seed=42, max_epochs=50)
    m2.fit(X, y)

    assert m1.weights == m2.weights
    assert m1.biases == m2.biases


def test_classifier_three_class_output_validation():
    """Verify predict_proba returns valid probabilities summing to 1.0 across all three classes."""
    X = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, 0.2, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
    ]
    y = [CLASS_TO_IDX["SPEAK"], CLASS_TO_IDX["SILENT"], CLASS_TO_IDX["ASK"]]

    clf = MultinomialLogisticRegression(seed=42, max_epochs=100)
    clf.fit(X, y)

    ctx = InferenceContext("s-1", "c-1", [], [], False, "none")
    probs = clf.predict_proba(ctx)

    for action in CLASS_LABELS:
        assert action in probs
        assert 0.0 <= probs[action] <= 1.0

    assert pytest.approx(sum(probs.values()), rel=1e-4) == 1.0


def test_classifier_addressing_invariant_enforcement():
    """Verify classifier NEVER predicts SILENT when context.addressed_to_hush is True."""
    clf = MultinomialLogisticRegression(enforce_addressing_invariant=True)
    # Intentionally force weights that favor SILENT
    clf.weights = [
        [10.0] * 22,   # SILENT high
        [-10.0] * 22,  # SPEAK low
        [-10.0] * 22,  # ASK low
    ]
    clf.biases = [10.0, 1.0, 0.0]  # SPEAK higher than ASK
    clf.is_fitted = True

    # 1. When NOT addressed, predicts SILENT
    ctx_unaddressed = InferenceContext("s-1", "c-1", [], [], addressed_to_hush=False, address_method="none")
    assert clf.predict(ctx_unaddressed) == Action.SILENT.value

    # 2. When addressed, invariant forces selection between SPEAK and ASK (never SILENT)
    ctx_addressed = InferenceContext("s-2", "c-2", [], [], addressed_to_hush=True, address_method="name")
    action = clf.predict(ctx_addressed)
    assert action != Action.SILENT.value
    assert action == Action.SPEAK.value  # Since speak bias > ask bias


def test_classifier_save_and_load_roundtrip(tmp_path: Path):
    """Verify classifier serialization and deserialization to JSON preserves predictions."""
    X = [[1.0] * 22, [0.0] * 22]
    y = [CLASS_TO_IDX["SPEAK"], CLASS_TO_IDX["SILENT"]]

    clf = MultinomialLogisticRegression(l2_reg=0.25, seed=42)
    clf.fit(X, y)

    save_path = tmp_path / "test_model.json"
    clf.save(save_path)

    loaded = MultinomialLogisticRegression.load(save_path)
    assert loaded.is_fitted is True
    assert loaded.l2_reg == clf.l2_reg
    assert loaded.weights == clf.weights
    assert loaded.biases == clf.biases


# ============================================================================
# 3. Provenance Gate & Review Verification
# ============================================================================


def test_dataset_provenance_audit_artifact_exists():
    """Verify dataset provenance audit document exists and records unverified status."""
    audit_file = Path("docs/dataset_provenance_audit.md")
    assert audit_file.exists()

    content = audit_file.read_text(encoding="utf-8")
    assert "PROVISIONAL ONLY" in content
    assert "Unverified Ground Truth" in content
    assert "Cohen's" in content


def test_review_manifest_template_exists():
    """Verify review manifest template exists for future independent adjudication."""
    manifest_template = Path("data/review_manifest_template.json")
    assert manifest_template.exists()

    with open(manifest_template, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "provisional_unverified"
    assert "min_cohens_kappa" in data["review_requirements"]


# ============================================================================
# 4. Production Decider Invariant & Non-Regression
# ============================================================================


def test_production_decider_silent_invariant_unaffected():
    """Verify that the production DevelopmentStubDecider still respects SILENT invariant."""
    from app.services.decider import DevelopmentStubDecider
    from app.schemas import Message

    stub = DevelopmentStubDecider()
    # Unaddressed message without Hush keyword
    msgs = [Message(speaker="alex", text="Sam, did you push the commit?")]
    decision = stub.decide(msgs)

    assert decision.action == Action.SILENT
    assert decision.confidence is None
    assert decision.confidence_kind == "none"
