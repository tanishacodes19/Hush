"""Tests for Hush Baseline Deciders and Unified Evaluation Harness.

Verifies:
1. Strict feature isolation (no annotation or gold label leakage into InferenceContext).
2. Majority-class predictor (fitting on dev labels only, deterministic tie-breaking).
3. Always-reply predictor (always Action.SPEAK).
4. Rule-based baseline (adherence to addressing invariant, heuristic triggers).
5. LLM-as-decider output parser (strict JSON schema validation, graceful fallbacks).
6. Mock LLM decider (deterministic offline operation, simulated latency).
7. Metric mathematics against hand-calculated fixtures (Accuracy, Macro-F1, Confusion Matrix).
8. Restraint trade-off metrics (R_unnecessary, R_missed, Calls Avoided).
9. Missing-class edge cases (zero-division safety).
10. Bootstrap confidence interval reproducibility (fixed seed 42).
11. Paired significance tests (McNemar test with binomial and chi-square, paired Macro-F1 bootstrap).
12. Dataset evaluation integration with split files.
"""

import json
from pathlib import Path
import pytest

from app.schemas import Action
from app.services.baselines import (
    AlwaysReplyBaseline,
    InferenceContext,
    LLMDecisionResult,
    LLMOutputParser,
    MajorityClassBaseline,
    MockLLMDecider,
    RuleBasedBaseline,
    extract_inference_context,
)
from app.services.evaluation import (
    CANONICAL_LABELS,
    EvaluationReport,
    MetricWithCI,
    bootstrap_ci,
    compute_accuracy,
    compute_calls_avoided_rate,
    compute_confusion_matrix,
    compute_macro_f1,
    compute_missed_contribution_rate,
    compute_per_class_metrics,
    compute_unnecessary_response_rate,
    evaluate_baseline_on_file,
    evaluate_predictions,
    paired_bootstrap_macro_f1,
    paired_mcnemar_test,
)


# ============================================================================
# 1. Feature Isolation & Inference Context
# ============================================================================


def test_feature_isolation_extract_inference_context():
    """Verify that annotation-only fields are completely excluded from InferenceContext."""
    full_record = {
        "id": "s-999",
        "conversation_id": "c-999",
        "split": "dev",
        "category": "direct_question_addressed",
        "metadata_completeness": "full",
        "participants": [
            {"speaker_id": "u-alex", "display_name": "Alex", "role": "human"},
            {"speaker_id": "hush", "display_name": "Hush", "role": "assistant"},
        ],
        "messages": [
            {
                "message_id": "m1",
                "speaker_id": "u-alex",
                "text": "Hush, what is the build command?",
                "mentions": ["hush"],
                "reply_to": None,
                "ts": 1730000000.0,
                "internal_debug": "leaked_secret",  # Unallowed key
            }
        ],
        "addressed_to_hush": True,
        "address_method": "mention",
        # Annotation & Gold Metadata — MUST NOT LEAK
        "expected_action": "SPEAK",
        "rationale": "Direct question addressed to Hush",
        "source": "handwritten",
        "annotator_confidence": "high",
        "review_status": "finalized",
    }

    ctx = extract_inference_context(full_record)

    assert isinstance(ctx, InferenceContext)
    assert ctx.scenario_id == "s-999"
    assert ctx.conversation_id == "c-999"
    assert ctx.addressed_to_hush is True
    assert ctx.address_method == "mention"
    assert ctx.latest_text == "Hush, what is the build command?"

    # Ensure annotation fields are NOT attributes of context
    assert not hasattr(ctx, "expected_action")
    assert not hasattr(ctx, "rationale")
    assert not hasattr(ctx, "category")
    assert not hasattr(ctx, "source")
    assert not hasattr(ctx, "annotator_confidence")
    assert not hasattr(ctx, "review_status")
    assert not hasattr(ctx, "split")

    # Ensure messages were stripped of non-inference keys
    assert "internal_debug" not in ctx.messages[0]


# ============================================================================
# 2. Majority-Class Baseline
# ============================================================================


def test_majority_class_fitting_and_tie_breaking():
    """Verify majority class fits strictly on provided labels and breaks ties deterministically."""
    # Test standard majority
    b1 = MajorityClassBaseline().fit(["SILENT", "SILENT", "SPEAK", "ASK"])
    ctx = InferenceContext("s-1", "c-1", [], [], False, "none")
    assert b1.predict(ctx) == "SILENT"

    b2 = MajorityClassBaseline().fit(["SPEAK", "SPEAK", "SILENT"])
    assert b2.predict(ctx) == "SPEAK"

    # Test tie-breaking: SILENT preferred over SPEAK
    b_tie_1 = MajorityClassBaseline().fit(["SILENT", "SPEAK"])
    assert b_tie_1.predict(ctx) == "SILENT"

    # Test tie-breaking: SPEAK preferred over ASK
    b_tie_2 = MajorityClassBaseline().fit(["SPEAK", "ASK"])
    assert b_tie_2.predict(ctx) == "SPEAK"

    # Test error cases
    with pytest.raises(ValueError, match="empty labels list"):
        MajorityClassBaseline().fit([])

    with pytest.raises(ValueError, match="Invalid action"):
        MajorityClassBaseline(majority_class="INVALID_ACTION")


# ============================================================================
# 3. Always-Reply Baseline
# ============================================================================


def test_always_reply_baseline():
    """Verify always-reply baseline always predicts Action.SPEAK."""
    baseline = AlwaysReplyBaseline()
    ctx1 = InferenceContext("s-1", "c-1", [], [], False, "none")
    ctx2 = InferenceContext("s-2", "c-2", [], [], True, "mention")
    assert baseline.predict(ctx1) == Action.SPEAK.value
    assert baseline.predict(ctx2) == Action.SPEAK.value

    with pytest.raises(ValueError, match="Invalid action"):
        AlwaysReplyBaseline(action="NO_SUCH_ACTION")


# ============================================================================
# 4. Rule-Based Baseline
# ============================================================================


def test_rule_based_baseline_addressing_invariant():
    """Verify RuleBasedBaseline strictly preserves the addressing invariant (never SILENT when addressed)."""
    rule_decider = RuleBasedBaseline(allow_ask=True)

    # 1. Addressed + clear query -> SPEAK
    msg_clear = [{"message_id": "m1", "speaker_id": "u-1", "text": "Hush, how do I configure pytest?"}]
    ctx_clear = InferenceContext("s-1", "c-1", msg_clear, [], addressed_to_hush=True, address_method="name")
    assert rule_decider.predict(ctx_clear) == Action.SPEAK.value

    # 2. Addressed + short ambiguous query -> ASK
    msg_vague = [{"message_id": "m2", "speaker_id": "u-1", "text": "Hush, it's not working"}]
    ctx_vague = InferenceContext("s-2", "c-2", msg_vague, [], addressed_to_hush=True, address_method="name")
    assert rule_decider.predict(ctx_vague) == Action.ASK.value

    # Invariant: Neither can ever be SILENT
    assert rule_decider.predict(ctx_clear) != Action.SILENT.value
    assert rule_decider.predict(ctx_vague) != Action.SILENT.value


def test_rule_based_baseline_unaddressed_behavior():
    """Verify unaddressed messages branch between SILENT and naive SPEAK on question mark."""
    rule_decider = RuleBasedBaseline()

    # Statement without ? -> SILENT
    msg_stmt = [{"message_id": "m1", "speaker_id": "u-1", "text": "Pushed the branch to remote."}]
    ctx_stmt = InferenceContext("s-1", "c-1", msg_stmt, [], addressed_to_hush=False, address_method="none")
    assert rule_decider.predict(ctx_stmt) == Action.SILENT.value

    # Unaddressed question -> naive chatbot responds (SPEAK)
    msg_q = [{"message_id": "m2", "speaker_id": "u-1", "text": "Does anyone know the staging URL?"}]
    ctx_q = InferenceContext("s-2", "c-2", msg_q, [], addressed_to_hush=False, address_method="none")
    assert rule_decider.predict(ctx_q) == Action.SPEAK.value

    # Unaddressed question with human mention -> SILENT
    msg_q_mention = [
        {"message_id": "m3", "speaker_id": "u-1", "text": "@alex where is the key?", "mentions": ["u-alex"]}
    ]
    ctx_q_mention = InferenceContext("s-3", "c-3", msg_q_mention, [], addressed_to_hush=False, address_method="none")
    assert rule_decider.predict(ctx_q_mention) == Action.SILENT.value


# ============================================================================
# 5. LLM-as-Decider Parser & Mock Decider
# ============================================================================


def test_llm_output_parser_success():
    """Verify LLMOutputParser handles clean JSON, code blocks, and embedded text."""
    # 1. Clean JSON
    raw_1 = '{"action": "SPEAK", "rationale": "Clear request"}'
    res_1 = LLMOutputParser.parse(raw_1)
    assert res_1.success is True
    assert res_1.action == "SPEAK"
    assert res_1.rationale == "Clear request"

    # 2. Markdown fenced block
    raw_2 = "```json\n{\n  \"action\": \"ASK\",\n  \"rationale\": \"Ambiguous parameters\"\n}\n```"
    res_2 = LLMOutputParser.parse(raw_2)
    assert res_2.success is True
    assert res_2.action == "ASK"
    assert res_2.rationale == "Ambiguous parameters"

    # 3. Outer text wrapper
    raw_3 = 'Decider analysis result:\n{"action": "SILENT", "rationale": "Human discussion"}\nEnd of analysis.'
    res_3 = LLMOutputParser.parse(raw_3)
    assert res_3.success is True
    assert res_3.action == "SILENT"


def test_llm_output_parser_failure_fallbacks():
    """Verify LLMOutputParser gracefully falls back to SILENT on bad syntax or schema."""
    # Malformed JSON
    res_bad_json = LLMOutputParser.parse("This is not valid JSON at all.")
    assert res_bad_json.success is False
    assert res_bad_json.action == Action.SILENT.value
    assert "JSON parse error" in (res_bad_json.error or "")

    # Invalid action
    res_bad_action = LLMOutputParser.parse('{"action": "IGNORE_EVERYTHING", "rationale": "test"}')
    assert res_bad_action.success is False
    assert res_bad_action.action == Action.SILENT.value
    assert "Invalid action" in (res_bad_action.error or "")

    # Empty rationale
    res_no_rat = LLMOutputParser.parse('{"action": "SPEAK", "rationale": ""}')
    assert res_no_rat.success is False
    assert "must be a non-empty string" in (res_no_rat.error or "")


def test_mock_llm_decider_offline():
    """Verify MockLLMDecider operates offline with presets and default fallbacks."""
    presets = {
        "s-001": {"action": "SPEAK", "rationale": "Preset speak rationale"},
        "s-002": '{"action": "ASK", "rationale": "Preset ask rationale"}',
    }
    decider = MockLLMDecider(preset_responses=presets, default_action="SILENT")

    ctx1 = InferenceContext("s-001", "c-1", [], [], True, "mention")
    res1 = decider.decide(ctx1)
    assert res1.action == "SPEAK"
    assert res1.success is True
    assert res1.latency_ms > 0

    ctx2 = InferenceContext("s-002", "c-1", [], [], True, "mention")
    assert decider.predict(ctx2) == "ASK"

    ctx_unknown = InferenceContext("s-999", "c-2", [], [], False, "none")
    assert decider.predict(ctx_unknown) == "SILENT"


# ============================================================================
# 6. Evaluation Metric Math & Fixtures
# ============================================================================


def test_metrics_hand_calculated_fixtures():
    """Verify exact precision of metrics against hand-calculated 5-sample fixture."""
    y_true = ["SILENT", "SILENT", "SPEAK", "SPEAK", "ASK"]
    y_pred = ["SILENT", "SPEAK", "SPEAK", "SILENT", "ASK"]

    # 1. Accuracy: 3 correct / 5 = 0.60
    acc = compute_accuracy(y_true, y_pred)
    assert pytest.approx(acc) == 0.60

    # 2. Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm["SILENT"]["SILENT"] == 1
    assert cm["SILENT"]["SPEAK"] == 1
    assert cm["SILENT"]["ASK"] == 0

    assert cm["SPEAK"]["SILENT"] == 1
    assert cm["SPEAK"]["SPEAK"] == 1
    assert cm["SPEAK"]["ASK"] == 0

    assert cm["ASK"]["SILENT"] == 0
    assert cm["ASK"]["SPEAK"] == 0
    assert cm["ASK"]["ASK"] == 1

    # 3. Per-class metrics
    per_class = compute_per_class_metrics(y_true, y_pred)
    # SILENT: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
    assert pytest.approx(per_class["SILENT"]["precision"]) == 0.5
    assert pytest.approx(per_class["SILENT"]["recall"]) == 0.5
    assert pytest.approx(per_class["SILENT"]["f1"]) == 0.5

    # SPEAK: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
    assert pytest.approx(per_class["SPEAK"]["precision"]) == 0.5
    assert pytest.approx(per_class["SPEAK"]["recall"]) == 0.5
    assert pytest.approx(per_class["SPEAK"]["f1"]) == 0.5

    # ASK: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
    assert pytest.approx(per_class["ASK"]["precision"]) == 1.0
    assert pytest.approx(per_class["ASK"]["recall"]) == 1.0
    assert pytest.approx(per_class["ASK"]["f1"]) == 1.0

    # 4. Macro-F1: (0.5 + 0.5 + 1.0) / 3 = 2.0 / 3 = 0.6667
    macro_f1 = compute_macro_f1(y_true, y_pred)
    assert pytest.approx(macro_f1) == 2.0 / 3.0

    # 5. Restraint trade-off metrics
    # R_unnecessary = False responses / True SILENT = 1 / 2 = 0.50
    unrec = compute_unnecessary_response_rate(y_true, y_pred)
    assert pytest.approx(unrec) == 0.50

    # R_missed = False SILENT / True (SPEAK + ASK) = 1 / 3 = 0.3333
    missed = compute_missed_contribution_rate(y_true, y_pred)
    assert pytest.approx(missed) == 1.0 / 3.0

    # Calls avoided = Count(y_pred == SILENT) / 5 = 2 / 5 = 0.40
    avoided = compute_calls_avoided_rate(y_pred)
    assert pytest.approx(avoided) == 0.40


def test_missing_class_zero_division_safety():
    """Verify metric calculations do not crash or raise ZeroDivisionError when a class is absent."""
    y_true = ["SILENT", "SILENT", "SILENT"]
    y_pred = ["SILENT", "SILENT", "SILENT"]

    per_class = compute_per_class_metrics(y_true, y_pred)
    assert per_class["SPEAK"]["precision"] == 0.0
    assert per_class["SPEAK"]["recall"] == 0.0
    assert per_class["SPEAK"]["f1"] == 0.0
    assert per_class["ASK"]["f1"] == 0.0

    # R_missed when no true SPEAK or ASK instances -> 0.0
    assert compute_missed_contribution_rate(y_true, y_pred) == 0.0


# ============================================================================
# 7. Bootstrap CI & Paired Statistical Significance
# ============================================================================


def test_bootstrap_ci_reproducibility():
    """Verify bootstrap confidence interval is completely reproducible with fixed seed."""
    y_true = ["SILENT", "SILENT", "SPEAK", "SPEAK", "ASK"] * 10
    y_pred = ["SILENT", "SPEAK", "SPEAK", "SILENT", "ASK"] * 10

    nom1, low1, high1 = bootstrap_ci(y_true, y_pred, compute_accuracy, n_resamples=100, seed=42)
    nom2, low2, high2 = bootstrap_ci(y_true, y_pred, compute_accuracy, n_resamples=100, seed=42)

    assert nom1 == nom2
    assert low1 == low2
    assert high1 == high2
    assert low1 <= nom1 <= high1


def test_paired_mcnemar_identical_models():
    """Verify McNemar test returns p=1.0 and stat=0.0 for identical predictions."""
    y_true = ["SILENT", "SPEAK", "ASK"] * 5
    y_pred = ["SILENT", "SPEAK", "ASK"] * 5

    res = paired_mcnemar_test(y_true, y_pred, y_pred)
    assert res["statistic"] == 0.0
    assert res["p_value"] == 1.0
    assert res["significant_at_05"] is False


def test_paired_mcnemar_exact_binomial_and_chi2():
    """Verify paired McNemar computes exact binomial for small discordant pairs and chi2 for large."""
    # Small discordant (< 25)
    y_true = ["SILENT"] * 10
    y_pred_a = ["SILENT"] * 8 + ["SPEAK"] * 2  # 8 correct, 2 wrong
    y_pred_b = ["SILENT"] * 2 + ["SPEAK"] * 8  # 2 correct, 8 wrong
    res_small = paired_mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert res_small["method"] == "exact_binomial"
    assert res_small["b"] == 6
    assert res_small["c"] == 0
    assert res_small["significant_at_05"] is True

    # Large discordant (>= 25)
    y_true_large = ["SILENT"] * 60
    y_pred_a_large = ["SILENT"] * 40 + ["SPEAK"] * 20
    y_pred_b_large = ["SILENT"] * 10 + ["SPEAK"] * 50
    res_large = paired_mcnemar_test(y_true_large, y_pred_a_large, y_pred_b_large)
    assert res_large["method"] == "mcnemar_chi2"
    assert res_large["statistic"] > 0
    assert res_large["significant_at_05"] is True


def test_paired_bootstrap_macro_f1():
    """Verify paired bootstrap Macro-F1 delta test."""
    y_true = ["SILENT"] * 20 + ["SPEAK"] * 20 + ["ASK"] * 20
    # Model A is perfect
    y_pred_a = list(y_true)
    # Model B gets half wrong
    y_pred_b = ["SPEAK"] * 20 + ["SILENT"] * 20 + ["SILENT"] * 20

    res = paired_bootstrap_macro_f1(y_true, y_pred_a, y_pred_b, n_resamples=100, seed=42)
    assert res["delta"] > 0
    assert res["ci_lower"] > 0
    assert res["significant_at_05"] is True


# ============================================================================
# 8. End-to-End Dataset Evaluation Integration
# ============================================================================


def test_evaluate_dataset_file_integration():
    """Verify evaluate_baseline_on_file works with actual dev split file."""
    dev_path = Path("data/splits/dev.jsonl")
    if not dev_path.exists():
        pytest.skip("Dev split file not present")

    majority = MajorityClassBaseline.fit_from_file(dev_path)
    report = evaluate_baseline_on_file(
        majority,
        dev_path,
        model_name="Test Majority",
        n_resamples=50,
        seed=42,
    )

    assert isinstance(report, EvaluationReport)
    assert report.total_samples == 180
    assert report.accuracy.value == 0.45
    assert report.calls_avoided_rate.value == 1.0
    assert len(report.predictions) == 180

    md = report.to_markdown()
    assert "Evaluation Report: Test Majority" in md
    assert "Confusion Matrix" in md
