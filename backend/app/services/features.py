"""Inference-time context feature extraction for Hush Conversational Restraint Engine.

Extracts deterministic, interpretable context features from InferenceContext.
Strictly isolates inference features from dataset annotations (expected_action,
rationale, category, source, annotator_confidence, review_status, split).

All continuous features are bounded and normalized to [0.0, 1.0].
Pure Python standard library implementation with zero external dependencies.
"""

from __future__ import annotations

import re
from typing import Any, Sequence, Union

from app.services.baselines import InferenceContext, extract_inference_context

# Ordered feature column names
FEATURE_NAMES: list[str] = [
    # Group 1: Addressing & Routing Signals
    "f01_addressed_to_hush",
    "f02_method_is_mention",
    "f03_method_is_reply",
    "f04_method_is_name",
    "f05_method_is_request",
    "f06_method_is_none",
    # Group 2: Surface & Syntactic Cues
    "f07_has_question_mark",
    "f08_has_exclamation",
    "f09_char_length_norm",
    "f10_word_count_norm",
    # Group 3: Intent, Ambiguity & Linguistic Indicators
    "f11_has_action_request",
    "f12_has_underspecified_cue",
    "f13_has_technical_spec",
    "f14_has_social_banter",
    "f15_has_emotional_venting",
    # Group 4: Conversational Thread & Context Cues
    "f16_has_human_mention",
    "f17_is_reply_to_human",
    "f18_is_reply_to_assistant",
    "f19_thread_turns_norm",
    "f20_assistant_spoke_previously",
    "f21_last_speaker_was_assistant",
    "f22_prior_unanswered_question",
]

# Precompiled regex patterns for speed and determinism
_ACTION_REQUEST_PATTERN = re.compile(
    r"\b(can you|could you|please|deploy|fix|check|restart|debug|write|create|generate|test|show|explain|help|solve|investigate)\b",
    re.IGNORECASE,
)

_UNDERSPECIFIED_PATTERN = re.compile(
    r"\b((it|this|that|my query|the function|the build|the webhook|the pipeline|the job|the test)\s+"
    r"(failed|broke|broken|stopped|crashed|is down|is not working|throws an error|error 500|times? out))|"
    r"\b(fix (this|it|my|that)|help me with this|why is it|how do i do this|something is wrong|not working)\b",
    re.IGNORECASE,
)

_TECHNICAL_SPEC_PATTERN = re.compile(
    r"(`[^`]+`)|"  # backticks code
    r"(/[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+)|"  # file paths
    r"(\b[a-zA-Z0-9_\-]+\.(py|ts|js|json|yml|yaml|md|sql|sh|go|rs)\b)|"  # extensions
    r"(\b(https?://|localhost|127\.0\.0\.1)\S*)|"  # URLs
    r"(\b(GET|POST|PUT|DELETE|PATCH)\b)|"  # HTTP methods
    r"(\b(500|404|403|401|502|503|504)\b)|"  # HTTP status
    r"(\b(AWS_[A-Z0-9_]+|DOCKER|K8S|GIT|CI/CD|TRACEBACK)\b)",  # Constants
    re.IGNORECASE,
)

_SOCIAL_BANTER_PATTERN = re.compile(
    r"\b(thanks|thank you|thx|good morning|gm|hello|hey|hi|bye|goodbye|see ya|cool|nice|awesome|haha|lol|congrats)\b",
    re.IGNORECASE,
)

_EMOTIONAL_VENTING_PATTERN = re.compile(
    r"\b(hate|frustrated|annoying|terrible|awful|worst|ugh|argh|so tired of|sucks|ridiculous)\b",
    re.IGNORECASE,
)


def extract_features(target: Union[InferenceContext, dict[str, Any]]) -> dict[str, float]:
    """Extract a named dictionary of features from an InferenceContext or raw record.

    If a raw record dict is passed, it is strictly sanitized via extract_inference_context
    first to prevent annotation leakage.
    """
    if isinstance(target, dict):
        ctx = extract_inference_context(target)
    else:
        ctx = target

    messages = ctx.messages
    latest = ctx.latest_message
    latest_text = ctx.latest_text.strip()
    words = latest_text.split()

    features: dict[str, float] = {}

    # -------------------------------------------------------------------------
    # Group 1: Addressing & Routing Signals
    # -------------------------------------------------------------------------
    features["f01_addressed_to_hush"] = 1.0 if ctx.addressed_to_hush else 0.0
    features["f02_method_is_mention"] = 1.0 if ctx.address_method == "mention" else 0.0
    features["f03_method_is_reply"] = 1.0 if ctx.address_method == "reply" else 0.0
    features["f04_method_is_name"] = 1.0 if ctx.address_method == "name" else 0.0
    features["f05_method_is_request"] = 1.0 if ctx.address_method == "request" else 0.0
    features["f06_method_is_none"] = 1.0 if ctx.address_method == "none" else 0.0

    # -------------------------------------------------------------------------
    # Group 2: Surface & Syntactic Cues
    # -------------------------------------------------------------------------
    features["f07_has_question_mark"] = 1.0 if "?" in latest_text else 0.0
    features["f08_has_exclamation"] = 1.0 if "!" in latest_text else 0.0
    features["f09_char_length_norm"] = min(len(latest_text), 200) / 200.0
    features["f10_word_count_norm"] = min(len(words), 40) / 40.0

    # -------------------------------------------------------------------------
    # Group 3: Intent, Ambiguity & Linguistic Indicators
    # -------------------------------------------------------------------------
    features["f11_has_action_request"] = 1.0 if bool(_ACTION_REQUEST_PATTERN.search(latest_text)) else 0.0
    features["f12_has_underspecified_cue"] = 1.0 if bool(_UNDERSPECIFIED_PATTERN.search(latest_text)) else 0.0
    features["f13_has_technical_spec"] = 1.0 if bool(_TECHNICAL_SPEC_PATTERN.search(latest_text)) else 0.0
    features["f14_has_social_banter"] = 1.0 if bool(_SOCIAL_BANTER_PATTERN.search(latest_text)) else 0.0
    features["f15_has_emotional_venting"] = 1.0 if bool(_EMOTIONAL_VENTING_PATTERN.search(latest_text)) else 0.0

    # -------------------------------------------------------------------------
    # Group 4: Conversational Thread & Context Cues
    # -------------------------------------------------------------------------
    mentions = latest.get("mentions", []) if latest else []
    reply_to = latest.get("reply_to") if latest else None

    # Human mention: has mentions targeting anyone other than hush or assistant
    has_human_mention = any(m.lower() not in {"hush", "assistant"} for m in mentions)
    features["f16_has_human_mention"] = 1.0 if has_human_mention else 0.0

    # Resolve reply-to target author from conversation history
    is_reply_to_human = False
    is_reply_to_assistant = False
    if reply_to and len(messages) > 1:
        for prev in messages[:-1]:
            if prev.get("message_id") == reply_to:
                author_role = prev.get("role", "")
                author_id = prev.get("speaker_id", "").lower()
                if author_role == "assistant" or author_id in {"hush", "assistant"}:
                    is_reply_to_assistant = True
                else:
                    is_reply_to_human = True
                break

    features["f17_is_reply_to_human"] = 1.0 if is_reply_to_human else 0.0
    features["f18_is_reply_to_assistant"] = 1.0 if is_reply_to_assistant else 0.0
    features["f19_thread_turns_norm"] = min(len(messages), 10) / 10.0

    # Assistant past participation in thread
    assistant_spoke_previously = any(
        m.get("role") == "assistant" or m.get("speaker_id", "").lower() in {"hush", "assistant"}
        for m in messages[:-1]
    )
    features["f20_assistant_spoke_previously"] = 1.0 if assistant_spoke_previously else 0.0

    # Immediate previous speaker was assistant
    last_speaker_was_assistant = False
    if len(messages) > 1:
        prev = messages[-2]
        if prev.get("role") == "assistant" or prev.get("speaker_id", "").lower() in {"hush", "assistant"}:
            last_speaker_was_assistant = True
    features["f21_last_speaker_was_assistant"] = 1.0 if last_speaker_was_assistant else 0.0

    # Prior message had a question that wasn't answered
    prior_unanswered = False
    if len(messages) > 1:
        penultimate = messages[-2]
        penultimate_text = str(penultimate.get("text", ""))
        penultimate_author = str(penultimate.get("speaker_id", "")).lower()
        if "?" in penultimate_text and penultimate_author not in {"hush", "assistant"}:
            prior_unanswered = True
    features["f22_prior_unanswered_question"] = 1.0 if prior_unanswered else 0.0

    return features


def extract_feature_vector(target: Union[InferenceContext, dict[str, Any]]) -> list[float]:
    """Extract an ordered list of feature floats according to FEATURE_NAMES."""
    feat_dict = extract_features(target)
    return [feat_dict[name] for name in FEATURE_NAMES]
