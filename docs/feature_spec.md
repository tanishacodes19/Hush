# Hush — Inference-Time Context Feature Specification

**Date:** October 27, 2026  
**Component:** `backend/app/services/features.py`  
**Purpose:** Deterministic, low-cost feature extraction for the Conversational Restraint Classifier.

---

## 1. Principles & Constraints

1. **Strict Inference-Time Isolation:**
   - Features must be extractable purely from the incoming `InferenceContext` (`messages`, `participants`, `addressed_to_hush`, `address_method`).
   - Zero access to annotation fields: `expected_action`, `rationale`, `category`, `source`, `annotator_confidence`, `review_status`, `split`.
2. **Deterministic & Lightweight:**
   - Pure Python standard library implementation.
   - Zero GPU, zero embeddings, zero external API or network calls.
   - Computable in $< 0.1\text{ ms}$ per scenario.
3. **Bounded & Normalized:**
   - Continuous signals (lengths, counts) are clipped and scaled to $[0.0, 1.0]$.
   - Binary signals are represented as $1.0$ (present) or $0.0$ (absent).

---

## 2. Feature Schema & Definitions

The feature extractor produces an ordered vector of 22 interpretable features:

### Group 1: Addressing & Routing Signals (6 features)
- `f01_addressed_to_hush`: Binary (1.0 if `addressed_to_hush == True`, else 0.0). Primary invariant trigger.
- `f02_method_is_mention`: Binary (1.0 if `address_method == "mention"`, else 0.0).
- `f03_method_is_reply`: Binary (1.0 if `address_method == "reply"`, else 0.0).
- `f04_method_is_name`: Binary (1.0 if `address_method == "name"`, else 0.0).
- `f05_method_is_request`: Binary (1.0 if `address_method == "request"`, else 0.0).
- `f06_method_is_none`: Binary (1.0 if `address_method == "none"`, else 0.0).

### Group 2: Surface & Syntactic Cues (4 features)
- `f07_has_question_mark`: Binary (1.0 if latest message contains `?`, else 0.0).
- `f08_has_exclamation`: Binary (1.0 if latest message contains `!`, else 0.0).
- `f09_char_length_norm`: Normalized float $\min(\text{len}(\text{text}), 200) / 200.0$.
- `f10_word_count_norm`: Normalized float $\min(\text{word\_count}, 40) / 40.0$.

### Group 3: Intent, Ambiguity & Linguistic Indicators (5 features)
- `f11_has_action_request`: Binary (1.0 if text contains direct action verbs: `can you`, `could you`, `please`, `deploy`, `fix`, `check`, `restart`, `debug`, `write`, `test`, `explain`, `help`).
- `f12_has_underspecified_cue`: Binary (1.0 if text indicates an ambiguous problem without parameters: `it failed`, `it broke`, `broken`, `crashed`, `not working`, `throws 500`, `fix this`, `why is it`).
- `f13_has_technical_spec`: Binary (1.0 if text contains concrete technical markers: backticks, file paths `/`, extensions `.py`/`.ts`/`.json`, URLs `http`, HTTP status codes `500`/`404`, AWS/Git constants).
- `f14_has_social_banter`: Binary (1.0 if text matches casual chat/greetings: `thanks`, `thank you`, `good morning`, `gm`, `hello`, `hey`, `hi`, `bye`, `see ya`, `cool`, `haha`, `lol`).
- `f15_has_emotional_venting`: Binary (1.0 if text contains frustration cues: `hate`, `frustrated`, `annoying`, `terrible`, `awful`, `worst`, `ugh`, `so tired of`).

### Group 4: Conversational Thread & Context Cues (7 features)
- `f16_has_human_mention`: Binary (1.0 if latest message mentions a human participant, excluding Hush/Assistant).
- `f17_is_reply_to_human`: Binary (1.0 if latest message replies to a message authored by a human).
- `f18_is_reply_to_assistant`: Binary (1.0 if latest message replies to a message authored by Hush).
- `f19_thread_turns_norm`: Normalized float $\min(\text{len}(\text{thread}), 10) / 10.0$.
- `f20_assistant_spoke_previously`: Binary (1.0 if Hush already spoke in this conversation thread).
- `f21_last_speaker_was_assistant`: Binary (1.0 if the immediately preceding turn was from Hush).
- `f22_prior_unanswered_question`: Binary (1.0 if penultimate message contained `?` and was not addressed to/answered by Hush).

---

## 3. Leakage Prevention Protocol

The following fields are strictly quarantined from feature extraction:
- `expected_action` (Ground truth label)
- `rationale` (Human annotation rationale)
- `category` (Ground truth taxonomy label)
- `source` (Dataset provenance metadata)
- `annotator_confidence` (Annotation confidence level)
- `review_status` (Dataset review tracking flag)
- `split` (Split identifier: dev / test)

Unit test `test_feature_isolation` in `backend/tests/test_features_and_classifier.py` asserts that passing records with poisoned or corrupted annotation metadata produces byte-identical feature vectors to clean inputs.
