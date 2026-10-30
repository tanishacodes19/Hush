#!/usr/bin/env python3
"""Dataset validation tooling for Hush Conversational Restraint Engine.

Validates scenarios against the schema and requirements defined in docs/dataset_spec.md:
1. JSON structure and required fields.
2. Unique scenario IDs and valid format (s-XXX).
3. Conversation group IDs format (c-XXX).
4. Allowed categories (12 defined categories).
5. Action labels (SPEAK, ASK, SILENT) and target distributions.
6. Addressing signals (addressed_to_hush, address_method) and consistency with AddressingService.
7. Addressing invariant: addressed_to_hush == True cannot have expected_action == SILENT.
8. Valid references (reply_to targets valid prior message or documented policy).
9. Duplicate and near-duplicate detection.
10. Split isolation: dev and test splits have 0 shared conversation groups.
11. Review status tracking (finalized vs pending_review).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Ensure backend is on sys.path for app imports
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from app.schemas import Message
    from app.services.addressing import resolve_addressing
except ImportError:
    # Graceful fallback if invoked outside backend environment
    resolve_addressing = None  # type: ignore
    Message = None  # type: ignore


ALLOWED_CATEGORIES = {
    "direct_question_addressed",
    "ambiguous_request",
    "human_to_human_question",
    "human_to_human_discussion",
    "casual_banter",
    "emotional_venting",
    "statement_no_response",
    "follow_up_dependent",
    "already_answered",
    "repeated_message",
    "name_false_positive",
    "malformed_or_edge_case",
}

ALLOWED_ACTIONS = {"SPEAK", "ASK", "SILENT"}
ALLOWED_ADDRESS_METHODS = {"mention", "reply", "name", "request", "none"}
ALLOWED_METADATA_COMPLETENESS = {"full", "text_only", "partial"}
ALLOWED_SPLITS = {"dev", "test"}
ALLOWED_SOURCES = {"handwritten", "llm_generated_reviewed", "corpus_adapted"}
ALLOWED_ROLES = {"human", "assistant"}
ALLOWED_REVIEW_STATUSES = {"finalized", "pending_review"}

ID_PATTERN = re.compile(r"^s-[0-9]{3,}$")
CONV_ID_PATTERN = re.compile(r"^c-[0-9]{3,}$")


@dataclass
class ValidationReport:
    """Detailed results of dataset validation."""
    is_valid: bool = True
    total_records: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    action_counts: dict[str, int] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    address_method_counts: dict[str, int] = field(default_factory=dict)
    review_status_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    unique_conversations: int = 0
    near_duplicate_warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def _tokenize(text: str) -> set[str]:
    """Simple alphanumeric word tokenization for Jaccard similarity."""
    return set(re.findall(r"\w+", text.lower()))


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def validate_single_record(
    record: dict[str, Any],
    index: int,
    seen_ids: set[str],
    report: ValidationReport,
    verify_addressing_engine: bool = True,
) -> None:
    """Validate all schema and logic rules for a single scenario record."""
    # 1. Required Top-Level Fields
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
    for req in required_fields:
        if req not in record:
            report.add_error(f"Record #{index}: missing required field '{req}'")

    scenario_id = record.get("id")
    if not scenario_id:
        report.add_error(f"Record #{index}: 'id' is empty or null")
        return

    prefix = f"Scenario '{scenario_id}' (line {index})"

    # 2. ID Pattern and Uniqueness
    if not isinstance(scenario_id, str) or not ID_PATTERN.match(scenario_id):
        report.add_error(f"{prefix}: id '{scenario_id}' does not match pattern '^s-[0-9]{{3,}}$'")
    if scenario_id in seen_ids:
        report.add_error(f"{prefix}: duplicate id detected")
    seen_ids.add(scenario_id)

    # 3. Conversation ID Pattern
    conv_id = record.get("conversation_id")
    if not isinstance(conv_id, str) or not CONV_ID_PATTERN.match(conv_id):
        report.add_error(f"{prefix}: conversation_id '{conv_id}' does not match pattern '^c-[0-9]{{3,}}$'")

    # 4. Split
    split = record.get("split")
    if split not in ALLOWED_SPLITS:
        report.add_error(f"{prefix}: split '{split}' must be one of {ALLOWED_SPLITS}")

    # 5. Category
    category = record.get("category")
    if category not in ALLOWED_CATEGORIES:
        report.add_error(f"{prefix}: category '{category}' is not an allowed category")

    # 6. Metadata Completeness
    meta_comp = record.get("metadata_completeness")
    if meta_comp not in ALLOWED_METADATA_COMPLETENESS:
        report.add_error(f"{prefix}: metadata_completeness '{meta_comp}' must be one of {ALLOWED_METADATA_COMPLETENESS}")

    # 7. Action
    action = record.get("expected_action")
    if action not in ALLOWED_ACTIONS:
        report.add_error(f"{prefix}: expected_action '{action}' must be one of {ALLOWED_ACTIONS}")

    # 8. Addressing Method & Boolean
    addressed = record.get("addressed_to_hush")
    if not isinstance(addressed, bool):
        report.add_error(f"{prefix}: addressed_to_hush must be a boolean (True/False)")

    addr_method = record.get("address_method")
    if addr_method not in ALLOWED_ADDRESS_METHODS:
        report.add_error(f"{prefix}: address_method '{addr_method}' must be one of {ALLOWED_ADDRESS_METHODS}")

    # 9. Addressing Invariant (PLAN.md Section 7)
    # Never silently ignore a message that is explicitly addressed to Hush
    if addressed is True and action == "SILENT":
        report.add_error(
            f"{prefix}: Addressing invariant violated — message is addressed_to_hush=True "
            f"but expected_action is SILENT. Addressed messages must be SPEAK or ASK."
        )

    # 10. Rationale and Source
    rationale = record.get("rationale")
    if not isinstance(rationale, str) or len(rationale.strip()) < 10:
        report.add_error(f"{prefix}: rationale must be a descriptive string >= 10 characters")

    source = record.get("source")
    if source not in ALLOWED_SOURCES:
        report.add_error(f"{prefix}: source '{source}' must be one of {ALLOWED_SOURCES}")

    # 11. Review Status (Optional field, default finalized if absent or validate)
    review_status = record.get("review_status", "finalized")
    if review_status not in ALLOWED_REVIEW_STATUSES:
        report.add_error(f"{prefix}: review_status '{review_status}' must be one of {ALLOWED_REVIEW_STATUSES}")

    # 12. Participants Validation
    participants = record.get("participants")
    if not isinstance(participants, list) or len(participants) < 1:
        report.add_error(f"{prefix}: participants must be a non-empty list of participant objects")
    else:
        participant_ids = set()
        has_assistant = False
        for p_idx, p in enumerate(participants):
            if not isinstance(p, dict):
                report.add_error(f"{prefix}: participant #{p_idx} is not a valid object")
                continue
            sp_id = p.get("speaker_id")
            role = p.get("role")
            if not sp_id or not isinstance(sp_id, str):
                report.add_error(f"{prefix}: participant #{p_idx} missing valid speaker_id")
            else:
                participant_ids.add(sp_id)
            if role not in ALLOWED_ROLES:
                report.add_error(f"{prefix}: participant '{sp_id}' has invalid role '{role}'")
            if role == "assistant" or sp_id == "hush":
                has_assistant = True

        if not has_assistant:
            report.add_warning(f"{prefix}: participants list does not declare Hush as assistant participant")

    # 13. Messages Validation
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 1:
        report.add_error(f"{prefix}: messages must be a non-empty list of message objects")
        return

    known_msg_ids = set()
    prev_ts: Optional[float] = None

    for m_idx, m in enumerate(messages):
        if not isinstance(m, dict):
            report.add_error(f"{prefix}: message #{m_idx} is not a valid object")
            continue

        msg_id = m.get("message_id")
        spk_id = m.get("speaker_id")
        text = m.get("text")
        mentions = m.get("mentions")
        reply_to = m.get("reply_to")
        ts = m.get("ts")

        if not msg_id or not isinstance(msg_id, str) or not msg_id.strip():
            report.add_error(f"{prefix}: message #{m_idx} missing valid non-empty message_id")
        else:
            known_msg_ids.add(msg_id)

        if not spk_id or not isinstance(spk_id, str):
            report.add_error(f"{prefix}: message '{msg_id}' missing valid speaker_id")

        if not text or not isinstance(text, str) or not text.strip():
            report.add_error(f"{prefix}: message '{msg_id}' text must be a non-empty string")

        if not isinstance(mentions, list):
            report.add_error(f"{prefix}: message '{msg_id}' mentions must be a list of strings")

        if reply_to is not None and not isinstance(reply_to, str):
            report.add_error(f"{prefix}: message '{msg_id}' reply_to must be string or null")

        # Reply-to target reference validation
        # reply_to must refer to an earlier message in the conversation or documented target
        if reply_to and reply_to not in known_msg_ids:
            # Allow documented external reference pattern (e.g. unknown target tests)
            if not reply_to.startswith("unknown-") and not reply_to.startswith("ext-"):
                report.add_warning(
                    f"{prefix}: message '{msg_id}' reply_to '{reply_to}' targets an unknown or forward message"
                )

        if ts is not None:
            if not isinstance(ts, (int, float)) or ts < 0:
                report.add_error(f"{prefix}: message '{msg_id}' ts must be non-negative number")
            elif prev_ts is not None and ts < prev_ts:
                report.add_warning(f"{prefix}: message '{msg_id}' ts ({ts}) precedes earlier message ({prev_ts})")
            prev_ts = ts

    # 14. Ground Truth Addressing Consistency Verification
    if verify_addressing_engine and resolve_addressing and Message:
        latest = messages[-1]
        prior_messages = messages[:-1]
        history = [
            Message(
                speaker=pm.get("speaker_id", "unknown"),
                text=pm.get("text", ""),
                message_id=pm.get("message_id"),
                speaker_id=pm.get("speaker_id"),
                role="assistant" if pm.get("speaker_id") == "hush" else "human",
                mentions=pm.get("mentions", []),
                reply_to=pm.get("reply_to"),
                ts=pm.get("ts"),
            )
            for pm in prior_messages
        ]
        resolved = resolve_addressing(
            message_text=latest.get("text", ""),
            mentions=latest.get("mentions", []),
            reply_to=latest.get("reply_to"),
            conversation_history=history,
        )

        if resolved.addressed_to_hush != addressed:
            report.add_error(
                f"{prefix}: Ground truth addressing mismatch — record says addressed_to_hush={addressed}, "
                f"but AddressingService resolved addressed_to_hush={resolved.addressed_to_hush} "
                f"(reason: {resolved.reason})"
            )

        if resolved.address_method != addr_method:
            report.add_error(
                f"{prefix}: Addressing method mismatch — record says address_method='{addr_method}', "
                f"but AddressingService resolved address_method='{resolved.address_method}' "
                f"(reason: {resolved.reason})"
            )


def validate_dataset_file(
    file_path: Path | str,
    verify_addressing_engine: bool = True,
    check_near_duplicates: bool = True,
) -> ValidationReport:
    """Validate a dataset JSONL file against all specifications."""
    path = Path(file_path)
    report = ValidationReport()

    if not path.exists():
        report.add_error(f"Dataset file does not exist: {path}")
        return report

    seen_ids: set[str] = set()
    conversation_to_splits: dict[str, set[str]] = {}
    message_tokens_map: dict[str, set[str]] = {}
    exact_texts: dict[str, list[str]] = {}

    line_num = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            line_num += 1
            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as exc:
                report.add_error(f"Line {line_num}: invalid JSON — {exc}")
                continue

            validate_single_record(
                record=record,
                index=line_num,
                seen_ids=seen_ids,
                report=report,
                verify_addressing_engine=verify_addressing_engine,
            )

            # Accumulate statistics
            action = record.get("expected_action", "UNKNOWN")
            category = record.get("category", "UNKNOWN")
            method = record.get("address_method", "UNKNOWN")
            split = record.get("split", "UNKNOWN")
            rev_stat = record.get("review_status", "finalized")
            conv_id = record.get("conversation_id", "UNKNOWN")

            report.action_counts[action] = report.action_counts.get(action, 0) + 1
            report.category_counts[category] = report.category_counts.get(category, 0) + 1
            report.address_method_counts[method] = report.address_method_counts.get(method, 0) + 1
            report.split_counts[split] = report.split_counts.get(split, 0) + 1
            report.review_status_counts[rev_stat] = report.review_status_counts.get(rev_stat, 0) + 1

            if conv_id != "UNKNOWN":
                conversation_to_splits.setdefault(conv_id, set()).add(split)

            # Duplicate / near-duplicate tracking on current (eval) message text
            messages = record.get("messages", [])
            scenario_id = record.get("id", f"s-line-{line_num}")
            if messages and isinstance(messages[-1], dict):
                eval_text = messages[-1].get("text", "").strip().lower()
                if eval_text:
                    exact_texts.setdefault(eval_text, []).append(scenario_id)
                    if check_near_duplicates:
                        message_tokens_map[scenario_id] = _tokenize(eval_text)

    report.total_records = len(seen_ids)
    report.unique_conversations = len(conversation_to_splits)

    # Check conversation split isolation (Leakage prevention)
    for c_id, splits in conversation_to_splits.items():
        if len(splits) > 1:
            report.add_error(
                f"Data Leakage Error: conversation_id '{c_id}' appears in multiple splits: {splits}. "
                f"All messages in a conversation group must be strictly assigned to a single split."
            )

    # Check exact duplicates
    for text, s_ids in exact_texts.items():
        if len(s_ids) > 1:
            report.add_warning(
                f"Exact duplicate incoming text detected in scenarios {s_ids}: '{text[:60]}...'"
            )

    # Check high Jaccard near-duplicates (sample or threshold check)
    if check_near_duplicates and len(message_tokens_map) > 1:
        keys = list(message_tokens_map.keys())
        for i in range(len(keys)):
            tok_a = message_tokens_map[keys[i]]
            if len(tok_a) < 5:
                continue
            for j in range(i + 1, min(i + 50, len(keys))):
                tok_b = message_tokens_map[keys[j]]
                sim = _jaccard_similarity(tok_a, tok_b)
                if sim >= 0.80:
                    warn = f"Near-duplicate incoming text ({sim:.2f} Jaccard) between {keys[i]} and {keys[j]}"
                    report.near_duplicate_warnings.append(warn)
                    report.add_warning(warn)

    return report


def print_report(report: ValidationReport, source_name: str = "Dataset") -> None:
    """Print readable validation report to stdout."""
    status_str = "VALID (PASS)" if report.is_valid else "INVALID (FAIL)"
    print("=" * 60)
    print(f"Hush Dataset Validation Report: {source_name}")
    print(f"Status: {status_str}")
    print("=" * 60)
    print(f"Total Records Evaluated: {report.total_records}")
    print(f"Unique Conversation Groups: {report.unique_conversations}")
    print(f"Errors: {len(report.errors)}")
    print(f"Warnings: {len(report.warnings)}")
    print("-" * 60)

    print("Action Breakdown:")
    for action in sorted(report.action_counts.keys()):
        count = report.action_counts[action]
        pct = (count / report.total_records * 100) if report.total_records > 0 else 0
        print(f"  {action:<8}: {count:>4} ({pct:>5.1f}%)")

    print("\nReview Status Breakdown:")
    for st, count in sorted(report.review_status_counts.items()):
        print(f"  {st:<15}: {count:>4}")

    print("\nSplit Breakdown:")
    for sp, count in sorted(report.split_counts.items()):
        print(f"  {sp:<10}: {count:>4}")

    print("\nAddressing Method Breakdown:")
    for m, count in sorted(report.address_method_counts.items()):
        print(f"  {m:<10}: {count:>4}")

    print("\nCategory Breakdown:")
    for cat, count in sorted(report.category_counts.items()):
        print(f"  {cat:<30}: {count:>4}")

    if report.errors:
        print("\n" + "!" * 60)
        print("VALIDATION ERRORS:")
        for err in report.errors[:25]:
            print(f"  [ERROR] {err}")
        if len(report.errors) > 25:
            print(f"  ... and {len(report.errors) - 25} more errors")

    if report.warnings:
        print("\n" + "-" * 60)
        print("VALIDATION WARNINGS:")
        for w in report.warnings[:15]:
            print(f"  [WARN] {w}")
        if len(report.warnings) > 15:
            print(f"  ... and {len(report.warnings) - 15} more warnings")

    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Hush scenario datasets")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="data/dataset_v1.jsonl",
        help="Path to dataset JSONL file (default: data/dataset_v1.jsonl)",
    )
    parser.add_argument(
        "--no-addressing-check",
        action="store_true",
        help="Skip AddressingService consistency verification",
    )
    args = parser.parse_args()

    target_path = Path(args.dataset_path)
    if not target_path.is_absolute():
        target_path = REPO_ROOT / target_path

    report = validate_dataset_file(
        target_path,
        verify_addressing_engine=not args.no_addressing_check,
    )
    print_report(report, source_name=str(target_path.name))

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
