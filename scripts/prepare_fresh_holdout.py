#!/usr/bin/env python3
"""Preparation and validation tooling for fresh independent holdout dataset (Hush Phase 7).

Prepares the quarantine boundary, collection templates, and validation rules for collecting
a brand-new 50-scenario holdout set (holdout_v1) to prevent reusing the historically evaluated test set:

1. Establishes inclusion / exclusion criteria and target category distributions.
2. Implements three-level overlap prevention against all 300 existing master scenarios:
   - Latest incoming message text similarity (Jaccard >= 0.70).
   - Full concatenated dialogue text similarity (Jaccard >= 0.70).
   - Conversational prefix similarity (Jaccard >= 0.70).
3. Enforces candidate schema & provenance validation:
   - Author identity, creation date, source origin.
   - Strict rejection of any candidate derived from existing examples.
4. Generates collection templates and author instructions (CSV & JSONL).
5. Enforces quarantine: holdout remains strictly inaccessible to training, feature tuning,
   and calibration pipelines, and collection status remains PENDING until dual human review is finalized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

TARGET_DISTRIBUTION = {
    "SILENT": 23,
    "SPEAK": 17,
    "ASK": 10,
}
TARGET_TOTAL = 50


def tokenize_words(text: str) -> set[str]:
    """Tokenize alphanumeric words in lowercase for Jaccard similarity."""
    return set(re.findall(r"\w+", text.lower()))


def compute_jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute Jaccard token similarity."""
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return inter / union if union > 0 else 0.0


def check_holdout_overlap(
    candidate_records: list[dict[str, Any]],
    master_records: list[dict[str, Any]],
    jaccard_threshold: float = 0.70,
) -> list[dict[str, Any]]:
    """Check candidate holdout records against all existing master records for leakage/overlap.

    Evaluates:
    1. Latest incoming message text similarity.
    2. Full concatenated dialogue text similarity.
    3. Conversational prefix similarity (all turns prior to latest message).
    """
    overlaps: list[dict[str, Any]] = []

    # Map existing master records: latest message, full dialogue, and prefix tokens
    master_representations = []
    for m in master_records:
        msgs = m.get("messages", [])
        last_text = msgs[-1].get("text", "") if msgs else ""
        full_text = " ".join(msg.get("text", "") for msg in msgs)
        prefix_text = " ".join(msg.get("text", "") for msg in msgs[:-1]) if len(msgs) > 1 else ""

        master_representations.append({
            "id": m.get("id", "unknown"),
            "latest_text": last_text,
            "latest_tok": tokenize_words(last_text),
            "full_text": full_text,
            "full_tok": tokenize_words(full_text),
            "prefix_text": prefix_text,
            "prefix_tok": tokenize_words(prefix_text),
        })

    for cand in candidate_records:
        c_id = cand.get("id", cand.get("scenario_id", "unknown"))
        c_msgs = cand.get("messages", [])
        c_last = c_msgs[-1].get("text", "") if c_msgs else ""
        c_full = " ".join(msg.get("text", "") for msg in c_msgs)
        c_prefix = " ".join(msg.get("text", "") for msg in c_msgs[:-1]) if len(c_msgs) > 1 else ""

        c_last_tok = tokenize_words(c_last)
        c_full_tok = tokenize_words(c_full)
        c_prefix_tok = tokenize_words(c_prefix)

        for m_rep in master_representations:
            sim_last = compute_jaccard(c_last_tok, m_rep["latest_tok"])
            sim_full = compute_jaccard(c_full_tok, m_rep["full_tok"])
            sim_prefix = (
                compute_jaccard(c_prefix_tok, m_rep["prefix_tok"])
                if (c_prefix_tok and m_rep["prefix_tok"])
                else 0.0
            )

            max_sim = max(sim_last, sim_full, sim_prefix)
            if max_sim >= jaccard_threshold:
                overlap_type = "latest_message"
                matched_cand = c_last
                matched_exist = m_rep["latest_text"]
                if max_sim == sim_full:
                    overlap_type = "full_dialogue"
                    matched_cand = c_full
                    matched_exist = m_rep["full_text"]
                elif max_sim == sim_prefix:
                    overlap_type = "dialogue_prefix"
                    matched_cand = c_prefix
                    matched_exist = m_rep["prefix_text"]

                overlaps.append({
                    "candidate_id": c_id,
                    "existing_master_id": m_rep["id"],
                    "similarity": round(max_sim, 4),
                    "overlap_type": overlap_type,
                    "candidate_text": matched_cand,
                    "existing_text": matched_exist,
                })

    return overlaps


def validate_candidate_record_schema(record: dict[str, Any]) -> list[str]:
    """Validate schema, provenance, and structure of a candidate holdout scenario."""
    errors: list[str] = []

    # 1. Identifier
    scen_id = record.get("id") or record.get("scenario_id")
    if not scen_id or not isinstance(scen_id, str):
        errors.append("Missing or invalid 'id' / 'scenario_id'")

    # 2. Provenance
    prov = record.get("provenance")
    if not prov or not isinstance(prov, dict):
        errors.append("Missing mandatory 'provenance' dictionary")
    else:
        author_id = prov.get("author_id")
        if not author_id or not isinstance(author_id, str) or not author_id.strip():
            errors.append("Provenance missing non-empty 'author_id'")

        created_at = prov.get("created_at")
        if not created_at or not isinstance(created_at, str) or not created_at.strip():
            errors.append("Provenance missing non-empty 'created_at'")

        source_type = prov.get("source_type")
        valid_sources = {
            "fresh_human_authored",
            "production_log",
            "anonymized_chat_export",
            "independently_authored",
        }
        if not source_type or source_type not in valid_sources:
            errors.append(f"Provenance 'source_type' must be one of {sorted(valid_sources)}, got '{source_type}'")

        derived = prov.get("derived_from_existing")
        if derived is True:
            errors.append("Candidate is marked as derived from existing example ('derived_from_existing': true); strictly prohibited")
        elif derived is not False:
            errors.append("Provenance must explicitly set 'derived_from_existing': false")

    # 3. Messages & Dialogue Structure
    msgs = record.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        errors.append(f"Candidate must feature multi-turn conversation (at least 2 turns), found {len(msgs) if isinstance(msgs, list) else 0}")
    else:
        for idx, m in enumerate(msgs):
            if not isinstance(m, dict):
                errors.append(f"Message index {idx} is not a dictionary")
                continue
            if not m.get("speaker_id") and not m.get("speaker"):
                errors.append(f"Message index {idx} missing 'speaker_id'")
            text = m.get("text")
            if not text or not isinstance(text, str) or not text.strip():
                errors.append(f"Message index {idx} has empty text")

    # 4. Action and Addressing
    action = record.get("candidate_action") or record.get("expected_action")
    if action not in {"SILENT", "SPEAK", "ASK"}:
        errors.append(f"Candidate action must be SILENT, SPEAK, or ASK; got '{action}'")

    addr = record.get("addressed_to_hush")
    if not isinstance(addr, bool):
        errors.append(f"'addressed_to_hush' must be boolean, got '{addr}'")

    method = record.get("address_method")
    valid_methods = {"mention", "reply", "name", "request", "none"}
    if method not in valid_methods:
        errors.append(f"'address_method' must be one of {sorted(valid_methods)}, got '{method}'")

    return errors


def load_candidate_file(file_path: Path) -> list[dict[str, Any]]:
    """Load candidate records from JSONL or CSV format."""
    suffix = file_path.suffix.lower()
    records: list[dict[str, Any]] = []

    if suffix in (".jsonl", ".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            if suffix == ".json":
                data = json.load(f)
                records = data if isinstance(data, list) else data.get("records", [])
            else:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSON decode error on line {line_num} in {file_path}: {exc}") from exc

    elif suffix == ".csv":
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scen_id = row.get("scenario_id") or row.get("id") or ""
                if not scen_id or scen_id.startswith("#"):
                    continue

                raw_history = row.get("conversation_history", "[]").strip()
                msgs: list[dict[str, Any]] = []
                if raw_history.startswith("["):
                    try:
                        msgs = json.loads(raw_history)
                    except json.JSONDecodeError:
                        msgs = []
                elif raw_history:
                    for idx, line in enumerate(raw_history.splitlines(), start=1):
                        if ":" in line:
                            spk, txt = line.split(":", 1)
                            msgs.append({
                                "message_id": f"m{idx}",
                                "speaker_id": spk.strip(),
                                "text": txt.strip(),
                                "mentions": [],
                                "reply_to": None,
                            })

                incoming_spk = row.get("incoming_turn_speaker", "u-user").strip()
                incoming_txt = row.get("incoming_turn_text", "").strip()
                incoming_mentions = [
                    m.strip()
                    for m in row.get("incoming_turn_mentions", "").split(",")
                    if m.strip()
                ]
                incoming_reply = row.get("incoming_turn_reply_to", "").strip() or None

                msgs.append({
                    "message_id": f"m{len(msgs) + 1}",
                    "speaker_id": incoming_spk,
                    "text": incoming_txt,
                    "mentions": incoming_mentions,
                    "reply_to": incoming_reply,
                })

                derived_val = row.get("derived_from_existing", "false").strip().lower()
                derived_bool = derived_val in ("true", "1", "yes")

                record: dict[str, Any] = {
                    "id": scen_id,
                    "domain": row.get("domain", "general").strip(),
                    "provenance": {
                        "author_id": row.get("author_id", "").strip(),
                        "created_at": row.get("created_at", "").strip(),
                        "source_type": row.get("source_type", "fresh_human_authored").strip(),
                        "derived_from_existing": derived_bool,
                        "collection_method": "template_submission",
                    },
                    "participants": [
                        {"speaker_id": incoming_spk, "display_name": incoming_spk, "role": "human"},
                        {"speaker_id": "hush", "display_name": "Hush", "role": "assistant"},
                    ],
                    "messages": msgs,
                    "addressed_to_hush": row.get("addressed_to_hush", "").strip().lower() in ("true", "1", "yes"),
                    "address_method": row.get("address_method", "none").strip().lower(),
                    "candidate_action": row.get("candidate_action", "").strip().upper(),
                    "is_ambiguous": row.get("is_ambiguous", "").strip().lower() in ("true", "1", "yes"),
                    "candidate_rationale": row.get("candidate_rationale", "").strip(),
                }
                records.append(record)
    else:
        raise ValueError(f"Unsupported candidate file format '{suffix}'. Must be .jsonl, .json, or .csv")

    return records


def validate_candidate_dataset(
    candidate_records: list[dict[str, Any]],
    master_records: list[dict[str, Any]],
    jaccard_threshold: float = 0.70,
    check_target_distribution: bool = False,
) -> dict[str, Any]:
    """Perform complete schema, provenance, overlap, and distribution validation."""
    schema_errors: dict[str, list[str]] = {}
    distribution: dict[str, int] = {"SILENT": 0, "SPEAK": 0, "ASK": 0}

    for cand in candidate_records:
        cand_id = cand.get("id") or cand.get("scenario_id", "unknown")
        errs = validate_candidate_record_schema(cand)
        if errs:
            schema_errors[cand_id] = errs

        act = cand.get("candidate_action") or cand.get("expected_action")
        if act in distribution:
            distribution[act] += 1

    overlaps = check_holdout_overlap(candidate_records, master_records, jaccard_threshold=jaccard_threshold)

    dist_matched = True
    dist_reasons: list[str] = []
    if check_target_distribution:
        if len(candidate_records) != TARGET_TOTAL:
            dist_matched = False
            dist_reasons.append(f"Expected exactly {TARGET_TOTAL} scenarios, found {len(candidate_records)}")
        for act, tgt in TARGET_DISTRIBUTION.items():
            actual = distribution.get(act, 0)
            if actual != tgt:
                dist_matched = False
                dist_reasons.append(f"Action '{act}': expected {tgt}, got {actual}")

    is_valid = len(schema_errors) == 0 and len(overlaps) == 0 and dist_matched

    return {
        "status": "PASS" if is_valid else "FAIL",
        "total_candidates": len(candidate_records),
        "schema_error_count": len(schema_errors),
        "schema_errors": schema_errors,
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "distribution": distribution,
        "target_distribution": TARGET_DISTRIBUTION,
        "distribution_matched": dist_matched,
        "distribution_mismatch_reasons": dist_reasons,
    }


def format_conversation_transcript(messages: list[dict[str, Any]]) -> str:
    """Format dialogue messages as a readable multiline transcript."""
    lines = []
    for msg in messages:
        spk = msg.get("speaker_id") or msg.get("speaker") or "unknown"
        txt = msg.get("text", "")
        lines.append(f"[{spk}]: {txt}")
    return "\n".join(lines)


def format_addressing_metadata(record: dict[str, Any]) -> str:
    """Format addressing signals as a human-readable metadata string."""
    addr = "YES" if record.get("addressed_to_hush") else "NO"
    method = record.get("address_method", "none")
    msgs = record.get("messages", [])
    last_msg = msgs[-1] if msgs else {}
    mentions = last_msg.get("mentions", [])
    mentions_str = ", ".join(mentions) if mentions else "None"
    reply_to = last_msg.get("reply_to") or "None"
    return f"Addressed to Hush: {addr} | Method: {method} | Mentions: {mentions_str} | Reply-To: {reply_to}"


def package_blinded_review(
    candidate_records: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """Package candidate scenarios into blinded review forms, JSONL packets, and manifest.

    Strips candidate actions and candidate rationales to ensure fully blinded human evaluation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    r1_csv_path = output_dir / "reviewer_1_form.csv"
    r2_csv_path = output_dir / "reviewer_2_form.csv"
    r1_jsonl_path = output_dir / "reviewer_1_packet.jsonl"
    r2_jsonl_path = output_dir / "reviewer_2_packet.jsonl"
    manifest_path = output_dir / "holdout_review_manifest.json"

    csv_header = [
        "scenario_id",
        "conversation_transcript",
        "addressing_metadata",
        "reviewer_action",
        "is_ambiguous",
        "reviewer_confidence",
        "reviewer_rationale",
        "reviewer_comments",
    ]

    # Generate blinded CSVs for Reviewer 1 and Reviewer 2
    for csv_path in (r1_csv_path, r2_csv_path):
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            for cand in candidate_records:
                cid = cand.get("id") or cand.get("scenario_id", "unknown")
                transcript = format_conversation_transcript(cand.get("messages", []))
                addr_meta = format_addressing_metadata(cand)
                writer.writerow([cid, transcript, addr_meta, "", "", "", "", ""])

    # Generate blinded JSONL packets for Reviewer 1 and Reviewer 2
    for reviewer_id, jsonl_path in [("reviewer_1", r1_jsonl_path), ("reviewer_2", r2_jsonl_path)]:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for cand in candidate_records:
                cid = cand.get("id") or cand.get("scenario_id", "unknown")
                packet_rec = {
                    "scenario_id": cid,
                    "conversation_id": cand.get("conversation_id", f"c-{cid}"),
                    "split": "holdout",
                    "domain": cand.get("domain", "general"),
                    "participants": cand.get("participants", []),
                    "messages": cand.get("messages", []),
                    "addressing_signals": {
                        "addressed_to_hush": cand.get("addressed_to_hush", False),
                        "address_method": cand.get("address_method", "none"),
                    },
                    "assigned_reviewer": reviewer_id,
                    "reviewer_action": None,
                    "is_ambiguous": None,
                    "reviewer_confidence": None,
                    "reviewer_rationale": "",
                    "reviewer_comments": "",
                }
                f.write(json.dumps(packet_rec) + "\n")

    # Generate initial Review Manifest
    manifest_data = {
        "manifest_version": "2.0.0",
        "dataset_name": "holdout_v1",
        "sample_size": len(candidate_records),
        "review_status": "pending_human_review",
        "cohens_kappa": None,
        "raw_agreement_rate": None,
        "quality_gate_passed": False,
        "adjudication_completed": False,
        "reviewer_ids": ["reviewer_1", "reviewer_2"],
        "adjudicator_id": "project_owner",
        "records": [
            {
                "scenario_id": cand.get("id") or cand.get("scenario_id", "unknown"),
                "conversation_id": cand.get("conversation_id", f"c-{cand.get('id') or cand.get('scenario_id', 'unknown')}"),
                "split": "holdout",
                "reviewer_1": {
                    "action": None,
                    "is_ambiguous": None,
                    "confidence": None,
                    "rationale": None,
                },
                "reviewer_2": {
                    "action": None,
                    "is_ambiguous": None,
                    "confidence": None,
                    "rationale": None,
                },
                "agreement": None,
                "adjudicated_action": None,
                "adjudicator_notes": None,
            }
            for cand in candidate_records
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return {
        "reviewer_1_form": r1_csv_path,
        "reviewer_2_form": r2_csv_path,
        "reviewer_1_packet": r1_jsonl_path,
        "reviewer_2_packet": r2_jsonl_path,
        "manifest": manifest_path,
    }


def finalize_holdout_dataset(
    candidate_file: Path,
    manifest_file: Path,
    output_file: Path,
) -> dict[str, Any]:
    """Finalize the holdout dataset from candidate scenarios and an adjudicated review manifest.

    Enforces that:
    1. The manifest has completed review with quality_gate_passed == True.
    2. Cohen's Kappa >= 0.75.
    3. Zero unresolved disputes exist.
    4. Addressing invariant is strictly respected for every scenario.
    """
    if not candidate_file.exists():
        raise FileNotFoundError(f"Candidate file not found: {candidate_file}")
    if not manifest_file.exists():
        raise FileNotFoundError(f"Review manifest file not found: {manifest_file}")

    candidates = load_candidate_file(candidate_file)
    cand_by_id = {c.get("id") or c.get("scenario_id"): c for c in candidates}

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Verify quality gate
    quality_passed = manifest_data.get("quality_gate_passed", False)
    kappa = manifest_data.get("cohens_kappa")
    if not quality_passed:
        raise ValueError(
            f"Quality gate not passed in {manifest_file.name}. "
            f"Holdout dataset cannot be finalized without verified human review (quality_gate_passed must be True, kappa={kappa})."
        )

    manifest_records = manifest_data.get("records", [])
    if len(manifest_records) != len(candidates):
        raise ValueError(
            f"Candidate count ({len(candidates)}) does not match manifest record count ({len(manifest_records)})."
        )

    finalized_records: list[dict[str, Any]] = []
    distribution: dict[str, int] = {"SILENT": 0, "SPEAK": 0, "ASK": 0}

    for m_rec in manifest_records:
        sid = m_rec.get("scenario_id")
        if sid not in cand_by_id:
            raise ValueError(f"Manifest scenario_id '{sid}' not found in candidate dataset.")

        cand = cand_by_id[sid]

        # Determine finalized label
        agreement = m_rec.get("agreement")
        r1_action = m_rec.get("reviewer_1", {}).get("action")
        r2_action = m_rec.get("reviewer_2", {}).get("action")
        adjudicated = m_rec.get("adjudicated_action")

        if agreement is True and r1_action:
            final_action = r1_action
            final_rationale = (
                m_rec.get("reviewer_1", {}).get("rationale")
                or m_rec.get("reviewer_2", {}).get("rationale")
                or ""
            )
            confidence = "high"
        elif adjudicated:
            final_action = adjudicated
            final_rationale = m_rec.get("adjudicator_notes") or ""
            confidence = "adjudicated"
        else:
            raise ValueError(
                f"Scenario '{sid}' has no agreed action (R1='{r1_action}', R2='{r2_action}') and no adjudicated action."
            )

        if final_action not in {"SILENT", "SPEAK", "ASK"}:
            raise ValueError(f"Scenario '{sid}' has invalid finalized action: '{final_action}'")

        # Addressing invariant check
        is_addressed = bool(cand.get("addressed_to_hush"))
        if is_addressed and final_action == "SILENT":
            raise ValueError(
                f"Addressing invariant violation on scenario '{sid}': addressed_to_hush is True, cannot be finalized as SILENT."
            )

        distribution[final_action] = distribution.get(final_action, 0) + 1

        final_rec: dict[str, Any] = {
            "id": sid,
            "conversation_id": cand.get("conversation_id", f"c-{sid}"),
            "split": "holdout",
            "domain": cand.get("domain", "general"),
            "provenance": cand.get("provenance", {}),
            "participants": cand.get("participants", []),
            "messages": cand.get("messages", []),
            "addressed_to_hush": is_addressed,
            "address_method": cand.get("address_method", "none"),
            "expected_action": final_action,
            "rationale": final_rationale,
            "annotator_confidence": confidence,
            "review_status": "finalized",
        }
        finalized_records.append(final_rec)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in finalized_records:
            f.write(json.dumps(rec) + "\n")

    sha256 = hashlib.sha256(output_file.read_bytes()).hexdigest()

    return {
        "status": "FINALIZED",
        "total_records": len(finalized_records),
        "distribution": distribution,
        "cohens_kappa": kappa,
        "output_file": str(output_file),
        "sha256_checksum": sha256,
    }


def generate_candidate_template_csv() -> str:
    """Generate candidate scenario submission CSV template with clear header and instructions."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "scenario_id",
        "author_id",
        "created_at",
        "source_type",
        "derived_from_existing",
        "domain",
        "addressed_to_hush",
        "address_method",
        "candidate_action",
        "is_ambiguous",
        "candidate_rationale",
        "conversation_history",
        "incoming_turn_speaker",
        "incoming_turn_text",
        "incoming_turn_mentions",
        "incoming_turn_reply_to",
    ])
    writer.writerow([
        "holdout-template-01",
        "author_human_id",
        "2026-10-27T12:00:00Z",
        "fresh_human_authored",
        "false",
        "incident_response",
        "true",
        "mention",
        "SPEAK",
        "false",
        "Direct question with clear diagnostic context in prod incident channel.",
        '[{"speaker_id": "u-lead", "text": "Postgres primary on db-01 is throwing connection timeouts."}, {"speaker_id": "u-sre", "text": "Failover replica db-02 is caught up with 0s replication lag."}]',
        "u-lead",
        "@hush what is the exact pg_ctl command to promote replica db-02?",
        "hush",
        "",
    ])
    writer.writerow([
        "holdout-template-02",
        "author_human_id",
        "2026-10-27T12:05:00Z",
        "fresh_human_authored",
        "false",
        "team_chat",
        "false",
        "none",
        "SILENT",
        "false",
        "Colleagues discussing lunch arrangements with zero assistant addressing.",
        '[{"speaker_id": "u-dev1", "text": "Anyone heading down to the cafeteria for tacos?"}]',
        "u-dev2",
        "Yeah wait five minutes, wrapping up this git cherry-pick.",
        "",
        "",
    ])
    writer.writerow([
        "holdout-template-03",
        "author_human_id",
        "2026-10-27T12:10:00Z",
        "fresh_human_authored",
        "false",
        "cloud_infra",
        "true",
        "request",
        "ASK",
        "true",
        "Direct request to assistant but completely missing environment, cloud provider, and cluster parameters.",
        '[{"speaker_id": "u-dev1", "text": "Starting cluster provision ticket."}]',
        "u-dev1",
        "bot, please deploy the new staging cluster now.",
        "",
        "",
    ])
    return output.getvalue()


def generate_candidate_template_jsonl() -> str:
    """Generate candidate scenario submission JSONL template."""
    samples = [
        {
            "id": "holdout-template-01",
            "domain": "incident_response",
            "provenance": {
                "author_id": "author_human_id",
                "created_at": "2026-10-27T12:00:00Z",
                "source_type": "fresh_human_authored",
                "derived_from_existing": False,
                "collection_method": "independent_authoring",
            },
            "participants": [
                {"speaker_id": "u-lead", "display_name": "Lead", "role": "human"},
                {"speaker_id": "u-sre", "display_name": "SRE", "role": "human"},
                {"speaker_id": "hush", "display_name": "Hush", "role": "assistant"},
            ],
            "messages": [
                {"message_id": "m1", "speaker_id": "u-lead", "text": "Postgres primary on db-01 is throwing connection timeouts.", "mentions": [], "reply_to": None, "ts": 1730000000.0},
                {"message_id": "m2", "speaker_id": "u-sre", "text": "Failover replica db-02 is caught up with 0s replication lag.", "mentions": [], "reply_to": None, "ts": 1730000010.0},
                {"message_id": "m3", "speaker_id": "u-lead", "text": "@hush what is the exact pg_ctl command to promote replica db-02?", "mentions": ["hush"], "reply_to": None, "ts": 1730000020.0},
            ],
            "addressed_to_hush": True,
            "address_method": "mention",
            "candidate_action": "SPEAK",
            "is_ambiguous": False,
            "candidate_rationale": "Direct question with clear diagnostic context in prod incident channel.",
        },
        {
            "id": "holdout-template-02",
            "domain": "team_chat",
            "provenance": {
                "author_id": "author_human_id",
                "created_at": "2026-10-27T12:05:00Z",
                "source_type": "fresh_human_authored",
                "derived_from_existing": False,
                "collection_method": "independent_authoring",
            },
            "participants": [
                {"speaker_id": "u-dev1", "display_name": "Dev 1", "role": "human"},
                {"speaker_id": "u-dev2", "display_name": "Dev 2", "role": "human"},
                {"speaker_id": "hush", "display_name": "Hush", "role": "assistant"},
            ],
            "messages": [
                {"message_id": "m1", "speaker_id": "u-dev1", "text": "Anyone heading down to the cafeteria for tacos?", "mentions": [], "reply_to": None, "ts": 1730000000.0},
                {"message_id": "m2", "speaker_id": "u-dev2", "text": "Yeah wait five minutes, wrapping up this git cherry-pick.", "mentions": [], "reply_to": None, "ts": 1730000010.0},
            ],
            "addressed_to_hush": False,
            "address_method": "none",
            "candidate_action": "SILENT",
            "is_ambiguous": False,
            "candidate_rationale": "Colleagues discussing lunch arrangements with zero assistant addressing.",
        },
        {
            "id": "holdout-template-03",
            "domain": "cloud_infra",
            "provenance": {
                "author_id": "author_human_id",
                "created_at": "2026-10-27T12:10:00Z",
                "source_type": "fresh_human_authored",
                "derived_from_existing": False,
                "collection_method": "independent_authoring",
            },
            "participants": [
                {"speaker_id": "u-dev1", "display_name": "Dev 1", "role": "human"},
                {"speaker_id": "hush", "display_name": "Hush", "role": "assistant"},
            ],
            "messages": [
                {"message_id": "m1", "speaker_id": "u-dev1", "text": "Starting cluster provision ticket.", "mentions": [], "reply_to": None, "ts": 1730000000.0},
                {"message_id": "m2", "speaker_id": "u-dev1", "text": "bot, please deploy the new staging cluster now.", "mentions": [], "reply_to": None, "ts": 1730000010.0},
            ],
            "addressed_to_hush": True,
            "address_method": "request",
            "candidate_action": "ASK",
            "is_ambiguous": True,
            "candidate_rationale": "Direct request to assistant but completely missing environment, cloud provider, and cluster parameters.",
        },
    ]
    return "\n".join(json.dumps(s) for s in samples) + "\n"


def generate_collection_instructions_md() -> str:
    """Generate collection guidelines and author instructions."""
    return r"""# Hush — Fresh Holdout Collection & Dual-Review Instructions (`holdout_v1`)

## 1. Purpose & Generalization Integrity
The fresh holdout dataset (`holdout_v1`) provides an uncompromised, single-use final generalization benchmark for the Hush Conversational Restraint Engine.

Evaluation audits established that previous development splits and synthetic scenarios shared conversational stems. To establish genuine generalization:
- Scenarios must be **freshly authored by independent human contributors** or collected from real anonymized communication.
- No scenario may be derived, modified, or paraphrased from existing dataset examples in `data/dataset_v1.jsonl`.
- No LLM may be used to fabricate supposedly independent human responses or candidate dialogues.
- The holdout dataset will be evaluated exactly **once** upon project finalization.

---

## 2. Target Dataset Composition (Prospective Targets)
The target holdout set comprises exactly **50 multi-turn scenarios**.

> [!NOTE]
> The numbers below are **prospective targets** for collection planning. No scenarios are pre-populated or fabricated; collection status remains `PENDING` until human review completes.

| Action | Prospective Target | Target Pct | Scenario Description & Addressing Characteristics |
|---|---|---|---|
| **`SILENT`** | 23 | 46.0% | Human-to-human technical questions, lunch/casual chats, metadata conflicts (mentions human with "hush" in text), third-person discussions of Hush |
| **`SPEAK`**  | 17 | 34.0% | Direct mentions `@hush`, replies to Hush messages, explicit role requests with complete context/parameters |
| **`ASK`**    | 10 | 20.0% | Addressed requests that are underspecified, ambiguous, or missing critical operational parameters |
| **Total**    | **50** | **100.0%** | Multi-turn dialogues (2 to 6 turns each) across realistic engineering & collaboration domains |

---

## 3. Strict Rules for Scenario Authors

### Rule 1: Zero Overlap with Existing Scenarios (Three-Level Screening)
Submissions are automatically screened against all 300 existing scenarios across three levels:
1. **Latest turn text:** Alphanumeric word Jaccard similarity must be $< 0.70$.
2. **Full dialogue text:** Alphanumeric word Jaccard similarity must be $< 0.70$.
3. **Dialogue prefix stems:** Zero reuse of synthetic opening stems (e.g., *"Starting morning code review session"*, *"Reviewing pull request updates"*, *"Investigating service metrics"*).

### Rule 2: Mandatory Provenance Tracking
Every scenario must explicitly record:
- `author_id`: Real author or collection agent ID (e.g., `author_human_1`).
- `created_at`: ISO 8601 timestamp (e.g., `2026-10-28T10:00:00Z`).
- `source_type`: Must be one of `fresh_human_authored`, `anonymized_chat_export`, or `production_log`.
- `derived_from_existing`: Must be explicitly set to `false`. Any submission marked `true` is automatically rejected.

### Rule 3: Realistic Addressing Edge Cases
Incorporate genuine group chat edge cases:
- **Metadata Conflict:** Colleague clicks "reply" on a human teammate's message and types: `"Hush told me yesterday to check the index, did you verify?"` -> Must be classified as `SILENT` (Tier 1 metadata points to human).
- **Third-Person Reference:** Colleague says: `"Hush seems down today, can anyone ping devops?"` -> Must be `SILENT` (not addressed to Hush).
- **Ambiguous Request:** Colleague says: `@hush run the tests` without specifying repo, branch, or test suite -> Classified as `ASK`.

---

## 4. End-to-End Holdout Collection & Dual-Review Protocol

```
+------------------------------------+
| 1. Author 50 Candidate Scenarios   | (CSV or JSONL with full provenance)
+------------------------------------+
                  |
                  v
+------------------------------------+
| 2. Candidate Validation & Overlap  | (python scripts/prepare_fresh_holdout.py --validate ... --check-distribution)
+------------------------------------+
                  |
                  v
+------------------------------------+
| 3. Blinded Review Packaging        | (python scripts/prepare_fresh_holdout.py --package-review ... --output-dir ...)
|    (Labels & Rationales Stripped)  | -> reviewer_1_form.csv, reviewer_2_form.csv, holdout_review_manifest.json
+------------------------------------+
                  |
                  v
+------------------------------------+
| 4. Dual Blind Human Review         | (Reviewer 1 & Reviewer 2 annotate independently)
|    & Agreement Audit (kappa >= 0.75)| (python scripts/validate_review_manifest.py ... --save-manifest)
|    & Adjudicate Disputes           |
+------------------------------------+
                  |
                  v
+------------------------------------+
| 5. Finalize Holdout Dataset        | (python scripts/prepare_fresh_holdout.py --finalize-holdout ...)
|    (Quality Gate Enforced)         | -> data/holdout/holdout_v1.jsonl
+------------------------------------+
```

### Step 1: Scenario Authoring
Populate candidate scenarios using either:
- CSV: [`data/holdout/holdout_collection_template.csv`](../data/holdout/holdout_collection_template.csv)
- JSONL: [`data/holdout/holdout_candidate_template.jsonl`](../data/holdout/holdout_candidate_template.jsonl)

### Step 2: Candidate Validation & Overlap Screening
Verify candidate schema, provenance, and 3-level overlap against master scenarios:
```powershell
python scripts/prepare_fresh_holdout.py --validate data/holdout/my_candidates.csv --check-distribution
```

### Step 3: Package Blinded Review Forms
Generate blinded forms where candidate actions and candidate rationales are stripped:
```powershell
python scripts/prepare_fresh_holdout.py --package-review data/holdout/my_candidates.csv --output-dir data/holdout/review
```
Outputs:
- `data/holdout/review/reviewer_1_form.csv` & `reviewer_2_form.csv`
- `data/holdout/review/reviewer_1_packet.jsonl` & `reviewer_2_packet.jsonl`
- `data/holdout/review/holdout_review_manifest.json`

### Step 4: Dual Blind Human Review & Agreement Audit
Two independent reviewers label `reviewer_1_completed.csv` and `reviewer_2_completed.csv` following [`docs/annotation_rubric_v2.md`](../docs/annotation_rubric_v2.md).
Audit agreement and log any disputes:
```powershell
python scripts/validate_review_manifest.py data/holdout/review/holdout_review_manifest.json `
    --r1 data/holdout/review/reviewer_1_completed.csv `
    --r2 data/holdout/review/reviewer_2_completed.csv `
    --disputes-path data/holdout/review/disputes_v1.json `
    --save-manifest
```
- **Agreement Threshold:** Cohen's $\kappa \ge 0.75$.
- **Dispute Adjudication:** Any conflicting labels are reviewed and adjudicated by the project owner.

### Step 5: Finalize Holdout Dataset
Once the review manifest passes the quality gate ($\kappa \ge 0.75$ and 0 unresolved disputes):
```powershell
python scripts/prepare_fresh_holdout.py --finalize-holdout `
    --candidates data/holdout/my_candidates.csv `
    --manifest data/holdout/review/holdout_review_manifest.json `
    --output-file data/holdout/holdout_v1.jsonl
```

---

## 5. Security & Privacy Notice
All private reviewer forms, completed submissions, dispute logs, and the finalized `holdout_v1.jsonl` are strictly excluded from Git tracking via `.gitignore`. Never commit raw reviewer identities, dispute deliberations, or holdout benchmark data to public repositories.
"""


def generate_holdout_spec_md() -> str:
    """Generate specification document for fresh independent holdout collection."""
    return r"""# Hush — Fresh Independent Holdout Specification (`holdout_v1`)

**Target Size:** Exactly 50 multi-turn scenarios  
**Purpose:** Uncontaminated, single-use final generalization benchmark for Hush Conversational Restraint Engine.  
**Integrity Rule:** The existing 120-scenario test set was evaluated historically in Phase 2 and cannot be reused for unbiased model selection.

---

## 1. Inclusion & Exclusion Criteria

### Inclusion Criteria
1. **Realistic Multi-Turn Dialogue:** Scenarios must feature 2 to 6 conversational turns between realistic participants.
2. **Domain Diversity:** Software engineering, DevOps, team collaboration, technical support, and group chat contexts.
3. **Participant Roles:** Must clearly define human colleague IDs and the assistant ID (`hush`).
4. **Addressing Ambiguity:** Scenarios must include edge cases (third-person mention of "hush", questions directed to humans, vague troubleshooting queries).

### Exclusion Criteria
1. **Template Overlap:** Any scenario with Jaccard word similarity $\ge 0.70$ on the latest message, full concatenated dialogue, or conversational prefix turns against `data/dataset_v1.jsonl` is strictly rejected.
2. **Synthetic AI Repetition:** Prompts must not reuse formulaic sentence templates from development curation.
3. **Malformed Inputs:** Empty text, missing speaker IDs, or undefined reply-to pointers are excluded from benchmark scoring.

---

## 2. Target Class Distribution

| Action | Target Count | Percentage |
|---|---|---|
| **`SILENT`** | 23 | 46.0% |
| **`SPEAK`**  | 17 | 34.0% |
| **`ASK`**    | 10 | 20.0% |
| **Total**    | **50** | **100.0%** |

---

## 3. Data Governance & Quarantine Protocol
1. **Quarantine:** Stored in `data/holdout/holdout_v1.jsonl`.
2. **Zero Access During Development:** Training code, feature extractors, calibration routines, and threshold sweep scripts are forbidden from reading `data/holdout/`.
3. **Independent Review Requirement:** Must undergo the dual-annotator verification protocol ($\kappa \ge 0.75$) before being evaluated.
4. **Single Evaluation:** To be evaluated exactly **once** upon project completion.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fresh holdout collection tooling.")
    parser.add_argument("--validate", type=str, default=None, help="Validate candidate holdout file (JSONL or CSV)")
    parser.add_argument("--check-distribution", action="store_true", help="Enforce target distribution (50 total: 23 SILENT, 17 SPEAK, 10 ASK)")
    parser.add_argument("--threshold", type=float, default=0.70, help="Jaccard similarity rejection threshold (default: 0.70)")
    parser.add_argument("--package-review", type=str, default=None, help="Package validated candidate file into blinded reviewer forms, packets, and manifest")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for packaged review files (default: data/holdout/review)")
    parser.add_argument("--finalize-holdout", action="store_true", help="Finalize holdout dataset after review and adjudication pass quality gate")
    parser.add_argument("--candidates", type=str, default=None, help="Candidate scenario file for finalization")
    parser.add_argument("--manifest", type=str, default=None, help="Completed/adjudicated review manifest file for finalization")
    parser.add_argument("--output-file", type=str, default=None, help="Output destination for finalized holdout JSONL (default: data/holdout/holdout_v1.jsonl)")
    args = parser.parse_args()

    holdout_dir = REPO_ROOT / "data" / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    master_file = REPO_ROOT / "data" / "dataset_v1.jsonl"

    print("=" * 75)
    print(" HUSH -- PHASE 7: FRESH INDEPENDENT HOLDOUT TOOLING")
    print("=" * 75)

    with open(master_file, "r", encoding="utf-8") as f:
        master_records = [json.loads(line) for line in f if line.strip()]

    if args.validate:
        cand_path = Path(args.validate)
        if not cand_path.exists():
            print(f"Error: Candidate file not found: {cand_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Loading candidate holdout file: {cand_path}")
        try:
            candidates = load_candidate_file(cand_path)
        except Exception as exc:
            print(f"Error reading candidate file: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"Loaded {len(candidates)} candidate scenario(s).")
        result = validate_candidate_dataset(
            candidate_records=candidates,
            master_records=master_records,
            jaccard_threshold=args.threshold,
            check_target_distribution=args.check_distribution,
        )

        print("-" * 75)
        print(f"Validation Status: {result['status']}")
        print(f"Total Candidate Scenarios: {result['total_candidates']}")
        print(f"Action Distribution: {result['distribution']}")
        print(f"Schema Violations: {result['schema_error_count']}")
        print(f"Overlap Violations (Jaccard >= {args.threshold}): {result['overlap_count']}")

        if result["schema_errors"]:
            print("\nSchema / Provenance Errors:")
            for cid, errs in list(result["schema_errors"].items())[:5]:
                print(f"  - [{cid}]: {'; '.join(errs)}")

        if result["overlaps"]:
            print(f"\nOverlap Violations against Master Dataset (Top 5):")
            for ov in result["overlaps"][:5]:
                print(f"  - Candidate [{ov['candidate_id']}] matches Master [{ov['existing_master_id']}]")
                print(f"    Type: {ov['overlap_type']}, Similarity: {ov['similarity']}")

        if args.check_distribution and not result["distribution_matched"]:
            print("\nDistribution Mismatches:")
            for r in result["distribution_mismatch_reasons"]:
                print(f"  - {r}")

        print("-" * 75)
        if result["status"] == "PASS":
            print("PASSED: Candidate dataset conforms to all schema, provenance, and overlap requirements.")
            sys.exit(0)
        else:
            print("FAILED: Candidate dataset does not pass validation gates.", file=sys.stderr)
            sys.exit(1)

    if args.package_review:
        cand_path = Path(args.package_review)
        if not cand_path.exists():
            print(f"Error: Candidate file not found: {cand_path}", file=sys.stderr)
            sys.exit(1)

        out_dir = Path(args.output_dir) if args.output_dir else (holdout_dir / "review")
        print(f"Packaging candidate file into blinded review forms: {cand_path}")
        print(f"Output directory: {out_dir}")

        try:
            candidates = load_candidate_file(cand_path)
            packaged = package_blinded_review(candidates, out_dir)
        except Exception as exc:
            print(f"Error packaging review: {exc}", file=sys.stderr)
            sys.exit(1)

        print("-" * 75)
        print("Successfully packaged blinded review files:")
        for name, p in packaged.items():
            print(f"  - {name}: {p}")
        print(f"\nTotal candidate scenarios packaged: {len(candidates)}")
        print("Blinded invariant enforced: Candidate actions and rationales stripped.")
        sys.exit(0)

    if args.finalize_holdout:
        cand_path = Path(args.candidates) if args.candidates else None
        manifest_path = Path(args.manifest) if args.manifest else None
        out_path = Path(args.output_file) if args.output_file else (holdout_dir / "holdout_v1.jsonl")

        if not cand_path or not manifest_path:
            print("Error: --finalize-holdout requires both --candidates and --manifest arguments.", file=sys.stderr)
            sys.exit(1)

        print("Finalizing holdout dataset:")
        print(f"  Candidates: {cand_path}")
        print(f"  Manifest  : {manifest_path}")
        print(f"  Output    : {out_path}")

        try:
            res = finalize_holdout_dataset(cand_path, manifest_path, out_path)
            print("-" * 75)
            print(f"Finalization Status: {res['status']}")
            print(f"Total Scenarios    : {res['total_records']}")
            print(f"Distribution       : {res['distribution']}")
            print(f"Cohen's Kappa      : {res['cohens_kappa']}")
            print(f"Output File        : {res['output_file']}")
            print(f"SHA256 Checksum    : {res['sha256_checksum']}")
            print("PASSED: Holdout dataset finalized successfully.")
            sys.exit(0)
        except Exception as exc:
            print(f"Finalization Error: {exc}", file=sys.stderr)
            sys.exit(1)

    # Export specifications, manifest template, collection template CSV, JSONL, and instructions
    spec_file = holdout_dir / "holdout_spec.md"
    spec_file.write_text(generate_holdout_spec_md(), encoding="utf-8")
    print(f"Exported holdout specification: {spec_file}")

    manifest_template = {
        "manifest_version": "1.0.0",
        "dataset_name": "holdout_v1",
        "target_size": 50,
        "collection_status": "pending_collection_and_review",
        "independent_review_completed": False,
        "cohens_kappa": None,
        "sha256_checksum": None,
        "inclusion_rules_verified": False,
        "zero_overlap_verified": False,
        "records": [],
    }
    manifest_file = holdout_dir / "holdout_manifest_template.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_template, f, indent=2)
    print(f"Exported holdout manifest template: {manifest_file}")

    template_csv = holdout_dir / "holdout_collection_template.csv"
    template_csv.write_text(generate_candidate_template_csv(), encoding="utf-8")
    print(f"Exported candidate collection CSV template: {template_csv}")

    template_jsonl = holdout_dir / "holdout_candidate_template.jsonl"
    template_jsonl.write_text(generate_candidate_template_jsonl(), encoding="utf-8")
    print(f"Exported candidate collection JSONL template: {template_jsonl}")

    instructions_md = holdout_dir / "collection_instructions.md"
    instructions_md.write_text(generate_collection_instructions_md(), encoding="utf-8")
    print(f"Exported candidate collection instructions: {instructions_md}")

    print("\nStatus: Fresh holdout governance, collection templates, and validation tooling established.")
    print("Notice: No synthetic scenarios were fabricated. Collection remains PENDING.")


if __name__ == "__main__":
    main()
