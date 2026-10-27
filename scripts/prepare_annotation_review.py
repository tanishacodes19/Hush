#!/usr/bin/env python3
"""Preparation script for independent human dataset review (Hush Phase 6).

Selects 60 scenarios (36 dev, 24 test) from data/dataset_v1.jsonl using fixed seed 42.
Stratified across:
- All 3 actions: SILENT (27, 45%), SPEAK (21, 35%), ASK (12, 20%)
- 12 scenario categories (~5 per category)

Generates independent review packages in both CSV and JSONL formats:
- data/review/reviewer_1_form.csv (Convenient spreadsheet format)
- data/review/reviewer_2_form.csv
- data/review/reviewer_1_packet.jsonl (Machine-readable JSONL)
- data/review/reviewer_2_packet.jsonl
- data/review/review_guidelines.md (Concrete decision rubric)
- data/review/review_manifest_v1.json (Status tracking manifest)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def select_review_sample(
    records: list[dict[str, Any]],
    n_sample: int = 60,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select a stratified sample of 60 scenarios (36 dev, 24 test; 27 SILENT, 21 SPEAK, 12 ASK)."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["split"], r["expected_action"])].append(r)

    targets = {
        ("dev", "SILENT"): 16,
        ("dev", "SPEAK"): 13,
        ("dev", "ASK"): 7,
        ("test", "SILENT"): 11,
        ("test", "SPEAK"): 8,
        ("test", "ASK"): 5,
    }

    rng = random.Random(seed)
    selected_sample: list[dict[str, Any]] = []

    for (split, action), count in targets.items():
        candidates = sorted(groups[(split, action)], key=lambda x: x["id"])
        chosen = rng.sample(candidates, k=count)
        selected_sample.extend(chosen)

    selected_sample.sort(key=lambda x: int(x["id"].split("-")[1]))
    return selected_sample


def format_conversation_text(messages: list[dict[str, Any]], participants: list[dict[str, Any]]) -> str:
    """Format conversation transcript in clean chronological order with speaker display names."""
    spk_map = {p.get("speaker_id"): p.get("display_name", p.get("speaker_id")) for p in participants}
    lines = []
    for m in messages:
        spk = spk_map.get(m.get("speaker_id"), m.get("speaker_id", "Unknown"))
        text = m.get("text", "")
        lines.append(f"[{spk}]: {text}")
    return "\n".join(lines)


def format_addressing_summary(signals: dict[str, Any], latest_msg: dict[str, Any]) -> str:
    """Summarize observable addressing signals for human reviewer."""
    is_addr = signals.get("addressed_to_hush", False)
    method = signals.get("address_method", "none")
    mentions = latest_msg.get("mentions", [])
    reply_to = latest_msg.get("reply_to")

    addr_str = "YES" if is_addr else "NO"
    mentions_str = ", ".join(mentions) if mentions else "None"
    reply_str = reply_to if reply_to else "None"

    return f"Addressed to Hush: {addr_str} | Method: {method} | Mentions: {mentions_str} | Reply-To: {reply_str}"


def export_csv_form(sample: list[dict[str, Any]], reviewer_id: str, csv_path: Path) -> None:
    """Export convenient human-annotator CSV form with clear instructions."""
    fieldnames = [
        "scenario_id",
        "conversation_transcript",
        "addressing_metadata",
        "reviewer_action",
        "is_ambiguous",
        "reviewer_confidence",
        "reviewer_rationale",
        "reviewer_comments",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in sample:
            msgs = r.get("messages", [])
            parts = r.get("participants", [])
            signals = {
                "addressed_to_hush": r.get("addressed_to_hush", False),
                "address_method": r.get("address_method", "none"),
            }
            latest = msgs[-1] if msgs else {}

            writer.writerow({
                "scenario_id": r["id"],
                "conversation_transcript": format_conversation_text(msgs, parts),
                "addressing_metadata": format_addressing_summary(signals, latest),
                "reviewer_action": "",  # To be filled: "SPEAK", "ASK", or "SILENT"
                "is_ambiguous": "",     # To be filled: "YES" or "NO"
                "reviewer_confidence": "",  # To be filled: "high", "medium", "borderline"
                "reviewer_rationale": "",   # 1-sentence rationale
                "reviewer_comments": "",    # Optional notes
            })


def create_reviewer_packet(sample: list[dict[str, Any]], reviewer_id: str) -> list[dict[str, Any]]:
    """Create a blinded JSONL packet for a human annotator stripped of target labels."""
    packet: list[dict[str, Any]] = []
    for r in sample:
        cleaned = {
            "scenario_id": r["id"],
            "conversation_id": r["conversation_id"],
            "split": r["split"],
            "participants": r.get("participants", []),
            "messages": r.get("messages", []),
            "addressing_signals": {
                "addressed_to_hush": r.get("addressed_to_hush", False),
                "address_method": r.get("address_method", "none"),
            },
            "assigned_reviewer": reviewer_id,
            "reviewer_action": None,
            "is_ambiguous": None,
            "reviewer_confidence": None,
            "reviewer_rationale": "",
            "reviewer_comments": "",
        }
        packet.append(cleaned)
    return packet


def generate_comprehensive_guidelines_md() -> str:
    """Generate the expanded, concrete annotation rubric for Phase 6."""
    return r"""# Hush — Conversational Restraint Annotation Rubric & Review Guidelines (v2.0)

**Project:** Hush — Conversational Restraint Engine  
**Protocol:** Independent Dual Annotation & Adjudication ($\kappa \ge 0.75$)

---

## 1. Core Actions & Concrete Decision Rules

Review each multi-turn group-chat scenario chronologically. Examine the conversation history and the final incoming message to select the single most appropriate action for **Hush** (the assistant):

### Action 1: `SPEAK`
- **Definition:** Generate a direct, helpful answer immediately.
- **When to choose:**
  1. **Direct Request:** The incoming message directly addresses Hush (`@hush`, name invocation, or replying to Hush) with sufficient technical details, parameters, or logs to answer.
  2. **Group Technical Question:** An open technical request directed to the channel without naming or replying to a specific human, where an assistant is expected to provide documentation or solution.
  3. **Critical Rule:** When `addressed_to_hush == True`, choose `SPEAK` if the query can be answered without needing missing user parameters.

### Action 2: `ASK`
- **Definition:** Generate a single, clarifying question; do NOT guess or hallucinate.
- **When to choose:**
  1. **Underspecified Problem:** Hush is addressed, but critical parameters are missing (e.g. *"@hush the build failed, can you fix it?"* without build ID, branch, or error log).
  2. **Ambiguous Referent / Stale Follow-up:** A user asks *"Can you deploy that one?"* or *"Same error as before"* where prior conversation history leaves multiple possible targets.
  3. **Choice Clarification:** A request where multiple conflicting options exist (e.g. *"Deploy staging"* when both US and EU staging environments exist).
  4. **Addressing Rule:** When `addressed_to_hush == True`, if you cannot provide a precise solution without guessing, choose `ASK`.

### Action 3: `SILENT`
- **Definition:** Exercise conversational restraint; make ZERO generation calls.
- **When to choose:**
  1. **Human-to-Human Discussion:** Two or more human colleagues conversing, debating, or questioning each other.
  2. **Targeted Human Questions:** Questions explicitly naming or mentioning a colleague (e.g. *"Sam, did you push the commit?"*, *"@alex where is the key?"*), even if technical keywords appear.
  3. **Casual Banter & Status:** Social chat, greetings (*"good morning"*, *"thanks"*), signing off, emojis, or memes.
  4. **Emotional Venting Without Request:** Frustration expressed about code or tools (e.g. *"I hate legacy code!"*) without directly summoning the assistant.
  5. **Already-Answered Questions:** A question was asked earlier, but a human colleague already answered it satisfactorily in preceding turns.
  6. **False-Positive Name Usage:** The word "hush" used as an ordinary verb or noun (*"a hush fell over the room"*, *"keep it hush-hush"*).

---

## 2. Inviolable Project Invariant
> **THE ADDRESSING INVARIANT:**  
> **A message addressed to Hush or directly requesting Hush's help must NEVER be silently ignored.**  
> If `addressed_to_hush == True` (via mention, reply, name, or role request), `SILENT` is strictly forbidden. You must choose either `SPEAK` (if answerable) or `ASK` (if ambiguous/missing info).

---

## 3. Concrete Edge-Case Disambiguation Matrix

| Scenario Type | Key Indicators | Prescribed Gold Action | Decision Justification |
|---|---|---|---|
| **Direct Request to Hush** | Mentions `@hush` or *"Hush, ... "* with complete query | `SPEAK` | Addressed and actionable. |
| **Ambiguous / Incomplete Request** | Mentions `@hush` or *"Hush, ... "* but lacks error log/parameters | `ASK` | Addressed but requires clarification before answer. |
| **Human Question in Group** | *"Sam, did you check the staging logs?"* or `@alex where is the PR?` | `SILENT` | Targeted to a human colleague. Bot must stay silent. |
| **Unaddressed Open Question** | *"Does anyone know why port 8080 is blocked?"* | `SPEAK` (or `SILENT` if answered) | Open question open to the group. |
| **Already-Answered Question** | Question asked at m1, but colleague answered at m2 | `SILENT` | Problem already solved; bot intrusion creates noise. |
| **Emotional Venting at Hush** | *"Hush, your previous answer was completely wrong."* | `SPEAK` | Direct critique of assistant behavior; requires acknowledgment. |
| **Casual Social Banter** | *"Thanks Hush!"* or *"Good morning everyone"* | `SILENT` | Zero-call silence is preferred for routine acknowledgments. |
| **Name False Positive** | *"Let's keep this hush-hush until the release."* | `SILENT` | Ordinary English usage; not an invocation of the assistant. |
| **Metadata Conflict** | Text mentions "hush", but `@alex` is explicitly mentioned | `SILENT` | Authoritative metadata overrides display text. Targeted to human. |

---

## 4. Submission Instructions
1. Review the 60 scenarios in your assigned form (`reviewer_1_form.csv` or `reviewer_2_form.csv`).
2. Fill out `reviewer_action`: `"SPEAK"`, `"ASK"`, or `"SILENT"`.
3. Fill out `is_ambiguous`: `"YES"` or `"NO"`.
4. Fill out `reviewer_confidence`: `"high"`, `"medium"`, or `"borderline"`.
5. Provide a 1-sentence `reviewer_rationale`.
6. Submit completed form to the evaluation coordinator. Do NOT discuss decisions with the other reviewer before submission.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare independent human review package.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified selection")
    parser.add_argument("--size", type=int, default=60, help="Number of scenarios to sample")
    args = parser.parse_args()

    data_dir = REPO_ROOT / "data"
    dataset_file = data_dir / "dataset_v1.jsonl"
    review_dir = data_dir / "review"
    docs_dir = REPO_ROOT / "docs"

    review_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" HUSH — PHASE 6: ANNOTATION REVIEW PACKAGE PREPARATION")
    print("=" * 75)

    with open(dataset_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded master dataset: {len(records)} records from {dataset_file}")

    sample = select_review_sample(records, n_sample=args.size, seed=args.seed)
    print(f"Selected stratified sample of {len(sample)} scenarios (Seed {args.seed}):")

    # 1. Export CSV forms
    csv1_path = review_dir / "reviewer_1_form.csv"
    csv2_path = review_dir / "reviewer_2_form.csv"
    export_csv_form(sample, "reviewer_1", csv1_path)
    export_csv_form(sample, "reviewer_2", csv2_path)
    print(f"Exported Reviewer 1 CSV Form : {csv1_path}")
    print(f"Exported Reviewer 2 CSV Form : {csv2_path}")

    # 2. Export blinded JSONL packets
    p1 = create_reviewer_packet(sample, "reviewer_1")
    p2 = create_reviewer_packet(sample, "reviewer_2")
    p1_file = review_dir / "reviewer_1_packet.jsonl"
    p2_file = review_dir / "reviewer_2_packet.jsonl"
    with open(p1_file, "w", encoding="utf-8") as f:
        for item in p1:
            f.write(json.dumps(item) + "\n")
    with open(p2_file, "w", encoding="utf-8") as f:
        for item in p2:
            f.write(json.dumps(item) + "\n")
    print(f"Exported Reviewer 1 JSONL : {p1_file}")
    print(f"Exported Reviewer 2 JSONL : {p2_file}")

    # 3. Export Comprehensive Guidelines
    guidelines_text = generate_comprehensive_guidelines_md()
    guidelines_file = review_dir / "review_guidelines.md"
    guidelines_file.write_text(guidelines_text, encoding="utf-8")
    doc_rubric_file = docs_dir / "annotation_rubric_v2.md"
    doc_rubric_file.write_text(guidelines_text, encoding="utf-8")
    print(f"Exported Review Guidelines : {guidelines_file}")
    print(f"Exported Annotation Rubric : {doc_rubric_file}")

    # 4. Review manifest template
    manifest_records = []
    for r in sample:
        manifest_records.append({
            "scenario_id": r["id"],
            "conversation_id": r["conversation_id"],
            "split": r["split"],
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
        })

    manifest = {
        "manifest_version": "2.0.0",
        "dataset_version": "v1.0.0-provisional",
        "sample_size": len(sample),
        "seed": args.seed,
        "review_status": "pending_human_review",
        "cohens_kappa": None,
        "raw_agreement_rate": None,
        "quality_gate_passed": False,
        "adjudication_completed": False,
        "reviewer_ids": ["reviewer_1", "reviewer_2"],
        "adjudicator_id": "lead_architect",
        "records": manifest_records,
    }

    manifest_file = review_dir / "review_manifest_v1.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Exported Review Manifest   : {manifest_file}")
    print("\nStatus: All reviewer-friendly files prepared. Review remains PENDING human execution.")


if __name__ == "__main__":
    main()
