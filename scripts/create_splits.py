#!/usr/bin/env python3
"""Split generation tooling for Hush Conversational Restraint Engine.

Produces leakage-resistant development and held-out test splits from a master dataset:
1. Strict conversation grouping: All messages with the same conversation_id belong to exactly one split.
2. Deterministic: Governed by a documented, fixed random seed (default 42).
3. Target split proportions: ~60% development (~180 scenarios) / ~40% held-out test (~120 scenarios).
4. Stratification across conversation groups: Balances SILENT (45%), SPEAK (35%), ASK (20%).
5. Generates data/splits/dev.jsonl, data/splits/test.jsonl, and data/splits/manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def compute_sha256(file_path: Path) -> str:
    """Compute sha256 checksum for a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def partition_conversations(
    conversations: dict[str, list[dict[str, Any]]],
    target_dev_ratio: float = 0.60,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Partition conversation IDs into dev and test splits maintaining class balance."""
    rng = random.Random(seed)

    # Group conversations by dominant action for stratified allocation
    action_groups: dict[str, list[str]] = {"SILENT": [], "SPEAK": [], "ASK": []}
    for conv_id, sc_list in conversations.items():
        # Determine primary action for this conversation
        act = sc_list[0].get("expected_action", "SILENT")
        if act not in action_groups:
            action_groups[act] = []
        action_groups[act].append(conv_id)

    dev_convs: list[str] = []
    test_convs: list[str] = []

    for act in sorted(action_groups.keys()):
        c_list = action_groups[act]
        rng.shuffle(c_list)
        split_idx = round(len(c_list) * target_dev_ratio)
        dev_convs.extend(c_list[:split_idx])
        test_convs.extend(c_list[split_idx:])

    dev_convs.sort()
    test_convs.sort()
    return dev_convs, test_convs


def generate_splits(
    dataset_path: Path | str,
    output_dir: Path | str = "data/splits",
    target_dev_ratio: float = 0.60,
    seed: int = 42,
    dataset_version: str = "1.0.0",
) -> dict[str, Any]:
    """Generate dev.jsonl, test.jsonl, and manifest.json from dataset file."""
    src_path = Path(dataset_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_path}")

    # Read and group by conversation_id
    records: list[dict[str, Any]] = []
    conversations: dict[str, list[dict[str, Any]]] = {}

    with open(src_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            record = json.loads(line_str)
            records.append(record)
            conv_id = record.get("conversation_id", f"c-unknown-{line_num}")
            conversations.setdefault(conv_id, []).append(record)

    # Deterministic split
    dev_conv_ids, test_conv_ids = partition_conversations(
        conversations=conversations,
        target_dev_ratio=target_dev_ratio,
        seed=seed,
    )

    dev_conv_set = set(dev_conv_ids)
    test_conv_set = set(test_conv_ids)

    # Verify zero overlap (leakage guard)
    intersection = dev_conv_set & test_conv_set
    if intersection:
        raise ValueError(f"CRITICAL: Conversation groups overlap between dev and test: {intersection}")

    dev_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    dev_actions: Counter[str] = Counter()
    test_actions: Counter[str] = Counter()

    for r in records:
        conv_id = r["conversation_id"]
        record_copy = dict(r)
        if conv_id in dev_conv_set:
            record_copy["split"] = "dev"
            dev_records.append(record_copy)
            dev_actions[record_copy["expected_action"]] += 1
        elif conv_id in test_conv_set:
            record_copy["split"] = "test"
            test_records.append(record_copy)
            test_actions[record_copy["expected_action"]] += 1
        else:
            raise ValueError(f"Conversation {conv_id} not assigned to any split")

    dev_path = out_dir / "dev.jsonl"
    test_path = out_dir / "test.jsonl"
    manifest_path = out_dir / "manifest.json"

    # Write dev.jsonl
    with open(dev_path, "w", encoding="utf-8") as f:
        for rec in dev_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write test.jsonl
    with open(test_path, "w", encoding="utf-8") as f:
        for rec in test_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Also update master dataset records if writing back with consistent split field
    with open(src_path, "w", encoding="utf-8") as f:
        for rec in dev_records + test_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Build manifest
    manifest: dict[str, Any] = {
        "dataset_version": dataset_version,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_scenarios": len(records),
        "total_conversations": len(conversations),
        "split_summary": {
            "dev": {
                "count": len(dev_records),
                "conversations_count": len(dev_conv_ids),
                "actions": dict(dev_actions),
                "conversations": dev_conv_ids,
            },
            "test": {
                "count": len(test_records),
                "conversations_count": len(test_conv_ids),
                "actions": dict(test_actions),
                "conversations": test_conv_ids,
            },
        },
        "sha256": {
            "source": compute_sha256(src_path),
            "dev": compute_sha256(dev_path),
            "test": compute_sha256(test_path),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic Hush dataset splits")
    parser.add_argument(
        "--source",
        default="data/dataset_v1.jsonl",
        help="Source dataset file (default: data/dataset_v1.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits",
        help="Output directory for splits (default: data/splits)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for grouping allocation (default: 42)",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.60,
        help="Target dev split ratio (default: 0.60)",
    )
    args = parser.parse_args()

    src = Path(args.source)
    if not src.is_absolute():
        src = REPO_ROOT / src

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out

    manifest = generate_splits(
        dataset_path=src,
        output_dir=out,
        target_dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    print("=" * 60)
    print("Hush Split Generation Successful")
    print("=" * 60)
    print(f"Seed: {manifest['seed']}")
    print(f"Total Scenarios: {manifest['total_scenarios']}")
    print(f"Total Conversations: {manifest['total_conversations']}")
    print(f"Dev: {manifest['split_summary']['dev']['count']} scenarios ({manifest['split_summary']['dev']['conversations_count']} conversations)")
    print(f"     Actions: {manifest['split_summary']['dev']['actions']}")
    print(f"Test: {manifest['split_summary']['test']['count']} scenarios ({manifest['split_summary']['test']['conversations_count']} conversations)")
    print(f"      Actions: {manifest['split_summary']['test']['actions']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
