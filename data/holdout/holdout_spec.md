# Hush — Fresh Independent Holdout Specification (`holdout_v1`)

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
