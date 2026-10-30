# Hush -- Independent Dataset Review Status Report

- **Manifest File:** `review_manifest_v1.json`
- **Manifest Version:** `2.0.0`
- **Sample Size:** 60 stratified scenarios
- **Review Status:** `COMPLETED_REVIEW`

## 1. Reviewer Completion Status

| Reviewer | Assigned Scenarios | Completed Scenarios | Completion Rate | Status |
|---|---|---|---|---|
| **Reviewer 1** | 60 | 60 | 100.0% | COMPLETED |
| **Reviewer 2** | 60 | 60 | 100.0% | COMPLETED |

## 2. Agreement Metrics & Quality Gate

| Metric | Measured Value | Standard Required | Status |
|---|---|---|---|
| **Observed Raw Agreement** | 98.33% | Informative | - |
| **Cohen's Kappa ($\kappa$)** | 0.9740 | $\ge 0.75$ | PASS |
| **Unresolved Disputes** | 0 | $0$ | PASS |
| **Quality Gate Determination** | - | **ALL PASS** | **PASS** |

### Inter-Annotator Agreement Matrix (Reviewer 1 vs Reviewer 2)

| R1 \ R2 | SILENT | SPEAK | ASK |
|---|---|---|---|
| **SILENT** | 28 | 0 | 0 |
| **SPEAK** | 0 | 15 | 1 |
| **ASK** | 0 | 0 | 16 |

### Adjudicated Decisions

| Scenario ID | Reviewer 1 Action | Reviewer 2 Action | Adjudicated Action | Adjudicator | Rationale |
|---|---|---|---|---|---|
| `s-192` | `SPEAK` | `ASK` | **`SPEAK`** | `project_owner` | Project-owner adjudication: The user directly replies to Hush's preceding Python... |