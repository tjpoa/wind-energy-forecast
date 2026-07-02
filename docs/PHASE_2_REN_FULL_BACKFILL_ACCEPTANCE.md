# Phase 2 REN Full Backfill Acceptance Audit

## Audit Scope

This document records the Phase 2 Step 2A.11 local acceptance audit for the REN v2 raw production full backfill. It is an audit of the locally generated raw REN v2 backfill artifacts and manifest evidence only.

The audit covers coverage, status metadata, row counts, timestamp/DST behavior, value bounds, partition integrity, checksums, manifest consistency, frozen v1 overlap, and resume-safety implications.

No network calls were made for this audit. No notebooks, pipelines, data regeneration, model training, scaler fitting, or downstream aggregation were run. Ignored v2 backfill data was not altered by this documentation step.

## Verdict

Qualified acceptance: the raw REN v2 full backfill is locally auditable and checksum-consistent.

This verdict is qualified by known caveats:

- Six dates are unavailable in the local backfill.
- One date, `2010-03-28`, is marked incomplete because legacy status metadata expected 96 rows, while the physical Europe/Lisbon spring-DST expectation is 92 intervals and matches the CSV.
- The manifest requested-ranges union covers fewer dates than status coverage.
- REN timezone semantics, source license, and provisional/final source-status behavior remain unresolved.
- Frozen v1 overlap contains late-period revisions and therefore does not support replacing v1 without a separate approved migration, validation, and re-baselining step.

This audit does not approve downstream daily aggregation, v2 processed features, scaler validity, model validity, metric comparison, or replacement of frozen v1 artifacts.

## Repository And Data Locations

Repository context:

- Roadmap phase: Phase 2, data validation and sanity checks.
- Step: Phase 2 Step 2A.11, REN full backfill acceptance audit.
- Audit document: `docs/PHASE_2_REN_FULL_BACKFILL_ACCEPTANCE.md`.
- Commit message for this audit: `docs: accept REN full backfill audit`.

Data context:

- Raw REN v2 backfill artifacts remain under the ignored v2 production data area.
- Frozen v1 production data remains the immutable comparison baseline.
- No v1 raw data, processed data, models, scalers, notebooks, reports, or baselines were modified or approved for mutation.

## Evidence Summary

| Item | Evidence |
| --- | ---: |
| Coverage start | `2010-01-01` |
| Coverage end | `2026-06-27` |
| Calendar dates covered | `6,022` |
| Total files | `18,055` |
| Total storage | `279,440,105 bytes` |
| JSON files | `12,039` |
| CSV files | `6,016` |
| Status rows, complete | `6,015` |
| Status rows, incomplete | `1` |
| Status rows, unavailable | `6` |
| Status rows, invalid | `0` |
| Manifest row sum | `577,532` |
| Status row sum | `577,532` |
| Manifest SHA-256 | `97cdd28e609a7a4573c4dda4e5fddd55fd6be60f00742c7d03b14897ddd68cd0` |

## Coverage And Status

The local status coverage spans `2010-01-01` through `2026-06-27`, covering `6,022` calendar dates.

Status distribution:

- Complete: `6,015` dates.
- Incomplete: `1` date.
- Unavailable: `6` dates.
- Invalid: `0` dates.

Unavailable dates:

- `2014-05-03`
- `2016-02-03`
- `2016-02-04`
- `2021-10-03`
- `2023-08-30`
- `2025-08-02`

Incomplete date:

- `2010-03-28`

For `2010-03-28`, the CSV contains `92` rows, which matches the physical Europe/Lisbon spring-DST expectation. The incomplete status is therefore attributed to older status metadata that expected `96` rows for every date, not to physically missing intervals in the CSV.

The current DST-aware REN normalized-day validator accepts all `6,016` normalized CSV partitions as complete; the single incomplete status above is a persisted metadata caveat, not a current validation failure.

## DST And Timestamp Policy

The row evidence is consistent with physical Europe/Lisbon DST behavior:

| Interval class | Dates | Rows |
| --- | ---: | ---: |
| Ordinary days | `5,983` | `574,368` |
| Spring DST days | `17` | `1,564` |
| Fall DST days | `16` | `1,600` |

Timestamp checks:

- Timestamp identity duplicates: `0`.
- Physical missing intervals: `0`.
- Expected fall local-wall-clock duplicates: `16` dates / `128` rows.

Policy implication: fall-DST repeated local wall-clock intervals are expected physical behavior and must be preserved by timestamp handling. Spring-DST days may contain `92` physical intervals rather than `96`; legacy metadata expecting `96` rows requires interpretation rather than silent correction.

REN timezone semantics remain unresolved for downstream daily aggregation. This audit accepts only the local raw backfill integrity evidence and does not approve canonical daily aggregation boundaries.

## Value Validation

All `577,532` rows have finite, non-negative values in unit `MW`.

Observed production-value bounds:

- Minimum: `0.0 MW`.
- Maximum: `5094.6 MW`.

No value evidence in this audit indicates negative production, non-finite values, or mixed units.

## Partition And Checksum Integrity

The backfill tree contains `18,055` files totaling `279,440,105 bytes`:

- `12,039` JSON files.
- `6,016` CSV files.

Checksum and partition checks:

- Manifest records `18,054` file checksums.
- Missing files: `0`.
- Checksum mismatches: `0`.
- Raw/normalized partition mismatches: `0`.

The local partition evidence supports the qualified acceptance verdict for raw v2 storage integrity.

## Manifest Assessment

The manifest is parseable and deterministic serialization matches the file.

Manifest SHA-256:

```text
97cdd28e609a7a4573c4dda4e5fddd55fd6be60f00742c7d03b14897ddd68cd0
```

Manifest row evidence:

- Manifest row sum: `577,532`.
- Status row sum: `577,532`.
- Manifest checksum count: `18,054`.

Manifest caveat: the requested-ranges union covers `5,665` dates, while status coverage has `6,022` dates. Therefore, `357` status-covered dates are not represented in requested ranges.

The manifest is accepted as parseable and checksum-consistent for this local audit. It is not described as perfect because the requested-range gap remains a caveat to resolve or document before any stronger operational claim.

## Frozen V1 Overlap

Frozen v1 overlap through `2025-04-28` was compared to the REN v2 backfill.

| Metric | Result |
| --- | ---: |
| V1 rows | `537,308` |
| REN-aligned rows | `536,828` |
| Exact matches | `515,533` |
| Nonzero differences | `21,295` rows |
| Dates with nonzero differences | `258` |
| First nonzero-difference date | `2024-06-01` |
| Last nonzero-difference date | `2025-04-28` |
| MAE | `0.2445271111 MW` |
| Maximum absolute difference | `130.0 MW` |
| Correlation | `0.9999981644` |

REN aligns fewer rows than v1 because five unavailable dates fall inside v1 coverage.

Interpretation: the overlap is extremely close overall, but late-period differences confirm that v2 must not be silently substituted for frozen v1. The differences are consistent with unresolved source revision behavior or source-status differences, and this audit does not identify their cause.

## Resume Safety

Prior readiness evidence established the intended v2 raw layout and recovery behavior:

- Daily raw, normalized, and status partitions isolate failures by date.
- Verified partitions can be skipped with resume behavior.
- Strict status-only unavailable partitions can be retried through the approved retry path.
- Verified raw and normalized partitions must not be overwritten during recovery.

For this full-backfill audit, the checksum and partition evidence shows no missing files, no checksum mismatches, and no raw/normalized partition mismatches. That supports local resume-safety and inspectability of the partitioned raw v2 layout.

Any future retry or repair must remain explicit, bounded, and non-destructive. Corrupt, partial, raw-only, normalized-only, checksum-mismatched, or otherwise ambiguous partitions require manual inspection before choosing a repair path.

## Acceptance Checklist

- [x] Coverage spans `2010-01-01` through `2026-06-27`.
- [x] Status coverage includes `6,022` calendar dates.
- [x] Status counts are recorded: `6,015` complete, `1` incomplete, `6` unavailable, `0` invalid.
- [x] Unavailable dates are explicitly listed.
- [x] The single incomplete date is explained by legacy 96-row status metadata versus physical spring-DST expectation.
- [x] Manifest and status row sums match at `577,532`.
- [x] Ordinary, spring-DST, and fall-DST interval classes are recorded.
- [x] Timestamp identity duplicates are `0`.
- [x] Physical missing intervals are `0`.
- [x] Expected fall local-wall-clock duplicates are documented.
- [x] Values are finite, non-negative, and in `MW` for all rows.
- [x] File inventory and storage totals are recorded.
- [x] Manifest checksum evidence reports no missing files, checksum mismatches, or raw/normalized partition mismatches.
- [x] Manifest parseability, deterministic serialization, and SHA-256 are recorded.
- [x] Manifest requested-range gap is documented as a caveat.
- [x] Frozen v1 overlap is summarized without approving v1 replacement.
- [x] No network calls were made for this audit.
- [x] No downstream aggregation, scaler fitting, model training, notebook execution, or pipeline regeneration is approved by this document.

## Risks And Caveats

Known caveats:

- Six unavailable dates remain in the raw v2 backfill.
- The `2010-03-28` status is incomplete because of legacy expected-row metadata, even though the CSV matches physical spring-DST behavior.
- Manifest requested ranges do not cover all status-covered dates.
- REN timezone semantics remain unresolved.
- REN license and attribution requirements remain unresolved.
- REN provisional/final source-status behavior remains unresolved.
- Late v1 overlap revisions exist from `2024-06-01` through `2025-04-28`.
- The cause of the nonzero v1 overlap differences is not resolved by this audit.

Operational risks:

- Treating local raw integrity as daily aggregation approval could introduce timezone-boundary errors.
- Treating v2 overlap closeness as v1 replacement approval could invalidate existing baselines.
- Treating current v1 scalers or trained models as v2-compatible would violate the Phase 2 data-contract decision.

## Non-Goals

This audit does not:

- Run network calls.
- Download or regenerate data.
- Modify ignored v2 backfill data.
- Modify frozen v1 data.
- Execute notebooks.
- Run pipelines.
- Create processed v2 features.
- Approve downstream daily aggregation.
- Refit scalers.
- Retrain models.
- Re-baseline metrics.
- Approve v2 model or scaler validity.
- Approve replacing v1 with v2.
- Start Step 2B or Phase 3.

## Rollback And Containment

This documentation step creates only this audit file. Rollback for this step is to remove `docs/PHASE_2_REN_FULL_BACKFILL_ACCEPTANCE.md` before any future staging or commit, if the audit document is rejected.

Raw v2 backfill artifacts remain isolated under ignored v2 storage and were not modified by this documentation step. Any rollback, deletion, archiving, or repair of v2 data artifacts requires separate explicit approval.

Frozen v1 data, processed files, models, scalers, notebooks, reports, and baselines remain unchanged and must continue to be treated as the current reproducible baseline.

## Final Status

Phase 2 Step 2A.11 status: qualified local acceptance audit documented.

The raw REN v2 full backfill is locally auditable and checksum-consistent, subject to the caveats recorded above. This is not approval for downstream aggregation, v1 replacement, v2 model validity, scaler validity, or metric promotion.

Step 2B was not started. Phase 3 was not started.
