# Phase 2 REN Backfill Readiness

## Purpose

This document records Phase 2 Step 2A.10b REN backfill operational readiness after the January 2010 recovery pilot. It is a readiness note only.

## Scope And Non-Goals

Scope:

- Assess whether the current REN v2 raw ingestion implementation is operationally ready for a complete backfill.
- Record the January 2010 bounded live retry, resume/idempotency check, dry-run check, file inventory, and v1 comparison results.
- Preserve the operational decision for v2 raw file layout.

Non-goals:

- No full historical backfill was run.
- No v1 data, processed data, models, scalers, notebooks, or baselines were changed.
- No ERA5-Land, CDS, IPMA, WeatherAPI, Step 2B, feature regeneration, scaler refitting, model training, or metric re-baselining work was started.
- No new data source assumption is introduced beyond the existing REN v2 production-source decision.

Ignored v2 pilot outputs under `data/raw/v2/production` remain uncommitted.

## Implementation And Recovery Defect

The initial sandbox run attempted the January 2010 REN pilot. Dates `2010-01-04` through `2010-01-31` failed with `WinError 10013` network/socket permission errors and left status-only `unavailable` metadata partitions. That failure exposed a same-range resume defect: `--resume` correctly skipped verified partitions, but then refused to retry existing unverified status-only partitions; `--resume --overwrite` was intentionally disallowed.

The narrow correction adds `--retry-unavailable` for use with `--resume`. The retry path recognizes only strict status-only unavailable partitions: `status.json` exists, raw and normalized files do not exist, validation status is exactly `unavailable`, and status metadata has no raw or normalized paths or checksums. Verified partitions are still skipped and are never overwritten by retry recovery.

## Bounded Live Pilot

The bounded live retry used:

```python
run_ingestion(
    start_date="2010-01-01",
    end_date="2010-01-31",
    output_root=Path("data/raw/v2/production"),
    request_delay=1.0,
    resume=True,
    retry_unavailable=True,
    compare_v1_csv=Path("data/raw/ReparticaoProducao.csv"),
)
```

Validated result:

| Item | Result |
| --- | --- |
| Runtime seconds | `91.352694200119` |
| Requests made | `28` |
| Daily result status counts | `{"complete": 31}` |
| Manifest | `data/raw/v2/production/ren/manifests/ren_production_manifest.json` |
| Manifest SHA-256 | `a50b2ae3f50f868c5017e77914bb8860ab01a4add0b4cf7c96bc1e91b3c8d1cd` |

Previously verified smoke partition hashes for `2010-01-01` through `2010-01-03` and the separately verified `2026-06-27` partition were captured before the live retry and were preserved. The recovery request count of 28 matches only the Jan 4-31 status-only unavailable partitions.

## Monthly Recovery Metrics

| Item | Result |
| --- | --- |
| January raw files | `31` |
| January normalized files | `31` |
| January status files | `31` |
| January file count | `93` |
| January size | `1,403,959 bytes` |
| Total REN tree file count | `97` |
| Total REN tree size | `1,483,469 bytes` |
| Manifest size | `34,683 bytes` |
| January status counts | `complete: 31` |
| Warnings | `0` |
| Errors | None |
| Expected rows | `2,976` |
| Actual rows | `2,976` |
| First timestamp | `2010-01-01T00:00:00` |
| Last timestamp | `2010-01-31T23:45:00` |
| Missing expected timestamps | `0` |
| Duplicate timestamps | `0` |
| Unit | `MW` for 31 days |
| Manifest row count | `3,072`, including January plus `2026-06-27` |
| Manifest coverage start | `2010-01-01` |
| Manifest coverage end | `2026-06-27` |
| Unavailable dates | Empty |
| Incomplete dates | Empty |

## Resume And Idempotency

Resume verification used a guarded request function after the recovery was complete.

| Item | Result |
| --- | --- |
| Runtime seconds | `27.808218399994075` |
| Requests made reported | `0` |
| Guarded request calls | `0` |
| Daily result count | `31` |
| Complete results | `31` |
| Skipped existing count | `31` |
| Partition inventory unchanged | `true` |
| Partition file count before / after | `96 / 96` |
| Partition row counts unchanged | `true` |
| Manifest JSON parseable | `true` |
| Manifest hash changed | `true`, because run metadata changed while partition inventory stayed byte-stable |

This confirms that a same-range resume does not call the network and does not rewrite verified raw, normalized, or status partition files. Manifest metadata may change on a resumed run.
After the resume verification, the current manifest SHA-256 was `032c12b26db5499fc199186f002e41724e0ef336fb931b20fa2dc96219603bb4`.

## Dry-Run Verification

Dry-run verification used a guarded request function and performed no network calls or writes.

| Item | Result |
| --- | --- |
| Runtime seconds | `0.0037366000469774008` |
| `dry_run` | `true` |
| Network requests planned | `31` |
| Writes planned | `false` |
| Guarded request calls | `0` |
| Partition plan count | `31` |
| First planned date | `2010-01-01` |
| First planned paths | raw, normalized, and status paths for `date=2010-01-01` |
| Last planned date | `2010-01-31` |
| Last planned paths | raw, normalized, and status paths for `date=2010-01-31` |
| Inventory unchanged | `true` |
| File count before / after | `97 / 97` |

## V1 Comparison

The January 2010 recovered REN partitions were compared with the frozen v1 production CSV.

| Metric | Result |
| --- | --- |
| Aligned rows | `2,976` |
| Exact match count | `2,976` |
| Mean absolute error | `0.0` |
| Maximum absolute difference | `0.0` |
| Pearson correlation | `1.0` |
| Dates with nonzero differences | None |

This confirms exact overlap for the bounded January 2010 pilot. It does not prove full historical equivalence.

## File-Layout Decision

Decision A: keep daily raw, normalized, and status partitions for v2 raw REN ingestion.

Rationale:

- The partition layout isolated the sandbox failure to Jan 4-31 and allowed selective retry without replacing verified partitions.
- Per-day metadata, checksums, and validation status remain inspectable.
- A complete backfill can use the same layout and manifest conventions without a minimal further code change.

Compact derivatives may be produced separately in a later approved step. They should not replace the raw v2 partition layout. After transient unavailable failures, the required recovery path is to rerun with `--resume --retry-unavailable`.

## Operational Readiness

Assessment: ready for a complete REN production backfill only with explicit operator approval.

Readiness conditions:

- Use a conservative request delay.
- Keep outputs under the ignored v2 output path.
- Monitor request counts, status counts, file inventory, row counts, warnings, and manifest parseability.
- Preserve v1 as immutable; do not append to, overwrite, or reinterpret v1 files.
- Treat network/socket failures as recoverable only when they leave strict status-only unavailable partitions.
- Stop and inspect any corrupt, partial, raw-only, normalized-only, checksum-mismatched, incomplete, or schema-invalid partition.

No full backfill should be started automatically from this checkpoint.

## Future Full-Backfill Template

The latest separately verified date for this readiness note is `2026-06-27`.

NOT RUN:

```python
from pathlib import Path

from scripts.ingest_ren_production_v2 import run_ingestion

result = run_ingestion(
    start_date="2010-01-01",
    end_date="2026-06-27",
    output_root=Path("data/raw/v2/production"),
    request_delay=1.0,
    resume=True,
    retry_unavailable=True,
    compare_v1_csv=Path("data/raw/ReparticaoProducao.csv"),
)
```

The command requires explicit operator approval and live REN network access before use.

## Monitoring Checklist

Before running:

- Confirm `git status --short` and ensure only expected uncommitted files are present.
- Confirm output root is `data/raw/v2/production` and remains ignored.
- Confirm the requested end date and any separately verified latest-date evidence.
- Confirm `--resume --retry-unavailable` is used for recovery runs after transient unavailable failures.

During running:

- Track elapsed time, request count, request delay, and any HTTP or socket failures.
- Track daily result status counts and warnings.
- Stop on repeated non-transient failures, corrupt metadata, or validator errors.

After running:

- Count raw, normalized, and status files.
- Validate expected 96 rows per complete day unless a documented incomplete date is accepted.
- Check manifest JSON parseability, row count, coverage start/end, unavailable dates, incomplete dates, and checksums.
- Compare overlapping dates with frozen v1 where applicable.
- Confirm v1 files and existing model artifacts were not modified.

## Risks, Fallbacks, And Rollback

Risks:

- REN historical values may differ from frozen v1 outside the January 2010 pilot.
- REN service availability, throttling, socket permissions, or transient HTTP failures may interrupt a long run.
- Manifest metadata can change on resume even when partition files remain byte-stable.
- REN timezone semantics and source status remain unresolved for downstream daily aggregation and model use.

Fallbacks:

- For strict status-only unavailable partitions, rerun the same range with `--resume --retry-unavailable`.
- For verified partitions, rerun with plain `--resume` to skip without network calls.
- For any other existing unverified or corrupt partition, stop and inspect manually before choosing a repair path.

Rollback:

- Do not mutate v1.
- Because v2 outputs are isolated under the ignored v2 raw path, rollback can be handled by archiving or removing the affected v2 output tree only after explicit approval.
- Do not delete data, manifests, or pilot outputs without operator approval.

## Acceptance Criteria Checklist

- [x] January 2010 recovered to 31 complete daily partitions.
- [x] Jan 4-31 status-only unavailable partitions were retried without overwriting verified Jan 1-3 partitions.
- [x] Previously verified smoke partitions, including `2026-06-27`, were preserved.
- [x] Same-range resume made zero guarded network calls and skipped all verified partitions.
- [x] Dry run planned 31 partitions without network calls or writes.
- [x] January row count was `2,976` with no missing or duplicate timestamps.
- [x] January v1 overlap comparison was exact across all aligned rows.
- [x] Manifest was parseable and recorded coverage through `2026-06-27`.
- [x] V1 data, models, scalers, notebooks, and baselines were not changed.
- [x] Full backfill remains not run and requires explicit operator approval.

Step 2B and downstream training remain paused.
