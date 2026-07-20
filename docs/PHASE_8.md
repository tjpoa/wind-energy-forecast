# Phase 8 — Safe Incremental V2 Updates

## Status

Phase 8 implements an idempotent, transactional update path for the accepted
REN + ERA5-Land v2 dataset. It does not replace the accepted full-build
artifacts. Those files remain the immutable bootstrap baseline, while later
source observations and processed partitions are versioned separately.

This phase does not train models, refit scalers, execute notebooks, calculate
drift, operate a dashboard, or introduce Airflow.

## Operator command

Always plan a run first:

```powershell
.\venv\Scripts\python.exe .\scripts\update_v2_dataset.py `
  --through-date 2026-07-19 `
  --dry-run
```

Run the same command without `--dry-run` to refresh the planned REN dates and
ERA5-Land months, validate them, and publish an incremental release:

```powershell
.\venv\Scripts\python.exe .\scripts\update_v2_dataset.py `
  --through-date 2026-07-19
```

The command supports `--revision-lookback-days` (default 90),
`--recheck-min-age-hours` (default 24), repeated `--recheck-ren-date
YYYY-MM-DD`, and repeated `--recheck-era5-month YYYY-MM`. There is deliberately
no overwrite option. `--no-source-refresh` limits a recovery run to already
present local raw inputs and must not be used to claim that remote sources were
checked.

`--dry-run` is strictly read-only: it creates no directories, lock, run record,
manifest, or log, and does not invoke either provider adapter. It reports
missing and unavailable dates, revision rechecks, provider availability caps,
and the potentially affected integration and feature dates.

## Storage and publication contract

Processed incremental state is stored under
`data/processed/v2/incremental_update/`:

```text
runs/<run_id>/manifest.json
runs/<run_id>/events.jsonl
releases/<run_id>/integrated/date=YYYY-MM-DD/
releases/<run_id>/features/date=YYYY-MM-DD/
state/current.json
staging/<run_id>/
quarantine/<run_id>/
```

New or revised source observations are copied to immutable SHA-256-addressed
locations below `data/raw/v2/incremental_update/`. Provider payloads and status
evidence emitted by the existing ingestors are retained as supporting blobs.
Semantic checksums exclude retrieval-only metadata. A physically different but
semantically equal observation is retained without invalidating downstream
partitions.

The first real run accepts only the fixed `2010-01-01` through `2026-06-27`
bootstrap. It verifies the accepted integrated and feature manifests,
validation verdicts, recorded output checksums, exact calendars, ready-date
sets, schemas, null policy, and local source evidence before indexing any
baseline reference. Tests may override this calendar for isolated synthetic
fixtures; the operator CLI cannot.

Every current partition reference contains a deterministic partition key,
file checksums, lineage, and its storage origin. Baseline references select one
date from the accepted full CSV without copying or changing it. Incremental
references point to immutable release files. Consumers use
`materialize_current_integrated()` or `materialize_current_features()`; both
verify checksums and reject corrupt or ambiguous state.

Publication has one commit point: an atomic replacement of `state/current.json`.
The success manifest is written first and its checksum is recorded in the
pointer. A failed run before that step cannot expose staged partitions or
advance watermarks. Staging is quarantined, while valid content-addressed raw
observations may be reused. A crash immediately after pointer publication leaves
a valid committed generation; rerunning converges to `no_op`.

## Watermarks, revisions, and validation

- REN and ERA5-Land have separate `observed_through`,
  `validated_watermark`, `published_watermark`, and explicit gap lists.
  Watermarks are maxima, not claims of contiguous history.
- REN is eligible through the previous `Europe/Lisbon` civil day. ERA5-Land is
  conservatively eligible through local day D-6.
- Existing REN `unavailable` status partitions remain explicit gaps and do not
  block later watermarks. Old unavailable dates are retried only inside the
  revision window or through `--recheck-ren-date`.
- REN finality remains `unknown`. Semantic changes create immutable revisions
  linked by `supersedes_id`.
- ERA5-Land observations are operationally labelled `preliminary_window` until
  90 days after month end and `consolidated_window` afterwards. A finality
  transition is versioned but does not recalculate data when values are
  semantically unchanged.
- ERA5-Land source identity is `station_id + UTC month`, independent of a
  provider file's partial period label. A later partial-month extension is
  merged immutably with the prior observation, with the latest capture winning
  only on overlapping timestamps. Completeness is evaluated per station and
  month before a month is considered available.
- ERA5-Land revisions are compared by UTC timestamp and mapped to
  `Europe/Lisbon`, so only changed local dates are integrated, including month
  boundaries and DST days.
- A changed integrated date `S` invalidates feature outputs from `S` through
  `S+14`. Feature generation reads at least the preceding 14 calendar days and
  reuses the accepted 58-column formula and order.

Before publication the updater checks current-pointer and manifest integrity,
source schemas, timestamps, sorting, duplicates, finite required values,
station identity, 17-point coverage, REN 92/96/100 interval counts, ERA5-Land
23/24/25 hourly counts per point, integrated 27-column shape, feature
58-column shape, null policy, checksums, and unique daily keys. It fails closed;
it never repairs or overwrites a current partition silently.

## Observability and recovery

Run identifiers combine a UTC timestamp and random suffix. Real runs emit JSON
Lines to stdout and `events.jsonl`; the final manifest uses schema
`wind_forecast.v2_incremental_run.v1` and records normalized arguments, Git
commit, code/contract versions, plans, watermarks, source changes, affected
intervals, validation state, warnings, failures, and safeguards. Common secret
patterns are redacted from persisted errors.

An exclusive lock records host, PID, run ID, and creation time. A live owner is
rejected before run output is written. A stale same-host PID is recovered; its
original manifest is preserved, a separate `abandoned.json` recovery record is
written, and incomplete staging and unpublished release content are
quarantined. If the verified current pointer already names that run, it is
treated as committed rather than abandoned. A lock from another host is never
guessed stale.

Recovery procedure:

1. Inspect the failed or abandoned run manifest and `events.jsonl`.
2. Confirm `state/current.json` still verifies with the public materializers.
3. Correct the input/provider problem without editing existing raw or release
   files.
4. Rerun the identical command. The updater reuses valid observations and
   either publishes one complete generation or returns `no_op`.

## Acceptance evidence

`tests/test_incremental.py` uses synthetic local inputs and no network. It
covers read-only dry runs, two consecutive executions, missing-date discovery,
filling a gap older than the watermark, preservation of a complete REN
partition after a later unavailable observation, exact-date and partial-month
ERA5-Land revisions, semantic-equivalent physical captures,
preliminary-to-consolidated ERA5 policy, bounded 14-day feature invalidation,
DST alignment checks, duplicate/schema/null rejection, bootstrap truncation,
incremental-versus-clean-rebuild equivalence, failure injection after download,
source validation, integration, and on both sides of the atomic publish point,
live-lock rejection, committed-lock recognition, and stale-lock recovery.

The live REN/CDS refresh path has not been exercised as part of repository
tests. Operators must first review a dry run and use approved credentials and
network authorization. Current v1 and v2 model/scaler compatibility remains
unchanged and is not claimed by this phase.
