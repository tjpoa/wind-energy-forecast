# Controlled Retraining Decision Record

## Status

Approved contract. The policy/evidence contracts and manual monthly
eligibility evaluation are implemented. Training, MLflow/Registry writes,
deployment mutation, model-era monitoring, promotion, stabilization, rollback,
and automatic scheduling remain unimplemented until their separately reviewed
PRs.

The name "Stage 7 — Controlled Retraining" comes from the approved operational
plan. It does not replace or renumber roadmap Phase 7, which remains GitHub
Actions continuous integration.

## Scope

Controlled retraining applies only to the accepted v2 historical daily
hindcast used by the batch and monitoring workflow. The legacy v1 ANN/scaler
API, notebooks, D+1 forecasting, model-family search, cloud deployment, and
automatic model replacement are outside this contract.

Monitoring evidence supports a decision. It never trains, promotes, marks a
model stable, or rolls back a model without a separate operator action.

## Lifecycle Semantics

| Term | Contract |
| --- | --- |
| `candidate` | Registry version that passed validation and temporal backtesting; never used by the batch. |
| `champion` | Registry version referenced by the active deployment; it may be probationary or stable. |
| `probationary` | Active champion that has not completed the 90-observation stability gate. |
| `stable` | Last champion that completed observation and received a second manual approval. |

During probation, `champion` points to the newly activated model and `stable`
continues to point to the prior stable model. After manual stabilization, both
aliases point to the same version. No alias or lifecycle state changes
automatically.

The checksum-pinned local deployment pointer is the runtime source of truth.
MLflow aliases are a required governance mirror. A disagreement must stop the
batch rather than select either source silently.

## Policy

The tracked policy is `config/retraining_policy_v1.json`, with schema
`wind_forecast.retraining_policy.v1`.

- Evaluation is monthly on day 8 at 13:00 `Europe/Lisbon`, after the existing
  D+7 source-lateness boundary.
- Eligibility requires 90 new eligible observations and an active Phase 9
  drift or performance alert at warning or critical severity.
- The Phase 9 state remains the only persistence counter: the alert must
  already represent three consecutive distinct reporting dates.
- Quality alerts block evaluation. They do not trigger retraining.
- Training, promotion, and stabilization are manual.

## Manual Monthly Evaluation

PR 2 implements an offline, operator-invoked recommendation step:

```powershell
.\venv\Scripts\python.exe .\scripts\evaluate_retraining.py `
  --monitoring-report-path <exact-report.json> `
  --incumbent-id <phase9-model-snapshot-id> `
  --incumbent-fit-cutoff YYYY-MM-DD `
  --evaluated-at-utc YYYY-MM-DDTHH:MM:SSZ `
  --dry-run
```

The operator pins an exact verified Phase 9 report. The evaluator does not
infer "latest" or month-end evidence. The explicit incumbent identity and fit
cutoff are transitional inputs and are never described as a `champion`.
Evaluation is allowed only after day 8 at 13:00 `Europe/Lisbon` in the month
following the pinned cutoff and after that cutoff crosses D+7. A later manual
catch-up still seals the original evaluation month.

The evaluator verifies the report, its alert events, the derived report-state
pointer, and the Phase 9 ledger. It then uses the original `as_issued` feature
snapshots with the current verified target revision. Feature restatements are
recorded and ignored. The report's breach category is cross-checked against
its persistence state and immutable alert event.

Exactly one recommendation outcome is produced:

- `blocked_quality`;
- `insufficient_observations`;
- `no_trigger`;
- `eligible_for_manual_backtest`.

Before the monthly time gate, the outcome is `not_due` and no record is
written. `--dry-run` verifies and plans without creating an output directory.
Otherwise, the evaluator seals one content-addressed immutable JSON record
under
`data/processed/v2/retraining/evaluations/<YYYY-MM>/<evaluation-id>/evaluation.json`.
An identical rerun is idempotent. Different evidence for an already sealed
period fails closed.

This step emits a recommendation only. It does not increment Phase 9
persistence, train a candidate, write Registry state, promote or stabilize a
model, mutate deployment state, roll back, call the network, or create a
scheduler.

## Temporal Cutoffs

These fields are independent and must never be inferred from one another:

| Field | Meaning |
| --- | --- |
| `incumbent_fit_cutoff` | Last observation used to fit the incumbent. |
| `monitoring_evaluation_cutoff` | Latest report date considered by the monthly decision. |
| `data_snapshot_cutoff` | Latest observation pinned in the candidate snapshot. |
| `fold_train_cutoff` | Latest observation available to one fold's training step. |
| `fold_evaluation_start` / `fold_evaluation_end` | First and last eligible observations in one fold. |
| `candidate_fit_cutoff` | Latest observation used to fit the final candidate. |
| `promotion_effective_date` | First issuance assigned to the new deployment. |
| `observation_cutoff` | Latest probation observation included in stability review. |

The v1 ordering is:

```text
incumbent_fit_cutoff < data_snapshot_cutoff
data_snapshot_cutoff <= monitoring_evaluation_cutoff
incumbent_fit_cutoff < candidate_fit_cutoff <= data_snapshot_cutoff
monitoring_evaluation_cutoff < promotion_effective_date <= observation_cutoff
```

## Eligible Observations And Folds

An eligible observation has a unique ID and target date, pinned feature and
target revision IDs, source revisions, valid lineage and feature-schema
digests, accepted target/transformation contracts, finite feature and target
values, and no quality exclusion. Invalid or incompatible rows are reported
with reasons; they are never cleaned or silently coerced.

Eligible observations are ordered by target date. Backtesting uses complete,
non-overlapping blocks of exactly 30 observations, not 30 calendar days. Gaps
in calendar dates are retained and listed in each fold. At least three complete
folds are required and only the final incomplete block is excluded. A fold may
train only on evidence earlier than its first evaluation observation.

The later backtest compares the retrained incumbent recipe, frozen incumbent,
and `Wind_Production_Lag1` persistence on identical observation IDs. Candidate
aggregate MAE must be strictly lower than both incumbent aggregate MAE and
persistence aggregate MAE. In every fold, candidate MAE must be less than or
equal to each comparator's MAE.

The performance-breach gate reuses the incumbent calibration's
`thresholds.performance["30"]` numerical limits. Those limits are applied to
each 30-eligible-observation fold even when calendar gaps make its elapsed
span longer than 30 days. The fold records those gaps and the policy records
this choice as `incumbent_phase9_performance_30`; no unrecorded interpolation
or alternative threshold is allowed.

## Deployment Pointer

The only mutable deployment file will be `state/current.json`:

```json
{
  "schema_version": "wind_forecast.active_deployment_pointer.v1",
  "generation": 1,
  "deployment_id": "...",
  "deployment_state_id": "...",
  "state_manifest_path": "...",
  "state_manifest_sha256": "...",
  "updated_at_utc": "..."
}
```

It references an immutable `wind_forecast.deployment_state.v1` manifest.
Later increments will persist its model version, expected aliases,
bundle/calibration pins, monitoring era, fit and activation cutoffs, prior
deployment, and authorizing receipt. Mutations will require the expected
generation and deployment, write immutable evidence first, update the pointer
atomically, and fail closed on checksum or alias disagreement.

## Bootstrap Exception

The first initialization may designate the accepted v2
`selected_not_promoted` bundle as both champion and stable without rerunning
the new evaluation, backtest, or probation gates. This is a one-time migration
exception based on existing sealed-test, calibration, and checksum evidence
from Phases 4 and 9.

Bootstrap is allowed only when neither the deployment pointer nor v2 Registry
state exists. It requires manual approval and an immutable receipt with
`bootstrap_exception=true`, references the existing Phase 9 ledger as the
initial model era without rewriting it, and cannot be reused for another
version. Bootstrap initializes governance; it is not a normal promotion.

## Delivery Sequence

Implementation is split into separately reviewed PRs:

1. contracts and policy (implemented);
2. monthly evaluation (implemented);
3. temporal backtesting and v2 Registry;
4. bootstrap and deployment pointer;
5. model-era monitoring;
6. promotion, probation, and rollback;
7. stability and monthly scheduling.

Each PR must leave the repository safe and usable. The next PR starts only
after the previous PR has been reviewed and integrated.
