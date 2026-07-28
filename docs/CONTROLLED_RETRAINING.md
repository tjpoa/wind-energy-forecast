# Controlled Retraining Decision Record

## Status

Approved contract. The policy/evidence contracts, manual monthly eligibility
evaluation, manual temporal backtesting, v2 candidate Registry action, and
one-time deployment bootstrap and model-era monitoring are implemented.
Promotion, stabilization, rollback, and automatic scheduling remain
unimplemented until their separately reviewed PRs.

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

## Manual Temporal Backtest

PR 3 implements the separately invoked training and comparison step:

```powershell
.\venv\Scripts\python.exe .\scripts\backtest_retraining_candidate.py `
  --evaluation-path <exact-evaluation.json> `
  --monitoring-store-root <phase9-store> `
  --incumbent-bundle <accepted-v2-bundle> `
  --incumbent-base-dataset <exact-v2-csv> `
  --calibration-dir <exact-calibration-directory> `
  --policy-path .\config\retraining_policy_v1.json `
  --output-root <operator-selected-backtest-root> `
  --dry-run
```

The command accepts only an identity-verified
`eligible_for_manual_backtest` evaluation. It verifies the evaluation's
policy, report, report-state, ledger, calibration/reference, model snapshot,
and evidence IDs and hashes. It reconstructs each observation from the
immutable `as_issued` input snapshot and the current actual revision, recomputes
the observation ID and lineage hash, preserves the recorded order and calendar
gaps, and fails closed on any mismatch.

The incumbent base dataset must match the accepted v2 bundle checksum and exact
ordered schema. Only historical rows through `incumbent_fit_cutoff` enter the
first training window. Each later fold adds only observations from earlier
folds. The candidate recipe is the exact `get_params(deep=True)` mapping stored
in the incumbent model manifest and is allowlisted to
`ExtraTreesRegressor` or `RandomForestRegressor`; there is no family or
hyperparameter search.

Every comparator uses identical observation IDs. Each fold contains exactly 30
eligible observations and at least three complete folds are required. The
incomplete tail is excluded from comparison, but an accepted final model is
refit through `data_snapshot_cutoff`. Candidate aggregate MAE must be strictly
lower than both the frozen incumbent and `Wind_Production_Lag1`; candidate fold
MAE must be no worse than either comparator in every fold.

The no-performance-breach gate applies the exact incumbent calibration
`thresholds.performance["30"]` leaves and the public Phase 9 directional
threshold semantics to candidate `MAE`, `RMSE`, `MAPE_percent`, `R2`, and
`absolute_bias` (derived from the absolute value of signed `bias`). A warning
or critical result for any metric rejects the backtest.

`--dry-run` performs all validation and in-memory modelling but creates no
output directory. A non-dry run seals exactly one content-addressed result per
evaluation period:

```text
<output-root>/<evaluation_period>/<backtest_id>/
```

The strict root `bundle_manifest.json` has schema
`wind_forecast.retraining_backtest_bundle.v1`, embeds the
`wind_forecast.retraining_backtest.v1` record, and contains the SHA-256 of every
other file. The loader rejects undeclared files, directories, symbolic links,
unsafe paths, or any file-set difference. Every bundle contains
`backtest.json`, `predictions.csv`,
`fold_metrics.json`, `aggregate_metrics.json`, `lineage.json`,
`safeguards.json`, and `environment.json`. A rejected bundle deliberately has
no final model. An accepted bundle additionally contains `model.joblib`,
`model_manifest.json`, `dataset_manifest.json`, `reload_sample.csv`, and the
complete `training_evidence.csv` used for Registry reload equivalence.
The record distinguishes the IDs evaluated in complete folds from every
candidate-snapshot observation ID used by the final refit, including an
incomplete tail. The final-training identity hashes the base dataset, cutoffs,
row counts, and all candidate observation IDs; separate hashes pin the exact
final training table and serialized candidate model.
Identical reruns are idempotent. Different evidence for an already sealed
period fails closed.

## Manual v2 Candidate Registration

An accepted sealed bundle can be registered explicitly:

```powershell
.\venv\Scripts\python.exe .\scripts\register_retraining_candidate.py `
  --backtest-bundle <accepted-backtest-directory> `
  --run-id <finished-mlflow-run-id> `
  --registered-model-name <explicit-v2-model-name> `
  --expect-no-current-candidate `
  --output-root <operator-selected-receipt-root>
```

Use `--expected-current-candidate-version <exact-version>` instead when a
candidate already exists. One of these two optimistic state assertions is
required. The v2 registered-model name is also required: an empty value and
the legacy `DEFAULT_REGISTERED_MODEL_NAME` are rejected, and no v2 default is
invented.

Before mutation, the action validates the complete local accepted bundle,
clean Git lineage, a `FINISHED` MLflow run pinned to the backtest, the exact
run-artifact copy of `model.joblib`, logged model URI, ordered numeric input
signature and numeric output, run identity, estimator class/parameters/features,
and prediction equivalence over every row in the sealed
`training_evidence.csv`. It then verifies the expected candidate state, snapshots
`champion` and `stable`, creates and tags one version, rechecks all three aliases
before moving only `candidate`, and verifies that `champion` and `stable`
remain unchanged.

PR 3 does not create or log that MLflow run. Before invoking registration, the
operator must prepare the run through a separately controlled local MLflow
logging step. Its parameters must include `logged_model_uri`, `backtest_id`,
`git_sha`, `git_dirty=false`, and the safe run-relative
`candidate_model_artifact_path` ending in `model.joblib`. The referenced run
artifact must be byte-identical to the sealed candidate model, and the logged
model must carry the required signature.

Tags pin the evaluation/backtest/calibration/reference identities, policy,
feature and incumbent hashes, temporal cutoffs, aggregate candidate metrics,
source run, Git commit, candidate-model hash, and final-training dataset and
identity hashes. A successful action writes one immutable
content-addressed `wind_forecast.retraining_registration_receipt.v1` beneath
the operator-selected receipt root. If an alias update or receipt write fails,
the action restores the prior candidate alias when it can do so safely and
raises a reconciliation error if compensation itself cannot be trusted. A
created but unaliased Registry version is retained as audit evidence; this
action never changes `champion`, `stable`, deployment state, promotion state,
or batch serving.

MLflow has no compare-and-set alias API. Registry mutations therefore serialize
cooperating local CLIs with an atomic exclusive lock at a canonical path
derived from the explicit Registry-lock root and registered-model name.
Candidate registration retains its receipt output root as the compatible
default; operators using bootstrap and candidate registration for one model
must pass the same `--registry-lock-root`. The lock is acquired before the
first alias read and held through immutable evidence and postchecks; contention
fails closed.
This local protocol cannot serialize an unrelated external writer that bypasses
the CLI, so aliases are also re-read immediately before and after mutation.
Unsafe compensation emits recovery evidence for manual reconciliation.

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

It references an immutable `wind_forecast.deployment_state.v1` manifest. The
generation-one manifest persists its exact Registry version and expected
aliases, bundle/model/dataset/schema checksums, calibration/reference,
unchanged Phase 9 ledger and model snapshot, fit and activation cutoffs, null
predecessor, and checksum-pinned authorizing receipt. The loader rejects
absolute paths, traversal, symlinks, checksum mismatches, corrupt
content-addressed IDs, or disagreement with Registry aliases.

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

PR 4 implements that exception through one manual command. First run a dry
plan; it performs all local and MLflow reads, creates no lock or output, and
prints an approval payload template containing the observed checksum pins:

```powershell
.\venv\Scripts\python.exe .\scripts\bootstrap_v2_deployment.py `
  --model-bundle outputs\training\v2_reference_mlflow `
  --calibration-dir data\processed\v2\monitoring\reporting\calibrations\<ID> `
  --monitoring-store-root data\processed\v2\monitoring `
  --deployment-root data\processed\v2\deployment `
  --registry-lock-root data\processed\v2\registry-governance `
  --registered-model-name <explicit-v2-model-name> `
  --mlflow-tracking-uri http://127.0.0.1:5000 `
  --expect-no-deployment-pointer `
  --expect-no-v2-registry-state `
  --dry-run
```

An operator replaces only the three descriptive placeholders, stores the exact
`wind_forecast.bootstrap_approval.v1` JSON outside the deployment output, and
calculates its SHA-256. Remove `--dry-run` and add `--approval-path` plus
`--approval-sha256` to execute. Both absence assertions remain mandatory.

Before creating a version, the command verifies the exact accepted bundle file
set and checksums, no scaler, historical-hindcast and
`selected_not_promoted` contracts, raw and MLflow reload equivalence, local
MLflow receipt, `FINISHED` run, tags/parameters and ordered numeric signature,
calibration/reference, and the complete Phase 9 ledger chain. It rejects any
existing pointer, registered model, v2 version, or `candidate`, `champion`, or
`stable` alias.

Under the shared Registry lock, it repeats absence checks, creates and tags one
version, seals the content-addressed bootstrap receipt and deployment state,
sets `stable` and then `champion`, confirms `candidate` is absent, and publishes
`state/current.json` with atomic create-if-absent semantics. Failure before
pointer publication removes only aliases that still point to the created
version and preserves the orphan version plus immutable reconciliation
evidence. Failure after pointer publication never removes the pointer
automatically.

The repository checkout deliberately does not include
`data/processed/v2/monitoring/state/current.json`. Therefore the documented
real invocation currently fails before taking a lock or mutating MLflow. The
bootstrap must not be attempted until an operator supplies and verifies the
accepted Phase 9 ledger.

## Delivery Sequence

Implementation is split into separately reviewed PRs:

1. contracts and policy (implemented);
2. monthly evaluation (implemented);
3. temporal backtesting and v2 Registry (implemented);
4. bootstrap and deployment pointer (implemented);
5. model-era monitoring (implemented);
6. promotion, probation, and rollback;
7. stability and monthly scheduling.

Each PR must leave the repository safe and usable. The next PR starts only
after the previous PR has been reviewed and integrated.

## Model-Era Monitoring

Every Phase 9 prediction batch, report, local coordinator run, and Airflow run
now verifies the active deployment pointer, immutable state and receipt, the
explicit bundle and calibration, and all expected MLflow aliases. Any
disagreement fails closed, with verification repeated before derived pointers
advance.

The append-only ledger stores a content-addressed model-era record containing
deployment and Registry identities, fit/activation cutoffs, calibration and
reference IDs, and bundle/model/dataset/schema/calibration/ledger hashes.
Reporting windows and alert persistence never cross an era boundary.

The generation-one bootstrap adopts the earlier Phase 9 ledger without
rewriting it. Before the first v2 pointer update, the exact checksum-pinned
legacy state is sealed as immutable era evidence. Existing v1 reports remain
byte-identical and are labelled `bootstrap_adopted` only when calibration and
prediction lineage match; ambiguous history remains `legacy_unassociated`.
