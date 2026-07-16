# Phase 5B Dashboard Performance API Plan

## 1. Objective and scope

Phase 5B will extend the existing FastAPI application with a read-only
performance endpoint for a future single-page React dashboard. The dashboard
must be able to request a historical period and display actual wind production,
predicted wind production, the error for each observation, aggregate metrics,
the returned time interval, and minimum information about the result set used.

The first version includes:

- one backend endpoint over already-produced evaluation artifacts;
- optional inclusive start and end date filters;
- backend validation, filtering, error calculation, and metric calculation;
- a stable JSON response that does not expose the training feature contract or
  local artifact layout;
- explicit behavior for unavailable, empty, invalid, and out-of-range data.

The frontend must not know feature names or feature order, send feature-ready
records, read CSV or JSON artifact files directly, calculate backend-owned
metrics, or know local filesystem paths.

This phase does not include functional implementation, a database,
authentication, authorization, cloud deployment, Kubernetes, Airflow, PySpark,
streaming, WebSockets, real-time data, MLflow Registry model serving, complete
monitoring, alerts, or a multi-page frontend. It must not become a general
restructuring of the existing API.

## 2. Current repository evidence

### Existing API

`src/wind_forecast/api.py` currently exposes and must preserve:

- `GET /health`, which returns process health;
- `GET /model-info`, which reports model, scaler, and feature-reference
  readiness;
- `POST /predict`, which accepts feature-ready records and serves saved model
  predictions.

The API already uses a service dependency (`get_prediction_service`) and
FastAPI dependency overrides in `tests/test_api.py`. These patterns can be
reused for performance results. The current `PredictionService` is not reusable
for dashboard performance data because it prepares new inference inputs and
loads model-serving artifacts rather than reading historical evaluation
results.

`src/wind_forecast/config.py` currently configures only WeatherAPI access.
`src/wind_forecast/paths.py` resolves the project root and conventional data,
model, notebook, and script directories, but it does not define a training-run
or performance-results directory.

### Produced artifacts

`src/wind_forecast/training.py` defines the relevant filenames in
`OUTPUT_FILENAMES` and writes all three files under the `output_dir` supplied to
`run_baseline_training`:

| Artifact | Confirmed current format |
| --- | --- |
| `predictions.csv` | CSV with `Date`, `Actual_Wind_Production`, and `Predicted_Wind_Production`. Dates are written as `YYYY-MM-DD`; actual and predicted values are numeric; the file has a header, no index column, and LF line endings. |
| `metrics.json` | Non-empty JSON object containing numeric `R2`, `MAE`, `RMSE`, and `MAPE (%)`. JSON is UTF-8, indented, key-sorted, and LF-terminated. |
| `run_summary.json` | Non-empty JSON object containing run configuration, artifact references, row and feature counts, train/test coverage, metrics, and lineage metadata. JSON formatting matches `metrics.json`. |

The current `BaselineTrainingResult.summary()` emits these
`run_summary.json` fields:

- `model_type`, `seed`, `test_fraction`, and `n_estimators`;
- `input_path`, `output_dir`, and paths for the model, metrics, predictions,
  summary, plot, manifests, environment, and validation sample;
- `row_count`, `feature_count`, `train_row_count`, and `test_row_count`;
- `train_start_date`, `train_end_date`, `test_start_date`, and `test_end_date`;
- `metrics`, `input_sha256`, `feature_schema_sha256`, `feature_names`, and
  `dataset_version`.

The local `outputs/training/baseline_smoke` example confirms the CSV columns,
the four metric names, and the core run-summary fields. Its predictions contain
804 daily observations from `2021-10-19` through `2023-12-31`. Its summary was
produced before the current summary contract added fields such as
`input_sha256`, `feature_schema_sha256`, `feature_names`, and `dataset_version`.
The future reader should therefore require the stable core metadata and expose
`dataset_version` as nullable rather than silently rejecting this observed
older artifact solely for missing newer optional metadata.

`scripts/train_baseline.py` uses `outputs/training/baseline` as its default
output directory but accepts any `--output-dir`. Project-local output is
restricted to `outputs/`, and `outputs/training/` is intentionally ignored by
Git. Local MLflow runs also retain the artifacts under run-specific paths, but
the current FastAPI application does not consume MLflow runs or registry
aliases.

### Metrics and units

`calculate_regression_metrics` in `src/wind_forecast/training.py` calculates
metrics on the original target scale:

- `R2` using `r2_score`;
- `MAE` using `mean_absolute_error`;
- `RMSE` as the square root of mean squared error;
- `MAPE (%)` using `mean_absolute_percentage_error`, after replacing actual or
  predicted zeros with `1e-6`, then multiplying by 100.

Repository data-source documentation confirms MW for raw production
observations, but `predictions.csv`, `metrics.json`, and `run_summary.json` do
not record a unit for the daily aggregate target. The Phase 5B response must
therefore omit a unit instead of claiming that the artifact values are MW or
MWh. Establishing and versioning the aggregate target unit remains a separate
data-contract decision.

### Current API limitation

The API has no endpoint that reads evaluation artifacts, filters a historical
interval, returns actual and predicted values together, calculates per-row
errors, or calculates period metrics. `POST /predict` is deliberately not a
dashboard performance interface because it requires feature-ready inputs and
does not return observed production.

## 3. Proposed endpoint

### Request

- Path: `GET /performance`
- Optional query parameters:
  - `start_date`: ISO calendar date in `YYYY-MM-DD` format;
  - `end_date`: ISO calendar date in `YYYY-MM-DD` format.

Both bounds are inclusive. If neither is supplied, the service returns the
complete available evaluation period. If only `start_date` is supplied, the
service returns observations on or after that date. If only `end_date` is
supplied, it returns observations on or before that date. Observations are
always returned in ascending date order.

FastAPI/Pydantic must reject malformed dates and unknown query parameters with
HTTP 422. The domain service must reject `start_date > end_date` with HTTP 422.
A syntactically valid interval with no observations is a no-data result and
returns HTTP 404 as defined in the error model.

The first version has no `run_id`, artifact path, target type, feature payload,
sort control, pagination, or model-selection parameter. In particular, the
client cannot request a filesystem location or cause the backend to scan for a
"latest" run.

## 4. Response contract

### Field schema

| Field | JSON type | Meaning |
| --- | --- | --- |
| `interval` | object | Requested, available, and returned inclusive date bounds. |
| `interval.requested_start_date` | string or `null` | Echo of the validated `start_date`. |
| `interval.requested_end_date` | string or `null` | Echo of the validated `end_date`. |
| `interval.available_start_date` | string | Earliest valid observation in the selected artifact set. |
| `interval.available_end_date` | string | Latest valid observation in the selected artifact set. |
| `interval.returned_start_date` | string | First returned observation. |
| `interval.returned_end_date` | string | Last returned observation. |
| `observation_count` | integer | Number of returned observations; always greater than zero in a 200 response. |
| `metrics` | object | Metrics recalculated by the backend over exactly the returned observations. |
| `metrics.r2` | number or `null` | R2 for the returned observations; `null` when only one observation is returned. |
| `metrics.mae` | number | MAE for the returned observations. |
| `metrics.rmse` | number | RMSE for the returned observations. |
| `metrics.mape_percent` | number | MAPE percentage using the current training semantics. |
| `observations` | array | Date-ordered actual, predicted, and error values. |
| `observations[].date` | string | Observation date in `YYYY-MM-DD`. |
| `observations[].actual` | number | `Actual_Wind_Production` from the CSV. |
| `observations[].predicted` | number | `Predicted_Wind_Production` from the CSV. |
| `observations[].error` | number | `predicted - actual`; positive means overprediction. |
| `result` | object | Minimum non-path metadata about the evaluated result set. |
| `result.model_type` | string | Model type from the run summary. |
| `result.seed` | integer | Training seed from the run summary. |
| `result.test_fraction` | number | Test fraction from the run summary. |
| `result.dataset_version` | string or `null` | Dataset version when present in the run summary. |
| `result.evaluation_start_date` | string | Run-summary test start date, cross-checked with predictions. |
| `result.evaluation_end_date` | string | Run-summary test end date, cross-checked with predictions. |
| `result.artifact_metrics` | object | Full-run metrics mapped from `metrics.json` to API-safe field names. |

All response numbers must be finite JSON numbers. Filesystem paths,
`feature_names`, feature-schema hashes, input paths, model paths, and raw
artifact field names are not part of the public response.

### Metric convention

`metrics` is calculated over the filtered observations returned to the client.
`result.artifact_metrics` represents the complete evaluation run recorded in
`metrics.json`. For an unfiltered request, the recalculated values should match
the persisted full-run metrics within a documented floating-point tolerance;
otherwise the artifact set is inconsistent and unavailable.

For a one-observation interval, R2 is undefined and is returned as `null` while
MAE, RMSE, and MAPE remain numeric. MAPE retains the existing zero-replacement
behavior so that the dashboard and training reports do not silently use
different formulas.

### Example response

```json
{
  "interval": {
    "requested_start_date": "2021-10-19",
    "requested_end_date": "2021-10-20",
    "available_start_date": "2021-10-19",
    "available_end_date": "2023-12-31",
    "returned_start_date": "2021-10-19",
    "returned_end_date": "2021-10-20"
  },
  "observation_count": 2,
  "metrics": {
    "r2": -4.302526989457233,
    "mae": 37145.280000000006,
    "rmse": 37480.133772146546,
    "mape_percent": 26.576548276665918
  },
  "observations": [
    {
      "date": "2021-10-19",
      "actual": 123195.1,
      "predicted": 91048.68,
      "error": -32146.420000000013
    },
    {
      "date": "2021-10-20",
      "actual": 155748.0,
      "predicted": 113603.86000000002,
      "error": -42144.139999999985
    }
  ],
  "result": {
    "model_type": "extra_trees",
    "seed": 42,
    "test_fraction": 0.2,
    "dataset_version": null,
    "evaluation_start_date": "2021-10-19",
    "evaluation_end_date": "2023-12-31",
    "artifact_metrics": {
      "r2": 0.8597377487381633,
      "mae": 26802.371380597015,
      "rmse": 35180.82780767062,
      "mape_percent": 28.135658182443134
    }
  }
}
```

No `unit` field is included until the artifact contract records the aggregate
target unit unambiguously.

## 5. Artifact resolution

The application should receive one explicit artifact directory through a new
`WIND_FORECAST_PERFORMANCE_ARTIFACT_DIR` setting. A relative value is resolved
against `project_root()`; an absolute value may be used internally for a
read-only container mount or environment-specific location. Neither form is
returned to clients or included in public error messages.

The backend constructs all required paths from that one directory and the
confirmed filenames:

- `<artifact_dir>/predictions.csv`;
- `<artifact_dir>/metrics.json`;
- `<artifact_dir>/run_summary.json`.

It must not use path fields inside `run_summary.json` to locate other files,
accept a path from an HTTP request, search arbitrary directories, infer the
most recently modified output, or mutate any artifact. If the setting is
absent, the endpoint returns HTTP 503 rather than silently selecting
`outputs/training/baseline` or another local run.

The service constructor should accept a `PerformanceArtifactPaths` value or a
base `Path`. Unit tests pass a `tmp_path` directory directly, and endpoint tests
override the FastAPI service dependency. Tests therefore do not write to or
depend on repository-local outputs.

Training supports arbitrary output directories and MLflow can contain multiple
runs, but the current API has no approved active-run resolver. Phase 5B v1
serves only the explicitly configured directory. A later multi-run design must
define a stable logical identifier, authorization boundary, and resolution
policy before adding a `run_id` or catalog; this plan does not invent that
policy.

## 6. Error model

Errors use the existing FastAPI shape `{"detail": "..."}`. Artifact and
unexpected-error messages must be actionable without including absolute paths,
credentials, feature names, or internal exception details.

| Condition | HTTP status | Required behavior |
| --- | --- | --- |
| Malformed date or unknown query parameter | 422 Unprocessable Entity | Let typed request validation identify the invalid field. |
| `start_date` after `end_date` | 422 Unprocessable Entity | Return a stable interval-validation message. |
| Valid interval with no observations | 404 Not Found | Distinguish no data in the interval from invalid or missing artifacts. |
| Performance artifact directory not configured | 503 Service Unavailable | Report that performance artifacts are not configured. |
| Required file does not exist | 503 Service Unavailable | Name the missing artifact type, not its local path. |
| Required CSV or JSON file is empty | 503 Service Unavailable | Treat the artifact set as unavailable. |
| Malformed CSV/JSON or invalid schema | 503 Service Unavailable | Reject missing/extra prediction columns, missing metric keys, or missing required summary metadata. |
| Invalid artifact values | 503 Service Unavailable | Reject unparseable or duplicate dates, non-numeric or non-finite values, invalid metadata types, and inconsistent coverage/metrics. |
| Unexpected internal exception | 500 Internal Server Error | Return a generic detail and retain the underlying exception only for server-side diagnosis. |

`predictions.csv` validation must require the three exact columns, at least one
row, parseable unique dates in strict chronological order, and finite actual
and predicted numbers. It must not silently sort, coerce invalid values, drop
rows, fill nulls, or impose new physical bounds unsupported by the current
training contract.

`metrics.json` validation must require exactly the four known metric names with
finite numeric values. `run_summary.json` must require at least `model_type`,
`seed`, `test_fraction`, `test_start_date`, and `test_end_date` with valid
types. `dataset_version` is optional for compatibility with the observed older
local summary. When present, summary metrics and coverage are cross-checked
against the metrics and predictions artifacts.

## 7. Proposed internal design

Create a domain service separate from FastAPI. A minimal design is:

- `PerformanceArtifactPaths`: immutable value containing the three resolved
  paths, constructed from one configured directory;
- `PerformanceService`: reads and validates artifacts, filters inclusive date
  bounds, calculates errors and filtered metrics, and returns domain result
  objects;
- domain exceptions for invalid intervals, no observations, and artifacts that
  are unconfigured, missing, empty, malformed, or inconsistent;
- Pydantic response schemas such as `PerformanceInterval`,
  `PerformanceMetrics`, `PerformanceObservation`, `PerformanceResultInfo`, and
  `PerformanceResponse`;
- `get_performance_service`: cached FastAPI dependency that loads the explicit
  configuration and constructs the service, while remaining overrideable in
  tests.

The performance service should reuse the current
`calculate_regression_metrics` semantic contract rather than duplicating a
different formula. R2 for one observation is handled explicitly before calling
the current helper. Reuse must not change `training.py` in the first
implementation.

`api.py` remains responsible only for Pydantic HTTP schemas, dependency wiring,
the route, and mapping known domain exceptions to status codes. Artifact path
resolution, file I/O, pandas parsing, schema validation, filtering, error
calculation, metric calculation, artifact consistency checks, and path
redaction remain outside `api.py`.

## 8. Files for the future implementation

### Create

- `src/wind_forecast/performance.py`: artifact paths, domain objects, service,
  validation, filtering, metrics, and domain exceptions.
- `tests/test_performance.py`: isolated service tests over synthetic temporary
  artifacts.

### Modify

- `src/wind_forecast/api.py`: additive Pydantic schemas, dependency, endpoint,
  and exception mapping only.
- `src/wind_forecast/config.py`: additive performance-artifact directory
  configuration without import-time environment loading or I/O.
- `tests/test_api.py`: endpoint tests and compatibility assertions using
  dependency overrides.
- `docs/PHASE_5.md`: document the endpoint only when functional implementation
  is approved and completed.

### Keep unchanged

- `src/wind_forecast/training.py` and the existing training output contract;
- `src/wind_forecast/paths.py`, unless a later approved configuration design
  identifies a provider-neutral reusable helper;
- inference, feature engineering, data-source, validation, registry, and MLflow
  modules;
- notebooks, datasets, models, scalers, generated artifacts, dependencies, and
  CI workflows;
- behavior and contracts of `GET /health`, `GET /model-info`, and
  `POST /predict`.

## 9. Test plan

### Performance service unit tests

Use `tmp_path` to create only the three required synthetic artifacts. Cover:

- a valid full-period response and exact field mapping;
- inclusive start-only, end-only, and two-sided filtering;
- chronological response order and `predicted - actual` error sign;
- selected-period metrics, full-run artifact metrics, and floating-point
  consistency checks;
- one-observation metrics with `r2: null`;
- zero actual/predicted MAPE behavior matching current training semantics;
- inverted intervals and intervals without observations;
- unconfigured and missing artifact paths;
- empty CSV and empty JSON files;
- malformed CSV/JSON and missing or extra prediction columns;
- missing metric keys and invalid run-summary fields;
- invalid, duplicate, or unordered dates;
- null, non-numeric, NaN, and infinite values;
- inconsistent prediction coverage, summary coverage, or full-run metrics;
- assurance that failures do not modify source artifacts or expose paths.

### Endpoint tests

Use `create_app()`, dependency overrides, and a fake or temporary
`PerformanceService`. Cover:

- HTTP 200 schema and query forwarding;
- missing, one-sided, and two-sided optional date parameters;
- HTTP 422 for malformed dates, unknown parameters, and inverted ranges;
- HTTP 404 for a valid empty interval;
- HTTP 503 for every artifact-readiness class;
- HTTP 500 with a generic detail for an unexpected exception;
- response exclusion of filesystem paths, feature names, and feature-ready
  input details.

Retain and run all current API tests to prove that `/health`, `/model-info`, and
`/predict` are unchanged. The new tests must not load TensorFlow models, read
repository datasets, use local MLflow state, require WeatherAPI credentials, or
perform network calls.

## 10. Incremental implementation sequence

1. Implement `PerformanceService`, configuration, artifact validation, metric
   behavior, and unit tests without importing FastAPI into the domain module.
2. Add the Pydantic schemas, dependency, `GET /performance`, error mapping, and
   endpoint tests while running the existing API regression suite.
3. Add CORS only after the React development and deployed origins are known.
   Configure an explicit allowlist; do not use an unrestricted wildcard.
4. Build one React page that consumes only `GET /performance` and presents the
   comparison, errors, metrics, interval, and result metadata.

None of these implementation steps is performed by this planning task.

## 11. Risks and unresolved decisions

- **Run selection:** the configured directory is the only v1 selection
  mechanism. The project has no approved active-run, latest-run, or
  user-selectable run policy.
- **Clean-clone availability:** `outputs/training/` is ignored, so a clean clone
  normally has no performance artifacts. The expected endpoint behavior is
  HTTP 503 until an operator supplies a valid artifact directory.
- **Artifact versioning:** the three files have no shared schema-version field.
  The observed older summary lacks newer lineage fields, so strict versioned
  evolution cannot be enforced until a future artifact contract adds it.
- **Unit meaning:** the evaluated aggregate values do not carry a confirmed
  unit. The API must omit a unit until provenance and aggregation semantics make
  it explicit.
- **Persisted versus selected metrics:** `metrics.json` covers the full test set,
  while dashboard metrics cover the selected interval. The two sets must remain
  visibly distinct in the response.
- **MAPE around zero:** the existing `1e-6` substitution can produce very large
  percentages. Phase 5B preserves this behavior for compatibility rather than
  silently introducing a new metric definition.
- **Single-observation R2:** R2 is undefined for one observation and is nullable
  only in that case.
- **Response size:** v1 intentionally has no pagination because current output
  is daily and small. A materially larger future artifact requires an explicit
  pagination or aggregation decision rather than an undocumented response
  truncation.
- **Registry integration:** MLflow run IDs and aliases exist, but the API does
  not consume them. Connecting performance serving to MLflow is a separate
  approved phase, not an implicit fallback.

## 12. Acceptance criteria

The future implementation is complete only when:

- `GET /performance` accepts the documented optional inclusive bounds and
  returns the documented typed JSON contract;
- the response contains actual, predicted, and `predicted - actual` error values
  for every returned observation, selected-period metrics, available/returned
  intervals, and minimum result metadata;
- the backend alone reads artifacts, filters observations, and calculates
  errors and metrics;
- the frontend contract contains no feature names, feature order, feature-ready
  records, artifact paths, model paths, or unconfirmed units;
- artifact resolution uses one explicitly configured directory and never
  performs latest-run discovery or accepts client paths;
- invalid queries, inverted ranges, no-data intervals, missing/empty/invalid
  artifacts, invalid values, and unexpected failures map to the documented HTTP
  statuses without exposing internal paths;
- service and endpoint tests use `tmp_path`, synthetic data, and dependency
  overrides, with no dependency on local models, datasets, MLflow state,
  credentials, or network access;
- compatibility tests prove unchanged behavior for `GET /health`,
  `GET /model-info`, and `POST /predict`;
- no training output, dataset, model, scaler, notebook, dependency, or CI
  contract is modified as part of the endpoint;
- CORS and the single-page frontend remain separate incremental changes after
  the backend endpoint is validated.
