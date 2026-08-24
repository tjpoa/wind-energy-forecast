# ML Engineering Roadmap

This roadmap defines the phased evolution of the wind-energy forecasting project into a reproducible Data/ML Engineering and MLOps project. It is a planning document only; no phase should be implemented until explicitly requested.

## Core Requirement

The existing modelling workflow must remain usable throughout the roadmap. Current notebooks, scripts, trained models, datasets, column mappings and generated outputs should be preserved unless a phase explicitly introduces a tested, backward-compatible replacement.

## Roadmap Overview

| Phase | Focus | Expected outcome |
| --- | --- | --- |
| 0 | Repository audit, security and baseline | Current behaviour, assets and risks are documented. |
| 1 | Modular project structure and configuration | Reusable code and configuration are separated from notebooks. |
| 2 | Data validation and sanity checks | Raw, processed and prediction inputs have explicit validation. |
| 3 | Automated testing and code quality | Tests, linting and formatting support safer changes. |
| 4 | MLflow experiment tracking and model registry | Experiments, metrics, artifacts and model versions are tracked. |
| 5 | Prediction API with FastAPI | Saved models can serve predictions through an API. |
| 6 | Docker containerization | The project can run in a reproducible container. |
| 7 | GitHub Actions continuous integration | Pull requests run automated checks. |
| 8 | Idempotency, safe reruns and observability | Pipelines can be rerun safely with structured logs. |
| 9 | Data drift and model-performance monitoring | Input drift and prediction quality are monitored over time. |
| 10 | Batch orchestration with Apache Airflow | Batch workflows are scheduled and observable. |
| 11 | PySpark data-processing implementation | Scalable feature processing mirrors the pandas pipeline. |
| 12 | Azure and Databricks deployment design | A cloud deployment architecture is specified. |
| Operational extension | Operational Read-only Copilot | Planned outcome: grounded operational queries would be exposed through reviewable read-only layers. |

## Phase 0 — Repository audit, security and baseline

### Objective
Document the current repository state and establish a safe baseline before engineering changes begin.

### Main tasks
- Inventory datasets, notebooks, scripts, models, generated files and secrets-related files.
- Record current commands for data processing, prediction and notebook execution.
- Identify hardcoded paths, credentials risks and fragile assumptions.
- Capture baseline outputs and model metrics where reproducible.

### Technical requirements
- Do not change application code or modelling logic.
- Use repository-derived names, paths, columns and model artifacts only.
- Keep `.env` local and document required environment variables.

### Tests and validation
- Run existing smoke commands where credentials and data are available.
- Verify Git ignores local secrets.
- Compare generated output names and schemas with the existing workflow.

### Acceptance criteria
- A baseline audit exists and no existing workflow is broken.
- Security risks are documented without exposing secrets.

### Expected deliverables
- Repository audit notes.
- Baseline command list.
- Security and reproducibility issue list.

### Dependencies on previous phases
- None.

### Skills demonstrated
- Repository analysis, risk assessment, reproducibility baselining and secret hygiene.

## Phase 1 — Modular project structure and configuration

### Objective
Move reusable logic into a maintainable Python package while preserving script and notebook behaviour.

### Main tasks
- Define a package layout for data loading, feature engineering, prediction and configuration.
- Extract reusable functions from scripts and notebooks gradually.
- Introduce centralized configuration for paths, API settings and runtime options.
- Keep existing command entry points working.

### Technical requirements
- Use environment variables for secrets and environment-specific values.
- Resolve paths relative to the project root.
- Preserve current column mappings and saved model compatibility.

### Tests and validation
- Import package modules successfully.
- Run existing scripts after refactoring.
- Confirm generated schemas match the baseline.

### Acceptance criteria
- Reusable logic lives outside notebooks.
- Existing notebooks and scripts still run or have documented compatible entry points.

### Expected deliverables
- Modular source layout.
- Configuration module.
- Updated usage documentation.

### Dependencies on previous phases
- Phase 0 baseline.

### Skills demonstrated
- Python packaging, configuration management and backward-compatible refactoring.

## Phase 2 — Data validation and sanity checks

### Objective
Add explicit validation for raw data, processed features and prediction inputs.

### Main tasks
- Define schemas for raw E-REDES data, WeatherAPI responses, processed datasets and model inputs.
- Validate required columns, data types, date ranges, nulls and physical bounds.
- Add clear validation errors and warnings.
- Document expected data contracts.

### Technical requirements
- Do not invent new data fields.
- Validation must use the existing English schema and compatibility mappings.
- Failed validation should stop unsafe downstream processing.

### Tests and validation
- Test valid and invalid schemas.
- Test missing columns, bad dates, impossible values and null handling.
- Run validation against current sample datasets.

### Acceptance criteria
- Invalid inputs fail early with actionable messages.
- Current known-good datasets pass validation.

### Expected deliverables
- Data schema definitions.
- Validation utilities.
- Validation tests and documentation.

### Dependencies on previous phases
- Phases 0 and 1.

### Skills demonstrated
- Data contracts, defensive data engineering and quality checks.

## Phase 3 — Automated testing and code quality

### Objective
Create an automated quality foundation for future changes.

### Main tasks
- Add unit tests for schema mapping, feature engineering and prediction preparation.
- Add smoke tests for CLI/script entry points.
- Configure linting, formatting and import checks.
- Establish testing conventions for new modules.

### Technical requirements
- Keep tests deterministic and independent of real API calls.
- Mock external API access.
- Avoid committing generated secrets or environment-specific artifacts.

### Tests and validation
- Run the full test suite locally.
- Run formatting and linting commands.
- Confirm tests cover critical transformation logic.

### Acceptance criteria
- A contributor can run one command to validate code quality.
- Core reusable logic has automated tests.

### Expected deliverables
- `tests/` suite.
- Code quality configuration.
- Test-running documentation.

### Dependencies on previous phases
- Phases 0 through 2.

### Skills demonstrated
- Pytest, mocking, quality automation and regression prevention.

## Phase 4 — MLflow experiment tracking and model registry

### Objective
Track experiments, metrics, parameters, artifacts and model versions reproducibly.

### Main tasks
- Add MLflow tracking to training and evaluation workflows.
- Log model parameters, metrics, datasets references, plots and artifacts.
- Define model naming and versioning conventions.
- Register candidate and production-ready models.

### Technical requirements
- Preserve current saved model files and scaler compatibility.
- Use local tracking first before remote tracking.
- Avoid logging secrets or raw credentials.

### Tests and validation
- Run a small training/evaluation smoke run with MLflow logging.
- Verify metrics and artifacts appear in the tracking store.
- Reload a registered model and compare prediction behaviour.

### Acceptance criteria
- Experiments are traceable from data inputs to model artifacts.
- Existing modelling outputs remain reproducible.

### Expected deliverables
- MLflow integration.
- Experiment and registry documentation.
- Model promotion guidelines.

### Dependencies on previous phases
- Phases 0 through 3.

### Skills demonstrated
- Experiment tracking, artifact management and model governance.

## Phase 5 — Prediction API with FastAPI

### Objective
Expose saved model predictions through a small, tested HTTP API.

### Main tasks
- Build FastAPI endpoints for health checks, model metadata and prediction.
- Load saved models and scalers through the project configuration.
- Validate request payloads with typed schemas.
- Return predictions and basic metadata in a stable response format.

### Technical requirements
- API must not replace existing batch scripts.
- Model paths and feature order must come from repository configuration.
- Secrets and API keys must not be included in responses or logs.

### Tests and validation
- Test health and prediction endpoints.
- Test invalid payloads and missing model artifacts.
- Compare API predictions with batch prediction output for the same input.

### Acceptance criteria
- API serves predictions using the existing trained models.
- Batch workflow remains available and unchanged in intent.

### Expected deliverables
- FastAPI application.
- Request and response schemas.
- API tests and run instructions.

### Dependencies on previous phases
- Phases 0 through 4.

### Skills demonstrated
- API design, model serving, typed validation and service testing.

## Phase 6 — Docker containerization

### Objective
Package the project in a reproducible container for local and deployment use.

### Main tasks
- Create a Dockerfile for the API and supporting commands.
- Define runtime environment variables and mounted data/model locations.
- Add a minimal compose setup if needed for local execution.
- Document build and run commands.

### Technical requirements
- Do not bake secrets into images.
- Keep image layers reasonably small.
- Support current model and data artifact paths through configuration.

### Tests and validation
- Build the image locally.
- Run API health checks in the container.
- Run a smoke prediction with mounted artifacts.

### Acceptance criteria
- The container can run the supported project entry point.
- Local non-container workflows continue to work.

### Expected deliverables
- Dockerfile.
- Optional compose file.
- Container usage documentation.

### Dependencies on previous phases
- Phases 0 through 5.

### Skills demonstrated
- Containerization, runtime configuration and reproducible environments.

## Phase 7 — GitHub Actions continuous integration

### Objective
Run automated checks on repository changes.

### Main tasks
- Add CI workflows for tests, linting and formatting checks.
- Cache Python dependencies where appropriate.
- Separate checks that require secrets from public pull request checks.
- Publish test results or summaries.

### Technical requirements
- CI must not require real API keys for standard checks.
- Secrets must use GitHub Actions secret storage.
- Generated datasets and model artifacts should not be produced accidentally in CI.

### Tests and validation
- Trigger CI on a branch or pull request.
- Confirm failing tests block the workflow.
- Confirm no secrets are printed.

### Acceptance criteria
- Pull requests receive automated quality feedback.
- CI reflects the local validation commands.

### Expected deliverables
- GitHub Actions workflow files.
- CI documentation.
- Badge or status reference if desired.

### Dependencies on previous phases
- Phases 0 through 6.

### Skills demonstrated
- Continuous integration, secure automation and developer workflow design.

## Phase 8 — Idempotency, safe reruns and observability

### Objective
Make processing and prediction jobs safe to rerun and easier to diagnose.

### Main tasks
- Define run identifiers and output overwrite policies.
- Add structured logging for pipeline stages.
- Make reruns deterministic where inputs are fixed.
- Add clear failure modes and recovery guidance.

### Technical requirements
- Do not silently overwrite important data, models or reports.
- Preserve existing output compatibility or provide explicit migration notes.
- Logs must not expose secrets.

### Tests and validation
- Run the same job twice and verify expected output behaviour.
- Test failure paths for missing data, missing models and invalid config.
- Validate log structure and run metadata.

### Acceptance criteria
- Reruns are predictable, safe and observable.
- Operators can identify what ran, when, with which inputs.

### Expected deliverables
- Run metadata conventions.
- Logging updates.
- Safe rerun documentation.

### Dependencies on previous phases
- Phases 0 through 7.

### Skills demonstrated
- Operational reliability, observability and failure-mode design.

## Phase 9 — Data drift and model-performance monitoring

### Objective
Monitor whether live inputs and predictions remain aligned with the training baseline.

### Main tasks
- Define reference feature distributions from training data.
- Compute drift statistics for new API-featured datasets.
- Track prediction error when actual production values are available.
- Produce monitoring reports and alerts.

### Technical requirements
- Use validated feature names and schemas.
- Store monitoring outputs without overwriting historical reports.
- Keep thresholds configurable and documented.

### Tests and validation
- Test drift calculations on controlled datasets.
- Test alert thresholds.
- Compare monitoring metrics against known examples.

### Acceptance criteria
- Drift and performance reports can be generated for new batches.
- Monitoring does not change prediction behaviour.

### Expected deliverables
- Drift and performance monitoring utilities.
- Reference dataset metadata.
- Monitoring report documentation.

### Dependencies on previous phases
- Phases 0 through 8.

### Skills demonstrated
- Model monitoring, statistical drift detection and production ML diagnostics.

## Phase 10 — Batch orchestration with Apache Airflow

Status: implemented locally. The separate Airflow 3.3.0, PostgreSQL and
LocalExecutor stack passed structural checks and the required three-date
real-CLI backfill over generated synthetic evidence. It is not active in the
governed `local` environment, whose owner is Windows Task Scheduler.

### Objective
Schedule and observe the batch data-processing and prediction workflow.

### Main tasks
- Define DAGs for data ingestion, feature generation, prediction and monitoring.
- Add task-level retries, dependencies and failure notifications.
- Parameterize execution dates and runtime configuration.
- Document local Airflow setup and DAG operation.

### Technical requirements
- Tasks must call tested project modules or stable CLI entry points.
- DAGs must be idempotent and safe for backfills.
- Secrets must use Airflow connections or environment variables.

### Tests and validation
- Validate DAG imports.
- Run local DAG smoke tests.
- Test backfill behaviour on a limited date range.

### Acceptance criteria
- The batch workflow can be scheduled without manual notebook execution.
- Existing scripts remain usable outside Airflow.

### Expected deliverables
- Airflow DAG files.
- Local orchestration documentation.
- Operational runbook.

### Dependencies on previous phases
- Phases 0 through 9.

### Skills demonstrated
- Workflow orchestration, scheduling, retries and batch operations.

## Phase 11 — PySpark data-processing implementation

### Objective
Provide a scalable PySpark implementation of the core data-processing and feature-engineering logic.

### Main tasks
- Reimplement selected pandas transformations in PySpark.
- Define Spark schemas for raw and processed data.
- Compare PySpark outputs with pandas outputs.
- Document when to use pandas versus Spark.

### Technical requirements
- Preserve feature definitions, column names and model input ordering.
- Maintain parity with the validated pandas pipeline.
- Avoid changing model training or inference semantics.

### Tests and validation
- Run parity tests between pandas and PySpark outputs.
- Test Spark schema enforcement.
- Validate representative edge cases for dates, nulls and rolling features.

### Acceptance criteria
- PySpark output matches pandas output within documented tolerances.
- Existing pandas workflow remains supported.

### Expected deliverables
- PySpark processing modules or jobs.
- Parity tests.
- Spark usage documentation.

### Dependencies on previous phases
- Phases 0 through 10.

### Skills demonstrated
- Distributed data processing, schema design and pipeline parity testing.

## Phase 12 — Azure and Databricks deployment design

### Objective
Design a cloud deployment architecture for the mature project without forcing immediate migration.

### Main tasks
- Define target Azure architecture for storage, compute, secrets and monitoring.
- Map local components to Databricks jobs, MLflow, container services and data storage.
- Specify deployment environments, access controls and cost considerations.
- Document migration steps and operational responsibilities.

### Technical requirements
- Use managed secret storage such as Key Vault or platform-native equivalents.
- Preserve local development and existing modelling workflows.
- Avoid cloud-specific hardcoding in reusable project logic.

### Tests and validation
- Review architecture for security, reproducibility and operational feasibility.
- Validate that local configuration can map to cloud configuration.
- Identify proof-of-concept checks before implementation.

### Acceptance criteria
- A deployment design exists with clear trade-offs and migration steps.
- No cloud deployment is performed until explicitly requested.

### Expected deliverables
- Azure and Databricks architecture design.
- Environment and secret-management plan.
- Deployment backlog and risk register.

### Dependencies on previous phases
- Phases 0 through 11.

### Skills demonstrated
- Cloud architecture, Databricks platform design, MLOps deployment planning and cost-aware engineering.

## Operational Extension — Controlled Retraining

Status: implemented through the recommendation-only monthly scheduling
increment. Training and lifecycle transitions remain manually approval-gated.

This extension is also referred to as "Stage 7 — Controlled Retraining" in
its approved delivery plan. It does not replace roadmap Phase 7, which remains
the completed GitHub Actions continuous-integration phase.

### Objective

Use Phase 9 monitoring evidence to recommend reproducible v2 hindcast
retraining decisions without automatically training, promoting, stabilizing,
or rolling back a model.

### Contract

- Evaluate eligibility monthly after the D+7 source-lateness boundary.
- Require 90 new eligible observations and an already-persistent Phase 9 drift
  or performance alert.
- Backtest on complete folds of 30 eligible observations rather than calendar
  windows.
- Compare the candidate with the active v2 model and one-day persistence.
- Keep candidate, champion, probationary, and stable semantics explicit.
- Require manual promotion and a second manual stability approval.
- Preserve every model era and support checksum-pinned rollback receipts.

The full accepted decision is recorded in
[`CONTROLLED_RETRAINING.md`](CONTROLLED_RETRAINING.md). The legacy v1 API and
automatic model replacement remain out of scope.

## Operational Extension — Operational Read-only Copilot

Status: product contract, ADR, typed operational query layer, local-only
read-only API, and versioned offline evaluation dataset/harness implemented.
The separate PostgreSQL operational projection now includes its dedicated
foundation and migrations, manual projector/verifier, deterministic benchmark
with a superseding `GO`, and optional default-disabled `disabled|required`
query integration. Local sanitized observability and the provider-neutral
single-tool Copilot core/offline runner are implemented as separately
reviewed increments. The offline injected-candidate evaluation boundary and
additive receipt contract are also implemented with no egress, English-only
scope, 1-second selector/5-second total defaults, and digest-only retention.
The exact OpenAI `gpt-5.4-mini-2026-03-17` candidate adapter is now implemented
for the Responses API, sealed synthetic egress, `store=false`, one call per
case, zero retries, and a five-second remote selector/deadline. No live
candidate evaluation or receipt exists because no API key was available, and
the Copilot remains disabled. Later product increments have not started. See
[`OPERATIONAL_COPILOT_CANDIDATE_EVALUATION.md`](OPERATIONAL_COPILOT_CANDIDATE_EVALUATION.md).

This future extension must consume verified operational evidence through
read-only contracts. It must not mutate Phase 8/9 stores, MLflow, deployment
state, scheduler ownership, aliases, models, V1 artifacts, or API-serving
semantics. Technology choices listed later in the sequence are design targets,
not current repository capabilities.

### Objective

Allow operators to ask bounded questions about verified deployment,
monitoring, quality, drift, performance, and model metadata while preserving
the existing checksum-pinned loaders as the source of truth.

### Delivery sequence

1. Define the product contract and ADR: permitted questions, evidence and
   citation rules, read-only boundary, authentication expectations, failure
   semantics, and explicit non-goals. Accepted as the documentation-only
   [`operational_read_only_copilot_v1`](OPERATIONAL_COPILOT.md) contract.
2. Build a typed operational query layer over existing verified loaders, with
   deterministic inputs/outputs, explicit errors, timeouts, and synthetic-store
   tests. Implemented through `wind_forecast.operational_query_models`,
   `wind_forecast.operational_query`, verified reporting-attempt loaders, and
   dedicated zero-write acceptance tests.
3. Expose only that query layer through a separately reviewed read-only API
   endpoint. Implemented as local-only
   `POST /api/v1/operational-query`, with a 64 KiB body limit, server-generated
   request metadata, a maximum five-second cooperative deadline,
   socket-derived loopback authorization, and `OperationalAnswer` status
   mapping.
4. Create a versioned evaluation dataset and harness for correctness,
   groundedness, tool selection, refusal behaviour, and evidence attribution.
   Implemented offline with 88 synthetic English cases, deterministic scoring,
   sanitized stdout-only reports, and no Copilot candidate evaluated.
5. Deliver the PostgreSQL relational projection through the separately accepted
   [`operational_postgres_projection_v1`](OPERATIONAL_POSTGRES_PROJECTION.md)
   contract. Implemented through four separately reviewed gates: dedicated
   foundation and migrations, manual projector/verifier, deterministic
   benchmark with a superseding mandatory `GO`, and optional query-layer
   integration after that `GO`. Immutable files and verified loaders remain
   authoritative, PostgreSQL is never cited, and consumption is disabled by
   default.
6. Add local observability for requests and tool calls, including structured
   logs, correlation IDs, metrics, tracing, health checks, and secret/data
   sanitization. Implemented with lazy JSONL events, process-local counters,
   loopback-only health/metrics endpoints, writer-failure degradation, and no
   changes to operational query answers or authoritative stores.
7. Introduce a Copilot restricted to the single deterministic read-only
   `operational_query` tool. Implemented as the provider-neutral
   `OperationalCopilot` library and in-memory offline runner with strict
   `OperationalHttpRequest` validation, one selector/tool call, cooperative
   selector and total deadlines, zero retries, passthrough `OperationalAnswer`,
   and refusal while observability is degraded. No provider or candidate is
   evaluated or enabled by this item.
7a. Prepare candidate evaluation through an offline injected selector boundary
   and additive receipt. Implemented without provider SDKs, egress, or
   changes to the accepted harness; provider/model metadata remains required
   for a future candidate and no candidate is accepted by this increment.
7b. Implement the selected OpenAI candidate adapter over the same boundary.
   Implemented without a new dependency for the exact
   `gpt-5.4-mini-2026-03-17` snapshot, fixed Responses endpoint,
   environment-only secret, sealed synthetic egress, `store=false`, bounded
   responses, zero retries, fail-closed infrastructure handling, and a
   separate additive remote receipt. Live evaluation remains pending.
7c. Implement the alternative fixed Gemini `gemini-2.5-flash-lite` adapter
   over the same sealed 88-case boundary. Implemented with a separate REST
   transport and receipt; live evaluation and activation remain pending.
8. Add an MCP adapter over the same service contracts without creating a
   second business-logic or authorization path.
9. Add document-only RAG, with a small versioned corpus and optional
   `pgvector`, only when deterministic operational tools cannot answer the
   documentary question.
10. Design staging and cloud deployment separately, including identity,
    secrets, managed storage, network controls, CI/CD, rollback, and cost
    boundaries.

Each item is an independent, reviewable increment. No item authorizes automatic
training, lifecycle transitions, external notifications, live forecasting, or
Airflow activation in an environment owned by Windows Task Scheduler.
Items 1 through 7, the candidate-evaluation boundary, and the selected OpenAI
candidate adapter are implemented at their approved boundaries. Projection
consumption remains disabled by default. Actual candidate evaluation and
receipt, MCP, RAG, and cloud design require separate reviewed increments.
