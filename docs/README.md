# Documentation Index

This folder keeps project documentation organized by purpose and roadmap phase.

## Top-Level Documents

| Document | Purpose |
| --- | --- |
| [ML Engineering Roadmap](ML_ENGINEERING_ROADMAP.md) | Source-of-truth roadmap for the phased Data/ML Engineering evolution. |
| [Project Status](PROJECT_STATUS.md) | Current portfolio-facing status, capabilities, limitations, and next steps. |

## Portfolio Guide

| Document | Purpose |
| --- | --- |
| [Demo Guide](DEMO.md) | Full-stack local demo for preparing evaluation artifacts, running React with FastAPI, using Docker Compose, and validating the project. |
| [Azure Deployment](AZURE_DEPLOYMENT.md) | Protected Azure Container Apps deployment path for the synthetic portfolio demo, including rollback and cost boundaries. |
| [Reproducibility Guide](REPRODUCIBILITY.md) | MLflow Registry workflow and immutable artifact-bundle recovery. |
| [Controlled Retraining](CONTROLLED_RETRAINING.md) | Approved post-monitoring lifecycle contract, policy, cutoffs, deployment semantics, and staged delivery plan. |
| [Operational Read-only Copilot](OPERATIONAL_COPILOT.md) | Accepted product contract, typed query layer, and local-only HTTP API for bounded, grounded, read-only operational questions. |
| [PostgreSQL Operational Projection](OPERATIONAL_POSTGRES_PROJECTION.md) | Accepted derived-projection contract with dedicated schema/migrations, manual projector/verifier, benchmark `GO`, and optional default-disabled query integration. |

## Phase Documents

| Phase | Document | Contents |
| --- | --- | --- |
| 0 | [PHASE_0.md](PHASE_0.md) | Initial repository audit, baseline, security, and reproducibility notes. |
| 1 | [PHASE_1.md](PHASE_1.md) | Modularization closure summary and compatibility notes. |
| 2 | [PHASE_2.md](PHASE_2.md) | Data-source assessment, v2 data contracts, REN/ERA5-Land backfill records, and acceptance checks. |
| 4 | [PHASE_4.md](PHASE_4.md) | Baseline training CLI, model card, data card, and local experiment-tracking context. |
| 5 | [PHASE_5.md](PHASE_5.md) | FastAPI prediction, historical evidence, and local operational-query API documentation. |
| 8 | [PHASE_8.md](PHASE_8.md) | Transactional, idempotent v2 dataset updates, revision policy, observability, and recovery. |
| 9 | [PHASE_9.md](PHASE_9.md) | Delayed historical batch contract, immutable evidence, calibrated drift/performance reports, and local alerts. |
| 10 | [PHASE_10.md](PHASE_10.md) | Stable local/Task Scheduler orchestration plus the separate Airflow 3.3.0 DAG and recovery workflow. |

Other phases without a file do not yet have dedicated documentation in this
repository. Older audit records may describe the exact file paths that existed
when they were written; the current navigation source is this index. Phase
documentation is intentionally kept to one file per phase.
