# AGENTS.md

## Project objective

This repository contains a wind-energy production forecasting project based on historical E-REDES production data and meteorological data obtained through an external API.

The long-term objective is to evolve the existing academic project into a reproducible Data/ML Engineering and MLOps project without breaking the current modelling workflow.

The complete implementation roadmap is available at:

`docs/ML_ENGINEERING_ROADMAP.md`

## Working rules

Before changing code:

1. Inspect the relevant repository files and understand the existing implementation.
2. Briefly explain how the requested change integrates with the current project.
3. Preserve currently working behaviour unless the task explicitly requires changing it.
4. Do not invent column names, model paths, feature names, metrics or API fields.
5. Determine all implementation details from the repository.
6. Do not hardcode credentials, tokens, absolute paths or environment-specific values.
7. Prefer small, modular and testable changes over large rewrites.
8. Keep notebooks for exploration; reusable logic should live in Python modules.
9. Use type hints, docstrings and structured logging where appropriate.
10. Add or update tests for newly introduced behaviour.
11. Run relevant tests, linting and smoke checks before completing a task.
12. Review the final diff for regressions and unrelated changes.
13. Never start the next roadmap phase automatically.

## Security

* Never commit API keys, credentials, `.env` files or connection strings.
* Use environment variables for secrets.
* Do not display secrets in logs, tests or documentation.
* Ask before deleting data, models or important project artifacts.

## Completion report

At the end of each task, provide:

* Summary of the implementation.
* Files created, modified or removed.
* Commands required to run the functionality.
* Tests and checks executed.
* Test results.
* Assumptions and limitations.
* Remaining issues.
* Suggested Git commit message.

Stop after completing the explicitly requested phase.
