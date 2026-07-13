# AGENTS.md

## Project purpose

This repository forecasts Portuguese wind-energy production from historical production and meteorological data.

The long-term goal is to evolve the academic project into a reproducible Data/ML Engineering and MLOps project without breaking the current modelling workflow.

Treat the provenance and any replacement of historical data sources as explicit, versioned engineering decisions. Remain provider-neutral: do not assume a provider, dataset, or definition unless repository evidence or official primary documentation confirms it.

The implementation roadmap is [`docs/ML_ENGINEERING_ROADMAP.md`](docs/ML_ENGINEERING_ROADMAP.md). Read the applicable roadmap step and relevant phase audits, closure summaries, assessments, and decision records under `docs/` before planning or editing.

## Sources of truth

Apply this precedence when instructions or assumptions conflict:

1. The user's current explicit request.
2. This `AGENTS.md`.
3. The current roadmap and approved phase plan.
4. Relevant phase documents and decision records.
5. The repository implementation and Git history.
6. Official primary external documentation.

An older plan never overrides a newer user instruction. If repository evidence cannot resolve a material conflict, stop and report it rather than guessing.

Distinguish confirmed facts from inferences, estimates, and unresolved assumptions.

## Scope and working rules

- Identify whether the request is plan-only, implementation, review-only, research, or documentation work.
- A plan-only request authorizes inspection and a scoped plan, but no file changes, dependency installation, pipeline execution, staging, or commit.
- An implementation request authorizes only the requested or approved scope. If no plan exists for a non-trivial change, inspect first and produce a small plan before editing.
- A review-only request is read-only. Inspect the specified files, diff, commit, or behaviour; do not silently fix findings.
- Before work, read the applicable context, run `git status --short`, and protect unrelated uncommitted or untracked work.
- For non-trivial changes: inspect relevant context, define the objective, allowed files, non-goals, preserved behaviour, acceptance criteria, validation, risks, rollback, and stop condition; implement only that plan; validate; independently review the complete diff; correct blocking findings; then stop.
- Use separate agents for planning, implementation, and independent review when available. Do not run them concurrently against one working tree; parallelize only independent work in isolated worktrees.
- Modify only explicitly allowed files. If another file becomes necessary, stop and request approval before editing it.
- Work only on the requested task or roadmap step. Do not include unrelated cleanup or future-phase work.
- Preserve successful, deterministic behaviour and working data contracts unless change is explicitly authorized.
- Prefer small, modular, reversible, typed, documented, and testable changes over rewrites or premature abstractions.
- Derive column names, features, metrics, units, paths, filenames, model fields, API fields, and contracts from repository evidence and approved official documentation; never invent them.
- Avoid hidden cleaning, implicit coercion, silent fallbacks, and import-time side effects. Keep validation separate from cleaning and transformation unless explicitly designed otherwise.
- Add or update tests when the approved phase introduces behaviour and includes tests. Add logging only when it provides clear value and is in scope.
- Do not install dependencies or run destructive commands unless explicitly authorized.

## Data, models, notebooks, network and security

- Treat raw datasets as immutable unless replacement is explicitly authorized. Prefer versioned datasets and never overwrite v1 while piloting or building v2.
- Do not modify, overwrite, or delete datasets, models, scalers, reports, baselines, or important artifacts without explicit authorization.
- Do not commit large generated datasets unless repository policy allows it. Keep generated processed CSVs ignored when that is current policy, and keep temporary pilot outputs ignored or outside the repository.
- When introducing a dataset version, record checksums, coverage, units, source metadata, licenses, access dates, transformation versions, service limitations, and unresolved source assumptions as applicable.
- A material source, feature, or distribution change may invalidate datasets, scalers, models, and baselines. Do not claim compatibility without evidence; refit, retrain, and re-baseline when the approved contract change requires it.
- During behaviour-preserving refactors, preserve artifact filenames, hashes, timestamps, schemas, and numerical outputs where required.
- Keep notebooks exploratory, historical, training, or research interfaces unless explicitly changed. Put reusable production logic in Python modules.
- Do not execute notebooks without explicit authorization or clear outputs, alter execution counts, reformat unrelated cells, add `sys.path` manipulation, or migrate notebook logic without an explicit maintainability benefit and validation strategy.
- Do not run training, downloads, repeated API loops, live network calls, or artifact generation without authorization. Never perform network calls during import or bulk downloads during planning.
- For external technical or data-source research, use official primary documentation. Start with small, temporary, non-destructive probes before any approved full ingestion.
- Never hardcode, print, log, document, or commit secrets, API keys, credentials, `.env` files, credential files, or connection strings. Use approved environment variables or user-level credential stores, never production credentials in tests, and never weaken security or certificate validation.

## Validation and review

- Use the smallest checks that establish correctness: syntax/import checks, synthetic in-memory checks, read-only checks against local data, old-versus-new equivalence, targeted tests, then broader checks only when authorized.
- Run the relevant tests, linting, builds, equivalence, and smoke checks for the approved change.
- Do not claim a check passed unless it ran. Report skipped, unavailable, or failed checks honestly, and never silently replace a failed check with a weaker one or weaken validation to make it pass.
- Before commit, run `git diff --check`, inspect `git diff --stat`, `git status --short`, and the complete diff, and confirm that only approved files changed.
- Review must be independent and read-only. Compare the result with the plan and every acceptance criterion; check scope, regressions, side effects, data mutation, security, compatibility, numerical equivalence, Git state, and artifact safety as applicable.
- Classify findings as blocking, warning, or informational and give a verdict of `PASS`, `PASS WITH WARNINGS`, or `FAIL`. Do not approve unexplained failing checks.
- Correct all blocking findings within the approved scope and revalidate. If correction needs wider scope, return to the user for authorization. A task is not complete while a blocking finding remains.

## Git branch and Pull Request workflow

For every independent task, follow these two phases unless the user explicitly requests a narrower read-only workflow.

**Phase 1 — Implementation and user review**

1. Update local `master` from the remote without discarding user work.
2. Create a short, task-specific branch from the updated `master`.
3. Implement only the approved scope.
4. Run the relevant tests, linting, builds, and checks.
5. Review the complete diff and working-tree state.
6. Create one cohesive commit per small task unless an approved plan says otherwise.
7. Push the task branch.
8. Create a draft Pull Request.
9. Stop for user review.

During Phase 1, do not merge, delete the branch, or start the next task.

Stage files explicitly; never use `git add .` and never stage ignored generated files. Commit only when requested, all blocking checks pass, and the diff is scoped. Use the approved commit message when provided.

Do not reset, discard, or overwrite unrelated user work. Do not amend, rebase, force-push, or otherwise rewrite history without explicit authorization.

**Phase 2 — Authorized merge and cleanup**

Only after explicit user authorization:

1. Confirm that every check and CI job passed.
2. Confirm there are no conflicts or additional unreviewed changes.
3. Prefer `Squash and merge`.
4. Delete the remote branch.
5. Update local `master`.
6. Delete the local branch.
7. Prune stale remote-tracking references.
8. Confirm the repository is clean.

Never merge while checks are pending or failing, conflicts exist, or changes remain unreviewed.

## Completion report

Keep the final report short and include:

- summary;
- files changed;
- tests and checks executed;
- results, including relevant compatibility, equivalence, and safety evidence and anything not run;
- commit and Pull Request;
- assumptions, limitations, and deferred work;
- confirmation that the next task was not started.
