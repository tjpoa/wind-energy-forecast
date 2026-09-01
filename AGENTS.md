# AGENTS.md

## Purpose

This repository forecasts Portuguese wind-energy production and demonstrates
reproducible Data/ML Engineering. Preserve the existing v1 modelling workflow
while adding only explicit, tested, backward-compatible changes.

## Sources of truth

When instructions or assumptions conflict, use this order:

1. The user's current request.
2. Executable code, configuration, schemas, and tests.
3. Versioned data/model manifests, immutable receipts, and migrations.
4. Short operational documentation.
5. Git history and external primary documentation.

Distinguish confirmed facts from inference. Never invent providers, columns,
units, paths, model fields, or API contracts.

## Working rules

- Identify whether the task is implementation, review, planning, research, or
  documentation.
- Before work, inspect `git status --short`, protect unrelated changes, and
  define scope, non-goals, preserved behaviour, acceptance criteria, and
  validation.
- Modify only the approved files. Use `apply_patch` for edits.
- Keep notebooks exploratory/training interfaces; put reusable logic in
  `src/wind_forecast/`.
- Preserve filenames, schemas, feature order, model/scaler compatibility, and
  deterministic outputs unless the request explicitly changes them.
- Prefer small, typed, reversible, testable changes. Avoid hidden cleaning,
  coercion, fallbacks, import-time side effects, and duplicated contracts.
- Use separate planning and independent review when available; never edit a
  shared worktree concurrently.

## Data, security, and network

- Treat raw data, models, scalers, reports, and important artifacts as
  immutable. Use versioned paths for replacements.
- Record hashes, coverage, units, source metadata, licence, access date,
  transformation version, and unresolved assumptions in manifests.
- Never print, log, commit, or document secrets, credentials, `.env` files, or
  connection strings. Use approved environment variables or credential stores.
- Do not install dependencies, run notebooks/training, download data, call
  live providers, or mutate external systems without explicit authorization.
- Use bounded, non-destructive probes before approved provider operations.

## Validation and review

Use the smallest checks that establish correctness: syntax/import checks,
synthetic checks, targeted tests, equivalence checks, then broader suites.
Report skipped or failed checks honestly. Before commit, run:

```text
python -m pytest
python -m ruff check .
git diff --check
git diff --stat
git status --short
```

Review the complete diff independently for scope, regressions, data mutation,
security, compatibility, numerical equivalence, and artifact safety. Classify
findings as blocking, warning, or informational and give a
`PASS`, `PASS WITH WARNINGS`, or `FAIL` verdict. Correct blocking findings
within scope before completion.

## Git workflow

For an implementation task, use the two phases below. Keep the phases
separate: phase 1 ends at user review; phase 2 requires a new, explicit user
authorization.

### Phase 1 — Implementation and user review

1. Update clean `master` from `origin/master` without discarding user work.
2. Create a short task branch with the `codex/` prefix.
3. Implement only the approved scope.
4. Run the relevant tests, linting, builds, and safety checks.
5. Review the complete diff and working-tree state.
6. Stage files explicitly and create one cohesive commit.
7. Push the task branch and create a draft Pull Request.
8. Stop for user review. Do not merge, delete branches, or start another task.

### Phase 2 — Authorized merge and cleanup

Only after the user explicitly authorizes phase 2:

1. Confirm that every required check and CI job passed.
2. Confirm that the PR is mergeable, has no conflicts, and contains no
   unreviewed changes.
3. Prefer squash merge into `master`.
4. Delete the merged remote branch.
5. Fetch and fast-forward local `master` from `origin/master`.
6. Delete the merged local task branch.
7. Prune stale remote-tracking references.
8. Confirm that `master` is current and the working tree is clean.

Never merge while checks are pending or failing, conflicts exist, or changes
remain unreviewed. Do not rewrite history or use destructive reset/checkout
commands without explicit authorization.

Stage files explicitly; never use `git add .`. Preserve unrelated work and
never stage or overwrite another task's uncommitted changes.
