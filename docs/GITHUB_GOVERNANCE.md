# GitHub governance

This repository uses a single-maintainer governance profile. It is a public
portfolio project, not an organisation with independent separation of duties.
The controls below make changes traceable and machine-validated without
claiming a human approval that does not exist.

## Operating model

- Every change to `master` is associated with a pull request.
- The required `CI gate` is the merge gate; required human approvals are `0`.
- The maintainer completes the pull-request self-review checklist and records
  the validation evidence in the pull request.
- Independent human review is recorded as `N/A — single-maintainer project`
  unless an actual external reviewer participates.
- A second GitHub account, bot approval, or other identity must never be used
  to simulate independent review.

This profile demonstrates PR traceability, automated quality gates and
deployment control. It does not provide organisational separation of duties.

## `master` protection

The live branch protection must have this target configuration:

| Control | Required value |
| --- | --- |
| Pull request before merging | Enabled |
| Required human approvals | `0` |
| Required status check | `CI gate` |
| Branch up to date before merging | Enabled |
| Enforce rules for administrators | Enabled |
| Required conversation resolution | Enabled |
| Required linear history | Enabled |
| Force-pushes | Disabled |
| Branch deletion | Disabled |
| Bypass actors | None |

The zero-approval setting is intentional: GitHub supports requiring a pull
request as a change record without requiring an approval. See the official
[branch protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule)
and [ruleset](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)
documentation.

## `production` environment

Production deployments remain manually gated by the maintainer:

- `tjpoa` remains a required environment reviewer;
- `prevent_self_review` is disabled because the project has one maintainer;
- administrator bypass remains disabled;
- the environment remains restricted to protected branches.

This action is a `maintainer confirmation`, not an independent approval. The
release flag and Azure credentials remain separate fail-closed prerequisites;
this document does not enable releases or create any cloud resource.

GitHub documents that enabling `prevent_self_review` prevents the person who
initiated a deployment from approving it, which is appropriate when a second
reviewer exists but would deadlock this single-maintainer project. See the
official [environment protection](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments)
documentation.

## Evidence and receipts

Each release or rollback receipt must distinguish the approval mode from human
independence:

```json
{
  "approval": {
    "mode": "maintainer_confirmation",
    "independent_human_review": "not_applicable_single_maintainer"
  }
}
```

Receipts continue to record the source SHA, pull request, CI run, image
digests, Terraform plan results, deployment revisions, smoke tests and drift
result. No receipt may describe the maintainer's own action as an independent
review.

## Future upgrade

If a trusted collaborator becomes available, the repository can move to the
independent-review profile in a separate, explicitly reviewed change:

- grant the collaborator the repository permission required for branch review;
- require one human approval on `master`;
- dismiss stale approvals after code changes;
- require approval of the most recent reviewable push; and
- re-enable `prevent_self_review` for `production`.

Until then, the single-maintainer profile is the truthful and reproducible
governance boundary for this portfolio project.
