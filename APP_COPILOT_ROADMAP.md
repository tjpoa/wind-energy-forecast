# Local Copilot roadmap

This is an orientation document only. Python contracts, manifests, and tests
remain the source of truth.

## Scope

Build a local, single-user, read-only Copilot in Portuguese over verified
synthetic `demo/v1` evidence. Real historical v1 data is unavailable in the
public repository and must remain internal. Unsupported or unavailable data is
reported explicitly; no query writes data, trains models, promotes releases,
or calls Azure.

## Milestones

1. **Contracts and questions** — add typed query kinds for prediction versus
   observation and dataset coverage, including bounds, units, timezone,
   freshness, evidence IDs, and unavailable states. Gate: schemas and tests
   define every supported question.
2. **Deterministic local app** — expose the existing read-only query core through
   a local `/copilot` API route and React page using `demo/v1`. Gate: every
   answered fact has evidence; unsupported questions are refused; no external
   key is needed.
3. **Natural-language adapter (optional)** — evaluate Portuguese tool
   selection, refusals, citations, and timeouts before enabling one provider.
   Gate: the adapter can select only the typed operational tool and cannot
   produce facts itself.
4. **Local integrations (optional)** — add MCP over `stdio`, then documentary
   RAG over a small versioned corpus. Gate: REST and MCP are equivalent,
   documents have hashes/versions, and no secrets, paths, or raw data leak.

## Deferred

ANN v2 training and promotion, Azure deployment/publication, remote MCP/RAG,
autonomous agents, and future forecasts are outside this roadmap.
