# Operational Copilot Candidate Evaluation Boundary

## Decision status

This is an additive unblock record for the candidate-evaluation increment. It
does not select or accept a provider-backed candidate.

| Decision | Accepted boundary |
| --- | --- |
| Execution mode | `offline_injected`; the runner receives a candidate selector through a local protocol. |
| Egress | Disabled. No provider SDK, network call, remote model, or credential is used. |
| Language | English only (`en`), matching the sealed 88-case dataset. Portuguese remains `NO-GO` until a versioned dataset exists. |
| Selector timeout | `1.0` second, cooperative and fail-closed. |
| Total evaluation deadline | `5.0` seconds per case. |
| Retention | No prompts, candidate payloads, facts, citations, or answers in the receipt. Candidate response-set storage is caller-controlled and outside operational stores. |
| Receipt | Additive JSON receipt with candidate metadata, dataset/response/report/configuration digests, source commit, metrics, timestamp, and disabled-by-default state. Existing evaluation contracts remain unchanged. |
| Provider/model | Required metadata supplied by the future candidate owner; deliberately not invented or accepted by this record. |

## Implemented boundary

`wind_forecast.operational_candidate_evaluation` provides:

- a strict `CandidateInput` containing only the question, authorization, and
  the single `operational_query` tool schema;
- an injected `CandidateSelector` protocol with one selection call per case;
- a synthetic in-memory executor that uses the sealed oracle only after the
  candidate boundary and never exposes oracle fields or `evidence_scenario`;
- normalized `CandidateTrace` values scored by the existing
  `operational_evaluation` harness without weakening any gate;
- deterministic response-set digesting; and
- receipt construction only after a passed report, with no overwrite of an
  existing receipt path.

The CLI `scripts/evaluate_operational_copilot_candidate.py` accepts an
externally supplied response set and writes a receipt only when the existing
88-case harness reports `passed`. It emits only the sanitized report. The
runner and receipt writer do not read the API, query service, Registry,
loaders, PostgreSQL, scheduler, or any operational store.

This is an interface boundary, not a security sandbox. An injected selector
runs in the same Python process and must be trusted not to inspect frames,
filesystem state, or network services. The selector timeout and total deadline
are cooperative: a synchronous selector that blocks cannot be pre-empted by
this increment. These limits are why the boundary is restricted to offline
injected candidates and cannot authorize a provider-backed candidate.

## Acceptance and remaining gate

The existing harness remains authoritative:

- 100% is required for authorization, refusal, tool selection, arguments,
  factuality, grounding, citations, privacy, evidence states, and zero-write
  checks;
- at least 95% is required for paraphrases; and
- at most one safe paraphrase abstention is allowed.

Any wrong tool, wrong arguments, invalid output, timeout, unsupported
provider, or ungrounded fact fails closed. The CLI's receipt is an attestation
of the supplied response-set artifact: its source commit and candidate
metadata are caller-provided and do not prove that the traces were produced by
that candidate. The synthetic executor also means factuality, grounding, and
citation checks validate the sealed oracle/harness path, not end-to-end model
generation. A receipt is not an activation signal: the Copilot remains
disabled by default until a real candidate has explicitly supplied
provider/model metadata, produced a complete response set through an approved
isolated execution path, passed the gates, received human review, and merged
through the normal workflow.

This record does not authorize a provider, remote egress, dependency
installation, endpoint, UI, MCP, RAG, operational-store write, or Portuguese
evaluation.
