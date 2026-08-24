# Operational Copilot Candidate Evaluation Boundary

## Decision status

The original decision below remains the accepted offline boundary. A later
2026-08-24 decision selects one concrete remote candidate for a bounded
evaluation implementation; it does not claim that the live evaluation passed.

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

### Approved remote candidate

| Decision | Approved value |
| --- | --- |
| Provider/API | OpenAI Responses API at the fixed `https://api.openai.com/v1/responses` endpoint. |
| Model | Exact snapshot `gpt-5.4-mini-2026-03-17`. |
| Egress | Explicitly limited to each case's synthetic question, synthetic authorization context, the single tool schema, and static selector instructions. |
| Provider storage | `store=false`; this is not a Zero Data Retention claim. |
| Language | English only over the sealed 88-case dataset. |
| Selector/deadline | Five seconds per selector call and five seconds total per case, cooperative and fail-closed. |
| Calls | At most 88 calls, exactly one per case, with zero retries and no fallback model. |
| Secret | `OPENAI_API_KEY` from the process environment only; never from CLI arguments, files, receipts, logs, or exceptions. |
| Local retention | No provider response or normalized trace is persisted; only a passed, additive digest-only remote receipt may be written. |
| Activation | Disabled by default after evaluation; human review and merge remain mandatory. |

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

## Implemented OpenAI candidate adapter

`wind_forecast.operational_openai_candidate` implements the approved selector
without adding the OpenAI SDK or another dependency. It reuses `requests`,
fixes the endpoint and model, rejects HTTP redirects, disables environment
proxies, keeps normal TLS verification, bounds the response body, sends one strict function schema with
`tool_choice=auto`, disables parallel tool calls, and performs no retry.

The provider response may select exactly one tool or abstain. Multiple calls,
unknown output types, malformed arguments, an unexpected model, or an
incomplete response fail closed. Authentication, rate-limit, transport, TLS,
timeout, or service failures interrupt the run before another case is sent.
Generated text is never treated as an operational answer; the unchanged
synthetic executor and harness remain authoritative after tool selection.

The dedicated
`scripts/evaluate_openai_operational_copilot_candidate.py` CLI requires an
explicit `--confirm-synthetic-egress` flag, validates the sealed English
dataset against its pinned aggregate SHA-256, runs all traces in memory, prints only the existing sanitized report,
and writes the separate
`wind_forecast.operational_openai_candidate_evaluation_receipt.v1` contract
only after a complete pass. The remote receipt contains configuration,
digests, metrics, call count, source commit, and timestamp; it contains no API
key, prompt, question, arguments, provider response ID, facts, citations, or
answer payload.

The repository environment used to implement this increment did not contain
`OPENAI_API_KEY`. Consequently no live request, response set, candidate report,
or remote receipt was produced, and the candidate remains unevaluated.

The configured Requests timeout bounds connection and read inactivity rather
than providing hard wall-clock pre-emption of a slowly streamed response. The
harness checks the elapsed selector and total durations afterward and fails
closed, but the synchronous network operation cannot be interrupted at exactly
five seconds by this implementation.

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
disabled by default until the selected candidate produces a complete response
set through the approved CLI, passes the gates, receives human review, and
merges through the normal workflow.

The 2026-08-24 decision authorizes only the fixed OpenAI candidate and sealed
synthetic egress described above. It does not authorize dependency
installation, a public application endpoint, live operational evidence
egress, UI, MCP, RAG, operational-store writes, Portuguese evaluation, or
Copilot activation.

Official API references used for this decision:

- [GPT-5.4 mini model and snapshot](https://developers.openai.com/api/docs/models/gpt-5.4-mini), accessed 2026-08-24;
- [Responses API function-call contract](https://developers.openai.com/api/reference/typescript/resources/beta/subresources/responses/methods/create), accessed 2026-08-24; and
- [OpenAI API data controls](https://developers.openai.com/api/docs/guides/your-data#default-usage-policies-by-endpoint), accessed 2026-08-24.
