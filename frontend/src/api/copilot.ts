import { apiConfig } from "./config";

export interface CopilotFailure {
  readonly code: string;
  readonly message: string;
  readonly retryable: boolean;
  readonly evidence_state: string;
}

export interface CopilotAnswer {
  readonly query_kind: string | null;
  readonly status: string;
  readonly summary: string | null;
  readonly facts: readonly Record<string, unknown>[];
  readonly evidence: readonly Record<string, unknown>[];
  readonly limitations: readonly string[];
  readonly served_at_utc: string;
  readonly correlation_id: string;
  readonly failure: CopilotFailure | null;
}

export type CopilotResponse =
  | { readonly route: "operational"; readonly mode: "guided_local"; readonly answer: CopilotAnswer; readonly limitations: readonly string[]; readonly failure: null }
  | { readonly route: "documentary"; readonly mode: "rag_local" | "rag_openai"; readonly answer: { readonly summary: string; readonly evidence: readonly DocumentaryEvidence[] }; readonly limitations: readonly string[]; readonly failure: null; readonly provider_failure: { readonly code: string; readonly message: string } | null }
  | { readonly route: "refused"; readonly mode: "guided_local"; readonly answer: null; readonly limitations: readonly string[]; readonly failure: CopilotFailure };

export class CopilotApiError extends Error {}

export async function askCopilot(question: string, signal?: AbortSignal): Promise<CopilotResponse> {
  if (!apiConfig.baseUrl) throw new CopilotApiError("The API base URL is not configured.");
  let response: Response;
  try {
    response = await fetch(`${apiConfig.baseUrl}/api/v1/copilot`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ question }),
      signal,
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new CopilotApiError("The Copilot request could not be completed.");
  }
  if (!response.ok) throw new CopilotApiError(`The Copilot request failed with HTTP ${response.status}.`);
  const payload: unknown = await response.json();
  if (!payload || typeof payload !== "object" || !["operational", "documentary", "refused"].includes((payload as { route?: unknown }).route as string)) {
    throw new CopilotApiError("The API returned an invalid Copilot response.");
  }
  return payload as CopilotResponse;
}

export interface DocumentaryEvidence {
  readonly chunk_id: string;
  readonly document_id: string;
  readonly title: string;
  readonly heading: string;
  readonly uri: string;
  readonly sha256: string;
}
