export interface HealthResponse {
  readonly status: "ok";
}

export type HealthApiErrorKind =
  | "configuration"
  | "network"
  | "http"
  | "invalid-response";

export class HealthApiError extends Error {
  readonly kind: HealthApiErrorKind;
  readonly status: number | null;

  constructor(
    message: string,
    kind: HealthApiErrorKind,
    status: number | null = null,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "HealthApiError";
    this.kind = kind;
    this.status = status;
  }
}

function isHealthResponse(value: unknown): value is HealthResponse {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value) &&
    (value as { status?: unknown }).status === "ok"
  );
}

export async function getHealth(
  baseUrl: string | null,
  signal?: AbortSignal,
): Promise<HealthResponse> {
  const normalizedBaseUrl = baseUrl?.trim().replace(/\/+$/, "");
  if (!normalizedBaseUrl) {
    throw new HealthApiError(
      "The API base URL is not configured.",
      "configuration",
    );
  }

  let response: Response;
  try {
    response = await fetch(`${normalizedBaseUrl}/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (error) {
    if (signal?.aborted || (error instanceof Error && error.name === "AbortError")) {
      throw error;
    }
    throw new HealthApiError(
      "The health request could not be completed.",
      "network",
      null,
      { cause: error },
    );
  }

  if (!response.ok) {
    throw new HealthApiError(
      `The health request failed with HTTP ${response.status}.`,
      "http",
      response.status,
    );
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch (error) {
    throw new HealthApiError(
      "The API returned an invalid health response.",
      "invalid-response",
      response.status,
      { cause: error },
    );
  }

  if (!isHealthResponse(payload)) {
    throw new HealthApiError(
      "The API returned an invalid health response.",
      "invalid-response",
      response.status,
    );
  }

  return payload;
}
