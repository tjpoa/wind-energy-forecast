import type {
  PerformanceInterval,
  PerformanceMetrics,
  PerformanceObservation,
  PerformanceResponse,
  PerformanceResultInfo,
} from "../types/api";

const PERFORMANCE_ENDPOINT = "/api/v1/performance";

export type PerformanceApiErrorKind =
  | "configuration"
  | "network"
  | "http"
  | "invalid-response";

export class PerformanceApiError extends Error {
  readonly kind: PerformanceApiErrorKind;
  readonly status: number | null;

  constructor(
    message: string,
    kind: PerformanceApiErrorKind,
    status: number | null = null,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "PerformanceApiError";
    this.kind = kind;
    this.status = status;
  }
}

type UnknownRecord = Record<string, unknown>;

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value);
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function isPerformanceMetrics(value: unknown): value is PerformanceMetrics {
  if (!isRecord(value)) {
    return false;
  }

  return (
    (value.r2 === null || isFiniteNumber(value.r2)) &&
    isFiniteNumber(value.mae) &&
    isFiniteNumber(value.rmse) &&
    isFiniteNumber(value.mape_percent)
  );
}

function isPerformanceInterval(value: unknown): value is PerformanceInterval {
  if (!isRecord(value)) {
    return false;
  }

  return (
    isNullableString(value.requested_start_date) &&
    isNullableString(value.requested_end_date) &&
    typeof value.available_start_date === "string" &&
    typeof value.available_end_date === "string" &&
    typeof value.returned_start_date === "string" &&
    typeof value.returned_end_date === "string"
  );
}

function isPerformanceResultInfo(
  value: unknown,
): value is PerformanceResultInfo {
  if (!isRecord(value)) {
    return false;
  }

  return (
    typeof value.model_type === "string" &&
    isInteger(value.seed) &&
    isFiniteNumber(value.test_fraction) &&
    isNullableString(value.dataset_version) &&
    typeof value.evaluation_start_date === "string" &&
    typeof value.evaluation_end_date === "string" &&
    (value.artifact_metrics === null ||
      isPerformanceMetrics(value.artifact_metrics))
  );
}

function isPerformanceObservation(
  value: unknown,
): value is PerformanceObservation {
  if (!isRecord(value)) {
    return false;
  }

  return (
    typeof value.date === "string" &&
    isFiniteNumber(value.actual) &&
    isFiniteNumber(value.predicted) &&
    isFiniteNumber(value.error) &&
    isFiniteNumber(value.absolute_error)
  );
}

function isPerformanceResponse(value: unknown): value is PerformanceResponse {
  if (!isRecord(value) || !Array.isArray(value.observations)) {
    return false;
  }

  return (
    isPerformanceInterval(value.interval) &&
    isInteger(value.observation_count) &&
    value.observation_count >= 0 &&
    value.observation_count === value.observations.length &&
    isPerformanceMetrics(value.metrics) &&
    (value.result === null || isPerformanceResultInfo(value.result)) &&
    value.observations.every(isPerformanceObservation)
  );
}

function isAbortError(error: unknown, signal?: AbortSignal): boolean {
  return (
    signal?.aborted === true ||
    (error instanceof Error && error.name === "AbortError")
  );
}

export async function getPerformance(
  baseUrl: string | null,
  signal?: AbortSignal,
): Promise<PerformanceResponse> {
  const normalizedBaseUrl = baseUrl?.trim().replace(/\/+$/, "");

  if (!normalizedBaseUrl) {
    throw new PerformanceApiError(
      "The API base URL is not configured.",
      "configuration",
    );
  }

  let response: Response;
  try {
    response = await fetch(`${normalizedBaseUrl}${PERFORMANCE_ENDPOINT}`, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (error) {
    if (isAbortError(error, signal)) {
      throw error;
    }

    throw new PerformanceApiError(
      "The performance request could not be completed.",
      "network",
      null,
      { cause: error },
    );
  }

  if (!response.ok) {
    throw new PerformanceApiError(
      `The performance request failed with HTTP ${response.status}.`,
      "http",
      response.status,
    );
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch (error) {
    if (isAbortError(error, signal)) {
      throw error;
    }

    throw new PerformanceApiError(
      "The API returned an invalid performance response.",
      "invalid-response",
      response.status,
      { cause: error },
    );
  }

  if (!isPerformanceResponse(payload)) {
    throw new PerformanceApiError(
      "The API returned an invalid performance response.",
      "invalid-response",
      response.status,
    );
  }

  return payload;
}
