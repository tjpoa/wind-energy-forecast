import type {
  MonitoringAlert,
  MonitoringDrift,
  MonitoringHistoryResponse,
  MonitoringLatestResponse,
  MonitoringMetric,
  MonitoringReport,
  MonitoringRun,
  MonitoringRunResponse,
  MonitoringWindow,
} from "../types/monitoring";

type UnknownRecord = Record<string, unknown>;

export class MonitoringApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null, options?: ErrorOptions) {
    super(message, options);
    this.name = "MonitoringApiError";
    this.status = status;
  }
}

function record(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function nullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function oneOf(value: unknown, options: readonly string[]): boolean {
  return typeof value === "string" && options.includes(value);
}

function severity(value: unknown): boolean {
  return oneOf(value, ["not_available", "ok", "warning", "critical"]);
}

function isRun(value: unknown): value is MonitoringRun {
  if (!record(value) || (!record(value.failure) && value.failure !== null)) {
    return false;
  }
  const validFailure =
    value.failure === null ||
    (record(value.failure) &&
      nullableString(value.failure.failed_at_utc) &&
      nullableString(value.failure.error_type) &&
      typeof value.failure.message === "string");
  return (
    typeof value.run_id === "string" &&
    typeof value.attempted_at_utc === "string" &&
    nullableString(value.through_date) &&
    nullableString(value.source_pipeline_run_id) &&
    nullableString(value.source_pipeline_status) &&
    oneOf(value.status, ["succeeded", "failed", "in_progress"]) &&
    nullableString(value.report_id) &&
    Number.isInteger(value.active_alert_count) &&
    validFailure
  );
}

function isAlert(value: unknown): value is MonitoringAlert {
  return (
    record(value) &&
    typeof value.alert_event_id === "string" &&
    typeof value.rule_id === "string" &&
    typeof value.through_date === "string" &&
    oneOf(value.event_type, ["opened", "escalated", "resolved"]) &&
    severity(value.severity) &&
    nullableString(value.previous_alert_event_id)
  );
}

function isMetric(value: unknown): value is MonitoringMetric {
  return (
    record(value) &&
    typeof value.metric === "string" &&
    typeof value.label === "string" &&
    (value.value === null || finite(value.value)) &&
    typeof value.status === "string" &&
    severity(value.severity) &&
    finite(value.warning) &&
    finite(value.critical) &&
    oneOf(value.direction, ["upper", "lower"])
  );
}

function isDrift(value: unknown): value is MonitoringDrift {
  return (
    record(value) &&
    typeof value.feature === "string" &&
    oneOf(value.comparator, ["global", "seasonal"]) &&
    oneOf(value.detector, ["normalized_wasserstein", "ks_statistic"]) &&
    finite(value.value) &&
    severity(value.severity) &&
    finite(value.threshold) &&
    finite(value.threshold_ratio)
  );
}

function isWindow(value: unknown): value is MonitoringWindow {
  return (
    record(value) &&
    Number.isInteger(value.window_days) &&
    oneOf(value.status, ["available", "insufficient_data", "not_available"]) &&
    Number.isInteger(value.sample_count) &&
    (value.minimum_samples === null || Number.isInteger(value.minimum_samples)) &&
    nullableString(value.calendar_start) &&
    nullableString(value.calendar_end) &&
    (value.coverage_ratio === null || finite(value.coverage_ratio)) &&
    (value.coverage_severity === null || severity(value.coverage_severity)) &&
    Array.isArray(value.performance) &&
    value.performance.every(isMetric) &&
    Array.isArray(value.top_drift) &&
    value.top_drift.every(isDrift)
  );
}

function isReport(value: unknown): value is MonitoringReport {
  if (
    !record(value) ||
    !record(value.source_pipeline) ||
    !record(value.freshness) ||
    !record(value.windows) ||
    !Array.isArray(value.active_alerts)
  ) return false;
  const model = value.model;
  return (
    typeof value.report_id === "string" &&
    typeof value.reporting_run_id === "string" &&
    typeof value.created_at_utc === "string" &&
    typeof value.as_of_date === "string" &&
    typeof value.source_pipeline.run_id === "string" &&
    typeof value.source_pipeline.status === "string" &&
    oneOf(value.freshness.status, [
      "within_objective",
      "behind_objective",
      "late",
      "unknown",
    ]) &&
    nullableString(value.freshness.watermark_date) &&
    nullableString(value.freshness.objective_at) &&
    nullableString(value.freshness.late_at) &&
    value.freshness.timezone === "Europe/Lisbon" &&
    Number.isInteger(value.freshness.objective_days) &&
    Number.isInteger(value.freshness.late_days) &&
    (model === null ||
      (record(model) &&
        nullableString(model.snapshot_id) &&
        typeof model.checksum === "string" &&
        nullableString(model.model_type) &&
        nullableString(model.dataset_version) &&
        typeof model.dataset_checksum === "string" &&
        typeof model.transformation_version === "string" &&
        model.status === "selected_not_promoted")) &&
    isWindow(value.windows["30"]) &&
    isWindow(value.windows["90"]) &&
    value.active_alerts.every(isAlert) &&
    value.target_scale === "sum_of_15_minute_MW_observations"
  );
}

export function isMonitoringLatestResponse(
  value: unknown,
): value is MonitoringLatestResponse {
  return (
    record(value) &&
    oneOf(value.state, ["empty", "available"]) &&
    value.mode === "retrospective_historical_batch_not_real_time" &&
    typeof value.served_at_utc === "string" &&
    nullableString(value.message) &&
    (value.latest_attempt === null || isRun(value.latest_attempt)) &&
    (value.report === null || isReport(value.report)) &&
    (value.state !== "available" || value.report !== null)
  );
}

function page(value: unknown, validator: (item: unknown) => boolean): boolean {
  return (
    record(value) &&
    Array.isArray(value.items) &&
    value.items.every(validator) &&
    Number.isInteger(value.total) &&
    Number.isInteger(value.limit) &&
    Number.isInteger(value.offset)
  );
}

export function isMonitoringHistoryResponse(
  value: unknown,
): value is MonitoringHistoryResponse {
  return (
    record(value) &&
    oneOf(value.state, ["empty", "available"]) &&
    value.mode === "retrospective_historical_batch_not_real_time" &&
    page(value.runs, isRun) &&
    page(value.alerts, isAlert)
  );
}

export function isMonitoringRunResponse(
  value: unknown,
): value is MonitoringRunResponse {
  return (
    record(value) &&
    value.state === "available" &&
    value.mode === "retrospective_historical_batch_not_real_time" &&
    isRun(value.run) &&
    (value.report === null || isReport(value.report))
  );
}

async function request<T>(
  baseUrl: string | null,
  path: string,
  validator: (value: unknown) => value is T,
  signal?: AbortSignal,
): Promise<T> {
  const normalized = baseUrl?.trim().replace(/\/+$/, "");
  if (!normalized) {
    throw new MonitoringApiError("The API base URL is not configured.");
  }
  let response: Response;
  try {
    response = await fetch(`${normalized}${path}`, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (error) {
    if (signal?.aborted || (error instanceof Error && error.name === "AbortError")) {
      throw error;
    }
    throw new MonitoringApiError(
      "The monitoring request could not be completed.",
      null,
      { cause: error },
    );
  }
  if (!response.ok) {
    throw new MonitoringApiError(
      `The monitoring request failed with HTTP ${response.status}.`,
      response.status,
    );
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch (error) {
    throw new MonitoringApiError(
      "The API returned an invalid monitoring response.",
      response.status,
      { cause: error },
    );
  }
  if (!validator(payload)) {
    throw new MonitoringApiError(
      "The API returned an invalid monitoring response.",
      response.status,
    );
  }
  return payload;
}

export function getMonitoringLatest(baseUrl: string | null, signal?: AbortSignal) {
  return request(
    baseUrl,
    "/api/v1/monitoring/latest",
    isMonitoringLatestResponse,
    signal,
  );
}

export function getMonitoringHistory(baseUrl: string | null, signal?: AbortSignal) {
  return request(
    baseUrl,
    "/api/v1/monitoring/history",
    isMonitoringHistoryResponse,
    signal,
  );
}

export function getMonitoringRun(
  baseUrl: string | null,
  runId: string,
  signal?: AbortSignal,
) {
  return request(
    baseUrl,
    `/api/v1/monitoring/runs/${encodeURIComponent(runId)}`,
    isMonitoringRunResponse,
    signal,
  );
}
