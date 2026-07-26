import { describe, expect, it } from "vitest";

import {
  isMonitoringHistoryResponse,
  isMonitoringLatestResponse,
  isMonitoringRunResponse,
} from "../api/monitoring";

const run = {
  run_id: "run",
  attempted_at_utc: "2026-01-01T00:00:00Z",
  through_date: "2025-12-25",
  source_pipeline_run_id: "source",
  source_pipeline_status: "succeeded",
  status: "failed",
  report_id: null,
  active_alert_count: 0,
  failure: {
    failed_at_utc: "2026-01-01T00:01:00Z",
    error_type: "MonitoringReportingError",
    message: "The reporting attempt failed. Inspect local operator logs.",
  },
};

const emptyLatest = {
  state: "empty",
  mode: "retrospective_historical_batch_not_real_time",
  served_at_utc: "2026-01-01T00:00:00Z",
  message: "No reports.",
  latest_attempt: run,
  report: null,
};

const history = {
  state: "available",
  mode: "retrospective_historical_batch_not_real_time",
  runs: { items: [run], total: 1, limit: 20, offset: 0 },
  alerts: { items: [], total: 0, limit: 50, offset: 0 },
};

describe("monitoring response validation", () => {
  it("validates latest, history, and run contracts", () => {
    expect(isMonitoringLatestResponse(emptyLatest)).toBe(true);
    expect(isMonitoringHistoryResponse(history)).toBe(true);
    expect(
      isMonitoringRunResponse({
        state: "available",
        mode: "retrospective_historical_batch_not_real_time",
        run,
        report: null,
      }),
    ).toBe(true);
  });

  it("rejects inconsistent or unsanitized contracts", () => {
    expect(
      isMonitoringLatestResponse({ ...emptyLatest, state: "available" }),
    ).toBe(false);
    expect(
      isMonitoringHistoryResponse({
        ...history,
        runs: { ...history.runs, total: "1" },
      }),
    ).toBe(false);
    expect(
      isMonitoringRunResponse({
        state: "available",
        mode: "retrospective_historical_batch_not_real_time",
        run: { ...run, status: "exploded", local_path: "C:\\secret" },
        report: null,
      }),
    ).toBe(false);
    expect(
      isMonitoringHistoryResponse({
        ...history,
        runs: {
          ...history.runs,
          items: [{ ...run, status: ["failed"] }],
        },
      }),
    ).toBe(false);
  });
});
