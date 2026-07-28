import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "../App";

vi.mock("../api/config", () => ({
  apiConfig: { baseUrl: "http://api.test" },
}));

const emptyHistory = {
  state: "empty",
  mode: "retrospective_historical_batch_not_real_time",
  runs: { items: [], total: 0, limit: 20, offset: 0 },
  alerts: { items: [], total: 0, limit: 50, offset: 0 },
};

const run = {
  run_id: "report-run-1",
  attempted_at_utc: "2026-04-07T12:00:00Z",
  through_date: "2026-03-31",
  source_pipeline_run_id: "source-run-1",
  source_pipeline_status: "succeeded",
  status: "succeeded",
  report_id: "r".repeat(64),
  active_alert_count: 1,
  failure: null,
};

const alert = {
  alert_event_id: "a".repeat(64),
  rule_id: "feature_drift:x:30:global",
  through_date: "2026-03-31",
  event_type: "opened",
  severity: "warning",
  previous_alert_event_id: null,
};

const metric = {
  metric: "MAE",
  label: "MAE",
  value: 12.3,
  status: "available",
  severity: "warning",
  warning: 10,
  critical: 20,
  direction: "upper",
};

const drift = {
  feature: "wind_direction_current",
  comparator: "seasonal",
  detector: "ks_statistic",
  value: 0.3,
  severity: "critical",
  threshold: 0.2,
  threshold_ratio: 1.5,
};

const monitoringReport = {
  report_id: "r".repeat(64),
  reporting_run_id: "report-run-1",
  created_at_utc: "2026-04-07T12:00:00Z",
  as_of_date: "2026-03-31",
  source_pipeline: { run_id: "source-run-1", status: "succeeded" },
  freshness: {
    status: "late",
    watermark_date: "2026-03-31",
    objective_at: "2026-04-05T12:00:00+01:00",
    late_at: "2026-04-07T12:00:00+01:00",
    timezone: "Europe/Lisbon",
    objective_days: 5,
    late_days: 7,
  },
  model: {
    snapshot_id: "s".repeat(64),
    checksum: "c".repeat(64),
    model_type: "RandomForestRegressor",
    dataset_version: "v2",
    dataset_checksum: "d".repeat(64),
    transformation_version: "feature_ready_ren_era5_land_v2",
    status: "selected_not_promoted",
  },
  model_era: {
    model_era_id: "e".repeat(64),
    association_kind: "active_deployment",
    deployment_id: "a".repeat(64),
    deployment_state_id: "b".repeat(64),
    deployment_generation: 1,
    registered_model_name: "wind-v2",
    model_version: "1",
    cutoffs: {
      fit_cutoff: "2024-12-31",
      activation_cutoff: "2026-01-01",
    },
    pins: { model_sha256: "c".repeat(64) },
  },
  windows: {
    "30": {
      window_days: 30,
      status: "available",
      sample_count: 30,
      minimum_samples: null,
      calendar_start: "2026-03-02",
      calendar_end: "2026-03-31",
      coverage_ratio: 1,
      coverage_severity: "ok",
      performance: [metric],
      top_drift: [drift],
    },
    "90": {
      window_days: 90,
      status: "insufficient_data",
      sample_count: 30,
      minimum_samples: 45,
      calendar_start: null,
      calendar_end: null,
      coverage_ratio: null,
      coverage_severity: null,
      performance: [],
      top_drift: [],
    },
  },
  active_alerts: [alert],
  target_scale: "sum_of_15_minute_MW_observations",
};

const availableLatest = {
  state: "available",
  mode: "retrospective_historical_batch_not_real_time",
  served_at_utc: "2026-04-08T12:00:00Z",
  message: null,
  latest_attempt: run,
  report: monitoringReport,
};

const availableHistory = {
  ...emptyHistory,
  state: "available",
  runs: { ...emptyHistory.runs, items: [run], total: 1 },
  alerts: { ...emptyHistory.alerts, items: [alert], total: 1 },
};

const performancePayload = {
  interval: {
    requested_start_date: null,
    requested_end_date: null,
    available_start_date: "2026-01-01",
    available_end_date: "2026-01-02",
    returned_start_date: "2026-01-01",
    returned_end_date: "2026-01-02",
  },
  observation_count: 2,
  metrics: { r2: 0.91, mae: 12.3, rmse: 18.4, mape_percent: 5.6 },
  result: null,
  observations: [
    {
      date: "2026-01-01",
      actual: 100,
      predicted: 90,
      error: -10,
      absolute_error: 10,
    },
    {
      date: "2026-01-02",
      actual: 120,
      predicted: 130,
      error: 10,
      absolute_error: 10,
    },
  ],
};

function jsonResponse(payload: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(payload),
  } as unknown as Response;
}

function responseFor(url: string): Response {
  if (url.endsWith("/api/v1/monitoring/latest")) return jsonResponse(availableLatest);
  if (url.endsWith("/api/v1/monitoring/history")) return jsonResponse(availableHistory);
  if (url.endsWith("/api/v1/performance")) return jsonResponse(performancePayload);
  if (url.includes("/api/v1/monitoring/runs/")) {
    return jsonResponse({
      state: "available",
      mode: "retrospective_historical_batch_not_real_time",
      run,
      report: monitoringReport,
    });
  }
  return jsonResponse({}, 404);
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("App monitoring dashboard", () => {
  it("opens on monitoring, labels it as not real time, and shows loading", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => undefined)));
    render(<App />);

    expect(screen.getByRole("tab", { name: "Monitoring" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByText(/not real time/i)).toBeInTheDocument();
    expect(screen.getByText("Loading historical monitoring data…")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Connecting" })).toBeInTheDocument();
  });

  it("treats a verified empty monitoring store as connected", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | URL | Request) =>
        Promise.resolve(
          String(input).endsWith("/latest")
            ? jsonResponse({
                state: "empty",
                mode: "retrospective_historical_batch_not_real_time",
                served_at_utc: "2026-04-08T12:00:00Z",
                message: "No historical monitoring reports or runs are available.",
                latest_attempt: null,
                report: null,
              })
            : jsonResponse(emptyHistory),
        ),
      ),
    );
    render(<App />);

    expect(await screen.findByText("No monitoring reports yet")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Connected" })).toBeInTheDocument();
    expect(screen.getByText("No reporting runs are available.")).toBeInTheDocument();
  });

  it("renders delayed freshness, model state, rolling metrics, drift, and alerts", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | URL | Request) =>
        Promise.resolve(responseFor(String(input))),
      ),
    );
    render(<App />);

    expect(await screen.findByText(/API remains connected/)).toHaveTextContent("late");
    expect(screen.getByText(/selected, not promoted/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "30 days" })).toBeInTheDocument();
    expect(screen.getByText("wind_direction_current")).toBeInTheDocument();
    expect(screen.getAllByText("feature_drift:x:30:global")).toHaveLength(2);
    expect(screen.getByText(/30 samples; 45 required/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Connected" })).toBeInTheDocument();
  });

  it("renders null R² and a complete 90-day window explicitly", async () => {
    const r2Metric = {
      ...metric,
      metric: "R2",
      label: "R²",
      value: null,
      status: "insufficient_data",
      severity: "not_available",
      warning: 0.5,
      critical: 0.2,
      direction: "lower",
    };
    const completeReport = {
      ...monitoringReport,
      windows: {
        "30": {
          ...monitoringReport.windows["30"],
          performance: [metric, r2Metric],
        },
        "90": {
          ...monitoringReport.windows["30"],
          window_days: 90,
          sample_count: 90,
          calendar_start: "2026-01-01",
          performance: [metric],
        },
      },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | URL | Request) => {
        const url = String(input);
        if (url.endsWith("/latest")) {
          return Promise.resolve(
            jsonResponse({ ...availableLatest, report: completeReport }),
          );
        }
        return Promise.resolve(responseFor(url));
      }),
    );
    render(<App />);

    const r2Row = (await screen.findByRole("rowheader", { name: "R²" }))
      .parentElement;
    expect(r2Row).not.toBeNull();
    expect(within(r2Row!).getByText("Not available")).toBeInTheDocument();
    expect(within(r2Row!).getByText("insufficient data")).toBeInTheDocument();
    expect(screen.getByText(/90 samples, as issued/)).toBeInTheDocument();
  });

  it("renders opened, escalated, and resolved alert history", async () => {
    const alerts = [
      alert,
      {
        ...alert,
        alert_event_id: "b".repeat(64),
        event_type: "escalated",
        severity: "critical",
        previous_alert_event_id: alert.alert_event_id,
      },
      {
        ...alert,
        alert_event_id: "c".repeat(64),
        event_type: "resolved",
        severity: "ok",
        previous_alert_event_id: "b".repeat(64),
      },
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | URL | Request) => {
        const url = String(input);
        if (url.endsWith("/history")) {
          return Promise.resolve(
            jsonResponse({
              ...availableHistory,
              alerts: { ...availableHistory.alerts, items: alerts, total: 3 },
            }),
          );
        }
        return Promise.resolve(responseFor(url));
      }),
    );
    render(<App />);

    expect(await screen.findByText("escalated")).toBeInTheDocument();
    expect(screen.getByText("resolved")).toBeInTheDocument();
    expect(screen.getAllByText("opened")).not.toHaveLength(0);
  });

  it("shows a newer failed reporting attempt separately from the last report", async () => {
    const failedRun = {
      ...run,
      run_id: "failed-run",
      status: "failed",
      report_id: null,
      failure: {
        failed_at_utc: "2026-04-08T12:01:00Z",
        error_type: "MonitoringReportingError",
        message: "The reporting attempt failed. Inspect local operator logs.",
      },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | URL | Request) => {
        const url = String(input);
        if (url.endsWith("/latest")) {
          return Promise.resolve(
            jsonResponse({ ...availableLatest, latest_attempt: failedRun }),
          );
        }
        return Promise.resolve(responseFor(url));
      }),
    );
    render(<App />);

    const card = (await screen.findByRole("heading", {
      name: "Latest reporting attempt",
    })).parentElement;
    expect(card).not.toBeNull();
    expect(within(card!).getByText("failed")).toBeInTheDocument();
    expect(within(card!).getByText(/Inspect local operator logs/)).toBeInTheDocument();
  });

  it("loads a selected reporting run without exposing local paths", async () => {
    const fetchMock = vi.fn((input: string | URL | Request) =>
      Promise.resolve(responseFor(String(input))),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    const selector = await screen.findByLabelText("Inspect reporting run");
    fireEvent.change(selector, { target: { value: "report-run-1" } });

    expect(await screen.findByText(/Report rrrrrrrrrrrr/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/api/v1/monitoring/runs/report-run-1",
      expect.objectContaining({ method: "GET" }),
    );
    expect(screen.getAllByRole("heading", { name: "30 days" })).toHaveLength(2);
    expect(screen.getByRole("heading", { name: "Alerts active for this report" }))
      .toBeInTheDocument();
    expect(document.body).not.toHaveTextContent("C:\\");
  });

  it("shows a sanitized not-found state for an unknown reporting run", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | URL | Request) => {
        const url = String(input);
        if (url.includes("/api/v1/monitoring/runs/")) {
          return Promise.resolve(jsonResponse({ detail: "not found" }, 404));
        }
        return Promise.resolve(responseFor(url));
      }),
    );
    render(<App />);

    const selector = await screen.findByLabelText("Inspect reporting run");
    fireEvent.change(selector, { target: { value: "report-run-1" } });

    expect(
      await screen.findByText("The monitoring request failed with HTTP 404."),
    ).toBeInTheDocument();
  });

  it("cancels an older run request and ignores its late response", async () => {
    const secondRun = {
      ...run,
      run_id: "report-run-2",
      report_id: "2".repeat(64),
    };
    const secondReport = {
      ...monitoringReport,
      report_id: secondRun.report_id,
      reporting_run_id: secondRun.run_id,
      as_of_date: "2026-04-01",
    };
    let resolveFirst:
      | ((response: Response | PromiseLike<Response>) => void)
      | undefined;
    let firstSignal: AbortSignal | undefined;
    const fetchMock = vi.fn(
      (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/history")) {
          return Promise.resolve(
            jsonResponse({
              ...availableHistory,
              runs: {
                ...availableHistory.runs,
                items: [run, secondRun],
                total: 2,
              },
            }),
          );
        }
        if (url.endsWith(`/runs/${run.run_id}`)) {
          firstSignal = init?.signal ?? undefined;
          return new Promise<Response>((resolve) => {
            resolveFirst = resolve;
          });
        }
        if (url.endsWith(`/runs/${secondRun.run_id}`)) {
          return Promise.resolve(
            jsonResponse({
              state: "available",
              mode: "retrospective_historical_batch_not_real_time",
              run: secondRun,
              report: secondReport,
            }),
          );
        }
        return Promise.resolve(responseFor(url));
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    const selector = await screen.findByLabelText("Inspect reporting run");
    fireEvent.change(selector, { target: { value: run.run_id } });
    fireEvent.change(selector, { target: { value: secondRun.run_id } });

    expect(await screen.findByText(/Report 222222222222/)).toBeInTheDocument();
    expect(firstSignal?.aborted).toBe(true);
    resolveFirst?.(
      jsonResponse({
        state: "available",
        mode: "retrospective_historical_batch_not_real_time",
        run,
        report: monitoringReport,
      }),
    );
    await Promise.resolve();
    expect(screen.queryByText(/Report rrrrrrrrrrrr/)).not.toBeInTheDocument();
  });

  it("aborts overview requests when leaving the monitoring view", () => {
    const signals: AbortSignal[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.includes("/api/v1/monitoring/")) {
          if (init?.signal) signals.push(init.signal);
          return new Promise<Response>(() => undefined);
        }
        return Promise.resolve(responseFor(url));
      }),
    );
    render(<App />);

    fireEvent.click(screen.getByRole("tab", { name: "Historical performance" }));

    expect(signals).toHaveLength(2);
    expect(signals.every((signal) => signal.aborted)).toBe(true);
  });

  it("preserves the historical performance dashboard behind its tab", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | URL | Request) =>
        Promise.resolve(responseFor(String(input))),
      ),
    );
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Historical performance" }));

    expect(
      await screen.findByRole("heading", { name: "Wind Energy Forecast Dashboard" }),
    ).toBeInTheDocument();
    const maeCard = screen.getByRole("heading", { level: 3, name: "MAE" }).parentElement;
    expect(maeCard).not.toBeNull();
    expect(within(maeCard!).getByText("12.3")).toBeInTheDocument();
  });

  it("supports arrow-key navigation between dashboard tabs", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => undefined)));
    render(<App />);

    const monitoringTab = screen.getByRole("tab", { name: "Monitoring" });
    fireEvent.keyDown(monitoringTab, { key: "ArrowRight" });

    expect(
      screen.getByRole("tab", { name: "Historical performance" }),
    ).toHaveAttribute("aria-selected", "true");
  });

  it("refreshes only on demand and surfaces a history error independently", async () => {
    let historyCalls = 0;
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const url = String(input);
      if (url.endsWith("/history")) {
        historyCalls += 1;
        return Promise.resolve(
          historyCalls === 1
            ? jsonResponse({ detail: "unavailable" }, 503)
            : jsonResponse(availableHistory),
        );
      }
      return Promise.resolve(responseFor(url));
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    expect(
      await screen.findByText("The monitoring request failed with HTTP 503."),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Connected" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));

    await waitFor(() => expect(historyCalls).toBe(2));
    expect(await screen.findByLabelText("Inspect reporting run")).toBeInTheDocument();
  });

  it("renders the responsive monitoring layout hooks at a mobile viewport", async () => {
    vi.stubGlobal("innerWidth", 375);
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | URL | Request) =>
        Promise.resolve(responseFor(String(input))),
      ),
    );
    render(<App />);

    expect(await screen.findByRole("heading", { name: "30 days" }))
      .toBeInTheDocument();
    expect(document.querySelector(".monitoring-summary-grid")).toBeInTheDocument();
    expect(document.querySelector(".monitoring-windows")).toBeInTheDocument();
    expect(document.querySelector(".monitoring-table")).toBeInTheDocument();
    expect(document.querySelector(".dashboard-tabs")).toBeInTheDocument();
  });
});
