import { useCallback, useEffect, useRef, useState } from "react";

import { apiConfig } from "../api/config";
import {
  getMonitoringHistory,
  getMonitoringLatest,
  getMonitoringRun,
  MonitoringApiError,
} from "../api/monitoring";
import { ApiStatus } from "../components/ApiStatus";
import type {
  MonitoringHistoryResponse,
  MonitoringLatestResponse,
  MonitoringReport,
  MonitoringRun,
  MonitoringRunResponse,
  MonitoringSeverity,
  MonitoringWindow,
} from "../types/monitoring";

type LatestState =
  | { readonly status: "loading" }
  | { readonly status: "empty"; readonly data: MonitoringLatestResponse }
  | { readonly status: "success"; readonly data: MonitoringLatestResponse }
  | { readonly status: "error"; readonly message: string };

type HistoryState =
  | { readonly status: "loading" }
  | { readonly status: "success"; readonly data: MonitoringHistoryResponse }
  | { readonly status: "error"; readonly message: string };

type RunState =
  | { readonly status: "idle" }
  | { readonly status: "loading" }
  | { readonly status: "success"; readonly data: MonitoringRunResponse }
  | { readonly status: "error"; readonly message: string };

const numberFormatter = new Intl.NumberFormat("en-GB", {
  maximumFractionDigits: 3,
});

function displayNumber(value: number | null): string {
  return value === null ? "Not available" : numberFormatter.format(value);
}

function severityClass(severity: MonitoringSeverity | string | null): string {
  return `status-chip status-chip--${severity ?? "not_available"}`;
}

function MonitoringWindowPanel({ window }: { window: MonitoringWindow }) {
  return (
    <article className="monitoring-window">
      <div className="monitoring-window__heading">
        <div>
          <p className="eyebrow">Rolling window</p>
          <h3>{window.window_days} days</h3>
        </div>
        <span className={severityClass(window.coverage_severity)}>
          {window.status.replaceAll("_", " ")}
        </span>
      </div>
      {window.status !== "available" ? (
        <p className="monitoring-muted">
          {window.status === "insufficient_data"
            ? `${window.sample_count} samples; ${window.minimum_samples ?? "more"} required.`
            : "No verified prediction ledger evidence is available for this window."}
        </p>
      ) : (
        <>
          <p className="monitoring-muted">
            {window.sample_count} samples, as issued
            {window.calendar_start && window.calendar_end
              ? ` (${window.calendar_start} to ${window.calendar_end})`
              : ""}
            .
          </p>
          <div className="performance-table-wrapper">
            <table className="performance-table monitoring-table">
              <caption>Performance compared with sealed-test v2 thresholds.</caption>
              <thead>
                <tr>
                  <th scope="col">Metric</th>
                  <th scope="col">Value</th>
                  <th scope="col">Warning</th>
                  <th scope="col">Critical</th>
                  <th scope="col">Direction</th>
                  <th scope="col">State</th>
                </tr>
              </thead>
              <tbody>
                {window.performance.length === 0 ? (
                  <tr>
                    <td colSpan={6}>Performance has insufficient actuals.</td>
                  </tr>
                ) : (
                  window.performance.map((metric) => (
                    <tr key={metric.metric}>
                      <th scope="row">{metric.label}</th>
                      <td>{displayNumber(metric.value)}</td>
                      <td>{displayNumber(metric.warning)}</td>
                      <td>{displayNumber(metric.critical)}</td>
                      <td>
                        {metric.metric === "bias"
                          ? "Absolute value above"
                          : metric.direction === "lower"
                            ? "Lower is worse"
                            : "Higher is worse"}
                      </td>
                      <td>
                        <span className={severityClass(metric.severity)}>
                          {metric.status === "available"
                            ? metric.severity.replaceAll("_", " ")
                            : metric.status.replaceAll("_", " ")}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          <div className="performance-table-wrapper monitoring-drift-table">
            <table className="performance-table monitoring-table">
              <caption>Top feature drift for this window.</caption>
              <thead>
                <tr>
                  <th scope="col">Feature</th>
                  <th scope="col">Comparator</th>
                  <th scope="col">Detector</th>
                  <th scope="col">Value / threshold</th>
                  <th scope="col">State</th>
                </tr>
              </thead>
              <tbody>
                {window.top_drift.length === 0 ? (
                  <tr>
                    <td colSpan={5}>No feature drift results are available.</td>
                  </tr>
                ) : (
                  window.top_drift.map((drift) => (
                    <tr key={drift.feature}>
                      <th scope="row">{drift.feature}</th>
                      <td>{drift.comparator}</td>
                      <td>{drift.detector.replaceAll("_", " ")}</td>
                      <td>{numberFormatter.format(drift.threshold_ratio)}×</td>
                      <td>
                        <span className={severityClass(drift.severity)}>
                          {drift.severity}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </article>
  );
}

function ReportSummary({
  report,
  latestAttempt,
}: {
  report: MonitoringReport;
  latestAttempt: MonitoringRun | null;
}) {
  const delayed = ["behind_objective", "late"].includes(report.freshness.status);
  return (
    <>
      {delayed && (
        <div className="delayed-banner" role="status">
          Source evidence is {report.freshness.status.replaceAll("_", " ")}.
          The API remains connected; this is a freshness state.
        </div>
      )}
      <section className="dashboard-section" aria-labelledby="monitoring-state-title">
        <div className="section-heading">
          <p className="eyebrow">Verified status</p>
          <h2 id="monitoring-state-title">Lifecycle, state and freshness</h2>
        </div>
        <div className="monitoring-summary-grid">
          <article className="metric-card">
            <h3>Source freshness</h3>
            <p className="metric-card__compact">
              {report.freshness.status.replaceAll("_", " ")}
            </p>
            <span>
              Watermark {report.freshness.watermark_date ?? "unknown"} · D+
              {report.freshness.objective_days}/D+{report.freshness.late_days}
            </span>
          </article>
          <article className="metric-card">
            <h3>Model version</h3>
            <p className="metric-card__compact">
              {report.model_era.model_version
                ? `v${report.model_era.model_version}`
                : report.model?.checksum.slice(0, 12) ?? "Not available"}
            </p>
            <span>
              {report.model
                ? `${report.model.model_type ?? "model"} · deployment ${report.model_era.deployment_id?.slice(0, 12) ?? "legacy"} · ${report.model_era.association_kind.replaceAll("_", " ")} · ${report.model.status === "champion" ? "champion" : "selected, not promoted"}`
                : "No verified model snapshot in the ledger"}
            </span>
          </article>
          <article className="metric-card">
            <h3>Source pipeline</h3>
            <p className="metric-card__compact">{report.source_pipeline.status}</p>
            <span>{report.source_pipeline.run_id}</span>
          </article>
          <article className="metric-card">
            <h3>Latest report</h3>
            <p className="metric-card__compact">Succeeded</p>
            <span>As of {report.as_of_date}</span>
          </article>
          <article className="metric-card">
            <h3>Latest reporting attempt</h3>
            <p className="metric-card__compact">
              {latestAttempt?.status ?? "Unknown"}
            </p>
            <span>
              {latestAttempt?.failure?.message ??
                latestAttempt?.attempted_at_utc ??
                "No run metadata"}
            </span>
          </article>
        </div>
        <p className="scale-note">
          Production and error values are daily sums of 15-minute MW readings.
          They are not energy values in MWh.
        </p>
      </section>
      <section className="dashboard-section" aria-labelledby="model-lifecycle-title">
        <div className="section-heading">
          <p className="eyebrow">Deployment attribution</p>
          <h2 id="model-lifecycle-title">Model lifecycle</h2>
        </div>
        <div className="overview-grid">
          <article className="metric-card">
            <h3>Registered model</h3>
            <p className="metric-card__compact">
              {report.model_era.registered_model_name ?? "Not available"}
            </p>
            <span>Report-scoped registry identity</span>
          </article>
          <article className="metric-card">
            <h3>Model version</h3>
            <p>{report.model_era.model_version ?? "Not available"}</p>
            <span>
              {report.model?.status === "champion"
                ? "Champion"
                : report.model?.status === "selected_not_promoted"
                  ? "Selected, not promoted"
                  : "No snapshot status"}
            </span>
          </article>
          <article className="metric-card">
            <h3>Association</h3>
            <p className="metric-card__compact">
              {report.model_era.association_kind.replaceAll("_", " ")}
            </p>
            <span>
              Deployment {report.model_era.deployment_id?.slice(0, 12) ?? "legacy"}
            </span>
          </article>
          <article className="metric-card">
            <h3>Generation</h3>
            <p>{report.model_era.deployment_generation ?? "Not available"}</p>
            <span>
              {report.model_era.cutoffs?.monitoring_evaluation_cutoff
                ? `Monitoring cutoff ${report.model_era.cutoffs.monitoring_evaluation_cutoff}`
                : "No cutoff recorded"}
            </span>
          </article>
        </div>
        <p className="scale-note">
          This is the existing sanitized monitoring projection. It does not
          claim a live registry or controlled-retraining lifecycle beyond the
          evidence attached to this report.
        </p>
      </section>
      <section className="dashboard-section" aria-labelledby="rolling-title">
        <div className="section-heading">
          <p className="eyebrow">Drift and performance</p>
          <h2 id="rolling-title">Rolling evidence</h2>
        </div>
        <div className="monitoring-windows">
          <MonitoringWindowPanel window={report.windows["30"]} />
          <MonitoringWindowPanel window={report.windows["90"]} />
        </div>
      </section>
      <section className="dashboard-section" aria-labelledby="active-alerts-title">
        <div className="section-heading">
          <p className="eyebrow">Current evidence</p>
          <h2 id="active-alerts-title">Active alerts</h2>
        </div>
        {report.active_alerts.length === 0 ? (
          <div className="performance-state">
            <p>No active local alerts.</p>
          </div>
        ) : (
          <div className="performance-table-wrapper">
            <table className="performance-table monitoring-table">
              <thead>
                <tr>
                  <th scope="col">Rule</th>
                  <th scope="col">Date</th>
                  <th scope="col">Event</th>
                  <th scope="col">Severity</th>
                </tr>
              </thead>
              <tbody>
                {report.active_alerts.map((alert) => (
                  <tr key={alert.alert_event_id}>
                    <th scope="row">{alert.rule_id}</th>
                    <td>{alert.through_date}</td>
                    <td>{alert.event_type}</td>
                    <td>
                      <span className={severityClass(alert.severity)}>
                        {alert.severity}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}

function SelectedRunDetail({ data }: { data: MonitoringRunResponse }) {
  return (
    <div className="selected-run-report">
      <div className="run-detail">
        <strong>{data.run.status}</strong>
        <span>
          Report {data.run.report_id?.slice(0, 12) ?? "not produced"} · source
          pipeline {data.run.source_pipeline_run_id ?? "unknown"}
        </span>
        {data.run.failure && <span>{data.run.failure.message}</span>}
      </div>
      {data.report && (
        <>
          <p className="monitoring-muted">
            Historical report as of {data.report.as_of_date}; source freshness is{" "}
            {data.report.freshness.status.replaceAll("_", " ")} and model{" "}
            {data.report.model?.checksum.slice(0, 12) ?? "is unavailable"}.
          </p>
          <div className="monitoring-windows">
            <MonitoringWindowPanel window={data.report.windows["30"]} />
            <MonitoringWindowPanel window={data.report.windows["90"]} />
          </div>
          <h3>Alerts active for this report</h3>
          {data.report.active_alerts.length === 0 ? (
            <p className="monitoring-muted">
              No alerts were active for this report.
            </p>
          ) : (
            <ul className="selected-run-alerts">
              {data.report.active_alerts.map((alert) => (
                <li key={alert.alert_event_id}>
                  {alert.through_date} · {alert.rule_id} · {alert.event_type} ·{" "}
                  {alert.severity}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}

export function MonitoringPage() {
  const [latest, setLatest] = useState<LatestState>({ status: "loading" });
  const [history, setHistory] = useState<HistoryState>({ status: "loading" });
  const [selectedRun, setSelectedRun] = useState("");
  const [runDetail, setRunDetail] = useState<RunState>({ status: "idle" });
  const activeOverview = useRef<AbortController | null>(null);
  const activeRun = useRef<AbortController | null>(null);
  const overviewSequence = useRef(0);
  const runSequence = useRef(0);

  const refresh = useCallback(() => {
    activeOverview.current?.abort();
    const controller = new AbortController();
    const requestId = ++overviewSequence.current;
    activeOverview.current = controller;
    setLatest({ status: "loading" });
    setHistory({ status: "loading" });

    void getMonitoringLatest(apiConfig.baseUrl, controller.signal)
      .then((data) => {
        if (controller.signal.aborted || requestId !== overviewSequence.current) return;
        setLatest(
          data.state === "empty"
            ? { status: "empty", data }
            : { status: "success", data },
        );
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted || requestId !== overviewSequence.current) return;
        setLatest({
          status: "error",
          message:
            error instanceof MonitoringApiError
              ? error.message
              : "Monitoring data could not be loaded.",
        });
      });

    void getMonitoringHistory(apiConfig.baseUrl, controller.signal)
      .then((data) => {
        if (controller.signal.aborted || requestId !== overviewSequence.current) return;
        setHistory({ status: "success", data });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted || requestId !== overviewSequence.current) return;
        setHistory({
          status: "error",
          message:
            error instanceof MonitoringApiError
              ? error.message
              : "Monitoring history could not be loaded.",
        });
      });
  }, []);

  useEffect(() => {
    refresh();
    return () => {
      activeOverview.current?.abort();
      activeRun.current?.abort();
    };
  }, [refresh]);

  const loadRun = useCallback((runId: string) => {
    setSelectedRun(runId);
    activeRun.current?.abort();
    if (!runId) {
      setRunDetail({ status: "idle" });
      return;
    }
    const controller = new AbortController();
    const requestId = ++runSequence.current;
    activeRun.current = controller;
    setRunDetail({ status: "loading" });
    void getMonitoringRun(apiConfig.baseUrl, runId, controller.signal)
      .then((data) => {
        if (controller.signal.aborted || requestId !== runSequence.current) return;
        setRunDetail({ status: "success", data });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted || requestId !== runSequence.current) return;
        setRunDetail({
          status: "error",
          message:
            error instanceof MonitoringApiError
              ? error.message
              : "Run detail could not be loaded.",
        });
      });
  }, []);

  const connectionState =
    latest.status === "loading"
      ? "connecting"
      : latest.status === "error"
        ? "unavailable"
        : "connected";

  return (
    <main className="dashboard-shell">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">Portuguese wind energy</p>
          <h1>Model Operations</h1>
          <p className="dashboard-header__description">
            Verified retrospective evidence for data freshness, drift, model
            errors, alerts, and report-scoped lifecycle state.
          </p>
        </div>
        <ApiStatus baseUrl={apiConfig.baseUrl} status={connectionState} />
      </header>

      <aside className="monitoring-banner" aria-label="Monitoring mode">
        <strong>Retrospective historical batch monitoring — not real time.</strong>
        <span>
          Evidence is refreshed only when this page opens or when you request it.
        </span>
        <button
          type="button"
          onClick={refresh}
          disabled={latest.status === "loading"}
        >
          {latest.status === "loading" ? "Refreshing…" : "Refresh"}
        </button>
      </aside>

      {latest.status === "loading" && (
        <section className="dashboard-section">
          <div className="performance-state" role="status">
            <p>Loading historical monitoring data…</p>
          </div>
        </section>
      )}
      {latest.status === "error" && (
        <section className="dashboard-section">
          <div className="performance-state performance-state--error" role="alert">
            <p>{latest.message}</p>
          </div>
        </section>
      )}
      {latest.status === "empty" && (
        <section className="dashboard-section" aria-labelledby="empty-title">
          <div className="section-heading">
            <p className="eyebrow">Connected</p>
            <h2 id="empty-title">No monitoring reports yet</h2>
          </div>
          <div className="performance-state" role="status">
            <p>{latest.data.message}</p>
          </div>
          {latest.data.latest_attempt && (
            <p className="monitoring-muted">
              Latest reporting attempt: {latest.data.latest_attempt.status}.
            </p>
          )}
        </section>
      )}
      {latest.status === "success" && latest.data.report && (
        <ReportSummary
          report={latest.data.report}
          latestAttempt={latest.data.latest_attempt}
        />
      )}

      <section className="dashboard-section" aria-labelledby="history-title">
        <div className="section-heading">
          <p className="eyebrow">Immutable history</p>
          <h2 id="history-title">Runs and alert events</h2>
        </div>
        {history.status === "loading" && (
          <div className="performance-state" role="status">
            <p>Loading monitoring history…</p>
          </div>
        )}
        {history.status === "error" && (
          <div className="performance-state performance-state--error" role="alert">
            <p>{history.message}</p>
          </div>
        )}
        {history.status === "success" && (
          <>
            <label className="run-selector">
              Inspect reporting run
              <select
                value={selectedRun}
                onChange={(event) => loadRun(event.target.value)}
              >
                <option value="">Select a run</option>
                {history.data.runs.items.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.through_date ?? "unknown date"} — {run.status}
                  </option>
                ))}
              </select>
            </label>
            {history.data.runs.items.length === 0 && (
              <p className="monitoring-muted">No reporting runs are available.</p>
            )}
            {runDetail.status === "loading" && <p role="status">Loading run detail…</p>}
            {runDetail.status === "error" && <p role="alert">{runDetail.message}</p>}
            {runDetail.status === "success" && (
              <SelectedRunDetail data={runDetail.data} />
            )}
            <div className="performance-table-wrapper monitoring-drift-table">
              <table className="performance-table monitoring-table">
                <caption>Causally ordered local alert history.</caption>
                <thead>
                  <tr>
                    <th scope="col">Date</th>
                    <th scope="col">Rule</th>
                    <th scope="col">Event</th>
                    <th scope="col">Severity</th>
                  </tr>
                </thead>
                <tbody>
                  {history.data.alerts.items.length === 0 ? (
                    <tr>
                      <td colSpan={4}>No alert events are available.</td>
                    </tr>
                  ) : (
                    history.data.alerts.items.map((alert) => (
                      <tr key={alert.alert_event_id}>
                        <td>{alert.through_date}</td>
                        <th scope="row">{alert.rule_id}</th>
                        <td>{alert.event_type}</td>
                        <td>
                          <span className={severityClass(alert.severity)}>
                            {alert.severity}
                          </span>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </main>
  );
}
