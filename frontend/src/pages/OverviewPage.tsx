import { useCallback, useEffect, useRef, useState } from "react";

import { apiConfig } from "../api/config";
import { getHealth, HealthApiError } from "../api/health";
import {
  getMonitoringLatest,
  MonitoringApiError,
} from "../api/monitoring";
import { ApiStatus } from "../components/ApiStatus";
import type { MonitoringLatestResponse, MonitoringReport } from "../types/monitoring";

type HealthState =
  | { readonly status: "loading" }
  | { readonly status: "success" }
  | { readonly status: "error"; readonly message: string };

type MonitoringState =
  | { readonly status: "loading" }
  | { readonly status: "empty"; readonly data: MonitoringLatestResponse }
  | { readonly status: "success"; readonly data: MonitoringLatestResponse }
  | { readonly status: "error"; readonly message: string };

function apiConnectionState(
  health: HealthState,
): "connecting" | "connected" | "unavailable" {
  if (health.status === "loading") return "connecting";
  if (health.status === "error") return "unavailable";
  return "connected";
}

function statusLabel(value: string | null | undefined): string {
  return value?.replaceAll("_", " ") ?? "Not available";
}

function modelSummary(report: MonitoringReport | null): string {
  if (!report) return "Not available";
  const name = report.model_era.registered_model_name;
  const version = report.model_era.model_version;
  if (name && version) return `${name} · v${version}`;
  if (version) return `Version ${version}`;
  return report.model?.snapshot_id
    ? `Snapshot ${report.model.snapshot_id.slice(0, 12)}`
    : "Not available";
}

function modelStatus(report: MonitoringReport | null): string {
  if (!report) return "No verified report";
  return statusLabel(report.model?.status ?? report.model_era.association_kind);
}

function pipelineSummary(data: MonitoringLatestResponse): {
  readonly id: string;
  readonly status: string;
} {
  const reportPipeline = data.report?.source_pipeline;
  if (reportPipeline) {
    return { id: reportPipeline.run_id, status: reportPipeline.status };
  }
  const attempt = data.latest_attempt;
  return {
    id: attempt?.source_pipeline_run_id ?? "Not available",
    status: attempt?.source_pipeline_status ?? "Not available",
  };
}

export function OverviewPage() {
  const [health, setHealth] = useState<HealthState>({ status: "loading" });
  const [monitoring, setMonitoring] = useState<MonitoringState>({
    status: "loading",
  });
  const activeRequest = useRef<AbortController | null>(null);

  const loadOverview = useCallback(() => {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setHealth({ status: "loading" });
    setMonitoring({ status: "loading" });

    void getHealth(apiConfig.baseUrl, controller.signal)
      .then(() => {
        if (!controller.signal.aborted) setHealth({ status: "success" });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted || (error instanceof Error && error.name === "AbortError")) {
          return;
        }
        setHealth({
          status: "error",
          message:
            error instanceof HealthApiError
              ? error.message
              : "The API health check could not be completed.",
        });
      });

    void getMonitoringLatest(apiConfig.baseUrl, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        setMonitoring(
          data.state === "empty"
            ? { status: "empty", data }
            : { status: "success", data },
        );
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted || (error instanceof Error && error.name === "AbortError")) {
          return;
        }
        setMonitoring({
          status: "error",
          message:
            error instanceof MonitoringApiError
              ? error.message
              : "Historical monitoring data could not be loaded.",
        });
      });

  }, []);

  useEffect(() => {
    loadOverview();
    return () => activeRequest.current?.abort();
  }, [loadOverview]);

  const report =
    monitoring.status === "success" ? monitoring.data.report : null;
  const pipeline =
    monitoring.status === "success" || monitoring.status === "empty"
      ? pipelineSummary(monitoring.data)
      : { id: "Not available", status: "Not available" };
  const connectionState = apiConnectionState(health);

  return (
    <main className="dashboard-shell">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">Portuguese wind energy</p>
          <h1>Forecast operations overview</h1>
          <p className="dashboard-header__description">
            A compact view of the verified model, the latest historical batch,
            and the data watermark behind this demonstration.
          </p>
        </div>
        <ApiStatus baseUrl={apiConfig.baseUrl} status={connectionState} />
      </header>

      <section className="dashboard-section overview-hero" aria-labelledby="overview-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Overview</p>
            <h2 id="overview-title">Retrospective historical batch monitoring</h2>
          </div>
          <button
            className="secondary-button"
            type="button"
            onClick={loadOverview}
            disabled={health.status === "loading" || monitoring.status === "loading"}
          >
            {health.status === "loading" || monitoring.status === "loading"
              ? "Refreshing…"
              : "Refresh"}
          </button>
        </div>
        <p className="monitoring-banner__copy">
          This workspace presents delayed, as-issued evidence. It is not a
          real-time or ex-ante forecasting service.
        </p>
      </section>

      <section className="dashboard-section" aria-labelledby="overview-signals-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Verified signals</p>
            <h2 id="overview-signals-title">Pipeline and model state</h2>
          </div>
        </div>
        {monitoring.status === "loading" && (
          <div className="performance-state" role="status">
            <p>Loading verified monitoring evidence…</p>
          </div>
        )}
        {monitoring.status === "error" && (
          <div className="performance-state performance-state--error" role="alert">
            <p>{monitoring.message}</p>
          </div>
        )}
        {monitoring.status === "empty" && (
          <div className="performance-state" role="status">
            <p>No monitoring report is available yet. The API is connected.</p>
          </div>
        )}
        {(monitoring.status === "success" || monitoring.status === "empty") && (
          <div className="overview-grid">
            <article className="metric-card">
              <h3>Model version</h3>
              <p>{modelSummary(report)}</p>
              <span>{modelStatus(report)}</span>
            </article>
            <article className="metric-card">
              <h3>Last pipeline run</h3>
              <p className="metric-card__compact">{pipeline.id}</p>
              <span>{statusLabel(pipeline.status)}</span>
            </article>
            <article className="metric-card">
              <h3>Data watermark</h3>
              <p>{report?.freshness.watermark_date ?? "Not available"}</p>
              <span>{statusLabel(report?.freshness.status)}</span>
            </article>
            <article className="metric-card">
              <h3>Evidence health</h3>
              <p>{monitoring.status === "success" ? "Available" : "Empty"}</p>
              <span>
                {report?.as_of_date
                  ? `As of ${report.as_of_date}`
                  : "No verified report"}
              </span>
            </article>
          </div>
        )}
      </section>

      {health.status === "error" && (
        <section className="dashboard-section" aria-labelledby="health-error-title">
          <div className="performance-state performance-state--error" role="alert">
            <h2 id="health-error-title">API health check unavailable</h2>
            <p>{health.message}</p>
            <p className="monitoring-muted">
              Monitoring evidence is shown independently and may still be
              available from a separate request.
            </p>
          </div>
        </section>
      )}
    </main>
  );
}
