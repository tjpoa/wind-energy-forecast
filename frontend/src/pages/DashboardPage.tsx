import { useEffect, useState } from "react";

import { apiConfig } from "../api/config";
import { getPerformance, PerformanceApiError } from "../api/performance";
import { ApiStatus } from "../components/ApiStatus";
import type { PerformanceResponse } from "../types/api";

type PerformancePageState =
  | { readonly status: "loading" }
  | { readonly status: "success"; readonly data: PerformanceResponse }
  | { readonly status: "empty" }
  | { readonly status: "error"; readonly message: string };

const numberFormatter = new Intl.NumberFormat("en-GB", {
  maximumFractionDigits: 2,
});

function formatNumber(value: number): string {
  return numberFormatter.format(value);
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <article className="metric-card">
      <h3>{label}</h3>
      <p>{value}</p>
    </article>
  );
}

function PerformanceMetrics({ data }: { data: PerformanceResponse }) {
  return (
    <div className="metrics-grid metrics-grid--performance">
      <MetricCard
        label="R²"
        value={data.metrics.r2 === null ? "Not available" : formatNumber(data.metrics.r2)}
      />
      <MetricCard label="MAE" value={formatNumber(data.metrics.mae)} />
      <MetricCard label="RMSE" value={formatNumber(data.metrics.rmse)} />
      <MetricCard
        label="MAPE"
        value={`${formatNumber(data.metrics.mape_percent)}%`}
      />
    </div>
  );
}

function PerformanceTable({ data }: { data: PerformanceResponse }) {
  const observations = data.observations.slice(-10);

  return (
    <>
      <p className="performance-summary">
        <strong>{data.observation_count}</strong> observations returned from{" "}
        <time dateTime={data.interval.returned_start_date}>
          {data.interval.returned_start_date}
        </time>{" "}
        to{" "}
        <time dateTime={data.interval.returned_end_date}>
          {data.interval.returned_end_date}
        </time>
        .
      </p>
      <div className="performance-table-wrapper">
        <table className="performance-table">
          <caption>
            Showing the {observations.length} most recent of{" "}
            {data.observation_count} observations.
          </caption>
          <thead>
            <tr>
              <th scope="col">Date</th>
              <th scope="col">Actual</th>
              <th scope="col">Predicted</th>
              <th scope="col">Error</th>
              <th scope="col">Absolute error</th>
            </tr>
          </thead>
          <tbody>
            {observations.map((observation) => (
              <tr key={observation.date}>
                <th scope="row">
                  <time dateTime={observation.date}>{observation.date}</time>
                </th>
                <td>{formatNumber(observation.actual)}</td>
                <td>{formatNumber(observation.predicted)}</td>
                <td>{formatNumber(observation.error)}</td>
                <td>{formatNumber(observation.absolute_error)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export function DashboardPage() {
  const [performanceState, setPerformanceState] =
    useState<PerformancePageState>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    void getPerformance(apiConfig.baseUrl, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) {
          return;
        }

        setPerformanceState(
          data.observation_count === 0
            ? { status: "empty" }
            : { status: "success", data },
        );
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }

        setPerformanceState({
          status: "error",
          message:
            error instanceof PerformanceApiError
              ? error.message
              : "Performance data could not be loaded.",
        });
      });

    return () => controller.abort();
  }, []);

  const connectionState =
    performanceState.status === "loading"
      ? "connecting"
      : performanceState.status === "error"
        ? "unavailable"
        : "connected";

  return (
    <main className="dashboard-shell">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">Portuguese wind energy</p>
          <h1>Wind Energy Forecast Dashboard</h1>
          <p className="dashboard-header__description">
            A focused workspace for exploring production forecasts and the
            conditions behind them.
          </p>
        </div>
        <ApiStatus baseUrl={apiConfig.baseUrl} status={connectionState} />
      </header>

      <section className="dashboard-section" aria-labelledby="filters-title">
        <div className="section-heading">
          <p className="eyebrow">Controls</p>
          <h2 id="filters-title">Filters</h2>
        </div>
        <div className="placeholder placeholder--compact">
          <p>Forecast filters will be available here.</p>
        </div>
      </section>

      <section className="dashboard-section" aria-labelledby="metrics-title">
        <div className="section-heading">
          <p className="eyebrow">Overview</p>
          <h2 id="metrics-title">Metrics</h2>
        </div>
        {performanceState.status === "loading" && (
          <div className="performance-state" role="status">
            <p>Loading performance data…</p>
          </div>
        )}
        {performanceState.status === "error" && (
          <div className="performance-state performance-state--error" role="alert">
            <p>{performanceState.message}</p>
          </div>
        )}
        {performanceState.status === "empty" && (
          <div className="performance-state" role="status">
            <p>No performance observations are available.</p>
          </div>
        )}
        {performanceState.status === "success" && (
          <PerformanceMetrics data={performanceState.data} />
        )}
      </section>

      {performanceState.status === "success" && (
        <section
          className="dashboard-section"
          aria-labelledby="performance-table-title"
        >
          <div className="section-heading">
            <p className="eyebrow">History</p>
            <h2 id="performance-table-title">Performance observations</h2>
          </div>
          <PerformanceTable data={performanceState.data} />
        </section>
      )}

      <section className="dashboard-section" aria-labelledby="chart-title">
        <div className="section-heading">
          <p className="eyebrow">Forecast</p>
          <h2 id="chart-title">Forecast chart</h2>
        </div>
        <div className="placeholder chart-placeholder">
          <p>The forecast visualization will be added in a future task.</p>
        </div>
      </section>
    </main>
  );
}
