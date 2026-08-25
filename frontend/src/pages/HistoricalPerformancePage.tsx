import { useCallback, useEffect, useRef, useState } from "react";

import { apiConfig } from "../api/config";
import { getPerformance, PerformanceApiError } from "../api/performance";
import { ApiStatus } from "../components/ApiStatus";
import { DateRangeFilter } from "../components/DateRangeFilter";
import {
  ErrorChart,
  ProductionChart,
} from "../components/PerformanceCharts";
import { PerformanceMetrics } from "../components/PerformanceMetrics";
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
      <p className="scale-note">
        Production and error values are daily sums of 15-minute MW readings.
        They are not energy values in MWh.
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

export function HistoricalPerformancePage() {
  const [performanceState, setPerformanceState] =
    useState<PerformancePageState>({ status: "loading" });
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [availableStartDate, setAvailableStartDate] = useState<string | null>(
    null,
  );
  const [availableEndDate, setAvailableEndDate] = useState<string | null>(null);
  const activeRequest = useRef<AbortController | null>(null);
  const requestSequence = useRef(0);

  const loadPerformance = useCallback(
    (filters: { readonly startDate?: string; readonly endDate?: string } = {}) => {
      activeRequest.current?.abort();
      const controller = new AbortController();
      const requestId = ++requestSequence.current;
      activeRequest.current = controller;
      setPerformanceState({ status: "loading" });

      void getPerformance(apiConfig.baseUrl, {
        ...filters,
        signal: controller.signal,
      })
        .then((data) => {
          if (controller.signal.aborted || requestId !== requestSequence.current) {
            return;
          }

          setAvailableStartDate(data.interval.available_start_date);
          setAvailableEndDate(data.interval.available_end_date);
          setStartDate((current) => current || data.interval.available_start_date);
          setEndDate((current) => current || data.interval.available_end_date);
          setPerformanceState(
            data.observation_count === 0
              ? { status: "empty" }
              : { status: "success", data },
          );
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted || requestId !== requestSequence.current) {
            return;
          }

          if (error instanceof PerformanceApiError && error.status === 404) {
            setPerformanceState({ status: "empty" });
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
    },
    [],
  );

  useEffect(() => {
    loadPerformance();

    return () => activeRequest.current?.abort();
  }, [loadPerformance]);

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
          <h1>Forecast Replay</h1>
          <p className="dashboard-header__description">
            Replay historical predictions against actual production over a
            chosen date range. This is a retrospective holdout view.
          </p>
        </div>
        <ApiStatus baseUrl={apiConfig.baseUrl} status={connectionState} />
      </header>

      <section className="dashboard-section" aria-labelledby="filters-title">
        <div className="section-heading">
          <p className="eyebrow">Controls</p>
          <h2 id="filters-title">Filters</h2>
        </div>
        <DateRangeFilter
          startDate={startDate}
          endDate={endDate}
          availableStartDate={availableStartDate}
          availableEndDate={availableEndDate}
          isLoading={performanceState.status === "loading"}
          onStartDateChange={setStartDate}
          onEndDateChange={setEndDate}
          onUpdate={() => loadPerformance({ startDate, endDate })}
        />
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
        <section className="dashboard-section" aria-labelledby="charts-title">
          <div className="section-heading">
            <p className="eyebrow">Forecast</p>
            <h2 id="charts-title">Performance over time</h2>
          </div>
          <p className="scale-note">
            All production and error values use the daily sum of 15-minute MW
            readings. This scale does not represent MWh.
          </p>
          <div className="charts-grid">
            <ProductionChart observations={performanceState.data.observations} />
            <ErrorChart observations={performanceState.data.observations} />
          </div>
        </section>
      )}

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
    </main>
  );
}
