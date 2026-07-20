import type { PerformanceResponse } from "../types/api";

const numberFormatter = new Intl.NumberFormat("en-GB", {
  maximumFractionDigits: 2,
});

function formatNumber(value: number): string {
  return numberFormatter.format(value);
}

interface MetricCardProps {
  readonly label: string;
  readonly value: string;
  readonly unit: string;
}

function MetricCard({ label, value, unit }: MetricCardProps) {
  return (
    <article className="metric-card">
      <h3>{label}</h3>
      <p>{value}</p>
      <span>{unit}</span>
    </article>
  );
}

export function PerformanceMetrics({ data }: { data: PerformanceResponse }) {
  const productionUnit = "Daily sum of 15-minute MW readings";

  return (
    <div className="metrics-grid metrics-grid--performance">
      <MetricCard
        label="MAE"
        value={formatNumber(data.metrics.mae)}
        unit={productionUnit}
      />
      <MetricCard
        label="RMSE"
        value={formatNumber(data.metrics.rmse)}
        unit={productionUnit}
      />
      <MetricCard
        label="R²"
        value={
          data.metrics.r2 === null
            ? "Not available"
            : formatNumber(data.metrics.r2)
        }
        unit="Dimensionless"
      />
      <MetricCard
        label="Observations"
        value={numberFormatter.format(data.observation_count)}
        unit="Daily records"
      />
    </div>
  );
}
