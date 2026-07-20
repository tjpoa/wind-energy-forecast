import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { PerformanceObservation } from "../types/api";

const numberFormatter = new Intl.NumberFormat("en-GB", {
  maximumFractionDigits: 2,
});

const axisNumberFormatter = new Intl.NumberFormat("en-GB", {
  maximumFractionDigits: 0,
  notation: "compact",
});

const dateFormatter = new Intl.DateTimeFormat("en-GB", {
  day: "2-digit",
  month: "short",
  timeZone: "UTC",
});

function formatDate(value: string): string {
  return dateFormatter.format(new Date(`${value}T00:00:00Z`));
}

function tooltipObservation(
  payload: readonly { readonly payload?: unknown }[] | undefined,
): PerformanceObservation | null {
  const value = payload?.[0]?.payload;
  if (typeof value !== "object" || value === null) {
    return null;
  }
  return value as PerformanceObservation;
}

interface PerformanceTooltipProps {
  readonly active?: boolean;
  readonly payload?: readonly { readonly payload?: unknown }[];
}

export function PerformanceTooltip({
  active,
  payload,
}: PerformanceTooltipProps) {
  const observation = tooltipObservation(payload);
  if (!active || !observation) {
    return null;
  }

  return (
    <div className="performance-tooltip">
      <p>
        <strong>Date</strong>
        <time dateTime={observation.date}>{observation.date}</time>
      </p>
      <p>
        <strong>Actual</strong>
        <span>{numberFormatter.format(observation.actual)}</span>
      </p>
      <p>
        <strong>Predicted</strong>
        <span>{numberFormatter.format(observation.predicted)}</span>
      </p>
      <p>
        <strong>Error</strong>
        <span>{numberFormatter.format(observation.error)}</span>
      </p>
    </div>
  );
}

interface PerformanceChartProps {
  readonly observations: readonly PerformanceObservation[];
}

function ErrorLegend() {
  return (
    <ul className="error-legend" aria-label="Error categories">
      <li><span className="error-legend__swatch error-legend__swatch--over" />Overprediction</li>
      <li><span className="error-legend__swatch error-legend__swatch--under" />Underprediction</li>
      <li><span className="error-legend__swatch error-legend__swatch--exact" />Exact</li>
    </ul>
  );
}

export function ProductionChart({ observations }: PerformanceChartProps) {
  const showDots = observations.length <= 20;

  return (
    <article className="chart-card">
      <div className="chart-card__heading">
        <h3>Actual and predicted production</h3>
        <p>Daily sum of 15-minute MW readings</p>
      </div>
      <div
        className="chart-container"
        role="img"
        aria-label="Time series chart comparing actual and predicted wind production"
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={[...observations]}
            margin={{ top: 12, right: 16, bottom: 8, left: 4 }}
          >
            <CartesianGrid stroke="#dce7e1" strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              minTickGap={24}
              tickFormatter={formatDate}
              tick={{ fill: "#526a62", fontSize: 12 }}
            />
            <YAxis
              tickFormatter={(value: number) => axisNumberFormatter.format(value)}
              tick={{ fill: "#526a62", fontSize: 12 }}
              width={54}
            />
            <Tooltip content={<PerformanceTooltip />} />
            <Legend wrapperStyle={{ paddingTop: 8 }} />
            <Line
              type="linear"
              dataKey="actual"
              name="Actual"
              stroke="#116149"
              strokeWidth={2.5}
              dot={showDots ? { r: 3 } : false}
              activeDot={{ r: 5 }}
              isAnimationActive={false}
            />
            <Line
              type="linear"
              dataKey="predicted"
              name="Predicted"
              stroke="#d66a3a"
              strokeWidth={2.5}
              strokeDasharray="7 5"
              dot={showDots ? { r: 3 } : false}
              activeDot={{ r: 5 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </article>
  );
}

export function ErrorChart({ observations }: PerformanceChartProps) {
  const largestAbsoluteError = Math.max(
    1,
    ...observations.map((observation) => Math.abs(observation.error)),
  );
  const overpredictionCount = observations.filter(
    (observation) => observation.error > 0,
  ).length;
  const underpredictionCount = observations.filter(
    (observation) => observation.error < 0,
  ).length;
  const exactPredictionCount = observations.length - overpredictionCount - underpredictionCount;

  return (
    <article className="chart-card">
      <div className="chart-card__heading">
        <h3>Signed forecast error</h3>
        <p>Predicted minus actual, on the same daily-sum scale</p>
      </div>
      <div
        className="chart-container chart-container--error"
        role="img"
        aria-label={`Bar chart of signed forecast errors: ${overpredictionCount} overpredictions, ${underpredictionCount} underpredictions, and ${exactPredictionCount} exact predictions`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={[...observations]}
            margin={{ top: 12, right: 16, bottom: 8, left: 4 }}
          >
            <CartesianGrid stroke="#dce7e1" strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              minTickGap={24}
              tickFormatter={formatDate}
              tick={{ fill: "#526a62", fontSize: 12 }}
            />
            <YAxis
              domain={[-largestAbsoluteError, largestAbsoluteError]}
              tickFormatter={(value: number) => axisNumberFormatter.format(value)}
              tick={{ fill: "#526a62", fontSize: 12 }}
              width={54}
            />
            <Tooltip content={<PerformanceTooltip />} />
            <Legend content={<ErrorLegend />} wrapperStyle={{ paddingTop: 8 }} />
            <ReferenceLine y={0} stroke="#17372f" strokeWidth={1.5} />
            <Bar
              dataKey="error"
              name="Forecast error"
              minPointSize={2}
              isAnimationActive={false}
            >
              {observations.map((observation) => (
                <Cell
                  key={observation.date}
                  fill={
                    observation.error > 0
                      ? "#d66a3a"
                      : observation.error < 0
                        ? "#347bb7"
                        : "#70857b"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </article>
  );
}
