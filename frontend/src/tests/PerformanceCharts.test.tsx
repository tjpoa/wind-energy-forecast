import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  ErrorChart,
  PerformanceTooltip,
  ProductionChart,
} from "../components/PerformanceCharts";
import type { PerformanceObservation } from "../types/api";

const observations: readonly PerformanceObservation[] = [
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
];

describe("performance charts", () => {
  it.each([1, 2, 25])("renders production charts with %i observations", (count) => {
    const chartObservations = Array.from({ length: count }, (_, index) => ({
      ...observations[index % observations.length],
      date: `2026-01-${String(index + 1).padStart(2, "0")}`,
    }));

    render(<ProductionChart observations={chartObservations} />);

    expect(
      screen.getByRole("img", {
        name: "Time series chart comparing actual and predicted wind production",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", {
        level: 3,
        name: "Actual and predicted production",
      }),
    ).toBeInTheDocument();
  });

  it("describes both signs in the error chart", () => {
    render(<ErrorChart observations={observations} />);

    expect(
      screen.getByRole("img", {
        name: /1 overpredictions, 1 underpredictions, and 0 exact predictions/,
      }),
    ).toBeInTheDocument();
  });

  it("renders a single exact prediction without collapsing the error scale", () => {
    const exactObservation: PerformanceObservation = {
      ...observations[0],
      predicted: observations[0].actual,
      error: 0,
      absolute_error: 0,
    };

    render(<ErrorChart observations={[exactObservation]} />);

    expect(
      screen.getByRole("img", {
        name: /0 overpredictions, 0 underpredictions, and 1 exact predictions/,
      }),
    ).toBeInTheDocument();
  });

  it("shows date, actual, predicted, and signed error in the tooltip", () => {
    render(
      <PerformanceTooltip
        active
        payload={[{ payload: observations[0] }]}
      />,
    );

    const tooltip = screen
      .getByText("Error")
      .closest<HTMLElement>(".performance-tooltip");
    expect(tooltip).not.toBeNull();
    if (!tooltip) {
      throw new Error("Expected the performance tooltip to be rendered.");
    }
    expect(within(tooltip).getByText("2026-01-01")).toBeInTheDocument();
    expect(within(tooltip).getByText("Actual")).toBeInTheDocument();
    expect(within(tooltip).getByText("Predicted")).toBeInTheDocument();
    expect(within(tooltip).getByText("100")).toBeInTheDocument();
    expect(within(tooltip).getByText("90")).toBeInTheDocument();
    expect(within(tooltip).getByText("-10")).toBeInTheDocument();
  });
});
