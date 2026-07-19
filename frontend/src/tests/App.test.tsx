import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App from "../App";

describe("App", () => {
  it("renders the dashboard structure and neutral API state", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "Wind Energy Forecast Dashboard",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "Filters" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "Metrics" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "Forecast chart" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "Not connected" }),
    ).toBeInTheDocument();
  });
});
