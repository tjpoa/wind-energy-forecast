import { StrictMode } from "react";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "../App";

vi.mock("../api/config", () => ({
  apiConfig: { baseUrl: "http://api.test" },
}));

const validPayload = {
  interval: {
    requested_start_date: null,
    requested_end_date: null,
    available_start_date: "2026-01-01",
    available_end_date: "2026-01-02",
    returned_start_date: "2026-01-01",
    returned_end_date: "2026-01-02",
  },
  observation_count: 2,
  metrics: {
    r2: 0.91,
    mae: 12.3,
    rmse: 18.4,
    mape_percent: 5.6,
  },
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

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve;
  });
  return { promise, resolve };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("App", () => {
  it("shows the dashboard structure and loading state during the initial request", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => undefined)));

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
      screen.getByRole("heading", { level: 2, name: "Forecast chart" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(
      "Loading performance data",
    );
    expect(
      screen.getByRole("heading", { level: 2, name: "Connecting" }),
    ).toBeInTheDocument();
  });

  it("renders API metrics, observations, and the connected state", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(validPayload)),
    );

    render(<App />);

    expect(
      await screen.findByRole("heading", { level: 2, name: "Connected" }),
    ).toBeInTheDocument();
    const maeCard = screen.getByRole("heading", { level: 3, name: "MAE" })
      .parentElement;
    expect(maeCard).not.toBeNull();
    expect(within(maeCard!).getByText("12.3")).toBeInTheDocument();
    expect(
      screen.getByRole("row", { name: /2026-01-02 120 130 10 10/ }),
    ).toBeInTheDocument();
    expect(screen.getByText(/observations returned from/)).toHaveTextContent(
      "2 observations returned",
    );
  });

  it("renders an empty state for a valid response without observations", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          ...validPayload,
          observation_count: 0,
          observations: [],
        }),
      ),
    );

    render(<App />);

    expect(
      await screen.findByText("No performance observations are available."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "Connected" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("renders an error state for an API error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "unavailable" }, 503)),
    );

    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The performance request failed with HTTP 503.",
    );
    expect(
      screen.getByRole("heading", { level: 2, name: "Unavailable" }),
    ).toBeInTheDocument();
  });

  it("rejects an invalid success payload safely", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ ...validPayload, observation_count: 99 }),
      ),
    );

    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The API returned an invalid performance response.",
    );
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("shows only the ten most recent observations", async () => {
    const observations = Array.from({ length: 12 }, (_, index) => {
      const day = String(index + 1).padStart(2, "0");
      return {
        date: `2026-01-${day}`,
        actual: index,
        predicted: index + 1,
        error: 1,
        absolute_error: 1,
      };
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          ...validPayload,
          observation_count: observations.length,
          observations,
        }),
      ),
    );

    render(<App />);

    const table = await screen.findByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(11);
    expect(within(table).queryByText("2026-01-01")).not.toBeInTheDocument();
    expect(within(table).getByText("2026-01-03")).toBeInTheDocument();
    expect(within(table).getByText("2026-01-12")).toBeInTheDocument();
  });

  it("ignores a late response from the StrictMode cleanup request", async () => {
    const firstResponse = deferred<Response>();
    const secondResponse = deferred<Response>();
    const fetchMock = vi
      .fn()
      .mockImplementationOnce(() => firstResponse.promise)
      .mockImplementationOnce(() => secondResponse.promise);
    vi.stubGlobal("fetch", fetchMock);

    render(
      <StrictMode>
        <App />
      </StrictMode>,
    );

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    await act(async () => {
      secondResponse.resolve(
        jsonResponse({
          ...validPayload,
          observation_count: 0,
          observations: [],
        }),
      );
    });
    expect(
      await screen.findByText("No performance observations are available."),
    ).toBeInTheDocument();

    await act(async () => {
      firstResponse.resolve(jsonResponse(validPayload));
    });
    expect(
      screen.getByText("No performance observations are available."),
    ).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });
});
