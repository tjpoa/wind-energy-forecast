import { afterEach, describe, expect, it, vi } from "vitest";

import { getPerformance } from "../api/performance";

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
  future_optional_field: "accepted",
};

function jsonResponse(payload: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(payload),
  } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.resetModules();
});

describe("API configuration", () => {
  it("reads and normalizes VITE_API_BASE_URL", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "  http://localhost:8000///  ");
    vi.resetModules();

    const { apiConfig } = await import("../api/config");

    expect(apiConfig.baseUrl).toBe("http://localhost:8000");
  });
});

describe("getPerformance", () => {
  it("requests the configured performance URL and returns a valid response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(validPayload));
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      getPerformance("http://localhost:8000///"),
    ).resolves.toEqual(validPayload);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/performance",
      {
        method: "GET",
        headers: { Accept: "application/json" },
        signal: undefined,
      },
    );
  });

  it("serializes one or both optional date filters", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(validPayload));
    vi.stubGlobal("fetch", fetchMock);

    await getPerformance("http://localhost:8000", {
      startDate: "2026-01-01",
    });
    await getPerformance("http://localhost:8000", {
      startDate: "2026-01-01",
      endDate: "2026-01-02",
    });
    await getPerformance("http://localhost:8000", {
      endDate: "2026-01-02",
    });

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://localhost:8000/api/v1/performance?start_date=2026-01-01",
      "http://localhost:8000/api/v1/performance?start_date=2026-01-01&end_date=2026-01-02",
      "http://localhost:8000/api/v1/performance?end_date=2026-01-02",
    ]);
  });

  it.each([
    ["0000-01-01", undefined, "Start date must be a valid date"],
    ["2026-02-30", undefined, "Start date must be a valid date"],
    [undefined, "2025-13-01", "End date must be a valid date"],
    ["01-01-2026", undefined, "Start date must be a valid date"],
    ["2026-01-02", "2026-01-01", "Start date cannot be later"],
  ])(
    "rejects invalid date ranges before fetch",
    async (startDate, endDate, message) => {
      const fetchMock = vi.fn();
      vi.stubGlobal("fetch", fetchMock);

      await expect(
        getPerformance("http://localhost:8000", { startDate, endDate }),
      ).rejects.toMatchObject({
        kind: "validation",
        status: null,
        message: expect.stringContaining(message),
      });
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );

  it("passes the supplied abort signal to fetch", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(validPayload));
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();

    await getPerformance("http://localhost:8000", {
      signal: controller.signal,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/performance",
      expect.objectContaining({ signal: controller.signal }),
    );
  });

  it("accepts optional result metadata and artifact metrics", async () => {
    const payloadWithResult = {
      ...validPayload,
      result: {
        model_type: "extra_trees",
        seed: 42,
        test_fraction: 0.2,
        dataset_version: "v1",
        evaluation_start_date: "2026-01-01",
        evaluation_end_date: "2026-01-02",
        artifact_metrics: validPayload.metrics,
      },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(payloadWithResult)),
    );

    await expect(
      getPerformance("http://localhost:8000"),
    ).resolves.toEqual(payloadWithResult);
  });

  it("rejects non-success HTTP responses without parsing their body", async () => {
    const response = jsonResponse({ detail: "not available" }, 503);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));

    await expect(getPerformance("http://localhost:8000")).rejects.toMatchObject(
      {
        name: "PerformanceApiError",
        kind: "http",
        status: 503,
        message: "The performance request failed with HTTP 503.",
      },
    );
    expect(response.json).not.toHaveBeenCalled();
  });

  it("rejects malformed JSON responses", async () => {
    const response = {
      ok: true,
      status: 200,
      json: vi.fn().mockRejectedValue(new SyntaxError("invalid JSON")),
    } as unknown as Response;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));

    await expect(getPerformance("http://localhost:8000")).rejects.toMatchObject(
      {
        kind: "invalid-response",
        status: 200,
      },
    );
  });

  it("preserves an abort raised while parsing the response", async () => {
    const abortError = new Error("aborted");
    abortError.name = "AbortError";
    const response = {
      ok: true,
      status: 200,
      json: vi.fn().mockRejectedValue(abortError),
    } as unknown as Response;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));

    await expect(getPerformance("http://localhost:8000")).rejects.toBe(
      abortError,
    );
  });

  it("rejects responses whose count and observations are inconsistent", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          jsonResponse({ ...validPayload, observation_count: 3 }),
        ),
    );

    await expect(getPerformance("http://localhost:8000")).rejects.toMatchObject(
      {
        kind: "invalid-response",
        status: 200,
      },
    );
  });

  it("wraps network failures in a stable error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));

    await expect(getPerformance("http://localhost:8000")).rejects.toMatchObject(
      {
        kind: "network",
        status: null,
        message: "The performance request could not be completed.",
      },
    );
  });

  it("rejects missing configuration without making a request", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(getPerformance(null)).rejects.toMatchObject({
      kind: "configuration",
      status: null,
      message: "The API base URL is not configured.",
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
