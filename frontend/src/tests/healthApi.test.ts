import { afterEach, describe, expect, it, vi } from "vitest";

import { getHealth } from "../api/health";

function jsonResponse(payload: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(payload),
  } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("health API", () => {
  it("validates a healthy response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ status: "ok" }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(getHealth("http://api.test/")).resolves.toEqual({ status: "ok" });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/health",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("rejects an invalid health payload", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ status: "degraded" })));

    await expect(getHealth("http://api.test")).rejects.toMatchObject({
      kind: "invalid-response",
      status: 200,
    });
  });

  it("surfaces HTTP failures without treating them as healthy", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ detail: "down" }, 503)));

    await expect(getHealth("http://api.test")).rejects.toMatchObject({
      kind: "http",
      status: 503,
    });
  });
});
