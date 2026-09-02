import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CopilotPage } from "../pages/CopilotPage";

vi.mock("../api/config", () => ({
  apiConfig: { baseUrl: "http://api.test" },
}));

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("CopilotPage", () => {
  it("submits a suggestion and renders evidence and limitations", async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve(
        jsonResponse({
          route: "operational",
          mode: "guided_local",
          answer: {
            summary: "Deployment verificado.",
            evidence: [{ evidence_id: "e1" }],
          },
          limitations: ["Evidência histórica."],
          failure: null,
        }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<MemoryRouter><CopilotPage /></MemoryRouter>);

    fireEvent.click(screen.getByRole("button", { name: "Que deployment está ativo?" }));

    expect(await screen.findByText("Deployment verificado.")).toBeInTheDocument();
    expect(screen.getByText("1 registo(s) verificado(s).")).toBeInTheDocument();
    expect(screen.getByText("Evidência histórica.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("shows the replay link for the dedicated refusal and clears history", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(jsonResponse({
      route: "refused",
      mode: "guided_local",
      answer: null,
      limitations: ["Use Forecast Replay."],
      failure: {
        code: "forecast_replay_required",
        message: "Pergunta fora do catálogo.",
        retryable: false,
        evidence_state: "unsupported",
      },
    }))));
    render(<MemoryRouter><CopilotPage /></MemoryRouter>);

    fireEvent.change(screen.getByLabelText("Pergunta"), {
      target: { value: "Mostra previsões históricas" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Perguntar" }));

    expect(await screen.findByRole("link", { name: "Abrir Forecast Replay" }))
      .toHaveAttribute("href", "/forecast-replay");
    fireEvent.click(screen.getByRole("button", { name: "Limpar conversa" }));
    await waitFor(() => expect(screen.queryByText("Pergunta fora do catálogo."))
      .not.toBeInTheDocument());
  });

  it("surfaces a transport error", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("private"))));
    render(<MemoryRouter><CopilotPage /></MemoryRouter>);

    fireEvent.click(screen.getByRole("button", { name: "Há alertas ativos?" }));

    expect(await screen.findByText("The Copilot request could not be completed."))
      .toBeInTheDocument();
  });
});
