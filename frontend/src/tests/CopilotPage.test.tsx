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
  it("renders documentary provenance and a visible provider fallback", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(jsonResponse({
      route: "documentary",
      mode: "rag_local",
      answer: { summary: "Resumo local.", evidence: [{
        chunk_id: "readme:1", document_id: "readme", title: "README",
        heading: "Local Copilot", uri: "docs://wind-forecast/readme#1",
        sha256: "a".repeat(64),
        updated_at: "2026-09-03",
      }] },
      limitations: ["Corpus fechado."],
      failure: null,
      provider_failure: { code: "document_synthesis_failed", message: "Fallback local ativo." },
    }))));
    render(<MemoryRouter><CopilotPage /></MemoryRouter>);
    fireEvent.click(screen.getByRole("button", { name: "Qual é a metodologia e quais são as limitações?" }));
    expect(await screen.findByText("Resumo local.")).toBeInTheDocument();
    expect(screen.getByText("Fallback local ativo.")).toBeInTheDocument();
    expect(screen.getByText("docs://wind-forecast/readme#1")).toBeInTheDocument();
    expect(screen.getByText(`SHA-256: ${"a".repeat(64)}`)).toBeInTheDocument();
  });

  it("submits a suggestion and renders evidence and limitations", async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve(
        jsonResponse({
          route: "operational",
          mode: "guided_local",
          answer: {
            summary: "Deployment verificado.",
            facts: [{ fact_id: "f1", name: "monitoring.freshness", value: { status: "fresh" }, unit_or_scale: "status", as_of: "2026-09-03", evidence_ids: ["e1"] }],
            evidence: [{ evidence_id: "e1", source_kind: "load_model_era", record_id: "r1", effective_at: "2026-09-03", observed_at_utc: "2026-09-03T10:00:00Z", sha256: "b".repeat(64), domain: "deployment", schema_version: "v1" }],
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
    expect(screen.getByText("Fonte:")).toBeInTheDocument();
    expect(screen.getByText(/load_model_era/)).toBeInTheDocument();
    expect(screen.getByText(/2026-09-03T10:00:00Z/)).toBeInTheDocument();
    expect(screen.getByText("Freshness:").parentElement).toHaveTextContent("fresh");
    expect(screen.getByText("Evidência histórica.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("shows unavailable update dates without inventing a value", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(jsonResponse({
      route: "operational",
      mode: "guided_local",
      answer: { summary: "Estado verificado.", facts: [], evidence: [{ evidence_id: "e1", source_kind: "source", record_id: "r1", effective_at: "", observed_at_utc: null, sha256: "c".repeat(64), domain: "monitoring", schema_version: "v1" }] },
      limitations: [],
      failure: null,
    }))));
    render(<MemoryRouter><CopilotPage /></MemoryRouter>);
    fireEvent.click(screen.getByRole("button", { name: "Que deployment está ativo?" }));
    expect(await screen.findAllByText(/Data de atualização indisponível/)).not.toHaveLength(0);
  });

  it("keeps conversation state in memory and does not use localStorage", async () => {
    const storage = vi.spyOn(Storage.prototype, "setItem");
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(jsonResponse({
      route: "refused", mode: "guided_local", answer: null, limitations: [],
      failure: { code: "ambiguous_question", message: "Pergunta ambígua.", retryable: false, evidence_state: "unsupported" },
    }))));
    render(<MemoryRouter><CopilotPage /></MemoryRouter>);
    fireEvent.change(screen.getByLabelText("Pergunta"), { target: { value: "Qual é o estado operacional e a metodologia?" } });
    fireEvent.click(screen.getByRole("button", { name: "Perguntar" }));
    expect(await screen.findByText("Pergunta ambígua.")).toBeInTheDocument();
    expect(storage).not.toHaveBeenCalled();
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
