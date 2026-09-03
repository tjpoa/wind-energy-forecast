import { useState } from "react";
import { Link } from "react-router-dom";
import { askCopilot, CopilotApiError, type CopilotResponse } from "../api/copilot";

const suggestions = [
  "Qual é o estado operacional verificado?",
  "Que deployment está ativo?",
  "Como estão a qualidade e a freshness dos dados?",
  "Qual foi a performance/MAE dos últimos 30 dias?",
  "Qual foi a performance/MAE dos últimos 90 dias?",
  "Há drift nos últimos 30 dias?",
  "Há alertas ativos?",
  "Que modelo está ativo e que metadados o identificam?",
  "Qual é a metodologia e quais são as limitações?",
  "Qual é o estado do roadmap do Copilot?",
  "Como executar o projeto localmente?",
];

export function CopilotPage() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<readonly { question: string; response?: CopilotResponse; error?: string }[]>([]);
  const [loading, setLoading] = useState(false);

  async function submit(value = question) {
    const text = value.trim();
    if (!text || text.length > 1000 || loading) return;
    setLoading(true);
    setQuestion(value);
    try {
      const response = await askCopilot(text);
      setHistory((items) => [...items, { question: text, response }]);
    } catch (error) {
      setHistory((items) => [...items, { question: text, error: error instanceof CopilotApiError ? error.message : "Não foi possível contactar o Copilot." }]);
    } finally {
      setLoading(false);
    }
  }

  return <main className="dashboard-shell copilot-page">
    <header className="dashboard-header"><div><p className="eyebrow">Local Copilot</p><h1>Copilot operacional</h1><p className="dashboard-header__description">Pergunte sobre o estado verificado do serviço e os dados locais.</p></div></header>
    <section className="dashboard-section"><label className="copilot-question" htmlFor="copilot-question">Pergunta</label><textarea id="copilot-question" maxLength={1000} value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ex.: Que deployment está ativo?" /><div className="copilot-compose"><span>{question.length}/1000</span><button className="primary-button" disabled={loading || !question.trim()} onClick={() => void submit()}>{loading ? "A consultar…" : "Perguntar"}</button></div><div className="copilot-suggestions"><p>Sugestões</p>{suggestions.map((suggestion) => <button className="secondary-button" key={suggestion} onClick={() => void submit(suggestion)}>{suggestion}</button>)}</div></section>
    <section className="copilot-history" aria-live="polite">{history.map((item, index) => <article className="dashboard-section copilot-card" key={`${item.question}-${index}`}><p className="copilot-question-label">Pergunta</p><p>{item.question}</p>{item.error ? <p className="copilot-error">{item.error}</p> : item.response && <ResponseCard response={item.response} />}</article>)}</section>
    {history.length > 0 && <button className="secondary-button" onClick={() => setHistory([])}>Limpar conversa</button>}
  </main>;
}

function ResponseCard({ response }: { response: CopilotResponse }) {
  if (response.route === "operational") return <><div className="copilot-mode">Modo: {response.mode}</div><p className="copilot-summary">{response.answer.summary ?? "Não existe um resumo disponível."}</p><Evidence answer={response.answer} limitations={response.limitations} /></>;
  if (response.route === "documentary") return <><div className="copilot-mode">Modo: {response.mode}</div>{response.provider_failure && <p className="copilot-warning">{response.provider_failure.message}</p>}<p className="copilot-summary">{response.answer.summary}</p><DocumentaryEvidenceList evidence={response.answer.evidence} /><Evidence answer={response.answer} limitations={response.limitations} /></>;
  return <><div className="copilot-mode">Modo: {response.mode}</div><p className="copilot-error">{response.failure.message}</p><ul>{response.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>{response.failure.code === "forecast_replay_required" && <Link to="/forecast-replay">Abrir Forecast Replay</Link>}</>;
}

function DocumentaryEvidenceList({ evidence }: { evidence: readonly import("../api/copilot").DocumentaryEvidence[] }) {
  return <ul className="copilot-document-evidence">{evidence.map((item) => <li key={item.chunk_id}><strong>{item.title}</strong> — {item.heading}<br /><code>{item.uri}</code><br /><small>SHA-256: {item.sha256}</small></li>)}</ul>;
}

function Evidence({ answer, limitations }: { answer: { readonly evidence: readonly unknown[] }; limitations: readonly string[] }) {
  return <><p><strong>Evidência:</strong> {answer.evidence.length ? `${answer.evidence.length} registo(s) verificado(s).` : "Não disponível."}</p>{limitations.length > 0 && <><strong>Limitações</strong><ul>{limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul></>}</>;
}
