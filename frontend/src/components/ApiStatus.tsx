import type { ApiStatusProps } from "../types/api";

const statusLabels = {
  "not-connected": "Not connected",
  connecting: "Connecting",
  connected: "Connected",
  unavailable: "Unavailable",
} as const;

export function ApiStatus({ baseUrl, status }: ApiStatusProps) {
  return (
    <section className="api-status" aria-labelledby="api-status-title">
      <div>
        <p className="eyebrow">API status</p>
        <h2 id="api-status-title">{statusLabels[status]}</h2>
      </div>
      <p className="api-status__endpoint">
        Endpoint: <span>{baseUrl ?? "Not configured"}</span>
      </p>
    </section>
  );
}
