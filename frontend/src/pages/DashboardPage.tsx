import { useRef, useState } from "react";
import type { KeyboardEvent } from "react";

import { HistoricalPerformancePage } from "./HistoricalPerformancePage";
import { MonitoringPage } from "./MonitoringPage";

type View = "monitoring" | "performance";

export function DashboardPage() {
  const [view, setView] = useState<View>("monitoring");
  const monitoringTab = useRef<HTMLButtonElement>(null);
  const performanceTab = useRef<HTMLButtonElement>(null);

  function handleTabKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    const nextView =
      event.key === "ArrowRight" || event.key === "End"
        ? "performance"
        : event.key === "ArrowLeft" || event.key === "Home"
          ? "monitoring"
          : null;
    if (nextView === null) return;
    event.preventDefault();
    setView(nextView);
    const nextTab =
      nextView === "monitoring" ? monitoringTab.current : performanceTab.current;
    nextTab?.focus();
  }

  return (
    <>
      <nav className="dashboard-tabs" aria-label="Dashboard views">
        <div role="tablist" aria-label="Dashboard views">
          <button
            ref={monitoringTab}
            id="monitoring-tab"
            type="button"
            role="tab"
            aria-selected={view === "monitoring"}
            aria-controls="monitoring-panel"
            tabIndex={view === "monitoring" ? 0 : -1}
            onClick={() => setView("monitoring")}
            onKeyDown={handleTabKeyDown}
          >
            Monitoring
          </button>
          <button
            ref={performanceTab}
            id="performance-tab"
            type="button"
            role="tab"
            aria-selected={view === "performance"}
            aria-controls="performance-panel"
            tabIndex={view === "performance" ? 0 : -1}
            onClick={() => setView("performance")}
            onKeyDown={handleTabKeyDown}
          >
            Historical performance
          </button>
        </div>
      </nav>
      <div
        id="monitoring-panel"
        role="tabpanel"
        aria-labelledby="monitoring-tab"
        hidden={view !== "monitoring"}
      >
        {view === "monitoring" && <MonitoringPage />}
      </div>
      <div
        id="performance-panel"
        role="tabpanel"
        aria-labelledby="performance-tab"
        hidden={view !== "performance"}
      >
        {view === "performance" && <HistoricalPerformancePage />}
      </div>
    </>
  );
}
