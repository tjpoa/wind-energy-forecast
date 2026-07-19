import { apiConfig } from "../api/config";
import { ApiStatus } from "../components/ApiStatus";

export function DashboardPage() {
  return (
    <main className="dashboard-shell">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">Portuguese wind energy</p>
          <h1>Wind Energy Forecast Dashboard</h1>
          <p className="dashboard-header__description">
            A focused workspace for exploring production forecasts and the
            conditions behind them.
          </p>
        </div>
        <ApiStatus baseUrl={apiConfig.baseUrl} status="not-connected" />
      </header>

      <section className="dashboard-section" aria-labelledby="filters-title">
        <div className="section-heading">
          <p className="eyebrow">Controls</p>
          <h2 id="filters-title">Filters</h2>
        </div>
        <div className="placeholder placeholder--compact">
          <p>Forecast filters will be available here.</p>
        </div>
      </section>

      <section className="dashboard-section" aria-labelledby="metrics-title">
        <div className="section-heading">
          <p className="eyebrow">Overview</p>
          <h2 id="metrics-title">Metrics</h2>
        </div>
        <div className="metrics-grid">
          <article className="placeholder metric-placeholder">
            <h3>Forecast output</h3>
            <p>Awaiting forecast data</p>
          </article>
          <article className="placeholder metric-placeholder">
            <h3>Wind conditions</h3>
            <p>Awaiting weather data</p>
          </article>
          <article className="placeholder metric-placeholder">
            <h3>Model performance</h3>
            <p>Awaiting model metrics</p>
          </article>
        </div>
      </section>

      <section className="dashboard-section" aria-labelledby="chart-title">
        <div className="section-heading">
          <p className="eyebrow">Forecast</p>
          <h2 id="chart-title">Forecast chart</h2>
        </div>
        <div className="placeholder chart-placeholder">
          <p>The forecast visualization will be added in a future task.</p>
        </div>
      </section>
    </main>
  );
}
