import { Link } from "react-router-dom";

export function AboutPage() {
  return (
    <main className="dashboard-shell">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">Project notes</p>
          <h1>About this application</h1>
          <p className="dashboard-header__description">
            The decisions, contracts, and limitations behind the Portuguese
            wind-energy forecast demonstration.
          </p>
        </div>
      </header>

      <section className="dashboard-section" aria-labelledby="architecture-title">
        <div className="section-heading">
          <p className="eyebrow">Architecture</p>
          <h2 id="architecture-title">A read-only evidence path</h2>
        </div>
        <div className="architecture-flow" aria-label="Application architecture">
          <article className="architecture-step">
            <span>01</span>
            <h3>React + TypeScript</h3>
            <p>
              Vite serves the accessible application shell, page routes, charts,
              and explicit loading and failure states.
            </p>
          </article>
          <div className="architecture-arrow" aria-hidden="true">→</div>
          <article className="architecture-step">
            <span>02</span>
            <h3>FastAPI + Pydantic</h3>
            <p>
              Typed, read-only HTTP contracts validate health, monitoring
              projections, and historical performance requests.
            </p>
          </article>
          <div className="architecture-arrow" aria-hidden="true">→</div>
          <article className="architecture-step">
            <span>03</span>
            <h3>Verified artifacts</h3>
            <p>
              Local Compose mounts the selected evidence read-only; the Azure
              demo bakes the synthetic bundle into an immutable image digest.
            </p>
          </article>
        </div>
      </section>

      <section className="dashboard-section" aria-labelledby="contracts-title">
        <div className="section-heading">
          <p className="eyebrow">Data contract</p>
          <h2 id="contracts-title">What the pages mean</h2>
        </div>
        <div className="about-grid">
          <article className="about-card">
            <h3>Overview and Model Operations</h3>
            <p>
              These pages project the historical batch monitoring contract:
              source freshness, a validated watermark, report-scoped model
              attribution, rolling errors, feature drift, and immutable alert
              or run history.
            </p>
          </article>
          <article className="about-card">
            <h3>Forecast Replay</h3>
            <p>
              Replay reads the unchanged performance endpoint and compares
              predicted and actual daily sums of 15-minute MW observations.
              The values are not MWh and are not future forecasts.
            </p>
          </article>
          <article className="about-card">
            <h3>Demo provenance</h3>
            <p>
              The clean-clone Compose experience uses the deterministic,
              synthetic <code>demo-v1</code> bundle. It requires no credentials
              or network calls and is not a REN or ERA5-Land release.
            </p>
          </article>
        </div>
      </section>

      <section className="dashboard-section" aria-labelledby="limitations-title">
        <div className="section-heading">
          <p className="eyebrow">Boundaries</p>
          <h2 id="limitations-title">Technology decisions and limitations</h2>
        </div>
        <ul className="about-list">
          <li>React, TypeScript, Vite, and Recharts provide the frontend surface.</li>
          <li>FastAPI and Pydantic keep the browser contract typed and sanitized.</li>
          <li>Docker Compose and Nginx package the local demonstration; Azure Container Apps can host the synthetic portfolio demo.</li>
          <li>Monitoring is retrospective and read-only, with no real-time or ex-ante path.</li>
          <li>The cloud path is not a production forecast service and has no automatic retraining, registry serving, or external alert delivery.</li>
        </ul>
        <p className="about-next-step">
          Explore the evidence in <Link to="/overview">Overview</Link>, or use
          <Link to="/forecast-replay"> Forecast Replay</Link> to inspect the
          historical holdout.
        </p>
      </section>
    </main>
  );
}
