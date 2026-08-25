import { NavLink, Outlet } from "react-router-dom";

export function DashboardPage() {
  return (
    <>
      <nav className="dashboard-tabs" aria-label="Application pages">
        <div>
          <NavLink to="/overview">Overview</NavLink>
          <NavLink to="/forecast-replay">Forecast Replay</NavLink>
          <NavLink to="/model-operations">Model Operations</NavLink>
          <NavLink to="/about">About</NavLink>
        </div>
      </nav>
      <Outlet />
    </>
  );
}
