import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { DashboardPage } from "./pages/DashboardPage";
import { AboutPage } from "./pages/AboutPage";
import { HistoricalPerformancePage } from "./pages/HistoricalPerformancePage";
import { ModelOperationsPage } from "./pages/ModelOperationsPage";
import { OverviewPage } from "./pages/OverviewPage";
import { CopilotPage } from "./pages/CopilotPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<DashboardPage />}>
          <Route index element={<Navigate to="/overview" replace />} />
          <Route path="overview" element={<OverviewPage />} />
          <Route path="forecast-replay" element={<HistoricalPerformancePage />} />
          <Route path="model-operations" element={<ModelOperationsPage />} />
          <Route path="about" element={<AboutPage />} />
          <Route path="copilot" element={<CopilotPage />} />
          <Route path="*" element={<Navigate to="/overview" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
