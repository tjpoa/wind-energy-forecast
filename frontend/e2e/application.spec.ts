import { expect, test } from "@playwright/test";

test("overview and model operations render packaged monitoring evidence", async ({
  page,
}) => {
  await page.goto("/overview");

  await expect(page.getByRole("heading", { name: "Forecast operations overview" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Connected" })).toBeVisible();
  await expect(page.getByText("demo-synthetic-wind-forecast · v1")).toBeVisible();
  await expect(page.getByText("demo-pipeline-20260824")).toBeVisible();
  await expect(page.getByText("2026-08-14", { exact: true })).toBeVisible();

  await page.getByRole("link", { name: "Model Operations" }).click();
  await expect(page).toHaveURL(/\/model-operations$/);
  await expect(page.getByRole("heading", { name: "Model Operations" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "30 days" })).toBeVisible();
  await expect(page.getByText("No active local alerts.")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Model lifecycle" })).toBeVisible();
});

test("forecast replay deep-link filters packaged predictions and actuals", async ({ page }) => {
  await page.goto("/forecast-replay");

  await expect(page.getByRole("heading", { name: "Forecast Replay" })).toBeVisible();
  await expect(page.getByText(/14 observations returned/)).toBeVisible();

  await page.getByLabel("Start date").fill("2026-08-05");
  await page.getByLabel("End date").fill("2026-08-07");
  await page.getByRole("button", { name: "Update dashboard" }).click();

  await expect(page.getByText(/3 observations returned/)).toBeVisible();
  const filteredRow = page.getByRole("row", { name: /2026-08-05/ });
  await expect(filteredRow).toContainText("1,140");
  await expect(filteredRow).toContainText("1,149");
  await expect(page.getByRole("row", { name: /2026-08-07/ })).toBeVisible();
});

test("keyless local Copilot answers a guided operational question", async ({ page }) => {
  await page.goto("/copilot");

  await expect(page.getByRole("heading", { name: "Copilot operacional" })).toBeVisible();
  await page.getByRole("button", { name: "Que deployment está ativo?" }).click();
  await expect(page.getByText("Modo: guided_local")).toBeVisible();
  await expect(page.locator(".copilot-card")).toContainText("Evidência:");
});
