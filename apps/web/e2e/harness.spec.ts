import { expect, test } from "@playwright/test";

test("renders the experiment observatory without backend data", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Signal Simulation Engine" }),
  ).toBeVisible();
  await expect(page.getByText("Experiment chat")).toBeVisible();
  await expect(page.getByText("Decision trace")).toBeVisible();
  await expect(page.getByText("Account inspector")).toBeVisible();
  await expect(page.getByRole("button", { name: "RUN" })).toBeVisible();
});

test("runs a complete experiment and drills into a recommendation", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await page.goto("/");
  await expect(page.getByText("ENGINE ONLINE")).toBeVisible({
    timeout: 15_000,
  });
  await page.getByRole("button", { name: "RUN" }).click();
  await expect(page.locator(".account-row").first()).toBeVisible({
    timeout: 90_000,
  });
  await expect(page.getByText("lightgbm-v1").first()).toBeVisible();
  await expect(page.getByText("Run completed").first()).toBeVisible();
  await page.setViewportSize({ width: 520, height: 900 });
  const closeInspector = page.getByRole("button", {
    name: "Close account inspector",
  });
  await expect(closeInspector).toBeVisible();
  await expect(page.getByText("ACCOUNT STORYCARD")).toBeVisible();
  await expect(page.getByText("MODEL CONSENSUS")).toBeVisible();
  await expect(page.getByText("ACTION SIMULATION")).toBeVisible();
  await expect(page.getByText(/ACCOUNT 01 \/ 25/)).toBeVisible();
  await page.getByRole("button", { name: "Next account" }).click();
  await expect(page.getByText(/ACCOUNT 02 \/ 25/)).toBeVisible();
});
