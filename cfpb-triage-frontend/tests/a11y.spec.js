import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const WCAG_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];
// The Gradio client resolves the Space on huggingface.co, then talks to *.hf.space.
const BACKEND = /hf\.space|huggingface\.co/;

async function expectNoViolations(page) {
  const results = await new AxeBuilder({ page }).withTags(WCAG_TAGS).analyze();
  const violations = results.violations.map(
    (violation) => `${violation.id}: ${violation.nodes.map((node) => node.target.join(" ")).join(" | ")}`,
  );
  expect(violations).toEqual([]);
}

test("while the backend wakes, the stored results meet WCAG 2.2 AA", async ({ page }) => {
  // Never answer the backend, so the app stays in its starting state with stored results.
  await page.route(BACKEND, () => {});
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Stored results while the system starts" }),
  ).toBeVisible();
  await expectNoViolations(page);
});

test("when the backend is unreachable, the error keeps stored results and meets WCAG 2.2 AA", async ({ page }) => {
  await page.route(BACKEND, (route) => route.abort());
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("Connection problem");
  await expect(page.getByRole("heading", { name: "Stored results from earlier runs" })).toBeVisible();
  await expectNoViolations(page);
});

test("a sleeping backend keeps the visitor waiting instead of showing an error", async ({ page }) => {
  // Every connection fails, as it does while a Space wakes, but the Hub says it is asleep.
  await page.route(BACKEND, (route) => route.abort());
  await page.route(/huggingface\.co\/api\/spaces\/[^/]+\/[^/?]+$/, (route) =>
    route.fulfill({ json: { runtime: { stage: "SLEEPING" } } }),
  );
  await page.goto("/");
  await expect(page.locator('li[aria-current="step"]')).toContainText("Waking the service");
  // The first attempt has failed by now; the app must still be waiting, not erroring.
  await page.waitForTimeout(3000);
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("requests to Hugging Face carry no Referer", async ({ page }) => {
  // Hugging Face refuses requests referred by some hosts (e.g. *.workers.dev), so none is sent.
  const referers = [];
  await page.route(BACKEND, (route) => {
    referers.push(route.request().headers()["referer"] ?? null);
    route.abort();
  });
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("Connection problem");
  expect(referers.length).toBeGreaterThan(0);
  expect(referers.filter(Boolean)).toEqual([]);
});

test("keyboard navigation starts at a visible skip link", async ({ page }) => {
  await page.route(BACKEND, () => {});
  await page.goto("/");
  await page.keyboard.press("Tab");
  const focused = page.locator(":focus");
  await expect(focused).toHaveText("Skip to content");
  await expect(focused).toBeVisible();
});

test("the page fits the screen without sideways scrolling", async ({ page }) => {
  await page.route(BACKEND, () => {});
  await page.goto("/");
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});

test("@live a real escalated result, with technical details, meets WCAG 2.2 AA", async ({ page }) => {
  test.setTimeout(300_000);
  await page.goto("/");
  await page.getByRole("button", { name: /Identity theft/ }).click({ timeout: 240_000 });
  await expect(page.locator("#result-title")).toBeFocused({ timeout: 60_000 });
  await expect(page.getByText("Human review")).toBeVisible();
  await expectNoViolations(page);
  await page.getByRole("button", { name: "Technical details" }).click();
  await expectNoViolations(page);
});
