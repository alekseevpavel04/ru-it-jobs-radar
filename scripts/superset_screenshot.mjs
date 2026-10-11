// Screenshots of the Superset dashboards (docs/img/superset_<name>.png). Signs in as admin, opens each
// dashboard, waits until every chart has rendered and fails when a chart shows an error or no data.
//
//   SUPERSET_ADMIN_PASSWORD=... node scripts/superset_screenshot.mjs [--only market] [--out-dir docs/img]
//
// Needs the local Superset loaded by scripts/superset_import.sh and Playwright from site/node_modules
// (browsers in PLAYWRIGHT_BROWSERS_PATH).

import {mkdirSync} from "node:fs";
import {resolve} from "node:path";
import {parseArgs} from "node:util";
import {chromium} from "../site/node_modules/playwright/index.mjs";

// File name -> dashboard slug (infra/superset/assets/dashboards).
const DASHBOARDS = {
  market: "radar-market",
  skills: "radar-skills",
  salaries: "radar-salaries",
  junior: "radar-junior",
  lifetime: "radar-lifetime"
};

const {values: args} = parseArgs({
  options: {
    url: {type: "string", default: "http://localhost:8088"},
    user: {type: "string", default: "admin"},
    only: {type: "string"},
    "out-dir": {type: "string", default: "docs/img"},
    width: {type: "string", default: "1600"}
  }
});

const password = process.env.SUPERSET_ADMIN_PASSWORD;
if (!password) {
  console.error("SUPERSET_ADMIN_PASSWORD is not set");
  process.exit(1);
}
const names = args.only ? args.only.split(",") : Object.keys(DASHBOARDS);
for (const name of names) {
  if (!DASHBOARDS[name]) {
    console.error(`unknown dashboard ${name}; known: ${Object.keys(DASHBOARDS).join(", ")}`);
    process.exit(2);
  }
}
const outDir = resolve(args["out-dir"]);
mkdirSync(outDir, {recursive: true});

// Chart holders; Markdown blocks share the holder class but hold no .chart-slice.
const CHARTS = ".dashboard-component-chart-holder:has(.chart-slice)";

// Every chart has rendered something and no spinner is left.
async function waitForCharts(page) {
  await page.locator(CHARTS).first().waitFor({timeout: 60_000});
  await page.waitForFunction(
    (selector) => {
      const charts = [...document.querySelectorAll(selector)];
      const busy = charts.some((c) => c.querySelector('[data-test="loading-indicator"], .loading'));
      const drawn = charts.every((c) => c.querySelector("canvas, table, .alert, [role='alert'], .ant-empty"));
      return charts.length > 0 && !busy && drawn;
    },
    CHARTS,
    {timeout: 120_000, polling: 500}
  );
  // ECharts animates its first frame.
  await page.waitForTimeout(1500);
}

async function problems(page) {
  return page.evaluate((selector) =>
    [...document.querySelectorAll(selector)]
      .map((c) => {
        const title = c.querySelector("[data-test='editable-title'], .header-title")?.textContent?.trim() ?? "?";
        const text = c.innerText;
        if (/No results|Нет результатов|No data/i.test(text)) return `${title}: no data`;
        const error = c.querySelector(".alert-danger, [role='alert'], [data-test='error-message']");
        return error ? `${title}: ${error.textContent.trim().slice(0, 200)}` : null;
      })
      .filter(Boolean),
    CHARTS
  );
}

const browser = await chromium.launch();
let failed = false;
try {
  const width = Number(args.width);
  const context = await browser.newContext({viewport: {width, height: 1000}, deviceScaleFactor: 1, locale: "ru-RU"});
  const page = await context.newPage();
  await page.goto(`${args.url}/login/`);
  await page.locator('input[name="username"], #username').first().fill(args.user);
  await page.locator('input[name="password"], #password').first().fill(password);
  await page.locator('button[type="submit"], input[type="submit"]').first().click();
  await page.waitForURL((url) => !url.pathname.startsWith("/login"), {timeout: 30_000});

  for (const name of names) {
    await page.setViewportSize({width, height: 1000});
    await page.goto(`${args.url}/superset/dashboard/${DASHBOARDS[name]}/?standalone=1`);
    await waitForCharts(page);
    // Charts below the fold render only when scrolled into view: grow the viewport to the whole page.
    const height = await page.evaluate(() => document.documentElement.scrollHeight);
    await page.setViewportSize({width, height});
    await waitForCharts(page);
    const found = await problems(page);
    const out = `${outDir}/superset_${name}.png`;
    await page.screenshot({path: out, fullPage: true});
    if (found.length) {
      failed = true;
      console.error(`${name}: ${found.join("; ")}`);
    }
    console.log(`${out} (${width}x${height})`);
  }
} finally {
  await browser.close();
}
process.exit(failed ? 1 : 0);
