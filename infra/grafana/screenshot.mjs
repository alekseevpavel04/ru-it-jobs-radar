// Screenshots of the Grafana dashboards (docs/img/grafana_<name>.png), as an anonymous viewer.
// Prints the panels that show "No data" or an error and exits 1 if there are any.
//
//   node infra/grafana/screenshot.mjs [--only data,airflow] [--from now-3h] [--out-dir docs/img]
//
// Needs the mlops profile (Grafana on 127.0.0.1:3030) and Playwright from site/node_modules
// (browsers in PLAYWRIGHT_BROWSERS_PATH).

import {mkdirSync} from "node:fs";
import {resolve} from "node:path";
import {parseArgs} from "node:util";
import {chromium} from "../../site/node_modules/playwright/index.mjs";

// File name -> dashboard uid (infra/grafana/dashboards).
const DASHBOARDS = {data: "radar-data", api: "radar-api", airflow: "radar-airflow"};

const {values: args} = parseArgs({
  options: {
    url: {type: "string", default: "http://localhost:3030"},
    only: {type: "string"},
    from: {type: "string", default: "now-3h"},
    "out-dir": {type: "string", default: "docs/img"},
    width: {type: "string", default: "1600"}
  }
});

const names = args.only ? args.only.split(",") : Object.keys(DASHBOARDS);
for (const name of names) {
  if (!DASHBOARDS[name]) {
    console.error(`unknown dashboard ${name}; known: ${Object.keys(DASHBOARDS).join(", ")}`);
    process.exit(2);
  }
}
const outDir = resolve(args["out-dir"]);
mkdirSync(outDir, {recursive: true});

const PANEL = "[data-viz-panel-key]";

async function problems(page) {
  return page.evaluate((selector) =>
    [...document.querySelectorAll(selector)]
      .map((p) => {
        const title = p.querySelector("h2")?.textContent?.trim() ?? "?";
        if (p.querySelector('[data-testid*="Panel status error"]')) return `${title}: error`;
        return /No data/.test(p.innerText) ? `${title}: no data` : null;
      })
      .filter(Boolean),
    PANEL
  );
}

const browser = await chromium.launch();
let failed = false;
try {
  const width = Number(args.width);
  const context = await browser.newContext({viewport: {width, height: 1000}, deviceScaleFactor: 1, locale: "ru-RU"});
  const page = await context.newPage();
  for (const name of names) {
    await page.setViewportSize({width, height: 1000});
    await page.goto(`${args.url}/d/${DASHBOARDS[name]}?orgId=1&kiosk&from=${args.from}&to=now`);
    await page.locator(PANEL).first().waitFor({timeout: 60_000});
    // Panels below the fold query only when visible: grow the viewport to the whole dashboard.
    const height = await page.evaluate(() => document.querySelector(".react-grid-layout")?.scrollHeight ?? 1000);
    await page.setViewportSize({width, height: height + 80});
    await page.waitForLoadState("networkidle");
    await page.waitForTimeout(1500);
    const found = await problems(page);
    const out = `${outDir}/grafana_${name}.png`;
    await page.screenshot({path: out, fullPage: true});
    if (found.length) {
      failed = true;
      console.error(`${name}: ${found.join("; ")}`);
    }
    console.log(`${out} (${width}x${height + 80})`);
  }
} finally {
  await browser.close();
}
process.exit(failed ? 1 : 0);
