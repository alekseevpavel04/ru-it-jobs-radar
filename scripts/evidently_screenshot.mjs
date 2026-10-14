// Screenshots of the local Evidently reports (docs/img/evidently_<name>.png), the summary and the first
// column cards: the HTML reports themselves stay local for their size (radar.monitor.drift).
//
//   node scripts/evidently_screenshot.mjs [--week 2026-10-05] [--only telegram,output_telegram]
//
// Playwright from site/node_modules (browsers in PLAYWRIGHT_BROWSERS_PATH).

import {existsSync, mkdirSync, readdirSync} from "node:fs";
import {resolve} from "node:path";
import {pathToFileURL} from "node:url";
import {parseArgs} from "node:util";
import {chromium} from "../site/node_modules/playwright/index.mjs";

const {values: args} = parseArgs({
  options: {
    week: {type: "string"},
    only: {type: "string"},
    "out-dir": {type: "string", default: "docs/img"},
    width: {type: "string", default: "1600"},
    height: {type: "string", default: "1560"}
  }
});

const root = resolve("reports/evidently");
const weeks = readdirSync(root).filter((d) => /^\d{4}-\d{2}-\d{2}$/.test(d)).sort();
const week = args.week ?? weeks.at(-1);
const names = args.only ? args.only.split(",") : ["telegram", "output_telegram"];
const outDir = resolve(args["out-dir"]);
mkdirSync(outDir, {recursive: true});

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: {width: Number(args.width), height: Number(args.height)},
  colorScheme: "light"
});
let failed = false;
for (const name of names) {
  const file = resolve(root, week, `${name}.html`);
  if (!existsSync(file)) {
    console.error(`no report ${file}`);
    failed = true;
    continue;
  }
  await page.goto(pathToFileURL(file).href);
  // The report renders its widgets from embedded JSON after load; plots are SVG.
  await page.locator("svg.main-svg").first().waitFor({timeout: 60_000});
  await page.waitForTimeout(2000);
  const out = resolve(outDir, `evidently_${name}.png`);
  await page.screenshot({path: out});
  console.log(out);
}
await browser.close();
process.exit(failed ? 1 : 0);
