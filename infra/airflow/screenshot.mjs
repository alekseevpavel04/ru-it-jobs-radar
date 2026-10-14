// Screenshots of the Airflow UI (docs/img/airflow_<name>.png): the DAG list with recent runs and the
// grid of one DAG. Signs in through the simple auth manager as admin.
//
//   node --env-file=.env infra/airflow/screenshot.mjs [--dag radar_dbt] [--out-dir docs/img]
//
// Needs the orchestration profile (Airflow on 127.0.0.1:8090) and Playwright from site/node_modules
// (browsers in PLAYWRIGHT_BROWSERS_PATH).

import {mkdirSync} from "node:fs";
import {resolve} from "node:path";
import {parseArgs} from "node:util";
import {chromium} from "../../site/node_modules/playwright/index.mjs";

const {values: args} = parseArgs({
  options: {
    url: {type: "string", default: "http://localhost:8090"},
    user: {type: "string", default: "admin"},
    dag: {type: "string", default: "radar_dbt"},
    "out-dir": {type: "string", default: "docs/img"},
    width: {type: "string", default: "1600"}
  }
});

const password = process.env.AIRFLOW_ADMIN_PASSWORD;
if (!password) {
  console.error("AIRFLOW_ADMIN_PASSWORD is not set");
  process.exit(1);
}
const outDir = resolve(args["out-dir"]);
mkdirSync(outDir, {recursive: true});

const browser = await chromium.launch();
const page = await browser.newPage({viewport: {width: Number(args.width), height: 1000}, colorScheme: "light"});

await page.goto(`${args.url}/auth/login`);
await page.getByLabel(/username/i).fill(args.user);
await page.getByLabel(/password/i).fill(password);
await page.getByRole("button", {name: /sign in|log in/i}).click();
await page.waitForURL((u) => !u.pathname.startsWith("/auth"), {timeout: 30_000});

async function shoot(path, file, ready) {
  await page.goto(`${args.url}${path}`);
  await page.locator(ready).first().waitFor({timeout: 60_000});
  await page.waitForLoadState("networkidle");
  await page.waitForTimeout(1500);
  const out = resolve(outDir, file);
  await page.screenshot({path: out, fullPage: true});
  console.log(out);
}

await shoot("/dags", "airflow_dags.png", "a[href*='/dags/radar_']");
await shoot(`/dags/${args.dag}`, `airflow_${args.dag}.png`, `text=${args.dag}`);
await browser.close();
