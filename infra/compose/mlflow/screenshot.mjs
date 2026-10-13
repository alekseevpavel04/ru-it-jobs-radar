// Screenshots of the MLflow model registry: the list of registered models with their aliases and the
// page of each model (docs/img/mlflow_registry.png, docs/img/mlflow_<model>.png).
//
//   node infra/compose/mlflow/screenshot.mjs [--url http://localhost:5000] [--out-dir docs/img]
//
// Needs the ml profile with the registry filled by `python -m radar.monitor.registry` and Playwright
// from site/node_modules (browsers in PLAYWRIGHT_BROWSERS_PATH).

import {mkdirSync} from "node:fs";
import {resolve} from "node:path";
import {parseArgs} from "node:util";
import {chromium} from "../../../site/node_modules/playwright/index.mjs";

const MODELS = ["radar-extractor", "radar-salary"];

const {values: args} = parseArgs({
  options: {
    url: {type: "string", default: "http://localhost:5000"},
    "out-dir": {type: "string", default: "docs/img"},
    width: {type: "string", default: "1600"}
  }
});
const outDir = resolve(args["out-dir"]);
mkdirSync(outDir, {recursive: true});

async function shoot(context, path, ready, out) {
  const page = await context.newPage();
  await page.goto(`${args.url}/#${path}`);
  await page.getByRole("link", {name: ready, exact: true}).first().waitFor({timeout: 60_000});
  // The UI keeps polling, so the network never goes idle: let the rest of the page render.
  await page.waitForTimeout(2000);
  await page.screenshot({path: out, fullPage: true});
  await page.close();
  console.log(out);
}

const browser = await chromium.launch();
try {
  const context = await browser.newContext({viewport: {width: Number(args.width), height: 900}, deviceScaleFactor: 1});
  // The promo dialog of the new registry UI and the assistant panel would cover the tables.
  await context.addInitScript(() => {
    localStorage.setItem("_mlflow_model_registry_promo_modal_dismissed_v1", "true");
    localStorage.setItem("mlflow.assistant.fab.autoOpenedOnFirstLoad_v1", "true");
    localStorage.setItem("mlflow.assistant.panelOpen_v1", "false");
  });
  await shoot(context, "/models", MODELS[0], `${outDir}/mlflow_registry.png`);
  for (const model of MODELS) {
    await shoot(context, `/models/${model}`, "Version 1", `${outDir}/mlflow_${model.replace("radar-", "")}.png`);
  }
} finally {
  await browser.close();
}
