// Screenshots of the Space running locally (docs/img/space_<width>.png): loads the page, runs the
// first example vacancy and waits for the fields and the salary estimate. Fails when the request
// errors, the fields stay empty or the footer has not loaded its numbers.
//
//   node scripts/hf_space_screenshot.mjs [--url http://127.0.0.1:7860] [--out-dir docs/img]
//
// Needs the image started as in `python -m radar.site.hf_space bench` and Playwright from
// site/node_modules (browsers in PLAYWRIGHT_BROWSERS_PATH).

import {mkdirSync} from "node:fs";
import {resolve} from "node:path";
import {parseArgs} from "node:util";
import {chromium} from "../site/node_modules/playwright/index.mjs";

const {values: args} = parseArgs({
  options: {
    url: {type: "string", default: "http://127.0.0.1:7860/"},
    "out-dir": {type: "string", default: "docs/img"}
  }
});

const WIDTHS = [
  {width: 1280, height: 900},
  {width: 390, height: 844}
];

mkdirSync(args["out-dir"], {recursive: true});
const browser = await chromium.launch();
try {
  for (const viewport of WIDTHS) {
    const page = await browser.newPage({viewport});
    await page.goto(args.url, {waitUntil: "networkidle"});
    await page.waitForFunction(() => document.querySelector("#footer").textContent.includes("macro F1"));
    await page.click("#example");
    await page.click("#run");
    await page.waitForFunction(() => /Ответ за|Ошибка/.test(document.querySelector("#status").textContent), null, {
      timeout: 300_000
    });
    const status = await page.textContent("#status");
    if (status.startsWith("Ошибка")) throw new Error(`the Space answered: ${status}`);
    if (!(await page.$("#fields table"))) throw new Error("no fields were rendered");
    if (!(await page.$("#salary .big"))) console.warn(`${viewport.width}: no salary estimate on the example`);
    const file = resolve(args["out-dir"], `space_${viewport.width}.png`);
    await page.screenshot({path: file, fullPage: true});
    console.log(`${file}: ${status}`);
    await page.close();
  }
} finally {
  await browser.close();
}
