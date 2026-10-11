// Screenshot of the weekly digest workflow on the n8n canvas (docs/img/n8n_digest.png). Signs in as
// the owner, opens the workflow and, with --run, runs it from the manual trigger first, so the shot
// shows every node after a real run with its item counts.
//
//   N8N_OWNER_PASSWORD=... node scripts/n8n_screenshot.mjs [--run] [--out docs/img/n8n_digest.png]
//
// Needs the local n8n set up by scripts/n8n_import.sh and Playwright from site/node_modules
// (browsers in PLAYWRIGHT_BROWSERS_PATH).

import {mkdirSync} from "node:fs";
import {dirname, resolve} from "node:path";
import {parseArgs} from "node:util";
import {chromium} from "../site/node_modules/playwright/index.mjs";

const {values: args} = parseArgs({
  options: {
    url: {type: "string", default: "http://localhost:5678"},
    workflow: {type: "string", default: "radarWeeklyDigest"},
    email: {type: "string", default: "owner@radar.local"},
    out: {type: "string", default: "docs/img/n8n_digest.png"},
    run: {type: "boolean", default: false}
  }
});

const password = process.env.N8N_OWNER_PASSWORD;
if (!password) {
  console.error("N8N_OWNER_PASSWORD is not set");
  process.exit(1);
}

const browser = await chromium.launch();
try {
  const page = await browser.newPage({viewport: {width: 1600, height: 900}, deviceScaleFactor: 1.5});
  await page.goto(`${args.url}/signin`);
  await page.locator('input[type="email"], input[name="emailOrLdapLoginId"]').first().fill(args.email);
  await page.locator('input[type="password"]').first().fill(password);
  await page.getByRole("button", {name: /sign in/i}).click();
  await page.waitForURL((url) => !url.pathname.startsWith("/signin"));

  await page.goto(`${args.url}/workflow/${args.workflow}`);
  await page.locator(".vue-flow__node").first().waitFor();
  // The checklist n8n opens over the canvas of a published workflow.
  const checklist = page.locator('[data-test-id="suggested-actions-close"]');
  if (await checklist.isVisible()) {
    await checklist.click();
  }

  if (args.run) {
    // The run button of the manual trigger shows on hover; the main button starts from the schedule.
    await page.locator('[data-test-id="canvas-node"][data-node-name="Run by hand"]').hover();
    await page.locator('[data-test-id="execute-workflow-button-Run by hand"]').click();
    const done = page.getByText(/executed successfully/i);
    const failed = page.getByText(/problem in node|execution failed/i);
    await Promise.race([done.first().waitFor({timeout: 120_000}), failed.first().waitFor({timeout: 120_000})]);
    if (await failed.count()) {
      throw new Error(`the workflow run failed: ${await failed.first().textContent()}`);
    }
    await page.waitForTimeout(4000);
  }

  // Fit the whole workflow into the view and move the pointer off the nodes.
  await page.keyboard.press("1");
  await page.mouse.move(5, 895);
  await page.waitForTimeout(800);
  const out = resolve(args.out);
  mkdirSync(dirname(out), {recursive: true});
  await page.screenshot({path: out});
  console.log(`saved ${out}`);
} finally {
  await browser.close();
}
