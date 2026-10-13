// The assembled static Space in Chromium, served from a local HTTP server, with the model repo's GGUF
// URL answered by a local copy: model load (fresh profile, then from the browser storage), a grammar
// probe, extraction times with all threads and with one, screenshots at 1280 and 390 px.
// Started by `python -m radar.site.hf_space bench`, which turns OUT.json into the result file.
//
//   node scripts/hf_space_bench.mjs --space storage/hf/space --gguf MODEL.gguf --texts TEXTS.json
//        --out OUT.json [--single 5] [--shots docs/img]
//   node scripts/hf_space_bench.mjs --space storage/hf/space --gguf MODEL.gguf --shots-only
//        only the two screenshots, for a page reassembled after the timed run
//
// No request leaves the machine: everything but the local server and the GGUF URL is aborted.
// Browser profiles live under --profiles (on D:): an in-memory profile on a nearly full system disk
// refuses to store the 1 GB file a second time.

import {createReadStream, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync} from "node:fs";
import {createServer} from "node:http";
import {cpus} from "node:os";
import {extname, join, resolve, sep} from "node:path";
import {parseArgs} from "node:util";
import {chromium} from "../site/node_modules/playwright/index.mjs";

const {values: args} = parseArgs({
  options: {
    space: {type: "string"},
    gguf: {type: "string"},
    texts: {type: "string"},
    out: {type: "string"},
    single: {type: "string", default: "5"},
    shots: {type: "string", default: "docs/img"},
    profiles: {type: "string", default: "logs/hf_space"},
    "shots-only": {type: "boolean", default: false}
  }
});

const ROOT = resolve(args.space);
const TEXTS = args["shots-only"] ? [] : JSON.parse(readFileSync(args.texts, "utf-8"));
const INFO = JSON.parse(readFileSync(join(ROOT, "assets", "space.json"), "utf-8"));
const MODEL_PATH = "__model.gguf";
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".wasm": "application/wasm", ".md": "text/plain; charset=utf-8"
};
// The headers the Space card asks Hugging Face for (custom_headers).
const ISOLATION = {
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Resource-Policy": "cross-origin"
};
const GRAMMAR_PROBE = {
  prompt: "<|im_start|>user\nОпиши погоду в JSON.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  schema: {type: "object", properties: {probe: {const: "grammar"}}, required: ["probe"], additionalProperties: false}
};
const LONG = 900_000;
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

function serve(isolated) {
  const server = createServer((req, res) => {
    const url = new URL(req.url, "http://localhost");
    let file;
    let headers;
    if (url.pathname === `/${MODEL_PATH}`) {
      file = args.gguf;
      headers = {
        "Content-Type": "application/octet-stream",
        "Access-Control-Allow-Origin": "*",
        "Cross-Origin-Resource-Policy": "cross-origin",
        ETag: `"${INFO.model.gguf_sha256}"`
      };
    } else {
      file = resolve(ROOT, "." + decodeURIComponent(url.pathname === "/" ? "/index.html" : url.pathname));
      if (!file.startsWith(ROOT + sep)) {
        res.writeHead(403).end();
        return;
      }
      headers = {"Content-Type": TYPES[extname(file)] ?? "application/octet-stream", ...(isolated ? ISOLATION : {})};
    }
    let size;
    try {
      size = statSync(file).size;
    } catch {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, {...headers, "Content-Length": size, "Cache-Control": "no-store"});
    if (req.method === "HEAD") res.end();
    else createReadStream(file).pipe(res);
  });
  return new Promise((ok) => server.listen(0, "127.0.0.1", () => ok(server)));
}

async function newContext(server) {
  const base = `http://127.0.0.1:${server.address().port}/`;
  const profile = mkdtempSync(join(resolve(args.profiles), "profile-"));
  profiles.push(profile);
  const context = await chromium.launchPersistentContext(profile);
  contexts.push(context);
  context.setDefaultTimeout(LONG);
  const blocked = [];
  await context.route("**/*", (route) => {
    const url = route.request().url();
    if (url.startsWith(base)) return route.continue();
    if (url === INFO.gguf.url) {
      return route.fulfill({status: 302, headers: {"access-control-allow-origin": "*", location: base + MODEL_PATH}});
    }
    blocked.push(url);
    return route.abort();
  });
  return {context, base, blocked};
}

async function openPage(context, base, viewport) {
  const page = await context.newPage();
  await page.setViewportSize(viewport);
  page.on("pageerror", (e) => log("page error:", e.message));
  await page.goto(base, {waitUntil: "load"});
  await page.waitForSelector("#load:not([disabled])");
  return page;
}

// The page's own functions, from the module instances it loaded: startModel is what the load button
// calls, runOn what the run button calls with the text.
const startModel = (page) =>
  page.evaluate(async () => (await import(new URL("js/app.js", location.href).href)).startModel());

const runOn = (page, text) =>
  page.evaluate(async (value) => (await import(new URL("js/app.js", location.href).href)).runOn(value), text);

const engine = (page, call, arg) =>
  page.evaluate(
    async ([name, value]) => {
      const e = await import(new URL("js/extract.js", location.href).href);
      return name === "runtime" ? e.runtime() : (await e.complete(value.prompt, value.schema, 32)).text;
    },
    [call, arg]
  );

async function timeLoad(page) {
  const t0 = performance.now();
  const state = await startModel(page);
  return {...state, wall_s: (performance.now() - t0) / 1000};
}

async function timeTexts(page, texts) {
  const out = [];
  for (const t of texts) {
    const t0 = performance.now();
    const r = await runOn(page, t.input);
    const wall = (performance.now() - t0) / 1000;
    out.push({
      id: t.id,
      wall_s: Math.round(wall * 1000) / 1000,
      prompt_ms: r.timings.prompt_ms,
      predicted_ms: r.timings.predicted_ms,
      prompt_n: r.timings.prompt_n,
      predicted_n: r.timings.predicted_n,
      valid: r.valid,
      finish: r.finish,
      fields: r.fields,
      salary: r.salary
    });
    log(`${t.id} ${wall.toFixed(1)} s, ${r.timings.prompt_n} + ${r.timings.predicted_n} tokens, valid ${r.valid}`);
  }
  return out;
}

async function shot(page, width) {
  await page.click("#example");
  await page.click("#run");
  await page.waitForFunction(() => /Ответ за|Ошибка/.test(document.querySelector("#status").textContent));
  const status = await page.textContent("#status");
  if (status.startsWith("Ошибка")) throw new Error(`the page answered: ${status}`);
  const file = resolve(args.shots, `space_${width}.png`);
  await page.screenshot({path: file, fullPage: true});
  log(`${file}: ${status}`);
}

function isProbeAnswer(text) {
  try {
    return JSON.stringify(JSON.parse(text)) === JSON.stringify({probe: "grammar"});
  } catch {
    return false;
  }
}

const profiles = [];
const contexts = [];
const isolated = await serve(true);
const plain = await serve(false);
const result = {host_cpu: cpus()[0].model.trim(), logical_cpus: cpus().length};
try {
  // Fresh profile, all threads: download from the local copy, grammar probe, timed texts, 1280 shot.
  const a = await newContext(isolated);
  let page = await openPage(a.context, a.base, {width: 1280, height: 900});
  result.browser = (await page.evaluate(() => navigator.userAgent)).match(/Chrome\/[\d.]+/)[0];
  const cold = await timeLoad(page);
  log(`cold load ${cold.wall_s.toFixed(1)} s, threads ${cold.threads}, multithread ${cold.multithread}`);
  if (!cold.multithread) throw new Error("the isolated page runs one thread");
  if (!args["shots-only"]) {
    Object.assign(result, await engine(page, "runtime"));
    const probe = await engine(page, "complete", GRAMMAR_PROBE);
    result.grammar_enforced = isProbeAnswer(probe);
    log(`grammar probe: ${probe.trim()}`);
    // The first answer after the load also reads the system prompt; later ones find it in the KV cache.
    [result.first] = await timeTexts(page, [{id: INFO.examples[0].id, input: INFO.examples[0].text}]);
    result.multi = await timeTexts(page, TEXTS);
  }
  await shot(page, 1280);
  await page.close();

  // Same profile, a new page: the load button reads the GGUF from the browser storage; 390 shot.
  page = await openPage(a.context, a.base, {width: 390, height: 844});
  const t0 = performance.now();
  await page.click("#load");
  await page.waitForFunction(() => /Модель готова|не загрузилась/.test(document.querySelector("#load-status").textContent));
  const cached = (performance.now() - t0) / 1000;
  const loadStatus = await page.textContent("#load-status");
  log(`cached load ${cached.toFixed(1)} s: ${loadStatus}`);
  if (!loadStatus.startsWith("Модель готова")) throw new Error(loadStatus);
  await shot(page, 390);
  await page.close();
  result.blocked_requests = a.blocked;
  await a.context.close();
  if (!args["shots-only"]) {
    // Without the isolation headers the browser gives no shared memory: one thread.
    const b = await newContext(plain);
    page = await openPage(b.context, b.base, {width: 1280, height: 900});
    const single = await timeLoad(page);
    log(`one thread: load ${single.wall_s.toFixed(1)} s, threads ${single.threads}, multithread ${single.multithread}`);
    if (single.multithread) throw new Error("the page without isolation headers still runs several threads");
    [result.first_single] = await timeTexts(page, [{id: INFO.examples[0].id, input: INFO.examples[0].text}]);
    result.single = await timeTexts(page, TEXTS.slice(0, Number(args.single)));
    result.blocked_requests.push(...b.blocked);
    await b.context.close();

    result.threads = cold.threads;
    result.load = {cold_s: cold.wall_s, cached_s: cached, single_cold_s: single.wall_s};
    writeFileSync(args.out, JSON.stringify(result, null, 1));
    log(`-> ${args.out}; blocked requests: ${result.blocked_requests.length}`);
  }
} finally {
  for (const context of contexts) await context.close();
  isolated.close();
  plain.close();
  for (const dir of profiles) rmSync(dir, {recursive: true, force: true});
}
