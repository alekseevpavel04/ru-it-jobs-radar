// Serves a built site from disk, opens each page in Chromium at desktop and phone width, saves a
// full-page screenshot and records every request the page makes. A request to any other host is
// an external dependency, which the public site must not have (it has to open from Russia without
// a VPN), so the run fails on one.
//
//   node scripts/shoot.mjs --dir dist --out ../logs/site/shots --pages index,skills [--base /ru-it-jobs-radar/] [--text]

import {createReadStream, existsSync, mkdirSync, statSync, writeFileSync} from "node:fs";
import {createServer} from "node:http";
import {extname, join, normalize, resolve} from "node:path";
import {parseArgs} from "node:util";
import {chromium} from "playwright";

const {values: args} = parseArgs({
  options: {
    dir: {type: "string", default: "dist"},
    out: {type: "string", default: "../logs/site/shots"},
    pages: {type: "string", default: "index"},
    base: {type: "string", default: "/"},
    widths: {type: "string", default: "1280,390"},
    text: {type: "boolean", default: false}
  }
});

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".csv": "text/csv; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".wasm": "application/wasm",
  ".woff2": "font/woff2",
  ".parquet": "application/octet-stream"
};

const root = resolve(args.dir);
const base = args.base.endsWith("/") ? args.base : `${args.base}/`;

// GitHub Pages serves /page as page.html and /dir/ as dir/index.html; this server does the same.
function locate(urlPath) {
  if (!urlPath.startsWith(base)) return null;
  const rel = normalize(decodeURIComponent(urlPath.slice(base.length).split("?")[0]));
  if (rel.startsWith("..")) return null;
  for (const candidate of [rel, `${rel}.html`, join(rel, "index.html")]) {
    const path = join(root, candidate);
    if (existsSync(path) && statSync(path).isFile()) return path;
  }
  return null;
}

const server = createServer((req, res) => {
  const path = locate(req.url);
  if (!path) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, {"content-type": TYPES[extname(path)] ?? "application/octet-stream"});
  createReadStream(path).pipe(res);
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const origin = `http://127.0.0.1:${server.address().port}`;

mkdirSync(args.out, {recursive: true});
const browser = await chromium.launch();
const report = [];
for (const width of args.widths.split(",").map(Number)) {
  const context = await browser.newContext({
    viewport: {width, height: width < 600 ? 844 : 900},
    deviceScaleFactor: 1,
    colorScheme: "light",
    locale: "ru-RU"
  });
  for (const name of args.pages.split(",")) {
    const page = await context.newPage();
    const external = new Set();
    const errors = [];
    let bytes = 0;
    page.on("request", (r) => {
      if (!r.url().startsWith(origin) && !r.url().startsWith("data:") && !r.url().startsWith("blob:")) {
        external.add(new URL(r.url()).host);
      }
    });
    page.on("response", async (r) => {
      const body = await r.body().catch(() => null);
      if (body) bytes += body.length;
    });
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
    const started = Date.now();
    const url = `${origin}${base}${name === "index" ? "" : name}`;
    const response = await page.goto(url, {waitUntil: "networkidle", timeout: 120000});
    const loadMs = Date.now() - started;
    const text = args.text ? await page.innerText("body") : null;
    const file = join(args.out, `${name.replaceAll("/", "_")}_${width}.png`);
    await page.screenshot({path: file, fullPage: true});
    report.push({
      page: name,
      width,
      status: response?.status() ?? null,
      load_ms: loadMs,
      bytes,
      external_hosts: [...external].sort(),
      errors,
      screenshot: file,
      ...(text === null ? {} : {text})
    });
    await page.close();
  }
  await context.close();
}
await browser.close();
server.close();

writeFileSync(join(args.out, "report.json"), JSON.stringify(report, null, 2));
for (const r of report) {
  console.log(
    `${r.page} ${r.width}px status=${r.status} load=${r.load_ms}ms bytes=${r.bytes} external=[${r.external_hosts}] errors=${r.errors.length}`
  );
}
const failed = report.filter((r) => r.status !== 200 || r.external_hosts.length || r.errors.length);
process.exit(failed.length ? 1 : 0);
