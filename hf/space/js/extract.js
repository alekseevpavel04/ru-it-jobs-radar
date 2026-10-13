// The student in the browser: wllama (llama.cpp compiled to WebAssembly) runs the Q4_K_M GGUF from
// the model repo on the visitor's processor. The prompt is rendered by js/prompt.js, decoding is greedy
// with the extraction JSON schema as the grammar, the answer is checked by js/schema.js and its fields
// go to the salary model (js/salary.js). Nothing leaves the browser.

import {Wllama} from "../vendor/wllama/index.js";
import {Tokenizer} from "../vendor/tokenizers/tokenizers.min.mjs";
import {addedTokens, promptFor} from "./prompt.js";
import {makeValidator} from "./schema.js";
import {OutOfScope, SalaryModel} from "./salary.js";

const base = new URL("..", import.meta.url);
const asset = (rel) => new URL(rel, base).href;
const getJson = async (rel) => {
  const r = await fetch(asset(rel));
  if (!r.ok) throw new Error(`${rel}: HTTP ${r.status}`);
  return r.json();
};

const wllama = new Wllama({default: asset("vendor/wllama/wllama.wasm")}, {suppressNativeLog: true});
// Browsers without WebAssembly JSPI or 64-bit memory (Safari, some Firefox versions) need wllama's
// compatibility build, which wllama takes from cdn.jsdelivr.net at its own version.
wllama.setCompat("default");

let info = null;
let tokenizer = null;
let added = null;
let validate = null;
let salary = null;

export async function loadInfo() {
  if (!info) info = await getJson("assets/space.json");
  return info;
}

export async function isCached() {
  const {gguf, model} = await loadInfo();
  const name = await wllama.cacheManager.getNameFromURL(gguf.url);
  return (await wllama.cacheManager.getSize(name)) === model.gguf_size_bytes;
}

// Tokenizer, answer check and salary model first (about 15 MB), then the GGUF: from the browser's
// storage when it is there, else downloaded once and stored.
export async function loadModel(onProgress) {
  const {gguf, json_schema: schema, decoding} = await loadInfo();
  const [tokenizerJson, tokenizerConfig, salaryDoc] = await Promise.all([
    getJson("assets/tokenizer/tokenizer.json"),
    getJson("assets/tokenizer/tokenizer_config.json"),
    getJson("assets/salary.json")
  ]);
  tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
  added = addedTokens(tokenizerJson);
  validate = makeValidator(schema);
  salary = new SalaryModel(salaryDoc);
  if (!wllama.isModelLoaded()) {
    await wllama.loadModelFromUrl(gguf.url, {
      n_ctx: decoding.n_ctx,
      n_gpu_layers: 0,
      useCache: true,
      progressCallback: onProgress
    });
  }
  return {threads: wllama.getNumThreads(), multithread: wllama.isMultithread()};
}

export const isLoaded = () => wllama.isModelLoaded();

export function runtime() {
  return {
    wllama: info?.libraries?.["@wllama/wllama"],
    libllama: Wllama.getLibllamaVersion(),
    threads: wllama.isModelLoaded() ? wllama.getNumThreads() : null
  };
}

export async function complete(prompt, schema, maxTokens) {
  const r = await wllama.createCompletion({
    prompt,
    n_predict: maxTokens,
    temperature: 0,
    cache_prompt: true,
    json_schema: schema
  });
  return {text: r.choices[0].text, finish: r.choices[0].finish_reason, usage: r.usage, timings: r.timings};
}

function estimateSalary(fields) {
  try {
    return [salary.predict(fields, info.salary_source), null];
  } catch (e) {
    if (e instanceof OutOfScope) return [null, "non_it"];
    throw e;
  }
}

export async function extract(text) {
  const started = performance.now();
  const {prompt, truncated} = promptFor(tokenizer, added, info.prompt, text);
  const out = await complete(prompt, info.json_schema, info.decoding.max_new_tokens);
  const afterModel = performance.now();
  const raw = out.text.trim();
  const fields = validate(raw);
  const [estimate, note] = fields === null ? [null, "invalid"] : estimateSalary(fields);
  return {
    fields,
    valid: fields !== null,
    raw,
    finish: out.finish,
    input_truncated: truncated,
    prompt_tokens: out.usage.prompt_tokens,
    completion_tokens: out.usage.completion_tokens,
    timings: out.timings,
    salary: estimate,
    salary_note: note,
    seconds: {model: (afterModel - started) / 1000, total: (performance.now() - started) / 1000}
  };
}
