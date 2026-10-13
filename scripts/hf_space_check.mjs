// The Space page's JavaScript in node, on cases written by `python -m radar.site.hf_space check`,
// which compares the answers with the Python the page repeats. Imports the files of the assembled
// Space (storage/hf/space), the same bytes the browser loads.
//
//   node scripts/hf_space_check.mjs CASES.json OUT.json

import {readFileSync, writeFileSync} from "node:fs";
import {join} from "node:path";
import {pathToFileURL} from "node:url";

const [casesPath, outPath] = process.argv.slice(2);
const cases = JSON.parse(readFileSync(casesPath, "utf-8"));
const space = cases.space;
const load = (rel) => import(pathToFileURL(join(space, rel)).href);
const readJson = (rel) => JSON.parse(readFileSync(join(space, rel), "utf-8"));

const {Tokenizer} = await load("vendor/tokenizers/tokenizers.min.mjs");
const {addedTokens, capInput, promptFor} = await load("js/prompt.js");
const {makeValidator} = await load("js/schema.js");
const {SalaryModel, OutOfScope, featureRow} = await load("js/salary.js");

const tokenizerJson = readJson("assets/tokenizer/tokenizer.json");
const tokenizer = new Tokenizer(tokenizerJson, readJson("assets/tokenizer/tokenizer_config.json"));
const added = addedTokens(tokenizerJson);
const info = readJson("assets/space.json");
const validate = makeValidator(info.json_schema);
const salary = new SalaryModel(readJson("assets/salary.json"));

const prompts = {};
for (const r of cases.prompts) prompts[r.id] = promptFor(tokenizer, added, info.prompt, r.input);

const cuts = cases.cuts.map((c) => capInput(tokenizer, added, c.input, c.budget)[0]);

const answers = cases.answers.map((raw) => validate(raw));

const salaryRows = cases.salary_rows.map((row) => salary.predictRow(row));

const salaryFields = cases.salary_fields.map(({fields, source}) => {
  try {
    featureRow(fields, source, salary.doc.maps);
  } catch (e) {
    if (e instanceof OutOfScope) return null;
    throw e;
  }
  return salary.predict(fields, source);
});

writeFileSync(outPath, JSON.stringify({prompts, cuts, answers, salary_rows: salaryRows, salary_fields: salaryFields}));
console.log(`${Object.keys(prompts).length} prompts, ${cuts.length} cuts, ${answers.length} answers, ` +
  `${salaryRows.length} salary rows, ${salaryFields.length} salary fields`);
