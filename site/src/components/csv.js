// A "CSV" link that hands the reader the numbers behind a chart: the file is built in the browser from
// the rows the chart draws, so the download always matches what is on screen.
import {html} from "npm:htl";

const cell = (v) => {
  const s = v == null ? "" : String(v);
  return /[",\n;]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

// columns: [[header, row => value], ...]
export function csvLink(rows, columns, filename) {
  const lines = [columns.map(([h]) => cell(h)).join(","), ...rows.map((r) => columns.map(([, f]) => cell(f(r))).join(","))];
  // The BOM makes Excel read the Cyrillic names as UTF-8.
  const url = URL.createObjectURL(new Blob(["﻿" + lines.join("\r\n")], {type: "text/csv;charset=utf-8"}));
  return html`<a class="csv-link" href=${url} download=${filename}>CSV</a>`;
}
