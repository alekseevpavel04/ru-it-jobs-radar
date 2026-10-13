// The salary model in the browser: radar.models.salary_features.feature_row and the boosting models
// of radar.models.salary, exported by radar.site.hf_space as plain trees (assets/salary.json).
//
// A tree node is [value] for a leaf, [feature, threshold, nan_left, left, right] for a numeric split
// (an infinite threshold written as "inf" or "-inf") and [feature, codes_left, nan_left, left, right]
// for a categorical one. Categorical columns hold the
// level index of the encoder; an unseen level is missing, and so is a level the model never saw in
// its own training rows (codes_left already folds sklearn's known-category check in).

const UNKNOWN = "unknown";
const MOSCOW = "Москва";
const SPB = "Санкт-Петербург";

export class OutOfScope extends Error {}

function regionGroup(cities, workFormat) {
  if (workFormat === "remote") return "remote";
  if (cities.includes(MOSCOW)) return "moscow";
  if (cities.includes(SPB)) return "spb";
  return cities.length ? "other" : UNKNOWN;
}

export function featureRow(fields, source, maps) {
  const role = fields.role;
  if (role === "non_it") throw new OutOfScope("the salary model covers IT vacancies only");
  const bucket = role ? (maps.role_bucket[role] ?? UNKNOWN) : UNKNOWN;
  const experience = fields.experience_years || {};
  const workFormat = fields.work_format;
  return {
    role_bucket: bucket,
    role_group: maps.role_group[bucket] ?? UNKNOWN,
    grade: fields.grade || UNKNOWN,
    region_group: regionGroup(fields.cities || [], workFormat),
    work_format: workFormat || UNKNOWN,
    employment_type: fields.employment_type || UNKNOWN,
    source,
    experience_min: experience.min ?? null,
    skills: [...new Set(fields.skills || [])].sort()
  };
}

// Python's round(): halves go to the even neighbour.
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function readNode(node) {
  if (node.length === 1) return node;
  const [feature, split, nanLeft, left, right] = node;
  if (Array.isArray(split)) return [feature, new Set(split), nanLeft, left, right];
  if (typeof split === "string") return [feature, split === "inf" ? Infinity : -Infinity, nanLeft, left, right];
  return node;
}

export class SalaryModel {
  constructor(doc) {
    this.doc = doc;
    this.levels = doc.encoder.categorical.map(([name, levels]) => [name, new Map(levels.map((v, i) => [v, i]))]);
    this.numeric = doc.encoder.numeric;
    this.skillIndex = new Map(doc.encoder.skill_vocab.map((s, i) => [s, i]));
    this.width = this.levels.length + this.numeric.length + doc.encoder.skill_vocab.length;
    this.models = Object.fromEntries(
      Object.entries(doc.models).map(([name, m]) => [
        name,
        {
          baseline: m.baseline,
          trees: m.trees.map((tree) => tree.map(readNode))
        }
      ])
    );
  }

  // The row of Encoder.tree_matrix: level indices (NaN when unseen), numbers (NaN when unknown), skills.
  matrixRow(row) {
    const x = new Float64Array(this.width);
    let j = 0;
    for (const [name, index] of this.levels) {
      const code = index.get(String(row[name]));
      x[j++] = code === undefined ? NaN : code;
    }
    for (const name of this.numeric) {
      const v = row[name];
      x[j++] = v === null || v === undefined ? NaN : Number(v);
    }
    for (const s of row.skills) {
      const k = this.skillIndex.get(s);
      if (k !== undefined) x[j + k] = 1;
    }
    return x;
  }

  raw(name, x) {
    const m = this.models[name];
    let out = m.baseline;
    for (const tree of m.trees) {
      let node = tree[0];
      while (node.length !== 1) {
        const [feature, split, nanLeft, left, right] = node;
        const v = x[feature];
        let goLeft;
        if (Number.isNaN(v)) goLeft = nanLeft === 1;
        else if (split instanceof Set) goLeft = split.has(v);
        else goLeft = v <= split;
        node = tree[goLeft ? left : right];
      }
      out += node[0];
    }
    return out;
  }

  // radar.models.salary.SalaryModel.predict_rows for one row, before rounding.
  predictRow(row) {
    const x = this.matrixRow(row);
    const point = this.raw("point", x);
    const low = Math.min(this.raw("lower", x) - this.doc.margin, point);
    const high = Math.max(this.raw("upper", x) + this.doc.margin, point);
    const from = this.raw("bound_from", x);
    const to = this.raw("bound_to", x);
    return {
      point: Math.exp(point),
      low: Math.exp(low),
      high: Math.exp(high),
      from: Math.exp(Math.min(from, to)),
      to: Math.exp(Math.max(from, to))
    };
  }

  nSimilar(row) {
    return this.doc.n_similar[[row.role_bucket, row.grade, row.region_group].join("|")] ?? 0;
  }

  // radar.models.salary.predict: monthly RUB, rounded to the thousand.
  predict(fields, source) {
    const row = featureRow(fields, source, this.doc.maps);
    const out = this.predictRow(row);
    const round = (v) => roundHalfEven(v / this.doc.round_rub) * this.doc.round_rub;
    return {
      point: round(out.point),
      low: round(out.low),
      high: round(out.high),
      typical_from: round(out.from),
      typical_to: round(out.to),
      level: this.doc.level,
      n_similar: this.nSimilar(row),
      source,
      model_version: this.doc.version
    };
  }
}
