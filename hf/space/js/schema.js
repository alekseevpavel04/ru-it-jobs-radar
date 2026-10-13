// The extraction schema check of radar.extract.schema (try_parse, then dump) for the model's answer.
// Keys and allowed values come from the JSON Schema the decoding is constrained by (assets/space.json,
// written from radar.extract.schema); the rules it cannot express are repeated here: a salary or an
// experience needs one bound and from <= to, a list item is checked after stripping.

class Invalid extends Error {}

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

function allowed(property) {
  const variants = property.anyOf ?? [property];
  const withEnum = variants.find((v) => Array.isArray(v.enum));
  return withEnum ? withEnum.enum : null;
}

function exactKeys(obj, keys) {
  if (!isObject(obj)) throw new Invalid("not an object");
  const got = Object.keys(obj);
  if (got.length !== keys.length || !keys.every((k) => Object.hasOwn(obj, k))) throw new Invalid("keys");
}

function choice(value, values, nullable = true) {
  if (value === null && nullable) return null;
  if (typeof value !== "string" || !values.includes(value)) throw new Invalid("enum");
  return value;
}

function items(value, list) {
  if (!Array.isArray(value) || value.length > list.maxItems) throw new Invalid("list");
  return value.map((item) => {
    if (typeof item !== "string") throw new Invalid("item");
    const s = item.trim();
    const n = [...s].length;
    if (n < list.items.minLength || n > list.items.maxLength) throw new Invalid("item length");
    return s;
  });
}

function wholeNumber(value) {
  if (value === null) return null;
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) throw new Invalid("amount");
  return value;
}

function years(value) {
  if (value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 50) throw new Invalid("years");
  return value;
}

export function makeValidator(schema) {
  const props = schema.properties;
  const salaryProps = schema.$defs.Salary.properties;
  const enums = {
    role: allowed(props.role),
    grade: allowed(props.grade),
    work_format: allowed(props.work_format),
    employment_type: allowed(props.employment_type),
    lang: allowed(props.lang),
    currency: allowed(salaryProps.currency),
    gross_net: allowed(salaryProps.gross_net),
    period: allowed(salaryProps.period)
  };
  const keys = schema.required;
  const salaryKeys = schema.$defs.Salary.required;
  const experienceKeys = schema.$defs.Experience.required;

  function salary(value) {
    if (value === null) return null;
    // The schema key is "from"; the Python model also takes its field name "from_".
    const fromKey = isObject(value) && !Object.hasOwn(value, "from") && Object.hasOwn(value, "from_") ? "from_" : "from";
    exactKeys(value, salaryKeys.map((k) => (k === "from" ? fromKey : k)));
    const out = {
      from: wholeNumber(value[fromKey]),
      to: wholeNumber(value.to),
      currency: choice(value.currency, enums.currency),
      gross_net: choice(value.gross_net, enums.gross_net),
      period: choice(value.period, enums.period)
    };
    if (out.from === null && out.to === null) throw new Invalid("salary without a bound");
    if (out.from !== null && out.to !== null && out.from > out.to) throw new Invalid("salary from > to");
    return out;
  }

  function experience(value) {
    if (value === null) return null;
    exactKeys(value, experienceKeys);
    const out = {min: years(value.min), max: years(value.max)};
    if (out.min === null && out.max === null) throw new Invalid("experience without a bound");
    if (out.min !== null && out.max !== null && out.min > out.max) throw new Invalid("experience min > max");
    return out;
  }

  // The fields of a valid answer in schema key order, null for an answer the schema rejects.
  return function validate(raw) {
    let obj;
    try {
      obj = JSON.parse(raw.trim());
    } catch {
      return null;
    }
    try {
      exactKeys(obj, keys);
      return {
        role: choice(obj.role, enums.role),
        grade: choice(obj.grade, enums.grade),
        skills: items(obj.skills, props.skills),
        salary: salary(obj.salary),
        work_format: choice(obj.work_format, enums.work_format),
        cities: items(obj.cities, props.cities),
        experience_years: experience(obj.experience_years),
        employment_type: choice(obj.employment_type, enums.employment_type),
        lang: choice(obj.lang, enums.lang, false)
      };
    } catch (e) {
      if (e instanceof Invalid) return null;
      throw e;
    }
  };
}
