// The Framework CLI with its usage telemetry off, the same on Windows and in CI:
//   node scripts/observable.mjs build | preview
process.env.OBSERVABLE_TELEMETRY_DISABLE = "true";
await import("@observablehq/framework/dist/bin/observable.js");
