#!/usr/bin/env node
/**
 * Which classes `app/globals.css` names that `src/` cannot produce.
 *
 * One scan, two callers: `prune-css.cjs --write` deletes the rules this names,
 * and the test suite fails if it ever names one. They must agree on what "dead"
 * means, so neither of them re-implements it.
 *
 * A class counts as live if its name appears anywhere in `src/` or `index.html`
 * as an identifier — not only inside a `className` attribute. That is deliberate:
 * classes are assembled at runtime from values that never appear as literals in
 * any attribute (`state-dot ${run.state}` reaches `.state-dot.completed` through
 * a union in `types.ts`), and deleting a rule that is in use is far worse than
 * keeping one that is not. The cost is the converse: a class whose name is also
 * an unrelated identifier — `.button` against `type="button"` — cannot be seen
 * as dead by this scan and has to be caught by reading the diff.
 */
const fs = require("node:fs");
const path = require("node:path");

const IDENTIFIER = /[A-Za-z][A-Za-z0-9_-]*/g;
const SELECTOR_CLASS = /\.(-?[A-Za-z_][A-Za-z0-9_-]*)/g;

const defaultRoot = path.join(__dirname, "..");

/** Every identifier the client source and the HTML shell contain. */
function sourceTokens(root = defaultRoot) {
  const tokens = new Set();
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else for (const match of fs.readFileSync(full, "utf8").matchAll(IDENTIFIER)) tokens.add(match[0]);
    }
  };
  walk(path.join(root, "src"));
  for (const match of fs.readFileSync(path.join(root, "index.html"), "utf8").matchAll(IDENTIFIER)) {
    tokens.add(match[0]);
  }
  return tokens;
}

/** The class names the stylesheet selects on that no source token can match. */
function deadClasses(root = defaultRoot) {
  const tokens = sourceTokens(root);
  const dead = new Set();
  const css = fs.readFileSync(path.join(root, "app", "globals.css"), "utf8");
  for (const match of css.matchAll(SELECTOR_CLASS)) {
    if (!tokens.has(match[1])) dead.add(match[1]);
  }
  return [...dead];
}

module.exports = { sourceTokens, deadClasses };
