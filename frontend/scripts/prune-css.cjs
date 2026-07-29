#!/usr/bin/env node
/**
 * Deletes every CSS rule whose selector names a class `src/` cannot produce. A
 * rule nothing can match is dead code, which this repo does not keep.
 *
 * `dead-css.cjs` decides what "cannot produce" means — the same scan the test
 * suite asserts against — and a selector survives only if every class token in
 * it is live. A selector list is rewritten to its living members. Surviving
 * lines are emitted byte-for-byte; nothing is reformatted.
 *
 * Run with `--write` to apply, without it to report.
 */
const fs = require("node:fs");
const path = require("node:path");
const { sourceTokens } = require("./dead-css.cjs");

const root = path.join(__dirname, "..");
const cssPath = path.join(root, "app", "globals.css");

const tokens = sourceTokens(root);
const classesOf = (selector) => [...selector.matchAll(/\.(-?[A-Za-z_][A-Za-z0-9_-]*)/g)].map((m) => m[1]);
const selectorLives = (selector) => classesOf(selector).every((name) => tokens.has(name));

const text = fs.readFileSync(cssPath, "utf8");
const lines = text.split("\n");
const keep = new Array(lines.length).fill(true);
const dropped = [];

/** Parses `lines[from..to]` as a rule list, marking dead rules for deletion.
 *  Returns the number of style rules that survived, so an emptied `@media` can
 *  be dropped whole. */
function scan(from, to, insideAtRule) {
  let survivors = 0;
  let index = from;
  let preludeStart = null;
  let prelude = "";
  while (index <= to) {
    const line = lines[index];
    if (preludeStart === null && line.trim() === "") { index++; continue; }
    if (preludeStart === null) preludeStart = index;
    prelude += `${line}\n`;
    if (!line.includes("{")) { index++; continue; }

    // Body: from the first `{` to its matching `}`.
    let depth = 0;
    let end = index;
    for (let scanLine = index; scanLine <= to; scanLine++) {
      for (const character of lines[scanLine]) {
        if (character === "{") depth++;
        else if (character === "}") depth--;
      }
      if (depth === 0) { end = scanLine; break; }
      end = scanLine;
    }
    const selector = prelude.slice(0, prelude.lastIndexOf("{")).trim();
    const at = selector.startsWith("@");
    if (at) {
      if (/^@(media|supports)/.test(selector)) {
        const inner = scan(index + 1, end - 1, true);
        if (inner === 0) {
          for (let l = preludeStart; l <= end; l++) keep[l] = false;
          dropped.push(selector);
        } else survivors++;
      } else survivors++; // @keyframes, @font-face, @property: kept.
    } else {
      const parts = selector.split(",").map((part) => part.trim()).filter(Boolean);
      const live = parts.filter(selectorLives);
      if (live.length === 0) {
        for (let l = preludeStart; l <= end; l++) keep[l] = false;
        dropped.push(selector.replace(/\s+/g, " "));
      } else if (live.length !== parts.length) {
        // Some selectors in the list died: rewrite the prelude to the live ones,
        // keeping whatever followed the `{` on the same line for a compact rule.
        const indent = lines[preludeStart].match(/^\s*/)[0];
        const tail = lines[index].slice(lines[index].lastIndexOf("{") + 1);
        for (let l = preludeStart; l < index; l++) keep[l] = false;
        lines[index] = `${indent}${live.join(",\n" + indent)} {${tail}`;
        dropped.push(...parts.filter((part) => !selectorLives(part)));
        survivors++;
      } else survivors++;
    }
    prelude = "";
    preludeStart = null;
    index = end + 1;
  }
  return survivors;
}

scan(0, lines.length - 1, false);

const out = lines.filter((_, index) => keep[index]).join("\n").replace(/\n{3,}/g, "\n\n");
const removed = lines.length - keep.filter(Boolean).length;
console.log(`${dropped.length} dead selectors, ${removed} of ${lines.length} lines`);
if (process.argv.includes("--verbose")) console.log(dropped.join("\n"));
if (process.argv.includes("--write")) {
  fs.writeFileSync(cssPath, out);
  console.log(`wrote ${cssPath}`);
}
