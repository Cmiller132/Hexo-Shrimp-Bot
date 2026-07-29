import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import deadCss from "../scripts/dead-css.cjs";

const url = (path) => new URL(path, import.meta.url);

async function built() {
  const html = await readFile(url("../dist/index.html"), "utf8");
  const script = html.match(/<script[^>]+src="([^"]+\.js)"/)?.[1];
  assert.ok(script, "built HTML names its client bundle");
  const styles = html.match(/<link[^>]+href="([^"]+\.css)"/)?.[1];
  assert.ok(styles, "built HTML names its stylesheet");
  return {
    html,
    bundle: await readFile(url(`../dist${script}`), "utf8"),
    css: await readFile(url(`../dist${styles}`), "utf8"),
  };
}

test("the built bundle carries every deck surface", async () => {
  const { html, bundle } = await built();
  assert.match(html, /<title>Shrimp Control Deck<\/title>/i);
  assert.match(html, /<div id="root"><\/div>/);

  const copy = [
    // shell and the four screens
    "SHRIMP", "CONTROL DECK", "Play", "Game history", "Live run", "Model lab",
    // the command palette is a real command list, not a nav duplicate
    "Jump to a screen, switch run, launch", "SCREENS", "RUNS", "ACTIONS",
    // the lab is a position explorer: a line, a cursor, a walk, diagnostics
    "POSITION EXPLORER", "Policy across the line", "Position read", "Candidates",
    "Restore the source line", "Run walk", "Diagnostics",
    // history: replay plus the stored trace, aggregates behind a button
    "Game browser", "Per-ply trace", "Opening atlas", "Largest value swings", "Not requested",
    // play reads the same position through the same board
    "Arena & quick suites", "Checkpoint read of this position",
    // the shared board, chart and transport
    "board overlay", "labelled cells", "playback speed", "ply cursor",
  ];
  for (const text of copy) assert.ok(bundle.includes(text), `bundle carries ${JSON.stringify(text)}`);

  // Surfaces the deck must not carry: a hand-typed JSON position input, a raw
  // response dump, the removed worker layer, invented sample data.
  const gone = [
    "Inspect position", "result-json", "lab-page", "scrubber", "legal-cell",
    "codex-preview", "react-loading-skeleton", "vinext", "wrangler",
    "48 actors", "Replay memory", "Final V", "checkpoint promotion",
  ];
  for (const text of gone) assert.ok(!bundle.includes(text), `bundle has dropped ${JSON.stringify(text)}`);
});

test("the stylesheet carries no rule src/ cannot match", async () => {
  const { css } = await built();
  // The same scan `npm run prune-css` deletes by, so the writer and the checker
  // of this invariant cannot drift. It is one-sided by design: it proves a rule
  // dead, never proves one live, so a class whose name is also an unrelated
  // identifier still has to be caught by reading the diff.
  assert.deepEqual(deadCss.deadClasses(), [], "globals.css names classes src/ cannot produce");

  // The shipped stylesheet is the wired one, not the abandoned design draft.
  for (const rule of [".deck-board", ".deck-chart", ".transport", ".explorer-page", ".history-page"]) {
    assert.ok(css.includes(rule), `stylesheet carries ${rule}`);
  }
});
