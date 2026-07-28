import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("Vite build carries the Shrimp control deck shell", async () => {
  const html = await readFile(new URL("../dist/index.html", import.meta.url), "utf8");
  assert.match(html, /<title>Shrimp Control Deck<\/title>/i);
  assert.match(html, /<div id="root"><\/div>/);
  const script = html.match(/<script[^>]+src="([^"]+\.js)"/)?.[1];
  assert.ok(script, "built HTML names its client bundle");
  const bundle = await readFile(new URL(`../dist${script}`, import.meta.url), "utf8");
  for (const copy of ["SHRIMP", "CONTROL DECK", "Game history", "Live run", "Model lab", "KLENT position read"]) {
    assert.match(bundle, new RegExp(copy));
  }
  assert.doesNotMatch(bundle, /codex-preview|react-loading-skeleton|vinext|wrangler/i);
  assert.doesNotMatch(bundle, /48 actors|Replay memory|Final V|checkpoint promotion/i);
});
