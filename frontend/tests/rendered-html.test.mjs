import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the Shrimp control deck", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Shrimp Control Deck<\/title>/i);
  assert.match(html, /SHRIMP/);
  assert.match(html, /CONTROL DECK/);
  assert.match(html, /Play &amp; pit/);
  assert.match(html, /MANTISNET MATCH CONSOLE/);
  assert.match(html, /KLENT position read/);
  assert.match(html, /exact loader contract/);
  assert.doesNotMatch(html, /48 actors|Replay memory|Final V|checkpoint promotion/i);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});
