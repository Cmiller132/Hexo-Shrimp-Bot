import { useEffect, useMemo, useState } from "react";
import { api, useApi } from "../api";
import Board from "../components/Board";
import { Empty, ErrorBox, format, Metric, Panel } from "../components/Ui";
import type { Inspect, Move, Run } from "../types";

function parseMoves(text: string): Move[] {
  const value = JSON.parse(text);
  if (!Array.isArray(value) || value.some((move) => !Array.isArray(move) || move.length !== 2 || move.some((n) => !Number.isInteger(n)))) throw new Error("Move list must be JSON integer pairs, for example [[0,0],[1,0]].");
  return value;
}

export default function Lab({ runs, handoff }: { runs: Run[]; handoff: Move[] }) {
  const checkpoints = useMemo(() => runs.flatMap((run) => run.checkpoints.map((checkpoint) => ({ ...checkpoint, run: run.name }))), [runs]);
  const [checkpoint, setCheckpoint] = useState(checkpoints.at(-1)?.path ?? "");
  const [compare, setCompare] = useState("");
  const [text, setText] = useState(JSON.stringify(handoff));
  const [read, setRead] = useState<Inspect>();
  const [other, setOther] = useState<Inspect>();
  const [overlay, setOverlay] = useState<"policy" | "q" | "improved" | "rank">("policy");
  const [module, setModule] = useState<"position" | "attention" | "d6" | "manifest">("position");
  const [result, setResult] = useState<unknown>();
  const [error, setError] = useState<string>();
  const probes = useApi<Array<Record<string, unknown>>>("/api/probes", []);
  const manifest = useApi<Record<string, unknown>>("/api/model", []);
  useEffect(() => setText(JSON.stringify(handoff)), [handoff]);
  useEffect(() => { if (!checkpoint && checkpoints.length) setCheckpoint(checkpoints.at(-1)!.path); }, [checkpoint, checkpoints]);

  async function inspect() {
    try {
      const moves = parseMoves(text);
      const current = await api<Inspect>("/api/inspect", { method: "POST", body: JSON.stringify({ checkpoint, moves }) });
      setRead(current);
      setOther(compare ? await api("/api/inspect", { method: "POST", body: JSON.stringify({ checkpoint: compare, moves }) }) : undefined);
      setModule("position"); setError(undefined);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  }
  async function diagnostic(kind: "attention" | "d6") {
    try { setResult(await api(`/api/inspect/${kind}`, { method: "POST", body: JSON.stringify({ checkpoint, moves: parseMoves(text) }) })); setModule(kind); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  }
  async function save() {
    try { await api("/api/probes", { method: "POST", body: JSON.stringify({ name: `probe ${new Date().toLocaleString()}`, checkpoint, moves: parseMoves(text), module }) }); void probes.refresh(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  }
  const diffs = read && other ? read.legal.map((row) => {
    const match = other.legal.find((candidate) => String(candidate.move) === String(row.move));
    return { move: row.move, policy: match ? row.policy - match.policy : NaN, q: match ? row.q - match.q : NaN, improved: match ? row.improved - match.improved : NaN };
  }).sort((a, b) => Math.abs(b.policy) - Math.abs(a.policy)) : [];
  return <div className="screen-grid lab-page">
    <aside className="side-column">
      <Panel title="Position editor">
        <label>Move prefix<textarea rows={9} value={text} onChange={(e) => setText(e.target.value)} /></label>
        <label>Checkpoint<select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}><option value="">No checkpoints</option>{checkpoints.map((item) => <option key={item.path} value={item.path}>{item.run} / {item.name}</option>)}</select></label>
        <label>Compare against<select value={compare} onChange={(e) => setCompare(e.target.value)}><option value="">None</option>{checkpoints.filter((item) => item.path !== checkpoint).map((item) => <option key={item.path} value={item.path}>{item.run} / {item.name}</option>)}</select></label>
        <div className="stack-buttons"><button onClick={() => void inspect()}>Inspect position</button><button onClick={() => void save()}>Save probe</button></div><ErrorBox message={error} />
      </Panel>
      <Panel title="Saved probes"><div className="data-list">{probes.data?.map((probe) => <button key={String(probe.probe_id)} onClick={() => { setCheckpoint(String(probe.checkpoint)); setText(JSON.stringify(probe.moves)); }}><span>{String(probe.name)}</span><b>{String(probe.module)}</b></button>) ?? <Empty />}</div></Panel>
    </aside>
    <main className="main-column">
      <Panel title="Model lab" action={<div className="inline-controls"><button onClick={() => void diagnostic("attention")}>Capture attention</button><button onClick={() => void diagnostic("d6")}>Check D6</button><button onClick={() => setModule("manifest")}>Manifest</button><select value={overlay} onChange={(e) => setOverlay(e.target.value as typeof overlay)}><option>policy</option><option>q</option><option>improved</option><option>rank</option></select></div>}>
        {read ? <><Board moves={read.moves} legal={read.legal.map((row) => row.move)} inspect={read} overlay={overlay} /><div className="metric-row"><Metric label="v̂" value={format(read.v_hat)} /><Metric label="KL" value={format(read.kl)} /><Metric label="H/log|A|" value={format(read.norm_entropy)} /><Metric label="legal" value={read.legal_count} /></div></> : <Empty>Paste a legal prefix and inspect a checkpoint.</Empty>}
      </Panel>
      {module === "attention" && <Panel title="Reference SDPA attention capture"><pre className="result-json">{JSON.stringify(result, null, 2)}</pre></Panel>}
      {module === "d6" && <Panel title="D6 invariance · 12 transforms"><pre className="result-json">{JSON.stringify(result, null, 2)}</pre></Panel>}
      {module === "manifest" && <Panel title="Representation & versions"><pre className="result-json">{JSON.stringify(manifest.data, null, 2)}</pre></Panel>}
      {diffs.length > 0 && <Panel title="Checkpoint delta"><div className="table-scroll"><table><thead><tr><th>move</th><th>Δ policy</th><th>Δ Q</th><th>Δ π′</th></tr></thead><tbody>{diffs.slice(0, 40).map((row) => <tr key={String(row.move)}><td>({row.move.join(", ")})</td><td>{format(row.policy)}</td><td>{format(row.q)}</td><td>{format(row.improved)}</td></tr>)}</tbody></table></div></Panel>}
    </main>
    <aside className="side-column">
      <Panel title="Policy · Q · π′"><div className="candidate-table">{read?.legal.slice().sort((a, b) => b[overlay] - a[overlay]).slice(0, 32).map((row) => <div key={String(row.move)}><b>({row.move.join(", ")})</b><span>π {format(row.policy)}</span><span>Q {format(row.q)}</span><span>π′ {format(row.improved)}</span><em>#{row.rank + 1}</em></div>) ?? <Empty />}</div></Panel>
    </aside>
  </div>;
}
