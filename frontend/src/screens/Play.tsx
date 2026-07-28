import { useEffect, useMemo, useState } from "react";
import { api, useApi } from "../api";
import Board from "../components/Board";
import { Empty, ErrorBox, format, Metric, Panel } from "../components/Ui";
import type { Checkpoint, Inspect, Move, Run } from "../types";

type Session = { session_id: string; moves: Move[]; seats: Array<Record<string, unknown>>; current_player: number | null; moves_remaining: number; terminal: boolean; winner: number | null; legal_count: number; legal_moves: Move[]; capped: boolean };

export default function Play({ runs }: { runs: Run[] }) {
  const checkpoints = useMemo(() => runs.flatMap((run) => run.checkpoints.map((checkpoint) => ({ ...checkpoint, run: run.name }))), [runs]);
  const [checkpoint, setCheckpoint] = useState(checkpoints.at(-1)?.path ?? "");
  const [opponent, setOpponent] = useState<"random" | "checkpoint" | "sealbot">("random");
  const [humanSeat, setHumanSeat] = useState(0);
  const [mode, setMode] = useState("argmax");
  const [overlay, setOverlay] = useState<"policy" | "q" | "improved" | "rank">("policy");
  const [session, setSession] = useState<Session>();
  const [inspect, setInspect] = useState<Inspect>();
  const [error, setError] = useState<string>();
  const jobs = useApi<Array<Record<string, unknown>>>("/api/matches", []);
  useEffect(() => { if (!checkpoint && checkpoints.length) setCheckpoint(checkpoints.at(-1)!.path); }, [checkpoint, checkpoints]);
  useEffect(() => {
    if (!session || !checkpoint || session.terminal) { setInspect(undefined); return; }
    void api<Inspect>(`/api/play/${session.session_id}/inspect?checkpoint=${encodeURIComponent(checkpoint)}`).then(setInspect).catch((reason) => setError(reason.message));
  }, [session, checkpoint]);

  async function create() {
    const bot = opponent === "checkpoint" ? { kind: "checkpoint", checkpoint, mode } : opponent === "sealbot" ? { kind: "sealbot", depth: 1 } : { kind: "random" };
    const seats = humanSeat === 0 ? [{ kind: "human" }, bot] : [bot, { kind: "human" }];
    try { setSession(await api("/api/play", { method: "POST", body: JSON.stringify({ seats }) })); setError(undefined); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  }
  async function move(value: Move) {
    if (!session) return;
    try { setSession(await api(`/api/play/${session.session_id}/moves`, { method: "POST", body: JSON.stringify({ move: value }) })); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  }
  async function arena(kind: "sealbot" | "checkpoint") {
    if (!checkpoint) return;
    const second = checkpoints.find((item) => item.path !== checkpoint);
    try {
      await api("/api/matches", { method: "POST", body: JSON.stringify({
        checkpoint_a: checkpoint, opponent: kind,
        checkpoint_b: kind === "checkpoint" ? second?.path : undefined, games: 8,
      }) });
      void jobs.refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  }
  return <div className="screen-grid play-page">
    <aside className="side-column">
      <Panel title="Seats">
        <label>Human seat<select value={humanSeat} onChange={(e) => setHumanSeat(Number(e.target.value))}><option value="0">P0 · blue</option><option value="1">P1 · red</option></select></label>
        <label>Opponent<select value={opponent} onChange={(e) => setOpponent(e.target.value as typeof opponent)}><option value="random">Random</option><option value="checkpoint">MantisNet checkpoint</option><option value="sealbot">SealBot depth 1</option></select></label>
        <label>Checkpoint<select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}><option value="">No checkpoints</option>{checkpoints.map((item) => <option key={item.path} value={item.path}>{item.run} / {item.name}</option>)}</select></label>
        {opponent === "checkpoint" && <label>Move mode<select value={mode} onChange={(e) => setMode(e.target.value)}><option value="argmax">argmax πθ</option><option value="sample">sample πθ</option><option value="improved">π′</option></select></label>}
        <button onClick={() => void create()}>New game</button>
        <ErrorBox message={error} />
      </Panel>
      <Panel title="Arena & quick suites"><div className="stack-buttons"><button onClick={() => void arena("sealbot")}>8-game SealBot set</button><button disabled={checkpoints.length < 2} onClick={() => void arena("checkpoint")}>Checkpoint cross-play</button></div><div className="data-list">{jobs.data?.slice(0, 5).map((job) => <div key={String(job.job_id)}><span>job {String(job.job_id)}</span><b>{String(job.status)}</b></div>)}</div></Panel>
    </aside>
    <main className="main-column">
      <Panel title={session ? `Session ${session.session_id}` : "Authoritative game board"} action={<select value={overlay} onChange={(e) => setOverlay(e.target.value as typeof overlay)}><option>policy</option><option>q</option><option>improved</option><option>rank</option></select>}>
        {session ? <><Board moves={session.moves} legal={session.legal_moves} inspect={inspect} overlay={overlay} onMove={(value) => void move(value)} /><div className="metric-row"><Metric label="ply" value={session.moves.length} /><Metric label="to move" value={session.terminal ? "finished" : `P${session.current_player}`} /><Metric label="placements left" value={session.moves_remaining} /><Metric label="legal" value={session.legal_count} /><Metric label="result" value={session.capped ? "capped" : session.terminal ? `P${session.winner} wins` : "live"} /></div></> : <Empty>Choose seats and start a game.</Empty>}
      </Panel>
    </main>
    <aside className="side-column">
      <Panel title="KLENT position read"><div className="metric-row wrap"><Metric label="v̂" value={format(inspect?.v_hat)} /><Metric label="KL" value={format(inspect?.kl)} /><Metric label="H/log|A|" value={format(inspect?.norm_entropy)} /><Metric label="legal" value={inspect?.legal_count ?? "—"} /></div></Panel>
      <Panel title="Candidates"><div className="candidate-table">{inspect?.legal.slice().sort((a, b) => b[overlay] - a[overlay]).slice(0, 20).map((row) => <div key={`${row.move}`}><b>({row.move.join(", ")})</b><span>π {format(row.policy)}</span><span>Q {format(row.q)}</span><span>π′ {format(row.improved)}</span><em>#{row.rank + 1}</em></div>) ?? <Empty />}</div></Panel>
    </aside>
  </div>;
}
