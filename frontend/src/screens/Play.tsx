import { useCallback, useEffect, useMemo, useState } from "react";
import { api, reasonText, useApi } from "../api";
import Board from "../components/Board";
import Transport from "../components/Replay";
import { Empty, ErrorBox, format, Metric, Notice, Panel } from "../components/Ui";
import { lineKey, moveKey } from "../lib/hex";
import { withPolicyRank } from "../lib/inspect";
import type { Candidate, Inspect, Move, Run } from "../types";

/** `/api/play/{id}` — the engine's own view of a session. `legal_moves` is the
 *  authoritative legality mask, so the board is playable with no model loaded. */
interface Session {
  session_id: string;
  moves: Move[];
  current_player: number | null;
  moves_remaining: number;
  terminal: boolean;
  winner: number | null;
  legal_count: number;
  legal_moves: Move[];
  capped: boolean;
}

interface MatchJob { job_id: number | string; status: string }

export default function Play({ runs }: { runs: Run[] }) {
  const checkpoints = useMemo(
    () => runs.flatMap((run) => run.checkpoints.map((checkpoint) => ({ ...checkpoint, run: run.name }))),
    [runs],
  );
  const [checkpoint, setCheckpoint] = useState(checkpoints.at(-1)?.path ?? "");
  const [opponent, setOpponent] = useState<"random" | "checkpoint" | "sealbot">("random");
  const [humanSeat, setHumanSeat] = useState(0);
  const [mode, setMode] = useState("argmax");
  const [session, setSession] = useState<Session>();
  const [error, setError] = useState<string>();
  // The live cursor equals `session.moves.length`; earlier cursors are read-only.
  const [cursor, setCursor] = useState(0);
  const [selected, setSelected] = useState<Move | null>(null);
  const jobs = useApi<MatchJob[]>("/api/matches", []);

  useEffect(() => { if (!checkpoint && checkpoints.length) setCheckpoint(checkpoints.at(-1)!.path); }, [checkpoint, checkpoints]);

  const live = session ? session.moves.length : 0;
  const atLive = cursor === live;
  useEffect(() => { setCursor(live); setSelected(null); }, [live, session?.session_id]);

  /* The checkpoint read is keyed by the complete session and accepted only when
     its move prefix still matches the live position. */
  const inspect = useApi<Inspect>(
    session && checkpoint && !session.terminal
      ? `/api/play/${session.session_id}/inspect?checkpoint=${encodeURIComponent(checkpoint)}`
      : null,
    [session],
  );

  // Loading, failed, or mismatched reads fall back to the engine legality mask.
  const read = useMemo(
    () => !inspect.loading && !inspect.error && inspect.data && session && checkpoint
      && lineKey(inspect.data.moves) === lineKey(session.moves)
      ? inspect.data
      : undefined,
    [inspect.data, inspect.loading, inspect.error, session, checkpoint],
  );
  // The board receives either scored candidates or an unscored legality mask.
  const candidates = useMemo((): Candidate[] | undefined => {
    if (!read || !session || !atLive || session.terminal) return undefined;
    return withPolicyRank(read.legal);
  }, [read, session, atLive]);
  const legalMask = !session || read || !atLive || session.terminal ? undefined : session.legal_moves;
  const topMove = useMemo(() => candidates?.find((row) => row.policyRank === 0)?.move, [candidates]);

  async function create() {
    const bot = opponent === "checkpoint" ? { kind: "checkpoint", checkpoint, mode }
      : opponent === "sealbot" ? { kind: "sealbot", depth: 1 }
      : { kind: "random" };
    const seats = humanSeat === 0 ? [{ kind: "human" }, bot] : [bot, { kind: "human" }];
    try { setSession(await api("/api/play", { method: "POST", body: JSON.stringify({ seats }) })); setError(undefined); }
    catch (reason) { setError(reasonText(reason)); }
  }

  const move = useCallback(async (value: Move) => {
    if (!session) return;
    try {
      setSession(await api(`/api/play/${session.session_id}/moves`, { method: "POST", body: JSON.stringify({ move: value }) }));
      setError(undefined);
    } catch (reason) { setError(reasonText(reason)); }
  }, [session]);

  async function arena(kind: "sealbot" | "checkpoint") {
    if (!checkpoint) return;
    const second = checkpoints.find((item) => item.path !== checkpoint);
    try {
      await api("/api/matches", { method: "POST", body: JSON.stringify({
        checkpoint_a: checkpoint, opponent: kind,
        checkpoint_b: kind === "checkpoint" ? second?.path : undefined, games: 8,
      }) });
      void jobs.refresh();
    } catch (reason) { setError(reasonText(reason)); }
  }

  return <div className="screen-grid play-page">
    <aside className="side-column">
      <Panel title="Seats">
        <label>Human seat<select value={humanSeat} onChange={(e) => setHumanSeat(Number(e.target.value))}><option value="0">P0 · blue</option><option value="1">P1 · red</option></select></label>
        <label>Opponent<select value={opponent} onChange={(e) => setOpponent(e.target.value as typeof opponent)}><option value="random">Random</option><option value="checkpoint">MantisNet checkpoint</option><option value="sealbot">SealBot depth 1</option></select></label>
        <label>Checkpoint<select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
          <option value="">No checkpoint · engine only</option>
          {runs.filter((run) => run.checkpoints.length).map((run) => <optgroup key={run.name} label={run.name}>
            {run.checkpoints.map((item) => <option key={item.path} value={item.path}>{item.name}{item.iteration != null ? ` · iter ${item.iteration}` : ""}</option>)}
          </optgroup>)}
        </select></label>
        {opponent === "checkpoint" && <label>Move mode<select value={mode} onChange={(e) => setMode(e.target.value)}><option value="argmax">argmax πθ</option><option value="sample">sample πθ</option><option value="improved">π′</option></select></label>}
        <button onClick={() => void create()}>New game</button>
        <ErrorBox message={error} />
      </Panel>
      <Panel title="Arena & quick suites">
        <div className="stack-buttons">
          <button onClick={() => void arena("sealbot")}>8-game SealBot set</button>
          <button disabled={checkpoints.length < 2} onClick={() => void arena("checkpoint")}>Checkpoint cross-play</button>
        </div>
        <div className="data-list">{jobs.data?.length
          ? jobs.data.slice(0, 5).map((job) => <div key={String(job.job_id)}><span>job {job.job_id}</span><b>{job.status}</b></div>)
          : <Empty>No arena jobs.</Empty>}</div>
      </Panel>
    </aside>

    <main className="main-column">
      <Panel title={session ? `Session ${session.session_id}` : "Authoritative game board"}>
        {!session ? <Empty>Choose seats and start a game.</Empty> : <>
          {!atLive && <Notice kind="info" action={<button type="button" onClick={() => setCursor(live)}>Back to the live position</button>}>
            Reviewing ply {cursor} of {live} — the engine only accepts a placement from the live position.
          </Notice>}
          <Board
            moves={session.moves}
            cursor={cursor}
            legal={candidates}
            mask={legalMask}
            selected={selected}
            onSelect={(value) => void move(value)}
            onStone={(ply) => setCursor(ply + 1)}
            interactive={atLive && !session.terminal}
            toolbar
            status={inspect.loading && atLive ? "pending" : "idle"}
            height={500}
            caption={session.terminal
              ? `Finished after ${session.moves.length} plies`
              : `Live position, P${session.current_player} to place`}
          />
          <Transport length={live} value={cursor} onChange={setCursor} />
          <div className="metric-row wrap">
            <Metric label="ply" value={`${cursor} / ${live}`} />
            <Metric label="to move" value={session.terminal ? "finished" : `P${session.current_player}`} />
            <Metric label="placements left" value={session.moves_remaining} />
            <Metric label="legal" value={session.legal_count} />
            <Metric label="result" value={session.capped ? "capped" : session.terminal ? `P${session.winner} wins` : "live"} />
          </div>
          <div className="kbd-hint">click a legal cell to place · click a stone to review that ply · ← → step · shift ±10 · space play</div>
        </>}
      </Panel>
    </main>

    <aside className="side-column">
      <Panel title="Checkpoint read of this position">
        <ErrorBox message={inspect.error} />
        {!session ? <Empty>Start a game to read a position.</Empty>
          : !checkpoint ? <Empty>No checkpoint selected — the engine is refereeing, but nothing is reading the position.</Empty>
          : session.terminal ? <Empty>The game is over — there is nothing to choose.</Empty>
          : !atLive ? <Empty>Reviewing ply {cursor}; the read follows the live position.</Empty>
          : !read ? <Empty>reading…</Empty>
          : <>
            <div className="metric-row wrap">
              <Metric label="v̂" value={format(read.v_hat)} />
              <Metric label="KL" value={format(read.kl)} />
              <Metric label="H/log|A|" value={format(read.norm_entropy)} />
              <Metric label="legal" value={read.legal_count} />
            </div>
            <div className="play-note">
              model's top move {topMove ? `(${topMove.join(", ")})` : "—"} · τ {read.tau} · λ {read.lam} · {read.moves_remaining} stone{read.moves_remaining === 1 ? "" : "s"} left in this turn
            </div>
          </>}
      </Panel>
      <Panel title="Candidates" action={<span>{read ? candidates?.length ?? 0 : "—"}</span>}>
        {read && candidates?.length
          ? <div className="candidate-table" style={{ ["--cand-cols" as string]: "1fr 50px 50px 50px 26px" }} data-pending={inspect.loading ? "" : undefined}>
            <div className="candidate-head"><b>move</b><span>π</span><span>Q</span><span>π′</span><em>#</em></div>
            {candidates.slice().sort((a, b) => a.policyRank - b.policyRank).slice(0, 24).map((row) => {
              const isSelected = selected != null && moveKey(selected) === moveKey(row.move);
              return <button
                key={moveKey(row.move)}
                type="button"
                aria-pressed={isSelected}
                onClick={() => setSelected(isSelected ? null : row.move)}
              >
                <b>({row.move.join(", ")})</b>
                <span>{format(row.policy)}</span>
                <span>{format(row.q)}</span>
                <span>{format(row.improved)}</span>
                <em>#{row.policyRank + 1}</em>
              </button>;
            })}
          </div>
          : <Empty>{session && !checkpoint ? "Select a checkpoint to rank the legal moves." : "No candidate set for this position."}</Empty>}
      </Panel>
    </aside>
  </div>;
}
