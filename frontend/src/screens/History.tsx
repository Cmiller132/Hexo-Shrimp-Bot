import { ChevronLeft, ChevronRight, FlaskConical, RotateCw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, query, useApi } from "../api";
import Board from "../components/Board";
import Chart, { type Series } from "../components/Chart";
import { usePaneActive } from "../components/Pane";
import Transport from "../components/Replay";
import { Empty, ErrorBox, Metric, Notice, Panel, Segmented, format } from "../components/Ui";
import { p1TurnBands, playerAt } from "../lib/hex";
import type { Game, GameRow, Move, Run, StoredPly } from "../types";

/** `/api/runs/{run}/calibration` — v̂ against the outcome it predicted, bucketed on
 *  the chosen axis. Finished self-play plies only. */
interface CalibrationRow {
  bucket: number;
  bucket_lo: number;
  plies: number;
  v_hat_mean: number;
  outcome_mean: number;
  mae: number;
}

/** `/api/runs/{run}/blunders` — a ply whose value collapsed on the next one.
 *  `swing` is in the mover's frame at `t`; `rank` is engine order, not quality. */
interface BlunderRow {
  game_id: number;
  iteration: number;
  t: number;
  mover: number;
  v_hat: number;
  v_hat_next: number;
  swing: number;
  rank: number;
  legal_count: number;
  norm_entropy: number;
  pi_chosen: number;
}

/** `/api/runs/{run}/openings` — the D6-canonical openings the run plays. */
interface OpeningRow {
  opening: Move[];
  games: number;
  p0_wins: number;
  p1_wins: number;
  capped: number;
  mean_length: number;
}

interface Filters {
  kind: string;
  winner: string;
  capped: string;
  order: string;
  from_iteration: string;
  to_iteration: string;
  min_length: string;
  max_length: string;
}

const NO_FILTERS: Filters = {
  kind: "", winner: "", capped: "", order: "recent",
  from_iteration: "", to_iteration: "", min_length: "", max_length: "",
};

const FILTER_FIELDS = Object.keys(NO_FILTERS) as Array<keyof Filters>;

/** Filter fields the server takes as integers, with the caption each is under. */
const NUMERIC_FIELDS = ["from_iteration", "to_iteration", "min_length", "max_length"] as const;
const FIELD_LABEL: Record<(typeof NUMERIC_FIELDS)[number], string> = {
  from_iteration: "iteration ≥", to_iteration: "iteration ≤",
  min_length: "plies ≥", max_length: "plies ≤",
};

const PAGE = 50;

/** Bucket width per calibration axis: v̂ is a unit interval, the other two are ply
 *  counts. Printed in the panel so the axis is never read without its width. */
const CALIBRATION_BUCKET: Record<string, number> = { v_hat: 0.1, ply: 10, length: 10 };

type TraceGroup = "value" | "swing" | "policy" | "entropy" | "kl" | "legal";

/** Elapsed-time pending state. Mounted only while a request is outstanding, so the
 *  clock starts when the request does. Every query on this screen has a measured
 *  cost in tens of seconds; a spinner with no number cannot be told from a hang. */
function Pending({ children }: { children: ReactNode }) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const started = Date.now();
    const timer = setInterval(() => setSeconds(Math.round((Date.now() - started) / 1000)), 500);
    return () => clearInterval(timer);
  }, []);
  return <div className="history-pending"><i /><span>{children}</span><b>{seconds}s</b></div>;
}

function elapsed(ms: number | null): string {
  if (ms == null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`;
}

export default function History({ run, openLab }: { run?: Run; openLab: (game: Game, ply: number) => void }) {
  /* ------------------------------------------------------------------ listing --
     Measured against the live deck's 1.4 GB telemetry database: an unfiltered page
     of 50 answers in ~40 ms, but adding `kind=selfplay` costs ~45 s, because that
     database was written before the browse indexes existed and the planner falls
     back to sorting every matching row. Nothing on this screen can fix that, so the
     screen is built so it never has to wait for it: filters are staged and applied
     deliberately rather than firing a query per keystroke, the cost of the last
     query is printed beside the row count, a game can be opened by id without the
     listing at all, and every panel loads independently. */
  const [draft, setDraft] = useState<Filters>(NO_FILTERS);
  const [applied, setApplied] = useState<Filters>(NO_FILTERS);
  const [offset, setOffset] = useState(0);
  const dirtyFilters = useMemo(
    () => FILTER_FIELDS.some((field) => draft[field] !== applied[field]),
    [draft, applied],
  );
  const isDefault = (filters: Filters) => FILTER_FIELDS.every((field) => filters[field] === NO_FILTERS[field]);
  // A non-numeric bound would be a 422 from the server. Say which field, rather
  // than dropping it and quietly querying something else.
  const badField = NUMERIC_FIELDS.find((field) => draft[field] !== "" && !/^\d+$/.test(draft[field].trim()));

  const listPath = run
    ? `/api/runs/${run.name}/games?${query({ ...applied, limit: PAGE, offset })}`
    : null;
  // `manual` so a filter change clears the held page instead of leaving last
  // query's rows under this query's heading; the effect below then asks for it.
  const games = useApi<GameRow[]>(listPath, [], { manual: true });
  const listRefresh = games.refresh;
  // Once per query, and never while the screen is hidden: this page costs tens of
  // seconds against the database the live run is writing to, so a run switch made
  // from another screen must not spend it.
  const onShow = usePaneActive();
  const queried = useRef<string | null>(null);
  useEffect(() => {
    if (!onShow || !listPath || queried.current === listPath) return;
    queried.current = listPath;
    void listRefresh();
  }, [onShow, listPath, listRefresh]);

  // What the last page actually cost. The spread between an unfiltered page and a
  // filtered one is three orders of magnitude, and only the deployed database can
  // say which one this is — so it is measured rather than predicted.
  const [listMs, setListMs] = useState<number | null>(null);
  const listStarted = useRef(0);
  const listWasLoading = useRef(false);
  useEffect(() => { setListMs(null); listStarted.current = Date.now(); }, [listPath]);
  useEffect(() => {
    if (games.loading) { listStarted.current = Date.now(); listWasLoading.current = true; return; }
    if (!listWasLoading.current) return;
    listWasLoading.current = false;
    setListMs(Date.now() - listStarted.current);
  }, [games.loading]);

  /* ------------------------------------------------------------------- a game -- */
  const [selected, setSelected] = useState<number>();
  const [ply, setPly] = useState(0);
  const wantPly = useRef<number | null>(null);
  const [gameIdInput, setGameIdInput] = useState("");

  const detail = useApi<Game>(run && selected != null ? `/api/runs/${run.name}/games/${selected}` : null, []);
  // Never render one game's payload under another game's heading while the fetch
  // for the second is still out.
  const game = detail.data?.game_id === selected ? detail.data : undefined;
  const moves = game?.moves ?? [];
  const plies = game?.plies ?? [];

  // Seed the cursor once per game, keyed on its id rather than on the payload
  // object: saving a review refetches the game, and that must not throw the
  // cursor back to the end of the line.
  const seeded = useRef<number | null>(null);
  useEffect(() => {
    if (!game || seeded.current === game.game_id) return;
    seeded.current = game.game_id;
    const want = wantPly.current;
    wantPly.current = null;
    setPly(want == null ? game.moves.length : Math.max(0, Math.min(game.moves.length, want)));
  }, [game]);

  useEffect(() => { setSelected(undefined); setOffset(0); seeded.current = null; }, [run?.name]);

  useEffect(() => {
    if (selected == null && games.data?.length) setSelected(games.data[0].game_id);
  }, [games.data, selected]);

  /** Open a game at a ply. Re-selecting the loaded game does not refetch, so the
   *  cursor moves directly; otherwise the wanted ply is applied on arrival. */
  const open = useCallback((gameId: number, at?: number) => {
    if (gameId === selected) {
      if (at != null) setPly(Math.max(0, Math.min(moves.length, at)));
      return;
    }
    wantPly.current = at ?? null;
    setSelected(gameId);
  }, [selected, moves.length]);

  const listed = games.data ?? [];
  const listIndex = listed.findIndex((row) => row.game_id === selected);

  /* ------------------------------------------------------------------- review -- */
  const [note, setNote] = useState("");
  const [tags, setTags] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string>();
  useEffect(() => {
    setNote(game?.review.note ?? "");
    setTags(game?.review.tags.join(", ") ?? "");
    setSaveError(undefined);
  }, [game]);
  const tagList = useMemo(() => tags.split(",").map((tag) => tag.trim()).filter(Boolean), [tags]);
  const reviewDirty = !!game
    && (note !== game.review.note || tagList.join("\u0000") !== game.review.tags.join("\u0000"));

  async function saveReview() {
    if (!run || !game) return;
    setSaving(true);
    setSaveError(undefined);
    try {
      await api(`/api/runs/${run.name}/games/${game.game_id}/review`, {
        method: "PUT", body: JSON.stringify({ note, tags: tagList }),
      });
      await detail.refresh();
    } catch (reason) {
      setSaveError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }

  /* -------------------------------------------------------------- the trace ---- */
  const [group, setGroup] = useState<TraceGroup>("value");
  const [frame, setFrame] = useState<"mover" | "p0">("mover");

  // The stored swing, ply by ply: the mover's own assessment across one placement,
  // with the sign flipped when the seat changes. Same definition the blunders query
  // ranks on, so a row there and this curve are the same number.
  const swingPoints = useMemo<Array<[number, number | null]>>(() => plies.map((row, index) => {
    const next = plies[index + 1];
    if (!next || next.t !== row.t + 1) return [row.t, null];
    return [row.t, (next.mover === row.mover ? next.v_hat : -next.v_hat) - row.v_hat];
  }), [plies]);

  const trace = useMemo<{ series: Series[]; yLabel: string; yDomain?: [number, number] }>(() => {
    const column = (pick: (row: StoredPly) => number): Array<[number, number | null]> =>
      plies.map((row) => [row.t, pick(row)]);
    switch (group) {
      case "value":
        return {
          yLabel: "v̂", yDomain: [-1, 1],
          series: [{
            id: "v_hat", tone: "mint",
            label: frame === "p0" ? "v̂ · P0 frame" : "v̂ · mover to play",
            points: column((row) => (frame === "p0" && row.mover === 1 ? -row.v_hat : row.v_hat)),
          }],
        };
      case "swing":
        return {
          yLabel: "Δv̂",
          series: [{ id: "swing", label: "swing · mover frame", tone: "amber", points: swingPoints }],
        };
      case "policy":
        return {
          yLabel: "π", yDomain: [0, 1],
          series: [
            { id: "pi_top1", label: "π top-1", tone: "mint", points: column((row) => row.pi_top1) },
            { id: "pi_chosen", label: "π played", tone: "blue", points: column((row) => row.pi_chosen) },
          ],
        };
      case "entropy":
        return {
          yLabel: "H", yDomain: [0, 1],
          series: [{ id: "norm_entropy", label: "normalised entropy", tone: "mint", points: column((row) => row.norm_entropy) }],
        };
      case "kl":
        return {
          yLabel: "KL",
          series: [{ id: "kl", label: "KL(π′ ‖ π)", tone: "blue", points: column((row) => row.kl) }],
        };
      case "legal":
        return {
          yLabel: "cells",
          series: [{ id: "legal_count", label: "legal moves", tone: "blue", points: column((row) => row.legal_count) }],
        };
    }
  }, [group, frame, plies, swingPoints]);

  const turnBands = useMemo(() => p1TurnBands(moves.length), [moves.length]);

  const endMarker = useMemo(() => {
    if (!game || !game.moves.length) return [];
    const label = game.capped ? "ply cap" : game.winner == null ? "end" : `P${game.winner} wins`;
    return [{ x: game.moves.length - 1, label }];
  }, [game]);

  /** The acting net's read at the cursor: `plies[t]` is the opinion it formed to
   *  choose move `t`, so a cursor with `t` stones down reads row `t`. */
  const read = plies[ply];

  /* --------------------------------------------------------------- aggregates -- */
  // The aggregates scan the same slice of the run the browser is filtered to, so a
  // window narrowed above narrows them too — and changing it clears their held
  // results, which described a different slice.
  const iterations = { from_iteration: applied.from_iteration, to_iteration: applied.to_iteration };
  const windowLabel = applied.from_iteration || applied.to_iteration
    ? `iterations ${applied.from_iteration || "0"}–${applied.to_iteration || "latest"}`
    : "the whole run";

  const [calibrationBy, setCalibrationBy] = useState("v_hat");
  const calibration = useApi<CalibrationRow[]>(
    run ? `/api/runs/${run.name}/calibration?${query({ by: calibrationBy, bucket: CALIBRATION_BUCKET[calibrationBy], ...iterations })}` : null,
    [], { manual: true },
  );

  const [threshold, setThreshold] = useState("0.5");
  const thresholdValid = /^\d*\.?\d+$/.test(threshold.trim());
  const blunders = useApi<BlunderRow[]>(
    run && thresholdValid ? `/api/runs/${run.name}/blunders?${query({ threshold, limit: 50, ...iterations })}` : null,
    [], { manual: true },
  );

  const [openingPlies, setOpeningPlies] = useState("4");
  const openings = useApi<OpeningRow[]>(
    run ? `/api/runs/${run.name}/openings?${query({ plies: openingPlies, kind: "selfplay", limit: 40, ...iterations })}` : null,
    [], { manual: true },
  );

  if (!run) return <Empty>Select a run with telemetry.</Empty>;

  const calibrationSeries: Series[] = calibration.data
    ? [
      { id: "outcome", label: "realized outcome", tone: "mint", points: calibration.data.map((row) => [row.bucket_lo, row.outcome_mean]) },
      { id: "v_hat", label: "v̂ mean", tone: "blue", points: calibration.data.map((row) => [row.bucket_lo, row.v_hat_mean]) },
      ...(calibrationBy === "v_hat"
        ? [{ id: "ideal", label: "perfect calibration", tone: "muted" as const, dash: "dotted" as const, points: calibration.data.map((row) => [row.bucket_lo, row.bucket_lo] as [number, number]) }]
        : []),
    ]
    : [];

  return <div className="history-page">
    <Panel
      title="Game browser"
      action={<span className="history-listing-cost">
        {games.loading ? "querying…" : `${listed.length} rows`}
        {listMs != null && <em> · {elapsed(listMs)}</em>}
      </span>}
    >
      <div className="history-filters">
        <label>kind<select value={draft.kind} onChange={(e) => setDraft({ ...draft, kind: e.target.value })}><option value="">all</option><option value="selfplay">selfplay</option><option value="eval">eval</option></select></label>
        <label>winner<select value={draft.winner} onChange={(e) => setDraft({ ...draft, winner: e.target.value })}><option value="">any</option><option value="0">P0</option><option value="1">P1</option></select></label>
        <label>finish<select value={draft.capped} onChange={(e) => setDraft({ ...draft, capped: e.target.value })}><option value="">cap or finish</option><option value="false">finished</option><option value="true">capped</option></select></label>
        <label>order<select value={draft.order} onChange={(e) => setDraft({ ...draft, order: e.target.value })}><option value="recent">recent</option><option value="oldest">oldest</option><option value="longest">longest</option><option value="shortest">shortest</option></select></label>
        <label>iteration ≥<input inputMode="numeric" value={draft.from_iteration} onChange={(e) => setDraft({ ...draft, from_iteration: e.target.value })} placeholder="0" /></label>
        <label>iteration ≤<input inputMode="numeric" value={draft.to_iteration} onChange={(e) => setDraft({ ...draft, to_iteration: e.target.value })} placeholder="latest" /></label>
        <label>plies ≥<input inputMode="numeric" value={draft.min_length} onChange={(e) => setDraft({ ...draft, min_length: e.target.value })} placeholder="0" /></label>
        <label>plies ≤<input inputMode="numeric" value={draft.max_length} onChange={(e) => setDraft({ ...draft, max_length: e.target.value })} placeholder="∞" /></label>
      </div>
      <div className="history-filter-actions">
        <button
          className={dirtyFilters && !badField ? "history-apply" : ""}
          disabled={!dirtyFilters || !!badField}
          onClick={() => { setApplied(draft); setOffset(0); }}
        >{dirtyFilters ? "Apply filters" : "Filters applied"}</button>
        <button disabled={isDefault(draft) && isDefault(applied) && !offset} onClick={() => { setDraft(NO_FILTERS); setApplied(NO_FILTERS); setOffset(0); }}>Reset</button>
        <button title="re-run this page" onClick={() => void games.refresh()}><RotateCw size={11} /></button>
        <span className="spacer" />
        <span className="history-page-range">{listed.length ? `${offset + 1}–${offset + listed.length}` : `from ${offset}`}</span>
        <button disabled={!offset || games.loading} onClick={() => setOffset(Math.max(0, offset - PAGE))}><ChevronLeft size={11} /></button>
        <button disabled={listed.length < PAGE || games.loading} onClick={() => setOffset(offset + PAGE)}><ChevronRight size={11} /></button>
        <form className="history-goto" onSubmit={(event) => {
          event.preventDefault();
          const id = Number(gameIdInput.trim());
          if (Number.isInteger(id) && id > 0) open(id);
        }}>
          <input aria-label="game id" inputMode="numeric" value={gameIdInput} onChange={(e) => setGameIdInput(e.target.value)} placeholder="game id" />
          <button type="submit" disabled={!/^\d+$/.test(gameIdInput.trim())}>Open</button>
        </form>
      </div>
      {badField
        ? <Notice kind="warn">“{FIELD_LABEL[badField]}” is not a whole number: “{draft[badField]}”.</Notice>
        : dirtyFilters && <Notice kind="warn">Filters changed but not applied — the table still shows the applied query.</Notice>}
      <ErrorBox message={games.error} />
      {games.loading
        ? <Pending>listing this page{applied.kind ? " · a kind filter is unindexed on a large run and can cost ~45 s" : ""} · the rest of the screen stays live</Pending>
        : !games.requested ? <Empty>Not requested.</Empty>
        : !listed.length ? <Empty>No games match this query.</Empty>
        : <div className="table-scroll history-table"><table>
          <thead><tr><th>ID</th><th>kind</th><th>iter</th><th>winner</th><th>plies</th><th>opening (D6 canonical)</th><th>eval seat / depth</th></tr></thead>
          <tbody>{listed.map((row) => <tr
            key={row.game_id}
            className={selected === row.game_id ? "selected" : ""}
            onClick={() => open(row.game_id)}
          >
            <td>{row.game_id}</td>
            <td>{row.kind}</td>
            <td>{row.iteration ?? "—"}</td>
            <td>{row.capped ? "cap" : row.winner == null ? "—" : `P${row.winner}`}</td>
            <td>{row.length}</td>
            <td>{row.opening.map((move) => `(${move})`).join(" ")}</td>
            <td>{row.kind === "eval" ? `P${row.model_seat} / ${format(row.opponent_depth_mean)}` : "—"}</td>
          </tr>)}</tbody>
        </table></div>}
    </Panel>

    <div className="screen-grid history-detail">
      <div className="main-column">
        <Panel
          title={game ? `Game ${game.game_id} · ${game.kind}` : "Replay"}
          action={<div className="history-panel-actions">
            <button disabled={listIndex <= 0} title="previous listed game" onClick={() => open(listed[listIndex - 1].game_id)}><ChevronLeft size={11} /></button>
            <button disabled={listIndex < 0 || listIndex >= listed.length - 1} title="next listed game" onClick={() => open(listed[listIndex + 1].game_id)}><ChevronRight size={11} /></button>
            <button disabled={!game} onClick={() => game && openLab(game, ply)}><FlaskConical size={11} /> Open in lab</button>
          </div>}
        >
          <ErrorBox message={detail.error} />
          {!game
            ? detail.loading ? <Pending>loading the game</Pending> : <Empty>Pick a game, or open one by id.</Empty>
            : <>
              <div className="history-board">
                <Board
                  moves={moves}
                  cursor={ply}
                  played={ply < moves.length ? moves[ply] : null}
                  onStone={(index) => setPly(index + 1)}
                  toolbar
                  height={460}
                  caption={`Game ${game.game_id} after ${ply} of ${moves.length} plies`}
                />
              </div>
              <Transport length={moves.length} value={ply} onChange={setPly} />
              <div className="metric-row wrap">
                <Metric label="ply" value={`${ply} / ${moves.length}`} />
                <Metric label="to play" value={ply < moves.length ? `P${playerAt(ply)}` : "—"} />
                <Metric label="plays" value={ply < moves.length ? `(${moves[ply]})` : "game over"} />
                <Metric label="acting v̂" value={format(read?.v_hat)} />
                <Metric label="KL" value={format(read?.kl)} />
                <Metric label="entropy" value={format(read?.norm_entropy)} />
                <Metric label="π top-1" value={format(read?.pi_top1)} />
                <Metric label="π played" value={format(read?.pi_chosen)} />
                <Metric label="legal" value={read ? read.legal_count : "—"} />
              </div>
            </>}
        </Panel>

        <Panel title="Per-ply trace · the acting net's own read">
          {!game ? <Empty />
            : !plies.length
              ? <Notice>Evaluation games store no per-ply trace — the deck records scalars for self-play only. The move list above replays in full.</Notice>
              : <>
                <div className="history-trace-controls">
                  <Segmented
                    label="trace series"
                    value={group}
                    onChange={setGroup}
                    options={[
                      { value: "value", label: "v̂", title: "value head, as the mover saw it" },
                      { value: "swing", label: "swing", title: "how far v̂ moved across one placement" },
                      { value: "policy", label: "π", title: "top-1 policy against the policy of the move played" },
                      { value: "entropy", label: "H", title: "normalised policy entropy" },
                      { value: "kl", label: "KL", title: "KL of the improved target against the policy" },
                      { value: "legal", label: "legal", title: "legal move count" },
                    ]}
                  />
                  {group === "value" && <Segmented
                    label="value frame"
                    value={frame}
                    onChange={setFrame}
                    options={[
                      { value: "mover", label: "mover", title: "as stored: the seat to play at that ply" },
                      { value: "p0", label: "P0", title: "sign flipped on P1 plies, so one curve reads as one player's fortunes" },
                    ]}
                  />}
                  <span className="history-trace-hint">click the plot to move the cursor</span>
                </div>
                <Chart
                  series={trace.series}
                  yLabel={trace.yLabel}
                  yDomain={trace.yDomain}
                  xLabel="ply"
                  xDomain={[0, Math.max(1, moves.length)]}
                  cursor={ply}
                  onSelect={setPly}
                  bands={turnBands}
                  markers={endMarker}
                  height={215}
                />
              </>}
        </Panel>
      </div>

      <aside className="side-column">
        <Panel title="Game">
          {!game ? <Empty /> : <div className="data-list">
            <div><span>id</span><b>{game.game_id}</b></div>
            <div><span>kind</span><b>{game.kind}</b></div>
            <div><span>iteration</span><b>{game.iteration}</b></div>
            <div><span>index</span><b>{game.match == null ? `#${game.game_index}` : `match ${game.match} · #${game.game_index}`}</b></div>
            <div><span>result</span><b>{game.capped ? "hit the ply cap" : game.winner == null ? "unfinished" : `P${game.winner} wins`}</b></div>
            <div><span>plies</span><b>{game.moves.length}</b></div>
            <div><span>stored plies</span><b>{plies.length || "none"}</b></div>
            {game.kind === "eval" && <>
              <div><span>model seat</span><b>P{game.model_seat}</b></div>
              <div><span>opponent depth</span><b>{format(game.opponent_depth_mean)}</b></div>
              <div><span>scripted opening</span><b>{game.opening_len ?? 0} plies</b></div>
            </>}
          </div>}
        </Panel>

        <Panel
          title="Tags & review note"
          action={reviewDirty ? <span className="history-dirty">unsaved</span> : undefined}
        >
          {!game ? <Empty>Pick a game to review it.</Empty> : <>
            <label>Tags<input value={tags} onChange={(e) => setTags(e.target.value)} placeholder="opening, blunder, inspect" /></label>
            {!!game.review.tags.length && <div className="history-tags">{game.review.tags.map((tag) => <i key={tag}>{tag}</i>)}</div>}
            <label>Note<textarea rows={7} value={note} onChange={(e) => setNote(e.target.value)} /></label>
            <ErrorBox message={saveError} />
            <button disabled={!reviewDirty || saving} onClick={() => void saveReview()}>{saving ? "Saving…" : "Save review"}</button>
          </>}
        </Panel>
      </aside>
    </div>

    <div className="history-aggregates">
      <Panel
        title="Calibration"
        action={<button disabled={calibration.loading} onClick={() => void calibration.refresh()}>{calibration.requested ? "Re-run" : "Run"}</button>}
      >
        <div className="history-trace-controls">
          <Segmented
            label="calibration axis"
            value={calibrationBy}
            onChange={setCalibrationBy}
            options={[
              { value: "v_hat", label: "by v̂", title: "reliability diagram, 0.1-wide buckets" },
              { value: "ply", label: "by ply", title: "10-ply buckets" },
              { value: "length", label: "by length", title: "10-ply buckets" },
            ]}
          />
          <span className="history-trace-hint">bucket {CALIBRATION_BUCKET[calibrationBy]} · {windowLabel}</span>
        </div>
        {calibration.loading ? <Pending>scanning every finished self-play ply · ~50 s on the live run</Pending>
          : calibration.error ? <ErrorBox message={calibration.error} />
          : !calibration.requested ? <Empty>Not requested — this reads every stored ply and took ~50 s on the live run.</Empty>
          : !calibration.data?.length ? <Empty>No finished self-play plies in {windowLabel}.</Empty>
          : <>
            <Chart
              series={calibrationSeries}
              xLabel={calibrationBy === "v_hat" ? "v̂ bucket" : calibrationBy === "ply" ? "ply bucket" : "game length bucket"}
              yLabel="outcome"
              height={190}
            />
            <div className="table-scroll history-table"><table>
              <thead><tr><th>bucket</th><th>plies</th><th>v̂ mean</th><th>outcome</th><th>MAE</th></tr></thead>
              <tbody>{calibration.data.map((row) => <tr key={row.bucket}>
                <td>{format(row.bucket_lo)}</td><td>{row.plies.toLocaleString()}</td>
                <td>{format(row.v_hat_mean)}</td><td>{format(row.outcome_mean)}</td><td>{format(row.mae)}</td>
              </tr>)}</tbody>
            </table></div>
          </>}
      </Panel>

      <Panel
        title="Largest value swings"
        action={<button disabled={blunders.loading || !thresholdValid} onClick={() => void blunders.refresh()}>{blunders.requested ? "Re-run" : "Run"}</button>}
      >
        <div className="history-trace-controls">
          <label className="history-inline-field">|swing| ≥<input inputMode="decimal" value={threshold} onChange={(e) => setThreshold(e.target.value)} /></label>
          <span className="history-trace-hint">{windowLabel} · a row opens that game at that ply</span>
        </div>
        {!thresholdValid && <Notice kind="warn">“{threshold}” is not a number, so there is no threshold to query.</Notice>}
        {blunders.loading ? <Pending>joining every ply to its successor · ~75 s on the live run</Pending>
          : blunders.error ? <ErrorBox message={blunders.error} />
          : !blunders.requested ? <Empty>Not requested — this joins every ply to its successor and took ~75 s on the live run.</Empty>
          : !blunders.data?.length ? <Empty>No swing of {threshold} or more in {windowLabel}.</Empty>
          : <div className="table-scroll history-table"><table>
            <thead><tr><th>game</th><th>iter</th><th>ply</th><th>mover</th><th>v̂</th><th>→ v̂</th><th>swing</th><th>order</th><th>legal</th></tr></thead>
            <tbody>{blunders.data.map((row) => <tr
              key={`${row.game_id}-${row.t}`}
              className={selected === row.game_id ? "selected" : ""}
              onClick={() => open(row.game_id, row.t)}
            >
              <td>{row.game_id}</td><td>{row.iteration}</td><td>{row.t}</td><td>P{row.mover}</td>
              <td>{format(row.v_hat)}</td><td>{format(row.v_hat_next)}</td>
              <td className={row.swing < 0 ? "history-fall" : "history-rise"}>{format(row.swing)}</td>
              <td>{row.rank}</td><td>{row.legal_count}</td>
            </tr>)}</tbody>
          </table></div>}
      </Panel>

      <Panel
        title="Opening atlas"
        action={<button disabled={openings.loading} onClick={() => void openings.refresh()}>{openings.requested ? "Re-run" : "Run"}</button>}
      >
        <div className="history-trace-controls">
          <Segmented
            label="opening depth"
            value={openingPlies}
            onChange={setOpeningPlies}
            options={[{ value: "2", label: "2" }, { value: "4", label: "4" }, { value: "6", label: "6" }, { value: "8", label: "8" }]}
          />
          <span className="history-trace-hint">plies, D6-canonical · self-play · {windowLabel}</span>
        </div>
        {openings.loading ? <Pending>canonicalising every self-play opening · ~50 s on the live run</Pending>
          : openings.error ? <ErrorBox message={openings.error} />
          : !openings.requested ? <Empty>Not requested — this reads every self-play move blob and took ~50 s on the live run.</Empty>
          : !openings.data?.length ? <Empty>No self-play game reaches {openingPlies} plies in {windowLabel}.</Empty>
          : <div className="table-scroll history-table"><table>
            <thead><tr><th>opening</th><th>games</th><th>P0 rate</th><th>P1 rate</th><th>cap</th><th>mean plies</th></tr></thead>
            <tbody>{openings.data.map((row) => <tr key={row.opening.map((move) => move.join(",")).join(";")} className="history-static-row">
              <td>{row.opening.map((move) => `(${move})`).join(" ")}</td>
              <td>{row.games.toLocaleString()}</td>
              <td>{format(row.p0_wins / row.games)}</td>
              <td>{format(row.p1_wins / row.games)}</td>
              <td>{row.capped}</td>
              <td>{format(row.mean_length)}</td>
            </tr>)}</tbody>
          </table></div>}
      </Panel>
    </div>
  </div>;
}
