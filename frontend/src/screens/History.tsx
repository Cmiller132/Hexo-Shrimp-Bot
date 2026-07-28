import { useEffect, useState } from "react";
import { api, query, useApi } from "../api";
import Board from "../components/Board";
import { Empty, ErrorBox, format, Metric, Panel } from "../components/Ui";
import type { Move, Run } from "../types";

type GameRow = { game_id: number; kind: string; iteration: number; winner: number | null; length: number; capped: number; model_seat: number | null; opening_len: number | null; opponent_depth_mean: number | null; opening: Move[] };
type Game = GameRow & { moves: Move[]; plies: Array<Record<string, number>>; review: { tags: string[]; note: string } };

export default function History({ run, openLab }: { run?: Run; openLab: (moves: Move[]) => void }) {
  const [filters, setFilters] = useState({ kind: "", winner: "", capped: "", offset: 0 });
  const [selected, setSelected] = useState<number>();
  const [ply, setPly] = useState(0);
  const games = useApi<GameRow[]>(run ? `/api/runs/${run.name}/games?${query({ ...filters, limit: 50 })}` : null, [run?.name, filters]);
  const detail = useApi<Game>(run && selected ? `/api/runs/${run.name}/games/${selected}` : null, [selected, run?.name]);
  const calibration = useApi<Array<Record<string, number>>>(run ? `/api/runs/${run.name}/calibration` : null, [run?.name]);
  const blunders = useApi<Array<Record<string, number>>>(run ? `/api/runs/${run.name}/blunders` : null, [run?.name]);
  const openings = useApi<Array<Record<string, unknown>>>(run ? `/api/runs/${run.name}/openings` : null, [run?.name]);
  useEffect(() => { if (games.data?.length && !selected) setSelected(games.data[0].game_id); }, [games.data, selected]);
  useEffect(() => setPly(detail.data?.moves.length ?? 0), [detail.data]);
  const [note, setNote] = useState("");
  const [tags, setTags] = useState("");
  useEffect(() => { setNote(detail.data?.review.note ?? ""); setTags(detail.data?.review.tags.join(", ") ?? ""); }, [detail.data]);
  async function saveReview() {
    if (!run || !selected) return;
    await api(`/api/runs/${run.name}/games/${selected}/review`, { method: "PUT", body: JSON.stringify({ note, tags: tags.split(",").map((tag) => tag.trim()).filter(Boolean) }) });
    void detail.refresh();
  }
  if (!run) return <Empty>Select a run with telemetry.</Empty>;
  return <div className="history-page">
    <Panel title="Game browser" action={<span>{games.data?.length ?? 0} rows</span>}>
      <div className="filter-row">
        <select value={filters.kind} onChange={(e) => setFilters({ ...filters, kind: e.target.value, offset: 0 })}><option value="">all kinds</option><option>selfplay</option><option>eval</option></select>
        <select value={filters.winner} onChange={(e) => setFilters({ ...filters, winner: e.target.value, offset: 0 })}><option value="">any winner</option><option value="0">P0</option><option value="1">P1</option></select>
        <select value={filters.capped} onChange={(e) => setFilters({ ...filters, capped: e.target.value, offset: 0 })}><option value="">cap or finish</option><option value="false">finished</option><option value="true">capped</option></select>
        <button disabled={!filters.offset} onClick={() => setFilters({ ...filters, offset: Math.max(0, filters.offset - 50) })}>Previous</button><button onClick={() => setFilters({ ...filters, offset: filters.offset + 50 })}>Next</button>
      </div>
      <ErrorBox message={games.error} />
      <div className="table-scroll"><table><thead><tr><th>ID</th><th>kind</th><th>iter</th><th>winner</th><th>plies</th><th>opening (D6 canonical)</th><th>eval seat / depth</th></tr></thead>
        <tbody>{games.data?.map((game) => <tr className={selected === game.game_id ? "selected" : ""} key={game.game_id} onClick={() => setSelected(game.game_id)}><td>{game.game_id}</td><td>{game.kind}</td><td>{game.iteration ?? "—"}</td><td>{game.capped ? "cap" : `P${game.winner}`}</td><td>{game.length}</td><td>{game.opening.map((m) => `(${m})`).join(" ")}</td><td>{game.kind === "eval" ? `P${game.model_seat} / ${format(game.opponent_depth_mean)}` : "—"}</td></tr>)}</tbody>
      </table></div>
    </Panel>
    <div className="screen-grid history-detail">
      <div className="main-column">
        <Panel title={detail.data ? `Game ${detail.data.game_id} replay` : "Replay"} action={detail.data && <button onClick={() => openLab(detail.data!.moves.slice(0, ply))}>Open prefix in lab</button>}>
          {detail.data ? <>
            <Board moves={detail.data.moves.slice(0, ply)} />
            <input className="scrubber" type="range" min="0" max={detail.data.moves.length} value={ply} onChange={(e) => setPly(Number(e.target.value))} />
            <div className="metric-row"><Metric label="ply" value={`${ply}/${detail.data.moves.length}`} /><Metric label="mover v̂" value={format(detail.data.plies[Math.max(0, ply - 1)]?.v_hat)} /><Metric label="KL" value={format(detail.data.plies[Math.max(0, ply - 1)]?.kl)} /><Metric label="entropy" value={format(detail.data.plies[Math.max(0, ply - 1)]?.norm_entropy)} /><Metric label="π′ chosen" value={format(detail.data.plies[Math.max(0, ply - 1)]?.pi_chosen)} /></div>
          </> : <Empty />}
        </Panel>
        <Panel title="Calibration"><div className="table-scroll"><table><thead><tr><th>bucket</th><th>plies</th><th>v̂ mean</th><th>outcome</th><th>MAE</th></tr></thead><tbody>{calibration.data?.map((row, i) => <tr key={i}><td>{format(row.bucket_lo)}</td><td>{row.plies}</td><td>{format(row.v_hat_mean)}</td><td>{format(row.outcome_mean)}</td><td>{format(row.mae)}</td></tr>)}</tbody></table></div></Panel>
        <Panel title="Largest value swings"><div className="table-scroll"><table><thead><tr><th>game</th><th>ply</th><th>swing</th><th>rank</th><th>legal</th></tr></thead><tbody>{blunders.data?.map((row, i) => <tr key={i} onClick={() => setSelected(row.game_id)}><td>{row.game_id}</td><td>{row.t}</td><td>{format(row.swing)}</td><td>{row.rank}</td><td>{row.legal_count}</td></tr>)}</tbody></table></div></Panel>
      </div>
      <aside className="side-column">
        <Panel title="Tags & review note"><label>Tags<input value={tags} onChange={(e) => setTags(e.target.value)} placeholder="opening, inspect" /></label><label>Note<textarea rows={7} value={note} onChange={(e) => setNote(e.target.value)} /></label><button onClick={() => void saveReview()}>Save review</button></Panel>
        <Panel title="Opening atlas"><div className="data-list">{openings.data?.map((row, i) => <div key={i}><span>{JSON.stringify(row.opening)}</span><b>{String(row.games)} games · {format(row.mean_length)} plies</b></div>) ?? <Empty />}</div></Panel>
      </aside>
    </div>
  </div>;
}
