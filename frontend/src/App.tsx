import { Activity, FlaskConical, Gamepad2, History as HistoryIcon, Menu, Rocket, RotateCw, Search, Waypoints, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, useApi } from "./api";
import Pane from "./components/Pane";
import { ErrorBox } from "./components/Ui";
import History from "./screens/History";
import Lab from "./screens/Lab";
import LiveRun from "./screens/LiveRun";
import Play from "./screens/Play";
import type { Game, LabHandoff, Run, Screen } from "./types";

const NAV: Array<{ id: Screen; label: string; icon: typeof Gamepad2 }> = [
  { id: "play", label: "Play", icon: Gamepad2 },
  { id: "history", label: "Game history", icon: HistoryIcon },
  { id: "live", label: "Live run", icon: Activity },
  { id: "lab", label: "Model lab", icon: FlaskConical },
];

const EYEBROW: Record<Screen, string> = {
  play: "ENGINE-AUTHORITATIVE SESSION",
  history: "TELEMETRY ARCHIVE",
  live: "OPERATIONS",
  lab: "POSITION EXPLORER",
};

interface Command {
  id: string;
  group: string;
  label: string;
  hint: string;
  icon: typeof Gamepad2;
  run: () => void;
}

export default function App() {
  const [screen, setScreen] = useState<Screen>(() => (location.hash.slice(1) as Screen) || "live");
  const runs = useApi<Run[]>("/api/runs", []);
  const [runName, setRunName] = useState(localStorage.getItem("deck-run") ?? "");
  const [palette, setPalette] = useState(false);
  const [mobile, setMobile] = useState(false);
  const [handoff, setHandoff] = useState<LabHandoff | null>(null);
  const handoffToken = useRef(0);
  const [launch, setLaunch] = useState(false);
  const [launchError, setLaunchError] = useState<string>();
  const activeRun = runs.data?.find((run) => run.name === runName) ?? runs.data?.find((run) => run.state === "active") ?? runs.data?.at(-1);
  // History queries the active run, so that is the run any handed-off game is from.
  const activeRunName = activeRun?.name ?? "";
  useEffect(() => { if (activeRun && activeRun.name !== runName) { setRunName(activeRun.name); localStorage.setItem("deck-run", activeRun.name); } }, [activeRun, runName]);
  useEffect(() => {
    const key = (event: KeyboardEvent) => { if ((event.metaKey || event.ctrlKey) && event.key === "k") { event.preventDefault(); setPalette((value) => !value); } };
    addEventListener("keydown", key); return () => removeEventListener("keydown", key);
  }, []);
  // The key guard in the shared transport stands down while a modal is open, so the
  // dialogs advertise themselves here rather than threading a prop through screens.
  useEffect(() => { document.body.dataset.modal = palette || launch ? "open" : ""; }, [palette, launch]);

  // A screen mounts the first time it is opened and then stays mounted; see
  // `components/Pane.tsx` for why. Nothing is mounted on behalf of a screen the
  // user has not asked for, so no query fires for an unopened one.
  const [visited, setVisited] = useState<Screen[]>(() => [screen]);
  useEffect(() => { setVisited((current) => (current.includes(screen) ? current : [...current, screen])); }, [screen]);

  const navigate = useCallback((next: Screen) => { setScreen(next); location.hash = next; setPalette(false); setMobile(false); }, []);
  const selectRun = useCallback((name: string) => { setRunName(name); localStorage.setItem("deck-run", name); }, []);

  // The whole game travels to the lab, never a truncated prefix: the lab holds a line
  // and a cursor, and the cursor is where History left it. The lab consumes the
  // hand-off once, on the token — it stays mounted across a nav, so returning to it
  // shows what the user left there rather than re-seeding the last handed-off game.
  const openLab = useCallback((game: Game, ply: number) => {
    setHandoff({
      token: ++handoffToken.current, run: activeRunName, gameId: game.game_id, kind: game.kind,
      iteration: game.iteration, winner: game.winner, capped: !!game.capped, moves: game.moves, plies: game.plies, ply,
    });
    navigate("lab");
  }, [navigate, activeRunName]);

  async function submitLaunch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const values = Object.fromEntries(new FormData(event.currentTarget));
    try {
      await api("/api/runs", { method: "POST", body: JSON.stringify({
        name: values.name, iterations: Number(values.iterations), games: Number(values.games),
        envs: Number(values.envs), seed: Number(values.seed), checkpoint_every: Number(values.checkpoint_every),
        device: values.device, resume: values.resume === "on",
        init_from: values.init_from || undefined,
      }) });
      setLaunch(false); setLaunchError(undefined); void runs.refresh();
    } catch (reason) { setLaunchError(reason instanceof Error ? reason.message : String(reason)); }
  }

  /* --------------------------------------------------------------- the palette --
     Every deck-wide action the header carries, plus one entry per run — switching
     runs is the header control most often reached for, and it is a select that has
     to be found with the mouse. Screen-local work stays on its screen. */
  const commands = useMemo((): Command[] => [
    ...NAV.map(({ id, label, icon }) => ({
      id: `screen:${id}`, group: "SCREENS", label, icon,
      hint: screen === id ? "current" : "open", run: () => navigate(id),
    })),
    ...(runs.data ?? []).map((run) => ({
      id: `run:${run.name}`, group: "RUNS", label: run.name, icon: Waypoints,
      hint: `${run.state} · iteration ${run.iteration} / ${run.iterations}${run.name === activeRun?.name ? " · active" : ""}`,
      run: () => { selectRun(run.name); setPalette(false); },
    })),
    {
      id: "action:launch", group: "ACTIONS", label: "Launch or resume training", icon: Rocket,
      hint: "opens the launch dialog", run: () => { setPalette(false); setLaunch(true); },
    },
    {
      id: "action:refresh", group: "ACTIONS", label: "Reload the run list", icon: RotateCw,
      hint: "re-reads /api/runs", run: () => { void runs.refresh(); setPalette(false); },
    },
  ], [navigate, runs, screen, activeRun?.name, selectRun]);

  const [paletteQuery, setPaletteQuery] = useState("");
  const [paletteAt, setPaletteAt] = useState(0);
  const matches = useMemo(() => {
    const needle = paletteQuery.trim().toLowerCase();
    if (!needle) return commands;
    return commands.filter((command) => `${command.label} ${command.group}`.toLowerCase().includes(needle));
  }, [commands, paletteQuery]);
  useEffect(() => { setPaletteAt(0); }, [paletteQuery, palette]);
  useEffect(() => { if (!palette) setPaletteQuery(""); }, [palette]);
  const groups = useMemo(() => {
    const out: Array<{ name: string; items: Command[] }> = [];
    matches.forEach((command) => {
      const last = out.at(-1);
      if (last?.name === command.group) last.items.push(command);
      else out.push({ name: command.group, items: [command] });
    });
    return out;
  }, [matches]);

  function paletteKey(event: React.KeyboardEvent) {
    if (event.key === "Escape") { setPalette(false); return; }
    if (event.key === "ArrowDown") { event.preventDefault(); setPaletteAt((at) => Math.min(matches.length - 1, at + 1)); }
    else if (event.key === "ArrowUp") { event.preventDefault(); setPaletteAt((at) => Math.max(0, at - 1)); }
    else if (event.key === "Enter") { event.preventDefault(); matches[paletteAt]?.run(); }
  }

  return <div className="app-shell">
    <header className="topbar">
      <button className="brand" onClick={() => navigate("live")}><span>🦐</span><b>SHRIMP</b><small>CONTROL DECK</small></button>
      <nav className={mobile ? "open" : ""}>{NAV.map(({ id, label, icon: Icon }) => <button className={screen === id ? "active" : ""} key={id} onClick={() => navigate(id)}><Icon size={15} /><span>{label}</span></button>)}</nav>
      <div className="top-actions">
        <select aria-label="Run selector" value={activeRun?.name ?? ""} onChange={(e) => selectRun(e.target.value)}>
          {!runs.data?.length && <option>No runs</option>}{runs.data?.map((run) => <option key={run.name} value={run.name}>{run.name} · {run.state}</option>)}
        </select>
        <button className="command-trigger" onClick={() => setPalette(true)}><Search size={14} /><span>Command</span><kbd>⌘ K</kbd></button>
        <button onClick={() => setLaunch(true)}>Launch / resume</button>
        <button className="mobile-menu" onClick={() => setMobile(!mobile)}><Menu size={18} /></button>
      </div>
    </header>
    <div className="contextbar"><span className={`state-dot ${activeRun?.state ?? "stopped"}`} /> <b>{activeRun?.name ?? "No run"}</b><span>{activeRun ? `iteration ${activeRun.iteration} / ${activeRun.iterations}` : "Create a run directory to begin"}</span><span className="spacer" /><span>API {runs.error ? "offline" : "connected"}</span></div>
    <main className="screen">
      <div className="screen-heading"><div><small>{EYEBROW[screen]}</small><h1>{NAV.find((item) => item.id === screen)?.label}</h1></div></div>
      <ErrorBox message={runs.error} />
      {visited.includes("live") && <Pane active={screen === "live"}><LiveRun run={activeRun} refreshRuns={runs.refresh} /></Pane>}
      {visited.includes("history") && <Pane active={screen === "history"}><History run={activeRun} openLab={openLab} /></Pane>}
      {visited.includes("play") && <Pane active={screen === "play"}><Play runs={runs.data ?? []} /></Pane>}
      {visited.includes("lab") && <Pane active={screen === "lab"}><Lab runs={runs.data ?? []} run={activeRun} handoff={handoff} /></Pane>}
    </main>

    {palette && <div className="command-backdrop" onClick={() => setPalette(false)}>
      <div className="command-palette" onClick={(e) => e.stopPropagation()} onKeyDown={paletteKey}>
        <div className="command-search">
          <Search size={16} />
          <input
            autoFocus
            aria-label="command"
            placeholder="Jump to a screen, switch run, launch…"
            value={paletteQuery}
            onChange={(e) => setPaletteQuery(e.target.value)}
          />
          <button onClick={() => setPalette(false)}><X size={16} /></button>
        </div>
        <div className="command-results">
          {groups.length ? groups.map((group) => <div className="command-group" key={group.name}>
            <span>{group.name}</span>
            {group.items.map((command) => <button
              key={command.id}
              data-active={matches[paletteAt]?.id === command.id ? "" : undefined}
              onMouseEnter={() => setPaletteAt(matches.indexOf(command))}
              onClick={command.run}
            ><command.icon size={15} /><b>{command.label}</b><small>{command.hint}</small></button>)}
          </div>) : <div className="command-group"><span>NO MATCH</span><b className="command-empty">nothing matches “{paletteQuery}”</b></div>}
        </div>
      </div>
    </div>}

    {launch && <div className="command-backdrop" onKeyDown={(e) => { if (e.key === "Escape") setLaunch(false); }}><form className="launch-dialog" onSubmit={(e) => void submitLaunch(e)}><header><b>Launch or resume training</b><button type="button" onClick={() => setLaunch(false)}><X size={16} /></button></header><label>Name<input name="name" required pattern="[A-Za-z0-9][A-Za-z0-9._-]{0,79}" /></label><div className="form-grid"><label>Iterations<input name="iterations" type="number" min="1" defaultValue="100" /></label><label>Games<input name="games" type="number" min="1" defaultValue="64" /></label><label>Envs<input name="envs" type="number" min="1" defaultValue="256" /></label><label>Seed<input name="seed" type="number" defaultValue="0" /></label><label>Checkpoint cadence<input name="checkpoint_every" type="number" min="1" defaultValue="25" /></label><label>Device<select name="device"><option value="cuda">CUDA</option><option value="cpu">CPU</option></select></label></div><label>Initialize from checkpoint<input name="init_from" placeholder="run/checkpoint_000025.pt (optional)" /></label><label className="checkbox-label"><input name="resume" type="checkbox" /> Resume this named run from its latest checkpoint</label><ErrorBox message={launchError} /><button type="submit">Start process</button></form></div>}
  </div>;
}
