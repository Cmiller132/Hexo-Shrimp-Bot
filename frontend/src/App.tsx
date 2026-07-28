import { Activity, FlaskConical, Gamepad2, History as HistoryIcon, Menu, Search, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api, useApi } from "./api";
import { ErrorBox } from "./components/Ui";
import History from "./screens/History";
import Lab from "./screens/Lab";
import LiveRun from "./screens/LiveRun";
import Play from "./screens/Play";
import type { Move, Run, Screen } from "./types";

const NAV: Array<{ id: Screen; label: string; icon: typeof Gamepad2 }> = [
  { id: "play", label: "Play", icon: Gamepad2 },
  { id: "history", label: "Game history", icon: HistoryIcon },
  { id: "live", label: "Live run", icon: Activity },
  { id: "lab", label: "Model lab", icon: FlaskConical },
];

export default function App() {
  const [screen, setScreen] = useState<Screen>(() => (location.hash.slice(1) as Screen) || "live");
  const runs = useApi<Run[]>("/api/runs", []);
  const [runName, setRunName] = useState(localStorage.getItem("deck-run") ?? "");
  const [palette, setPalette] = useState(false);
  const [mobile, setMobile] = useState(false);
  const [handoff, setHandoff] = useState<Move[]>([]);
  const [launch, setLaunch] = useState(false);
  const [launchError, setLaunchError] = useState<string>();
  const activeRun = runs.data?.find((run) => run.name === runName) ?? runs.data?.find((run) => run.state === "active") ?? runs.data?.at(-1);
  useEffect(() => { if (activeRun && activeRun.name !== runName) { setRunName(activeRun.name); localStorage.setItem("deck-run", activeRun.name); } }, [activeRun, runName]);
  useEffect(() => {
    const key = (event: KeyboardEvent) => { if ((event.metaKey || event.ctrlKey) && event.key === "k") { event.preventDefault(); setPalette((value) => !value); } };
    addEventListener("keydown", key); return () => removeEventListener("keydown", key);
  }, []);
  const navigate = useCallback((next: Screen) => { setScreen(next); location.hash = next; setPalette(false); setMobile(false); }, []);
  function openLab(moves: Move[]) { setHandoff(moves); navigate("lab"); }
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
  return <div className="app-shell">
    <header className="topbar">
      <button className="brand" onClick={() => navigate("live")}><span>🦐</span><b>SHRIMP</b><small>CONTROL DECK</small></button>
      <nav className={mobile ? "open" : ""}>{NAV.map(({ id, label, icon: Icon }) => <button className={screen === id ? "active" : ""} key={id} onClick={() => navigate(id)}><Icon size={15} /><span>{label}</span></button>)}</nav>
      <div className="top-actions">
        <select aria-label="Run selector" value={activeRun?.name ?? ""} onChange={(e) => { setRunName(e.target.value); localStorage.setItem("deck-run", e.target.value); }}>
          {!runs.data?.length && <option>No runs</option>}{runs.data?.map((run) => <option key={run.name} value={run.name}>{run.name} · {run.state}</option>)}
        </select>
        <button className="command-trigger" onClick={() => setPalette(true)}><Search size={14} /><span>Command</span><kbd>⌘ K</kbd></button>
        <button onClick={() => setLaunch(true)}>Launch / resume</button>
        <button className="mobile-menu" onClick={() => setMobile(!mobile)}><Menu size={18} /></button>
      </div>
    </header>
    <div className="contextbar"><span className={`state-dot ${activeRun?.state ?? "stopped"}`} /> <b>{activeRun?.name ?? "No run"}</b><span>{activeRun ? `iteration ${activeRun.iteration} / ${activeRun.iterations}` : "Create a run directory to begin"}</span><span className="spacer" /><span>API {runs.error ? "offline" : "connected"}</span></div>
    <main className="screen">
      <div className="screen-heading"><div><small>{screen === "live" ? "OPERATIONS" : screen === "history" ? "TELEMETRY ARCHIVE" : screen === "lab" ? "CHECKPOINT INSTRUMENTATION" : "ENGINE-AUTHORITATIVE SESSION"}</small><h1>{NAV.find((item) => item.id === screen)?.label}</h1></div></div>
      <ErrorBox message={runs.error} />
      {screen === "live" && <LiveRun run={activeRun} refreshRuns={runs.refresh} />}
      {screen === "history" && <History run={activeRun} openLab={openLab} />}
      {screen === "play" && <Play runs={runs.data ?? []} />}
      {screen === "lab" && <Lab runs={runs.data ?? []} handoff={handoff} />}
    </main>
    {palette && <div className="command-backdrop" onClick={() => setPalette(false)}><div className="command-palette" onClick={(e) => e.stopPropagation()}><div className="command-search"><Search size={16} /><b>Navigate the deck</b><button onClick={() => setPalette(false)}><X size={16} /></button></div><div className="command-group"><span>SCREENS</span>{NAV.map(({ id, label, icon: Icon }) => <button key={id} onClick={() => navigate(id)}><Icon size={15} /><b>{label}</b><small>open</small></button>)}</div></div></div>}
    {launch && <div className="command-backdrop"><form className="launch-dialog" onSubmit={(e) => void submitLaunch(e)}><header><b>Launch or resume training</b><button type="button" onClick={() => setLaunch(false)}><X size={16} /></button></header><label>Name<input name="name" required pattern="[A-Za-z0-9][A-Za-z0-9._-]{0,79}" /></label><div className="form-grid"><label>Iterations<input name="iterations" type="number" min="1" defaultValue="100" /></label><label>Games<input name="games" type="number" min="1" defaultValue="64" /></label><label>Envs<input name="envs" type="number" min="1" defaultValue="256" /></label><label>Seed<input name="seed" type="number" defaultValue="0" /></label><label>Checkpoint cadence<input name="checkpoint_every" type="number" min="1" defaultValue="25" /></label><label>Device<select name="device"><option value="cuda">CUDA</option><option value="cpu">CPU</option></select></label></div><label>Initialize from checkpoint<input name="init_from" placeholder="run/checkpoint_000025.pt (optional)" /></label><label className="checkbox-label"><input name="resume" type="checkbox" /> Resume this named run from its latest checkpoint</label><ErrorBox message={launchError} /><button type="submit">Start process</button></form></div>}
  </div>;
}
