import { useEffect, useMemo, useRef, useState } from "react";
import { api, useApi } from "../api";
import Chart from "../components/Chart";
import { Empty, ErrorBox, format, Metric, Panel } from "../components/Ui";
import type { Run } from "../types";

const METRICS = ["f", "acting_kl", "acting_norm_entropy", "buffer_samples", "seconds"] as const;
const DIAGNOSTICS = ["policy_loss", "q_loss", "fit_steps", "samples_per_s", "v_hat_mae", "won_length_mean"];
const HARDWARE = ["cpu_percent_mean", "rss_max", "sys_ram_used_mean", "gpu_util_mean", "gpu_power_w_mean", "gpu_mem_used_max"];
type Row = Record<string, number | null>;

export default function LiveRun({ run, refreshRuns }: { run?: Run; refreshRuns: () => void }) {
  const [metric, setMetric] = useState<(typeof METRICS)[number]>("f");
  const [window, setWindow] = useState(100);
  const [events, setEvents] = useState<Array<{ kind: string; data: Record<string, unknown> }>>([]);
  const [paused, setPaused] = useState(false);
  const [message, setMessage] = useState<string>();
  const columns = ["iteration", ...METRICS, ...DIAGNOSTICS, ...HARDWARE].join(",");
  const lo = run ? Math.max(0, run.iteration - window) : 0;
  const series = useApi<Row[]>(run ? `/api/runs/${run.name}/iterations?columns=${columns}&from_iteration=${lo}` : null, [run?.iteration, window]);
  const strength = useApi<Array<Record<string, unknown>>>(run ? `/api/runs/${run.name}/strength` : null, [run?.iteration]);
  const manifest = useApi<{ config: Record<string, unknown>; invocations: Record<string, unknown>[] }>(run ? `/api/runs/${run.name}/manifest` : null, [run?.name]);

  // The pushed heartbeat. `/api/runs` is polled only when an iteration commits —
  // minutes apart on a real run — but the collector, the fit chunk and the slot
  // plies move every second, and the stream already carries them: a heartbeat
  // event is the same `status.json` object `describe()` reports as `heartbeat`.
  // Whichever of the two was written last wins; both stamp `updated` in UTC, so
  // the ISO strings order correctly as text.
  const [beat, setBeat] = useState<Run["heartbeat"]>(null);
  useEffect(() => setBeat(null), [run?.name]);
  const heartbeat = useMemo(() => {
    const polled = run?.heartbeat ?? null;
    if (!beat) return polled;
    if (!polled) return beat;
    return beat.updated >= polled.updated ? beat : polled;
  }, [beat, run?.heartbeat]);

  // The handlers close over `series.refresh` and `strength.refresh`, whose
  // identities change on every iteration because the run's iteration is part of
  // their request path. Depending on them here would tear the stream down and
  // rebuild it once per iteration, and the server replays its whole history to
  // each new connection — which is why the screen arrived in bursts. The stream
  // is identified by its run alone; the handlers ride in a ref.
  const onEvent = useRef<(kind: string, data: Record<string, unknown>) => void>(() => {});
  const refreshAll = useRef<() => void>(() => {});
  refreshAll.current = () => { void series.refresh(); void strength.refresh(); refreshRuns(); };
  const settle = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(settle.current), []);

  onEvent.current = (kind, data) => {
    setEvents((current) => [...current.slice(-79), { kind, data }]);
    if (kind === "heartbeat") setBeat(data as unknown as Run["heartbeat"]);
    if (kind === "lifecycle") refreshRuns();
    if (kind === "iteration") {
      // A new connection replays every iteration the run has ever committed, so
      // refetching per event would fire three requests per iteration of history
      // at mount — thousands on a long run. The queries ask "what is true now",
      // and one answer after the burst settles says the same thing.
      clearTimeout(settle.current);
      settle.current = setTimeout(() => refreshAll.current(), 250);
    }
  };

  useEffect(() => {
    if (!run || paused) return;
    const stream = new EventSource(`/api/runs/${run.name}/events`);
    const kinds = ["iteration", "heartbeat", "eval", "checkpoint", "log", "lifecycle"];
    const bound = kinds.map((kind) => {
      const listener = (event: Event) => onEvent.current(kind, JSON.parse((event as MessageEvent).data));
      stream.addEventListener(kind, listener);
      return [kind, listener] as const;
    });
    stream.onerror = () => setMessage("Event stream disconnected; retrying.");
    return () => {
      bound.forEach(([kind, listener]) => stream.removeEventListener(kind, listener));
      stream.close();
    };
  }, [run?.name, paused]);

  const latest = series.data?.at(-1);
  const latestEval = strength.data?.at(-1);
  const slots = heartbeat?.collect?.slot_plies ?? [];
  const bins = useMemo(() => {
    const counts = [0, 0, 0, 0, 0];
    slots.forEach((ply) => counts[ply >= 512 ? 4 : ply >= 96 ? 3 : ply >= 32 ? 2 : ply >= 8 ? 1 : 0]++);
    return counts;
  }, [slots]);

  async function control(action: "checkpoint" | "stop" | "kill") {
    if (!run) return;
    if (action === "kill" && !globalThis.confirm(`SIGTERM ${run.name}? This bypasses the graceful sentinel.`)) return;
    try {
      await api(`/api/runs/${run.name}/${action}`, { method: "POST", body: action === "kill" ? '{"confirm":true}' : undefined });
      setMessage(`${action} request accepted`);
      refreshRuns();
    } catch (reason) { setMessage(reason instanceof Error ? reason.message : String(reason)); }
  }

  if (!run) return <Empty>No run directories were found. Launch one from the header.</Empty>;
  return <div className="screen-grid live-screen">
    <div className="main-column">
      <div className="status-strip">
        <Metric label="state" value={run.working ? "working" : run.state} />
        <Metric label="iteration" value={`${run.iteration} / ${run.iterations}`} />
        <Metric label="collect" value={heartbeat?.collect ? `${heartbeat.collect.finished}/${heartbeat.collect.quota}` : "idle"} />
        <Metric label="fit" value={heartbeat?.fit ? `${heartbeat.fit.chunk}/${heartbeat.fit.chunks}` : "idle"} />
        <Metric label="eval" value={heartbeat?.eval ? `iteration ${heartbeat.eval.iteration}` : "idle"} />
      </div>
      <ErrorBox message={series.error ?? message} />
      <Panel title="Iteration telemetry" action={<div className="inline-controls">
        <select value={metric} onChange={(event) => setMetric(event.target.value as typeof metric)}>{METRICS.map((name) => <option key={name}>{name}</option>)}</select>
        <select value={window} onChange={(event) => setWindow(Number(event.target.value))}>{[25, 100, 500, 2000].map((n) => <option key={n} value={n}>last {n}</option>)}</select>
      </div>}>
        <Chart
          title={metric}
          series={[{
            id: metric,
            label: metric,
            // The x is the row's own iteration. A row without one describes an
            // iteration nobody can name, and plotting it anywhere would invent a
            // measurement; the y may be null, and the chart breaks the path there.
            points: (series.data ?? []).map((row) => {
              const at = row.iteration;
              if (at == null) throw new Error(`iteration series row has no iteration column: ${JSON.stringify(row)}`);
              return [at, row[metric]] as [number, number | null];
            }),
          }]}
          xLabel="iteration"
        />
        <div className="metric-row"><Metric label="latest" value={format(latest?.[metric])} /><Metric label="points" value={series.data?.length ?? 0} /><Metric label="seconds / iteration" value={format(latest?.seconds)} /></div>
      </Panel>
      <Panel title="Pipeline">
        <div className="pipeline">
          <div className={heartbeat?.collect ? "active" : ""}><span>COLLECT</span><strong>{heartbeat?.collect?.steps ?? "—"} steps</strong></div>
          <b>→</b><div className={heartbeat?.fit ? "active" : ""}><span>FIT</span><strong>{heartbeat?.fit ? `${heartbeat.fit.chunk}/${heartbeat.fit.chunks}` : "idle"}</strong></div>
          <b>→</b><div className={heartbeat?.eval ? "active" : ""}><span>EVAL</span><strong>{heartbeat?.eval ? "running" : "cadence"}</strong></div>
        </div>
      </Panel>
      <div className="two-column">
        <Panel title="Latest evaluation">
          {latestEval ? <div className="metric-row wrap">
            <Metric label="score" value={format(latestEval.win_rate)} />
            <Metric label="95% CI" value={`${format(latestEval.ci_lo)}–${format(latestEval.ci_hi)}`} />
            <Metric label="Elo" value={format(latestEval.elo, 1)} />
            <Metric label="as P0 / P1" value={`${format(latestEval.score_as_p0)} / ${format(latestEval.score_as_p1)}`} />
          </div> : <Empty />}
        </Panel>
        <Panel title="Diagnostics">
          <div className="data-list">{DIAGNOSTICS.map((name) => <div key={name}><span>{name}</span><b>{format(latest?.[name])}</b></div>)}</div>
        </Panel>
      </div>
      <Panel title="Hardware">
        <div className="metric-row wrap">{HARDWARE.map((name) => <Metric key={name} label={name} value={format(latest?.[name])} />)}</div>
      </Panel>
    </div>
    <aside className="side-column">
      <Panel title="Run controls">
        <div className="stack-buttons"><button onClick={() => void control("checkpoint")}>Checkpoint now</button><button onClick={() => void control("stop")}>Stop after iteration</button><button className="danger" onClick={() => void control("kill")}>Kill process</button></div>
      </Panel>
      <Panel title={`Collector slots · ${slots.length}`}>
        <div className="slot-grid">{slots.map((ply, index) => <i key={index} title={`slot ${index}: ${ply} plies`} data-band={ply >= 512 ? 4 : ply >= 96 ? 3 : ply >= 32 ? 2 : ply >= 8 ? 1 : 0} />)}</div>
        <div className="slot-legend">{["0–7", "8–31", "32–95", "96–511", "at cap"].map((label, index) => <span key={label}><i data-band={index} />{label} ({bins[index]})</span>)}</div>
      </Panel>
      <Panel title="Artifacts">
        <div className="timeline">{run.checkpoints.map((checkpoint) => <div key={checkpoint.path}><i /><span>{checkpoint.name}</span><small>{new Date(checkpoint.modified).toLocaleString()}</small></div>)}</div>
      </Panel>
      <Panel title="Manifest">
        <pre className="manifest">{manifest.data ? JSON.stringify({ config: manifest.data.config, latest_invocation: manifest.data.invocations.at(-1) }, null, 2) : "Loading…"}</pre>
      </Panel>
      <Panel title="Event stream" action={<button onClick={() => setPaused(!paused)}>{paused ? "Resume" : "Pause"}</button>}>
        <div className="event-stream">{events.length ? [...events].reverse().map((event, index) => <div key={index}><b>{event.kind}</b><span>{JSON.stringify(event.data)}</span></div>) : <Empty />}</div>
      </Panel>
    </aside>
  </div>;
}
