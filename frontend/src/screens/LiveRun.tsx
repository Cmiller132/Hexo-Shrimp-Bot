import { useEffect, useMemo, useRef, useState } from "react";
import { api, useApi } from "../api";
import Chart, { type Series, type Tone } from "../components/Chart";
import { Empty, ErrorBox, format, Metric, Panel } from "../components/Ui";
import type { EvalReadout, HorizonReadout, Run } from "../types";

const METRICS = [
  "f", "acting_norm_entropy", "acting_kl", "policy_loss", "q_loss", "critic_ce",
  "won_length_mean", "first_stone_win_rate", "v_hat_winner_mean",
  "v_hat_loser_mean", "buffer_samples", "seconds",
] as const;
const DIAGNOSTICS = ["policy_loss", "q_loss", "critic_ce", "fit_steps", "samples_per_s", "v_hat_mae", "won_length_mean"];
const HARDWARE = ["cpu_percent_mean", "rss_max", "sys_ram_used_mean", "gpu_util_mean", "gpu_power_w_mean", "gpu_mem_used_max"];
const TONES: Tone[] = ["mint", "blue", "red", "amber"];
type Row = Record<string, number | null>;
type CompareWindow = "none" | "previous" | "first";

function horizonSeries(readout: HorizonReadout | undefined, outcome: "won" | "lost", field: "sign_accuracy" | "mean_abs_v_hat") {
  return (readout?.buckets ?? [])
    .filter((row) => row.outcome === outcome)
    .map((row) => [row.k_min, row[field]] as [number, number | null]);
}

function horizonMagnitude(readout: HorizonReadout | undefined) {
  const grouped = new Map<number, { total: number; weighted: number }>();
  for (const row of readout?.buckets ?? []) {
    const entry = grouped.get(row.k_min) ?? { total: 0, weighted: 0 };
    if (row.mean_abs_v_hat != null) {
      entry.total += row.count;
      entry.weighted += row.count * row.mean_abs_v_hat;
    }
    grouped.set(row.k_min, entry);
  }
  return [...grouped].sort(([a], [b]) => a - b).map(([k, value]) => [
    k, value.total ? value.weighted / value.total : null,
  ] as [number, number | null]);
}

function opponentSeries(rows: EvalReadout[], family: "anchor" | "h2h"): Series[] {
  const grouped = new Map<number, EvalReadout[]>();
  for (const row of rows) {
    const accepted = family === "h2h" ? row.family === "h2h" : row.family === "sealbot" || row.family === "seat";
    if (accepted && row.iteration != null) grouped.set(row.opponent, [...(grouped.get(row.opponent) ?? []), row]);
  }
  return [...grouped.entries()].map(([opponent, matches], index) => {
    matches.sort((a, b) => (a.iteration ?? 0) - (b.iteration ?? 0));
    const h2h = family === "h2h";
    return {
      id: `${family}-${opponent}`,
      label: matches[0].opponent_name,
      tone: TONES[index % TONES.length],
      points: matches.map((match) => [
        match.iteration as number, h2h ? match.elo : match.win_rate,
      ] as [number, number | null]),
      interval: matches.map((match) => [
        match.iteration as number,
        h2h ? match.elo_lo : match.ci_lo,
        h2h ? match.elo_hi : match.ci_hi,
      ] as [number, number | null, number | null]),
    };
  });
}

export default function LiveRun({ run, refreshRuns }: { run?: Run; refreshRuns: () => void }) {
  const [metric, setMetric] = useState<(typeof METRICS)[number]>("f");
  const [window, setWindow] = useState(100);
  const [horizonSpan, setHorizonSpan] = useState(6);
  const [compareWindow, setCompareWindow] = useState<CompareWindow>("none");
  const [events, setEvents] = useState<Array<{ kind: string; data: Record<string, unknown> }>>([]);
  const [paused, setPaused] = useState(false);
  const [message, setMessage] = useState<string>();
  const columns = [...new Set(["iteration", ...METRICS, ...DIAGNOSTICS, ...HARDWARE])].join(",");
  const lo = run ? Math.max(0, run.iteration - window) : 0;
  const series = useApi<Row[]>(run ? `/api/runs/${run.name}/iterations?columns=${columns}&from_iteration=${lo}` : null, [run?.iteration, window]);
  const strength = useApi<EvalReadout[]>(run ? `/api/runs/${run.name}/strength` : null, [run?.iteration]);
  const manifest = useApi<{ config: Record<string, unknown>; invocations: Record<string, unknown>[] }>(run ? `/api/runs/${run.name}/manifest` : null, [run?.name]);

  const latestIteration = Math.max(0, (run?.iteration ?? 1) - 1);
  const horizonLo = Math.max(0, latestIteration - horizonSpan + 1);
  const horizonPath = run ? `/api/runs/${run.name}/horizon?lo=${horizonLo}&hi=${latestIteration}` : null;
  const horizon = useApi<HorizonReadout>(horizonPath, [run?.iteration, horizonSpan]);
  const compareHi = compareWindow === "previous" ? horizonLo - 1 : Math.min(latestIteration, horizonSpan - 1);
  const compareLo = compareWindow === "previous" ? Math.max(0, compareHi - horizonSpan + 1) : 0;
  const comparePath = run && compareWindow !== "none" && compareHi >= compareLo
    ? `/api/runs/${run.name}/horizon?lo=${compareLo}&hi=${compareHi}` : null;
  const comparison = useApi<HorizonReadout>(comparePath, [run?.iteration, horizonSpan, compareWindow]);
  const comparisonData = compareWindow === "none" ? undefined : comparison.data;

  // Use the newest UTC-stamped heartbeat from polling or the event stream.
  const [beat, setBeat] = useState<Run["heartbeat"]>(null);
  useEffect(() => setBeat(null), [run?.name]);
  const heartbeat = useMemo(() => {
    const polled = run?.heartbeat ?? null;
    if (!beat) return polled;
    if (!polled) return beat;
    return beat.updated >= polled.updated ? beat : polled;
  }, [beat, run?.heartbeat]);

  // Handler refs keep the event stream keyed by run while invoking current refresh functions.
  const onEvent = useRef<(kind: string, data: Record<string, unknown>) => void>(() => {});
  const refreshAll = useRef<() => void>(() => {});
  refreshAll.current = () => {
    void series.refresh(); void strength.refresh(); void horizon.refresh();
    if (comparePath) void comparison.refresh();
    refreshRuns();
  };
  const settle = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(settle.current), []);

  onEvent.current = (kind, data) => {
    setEvents((current) => [...current.slice(-79), { kind, data }]);
    if (kind === "heartbeat") setBeat(data as unknown as Run["heartbeat"]);
    if (kind === "lifecycle") refreshRuns();
    if (kind === "iteration") {
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
  const evalRows = strength.data ?? [];
  const evalIterations = evalRows.flatMap((row) => row.iteration == null ? [] : [row.iteration]);
  const latestEvalIteration = evalIterations.length ? Math.max(...evalIterations) : null;
  const latestEvals = latestEvalIteration == null ? [] : evalRows.filter((row) => row.iteration === latestEvalIteration);
  const anchorSeries = opponentSeries(evalRows, "anchor");
  const h2hSeries = opponentSeries(evalRows, "h2h");
  const slots = heartbeat?.collect?.slot_plies ?? [];
  const bins = useMemo(() => {
    const counts = [0, 0, 0, 0, 0];
    slots.forEach((ply) => counts[ply >= 512 ? 4 : ply >= 96 ? 3 : ply >= 32 ? 2 : ply >= 8 ? 1 : 0]++);
    return counts;
  }, [slots]);

  const primaryLabel = `${horizon.data?.lo ?? horizonLo}–${horizon.data?.hi ?? latestIteration}`;
  const compareLabel = comparisonData ? `${comparisonData.lo}–${comparisonData.hi}` : `${compareLo}–${compareHi}`;
  const accuracySeries: Series[] = [
    { id: "recent-won", label: `won ${primaryLabel}`, tone: "mint", points: horizonSeries(horizon.data, "won", "sign_accuracy") },
    { id: "recent-lost", label: `lost ${primaryLabel}`, tone: "red", points: horizonSeries(horizon.data, "lost", "sign_accuracy") },
    ...(comparisonData ? [
      { id: "compare-won", label: `won ${compareLabel}`, tone: "mint" as Tone, dash: "dotted" as const, points: horizonSeries(comparisonData, "won", "sign_accuracy") },
      { id: "compare-lost", label: `lost ${compareLabel}`, tone: "red" as Tone, dash: "dotted" as const, points: horizonSeries(comparisonData, "lost", "sign_accuracy") },
    ] : []),
  ];
  const magnitudeSeries: Series[] = [
    { id: "recent-magnitude", label: `|v̂| ${primaryLabel}`, tone: "blue", points: horizonMagnitude(horizon.data) },
    ...(comparisonData ? [{ id: "compare-magnitude", label: `|v̂| ${compareLabel}`, tone: "amber" as Tone, dash: "dotted" as const, points: horizonMagnitude(comparisonData) }] : []),
  ];

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
      <ErrorBox message={series.error ?? strength.error ?? horizon.error ?? (compareWindow !== "none" ? comparison.error : undefined) ?? message} />
      <Panel title="Iteration telemetry" action={<div className="inline-controls">
        <select value={metric} onChange={(event) => setMetric(event.target.value as typeof metric)}>{METRICS.map((name) => <option key={name}>{name}</option>)}</select>
        <select value={window} onChange={(event) => setWindow(Number(event.target.value))}>{[25, 100, 500, 2000].map((n) => <option key={n} value={n}>last {n}</option>)}</select>
      </div>}>
        <Chart title={metric} series={[{
          id: metric, label: metric,
          points: (series.data ?? []).map((row) => {
            const at = row.iteration;
            if (at == null) throw new Error(`iteration series row has no iteration column: ${JSON.stringify(row)}`);
            return [at, row[metric]] as [number, number | null];
          }),
        }]} xLabel="iteration" />
        <div className="metric-row"><Metric label="latest" value={format(latest?.[metric])} /><Metric label="points" value={series.data?.length ?? 0} /><Metric label="seconds / iteration" value={format(latest?.seconds)} /></div>
      </Panel>
      <Panel title="Knowledge horizon" action={<div className="inline-controls">
        <select aria-label="Horizon primary window" value={horizonSpan} onChange={(event) => setHorizonSpan(Number(event.target.value))}>{[6, 12, 25, 50].map((n) => <option key={n} value={n}>recent {n}</option>)}</select>
        <select aria-label="Horizon comparison window" value={compareWindow} onChange={(event) => setCompareWindow(event.target.value as CompareWindow)}>
          <option value="none">no overlay</option><option value="previous">previous window</option><option value="first">first window</option>
        </select>
      </div>}>
        <Chart title="sign accuracy" series={accuracySeries} xLabel="plies from end (bucket start)" yLabel="accuracy" yDomain={[0, 1]} />
        <Chart title="mean |v̂|" series={magnitudeSeries} xLabel="plies from end (bucket start)" yLabel="|v̂|" height={170} />
        <div className="metric-row"><Metric label="window" value={primaryLabel} /><Metric label="decisive plies" value={(horizon.data?.buckets ?? []).reduce((sum, row) => sum + row.count, 0)} /><Metric label="overlay" value={compareWindow === "none" ? "off" : compareLabel} /></div>
      </Panel>
      <Panel title="Pipeline">
        <div className="pipeline">
          <div className={heartbeat?.collect ? "active" : ""}><span>COLLECT</span><strong>{heartbeat?.collect?.steps ?? "—"} steps</strong></div>
          <b>→</b><div className={heartbeat?.fit ? "active" : ""}><span>FIT</span><strong>{heartbeat?.fit ? `${heartbeat.fit.chunk}/${heartbeat.fit.chunks}` : "idle"}</strong></div>
          <b>→</b><div className={heartbeat?.eval ? "active" : ""}><span>EVAL</span><strong>{heartbeat?.eval ? "running" : "cadence"}</strong></div>
        </div>
      </Panel>
      <Panel title={`Latest evaluation${latestEvalIteration == null ? "" : ` · iteration ${latestEvalIteration}`}`}>
        {anchorSeries.length > 0 && <Chart title="anchored win rate" series={anchorSeries} xLabel="iteration" yLabel="win rate" yDomain={[0, 1]} />}
        {h2hSeries.length > 0 && <Chart title="paired head-to-head" series={h2hSeries} xLabel="iteration" yLabel="Elo" />}
        {latestEvals.length ? <div className="eval-grid">{latestEvals.map((entry) => <div className="eval-card" key={entry.match_id}>
          <div className="eval-card-head"><b>{entry.opponent_name}</b><span>{entry.family}</span></div>
          {entry.family === "h2h" ? <div className="metric-row wrap">
            <Metric label="Elo" value={format(entry.elo, 1)} />
            <Metric label="Elo interval" value={`${format(entry.elo_lo, 1)}–${format(entry.elo_hi, 1)}`} />
            <Metric label="sign p" value={format(entry.sign_test_p)} />
            <Metric label="decisive pairs" value={entry.decisive_pairs ?? "—"} />
          </div> : <div className="metric-row wrap">
            <Metric label="win rate" value={format(entry.win_rate)} />
            <Metric label="95% CI" value={`${format(entry.ci_lo)}–${format(entry.ci_hi)}`} />
            {entry.family === "sealbot" && <Metric label="opponent depth" value={format(entry.opponent_depth_mean)} />}
          </div>}
        </div>)}</div> : <Empty />}
      </Panel>
      <Panel title="Diagnostics"><div className="data-list">{DIAGNOSTICS.map((name) => <div key={name}><span>{name}</span><b>{format(latest?.[name])}</b></div>)}</div></Panel>
      <Panel title="Hardware"><div className="metric-row wrap">{HARDWARE.map((name) => <Metric key={name} label={name} value={format(latest?.[name])} />)}</div></Panel>
    </div>
    <aside className="side-column">
      <Panel title="Run controls"><div className="stack-buttons"><button onClick={() => void control("checkpoint")}>Checkpoint now</button><button onClick={() => void control("stop")}>Stop after iteration</button><button className="danger" onClick={() => void control("kill")}>Kill process</button></div></Panel>
      <Panel title={`Collector slots · ${slots.length}`}>
        <div className="slot-grid">{slots.map((ply, index) => <i key={index} title={`slot ${index}: ${ply} plies`} data-band={ply >= 512 ? 4 : ply >= 96 ? 3 : ply >= 32 ? 2 : ply >= 8 ? 1 : 0} />)}</div>
        <div className="slot-legend">{["0–7", "8–31", "32–95", "96–511", "at cap"].map((label, index) => <span key={label}><i data-band={index} />{label} ({bins[index]})</span>)}</div>
      </Panel>
      <Panel title="Artifacts"><div className="timeline">{run.checkpoints.map((checkpoint) => <div key={checkpoint.path}><i /><span>{checkpoint.name}</span><small>{new Date(checkpoint.modified).toLocaleString()}</small></div>)}</div></Panel>
      <Panel title="Manifest"><pre className="manifest">{manifest.data ? JSON.stringify({ config: manifest.data.config, latest_invocation: manifest.data.invocations.at(-1) }, null, 2) : "Loading…"}</pre></Panel>
      <Panel title="Event stream" action={<button onClick={() => setPaused(!paused)}>{paused ? "Resume" : "Pause"}</button>}><div className="event-stream">{events.length ? [...events].reverse().map((event, index) => <div key={index}><b>{event.kind}</b><span>{JSON.stringify(event.data)}</span></div>) : <Empty />}</div></Panel>
    </aside>
  </div>;
}
