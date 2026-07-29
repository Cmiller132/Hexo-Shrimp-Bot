import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { format } from "./Ui";

/** Categorical slots, assigned in this fixed order and never cycled. Validated
 *  all-pairs against the panel surface #121820 (worst CVD ΔE 9.1 deutan, worst
 *  normal-vision ΔE 16.5, all ≥ 3:1 contrast). `muted` is context ink, not an
 *  identity slot — it is only for a series the caller has explicitly greyed. */
export type Tone = "mint" | "blue" | "red" | "amber" | "muted";
const SLOTS: Tone[] = ["mint", "blue", "red", "amber"];

export interface Series {
  id: string;
  label: string;
  /** `[x, y]`; a null `y` breaks the path. A missing value is never inferred. */
  points: Array<[number, number | null]>;
  tone?: Tone;
  /** Redundant second cue: solid = this checkpoint now, dashed = the stored
   *  acting-net trace, dotted = the compare checkpoint. */
  dash?: "solid" | "dashed" | "dotted";
  /** Render the legend entry greyed and draw nothing — for a series that is
   *  genuinely unavailable, rather than drawing it flat at zero. */
  absent?: boolean;
  absentReason?: string;
}

export interface ChartProps {
  series: Series[];
  title?: string;
  xLabel?: string;
  yLabel?: string;
  xDomain?: [number, number];
  yDomain?: [number, number];
  yScale?: "linear" | "log";
  /** x of the current ply / iteration; drawn as a rule with a dot per series. */
  cursor?: number | null;
  /** Click or arrow keys pick the nearest x. */
  onSelect?: (x: number) => void;
  markers?: Array<{ x: number; label: string; tone?: Tone }>;
  bands?: Array<{ x0: number; x1: number; label?: string }>;
  /** Determinate bar over the plot; the already-drawn series stay visible. */
  progress?: { done: number; total: number } | null;
  height?: number;
  empty?: ReactNode;
}

const MARGIN = { top: 12, right: 58, bottom: 24, left: 46 };

function niceTicks(lo: number, hi: number, count: number): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return [lo];
  const raw = (hi - lo) / Math.max(1, count);
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? magnitude * 10;
  const out: number[] = [];
  for (let tick = Math.ceil(lo / step) * step; tick <= hi + step * 1e-6; tick += step) {
    out.push(Math.abs(tick) < step * 1e-6 ? 0 : tick);
  }
  return out;
}

function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    const value = Math.pow(10, e);
    if (value >= lo && value <= hi) out.push(value);
  }
  return out.length > 1 ? out : [lo, hi];
}

/**
 * One chart for the whole deck: LiveRun plots against iteration, History and the
 * Lab against ply. There is exactly one y-axis by design — two scales on one plot
 * invent a correlation that is not in the data.
 */
export default function Chart({
  series, title, xLabel, yLabel, xDomain, yDomain, yScale = "linear",
  cursor = null, onSelect, markers = [], bands = [], progress = null, height = 210, empty,
}: ChartProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const [hoverX, setHoverX] = useState<number | null>(null);

  useEffect(() => {
    const node = wrapRef.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width));
    observer.observe(node);
    setWidth(node.getBoundingClientRect().width);
    return () => observer.disconnect();
  }, []);

  const toned = useMemo(() => {
    let slot = 0;
    return series.map((item) => {
      if (item.tone) return { ...item, tone: item.tone };
      if (slot >= SLOTS.length) {
        throw new Error(`chart: ${series.length} series need explicit tones — the categorical order stops at ${SLOTS.length}; facet instead of cycling hues`);
      }
      return { ...item, tone: SLOTS[slot++] };
    });
  }, [series]);

  const visible = useMemo(
    () => toned.filter((item) => !item.absent && !hidden[item.id]),
    [toned, hidden],
  );

  const domains = useMemo(() => {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const item of visible) {
      for (const [x, y] of item.points) {
        if (Number.isFinite(x)) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); }
        if (y != null && Number.isFinite(y) && (yScale === "linear" || y > 0)) { y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
      }
    }
    const xd = xDomain ?? (Number.isFinite(x0) ? [x0, x1] as [number, number] : [0, 1] as [number, number]);
    let yd: [number, number];
    if (yDomain) yd = yDomain;
    else if (!Number.isFinite(y0)) yd = yScale === "log" ? [1, 10] : [0, 1];
    else if (yScale === "log") yd = [y0, y1 === y0 ? y0 * 10 : y1];
    else {
      const pad = (y1 - y0) * 0.08 || Math.abs(y1) * 0.1 || 1;
      yd = [y0 - pad, y1 + pad];
    }
    if (xd[0] === xd[1]) xd[1] = xd[0] + 1;
    if (yd[0] === yd[1]) yd[1] = yd[0] + 1;
    return { xd, yd };
  }, [visible, xDomain, yDomain, yScale]);

  const plotW = Math.max(10, width - MARGIN.left - MARGIN.right);
  const plotH = Math.max(10, height - MARGIN.top - MARGIN.bottom);
  const { xd, yd } = domains;

  const sx = useCallback((x: number) => MARGIN.left + ((x - xd[0]) / (xd[1] - xd[0])) * plotW, [xd, plotW]);
  const sy = useCallback((y: number) => {
    if (yScale === "log") {
      const lo = Math.log10(Math.max(1e-12, yd[0]));
      const hi = Math.log10(Math.max(lo + 1e-9, yd[1]));
      return MARGIN.top + plotH - ((Math.log10(Math.max(1e-12, y)) - lo) / (hi - lo)) * plotH;
    }
    return MARGIN.top + plotH - ((y - yd[0]) / (yd[1] - yd[0])) * plotH;
  }, [yd, plotH, yScale]);

  const paths = useMemo(() => visible.map((item) => {
    let line = "";
    let open = false;
    for (const [x, value] of item.points) {
      const usable = value != null && Number.isFinite(value) && (yScale === "linear" || value > 0);
      // A null breaks the path rather than joining across it: the gap is the fact.
      if (!usable) { open = false; continue; }
      const px = sx(x), py = sy(value as number);
      line += `${open ? "L" : "M"}${px.toFixed(2)},${py.toFixed(2)}`;
      open = true;
    }
    const last = [...item.points].reverse().find(([, value]) => value != null && Number.isFinite(value));
    return { item, line, last: last ? { x: sx(last[0]), y: sy(last[1] as number) } : null };
  }), [visible, sx, sy, yScale]);

  const xs = useMemo(() => {
    const set = new Set<number>();
    for (const item of visible) for (const [x] of item.points) if (Number.isFinite(x)) set.add(x);
    return [...set].sort((a, b) => a - b);
  }, [visible]);

  const nearest = useCallback((value: number) => {
    if (!xs.length) return null;
    let lo = 0, hi = xs.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] < value) lo = mid + 1; else hi = mid;
    }
    const candidate = xs[lo];
    const before = lo > 0 ? xs[lo - 1] : candidate;
    return Math.abs(before - value) <= Math.abs(candidate - value) ? before : candidate;
  }, [xs]);

  const fromClient = useCallback((clientX: number) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect || plotW <= 0) return null;
    const raw = xd[0] + ((clientX - rect.left - MARGIN.left) / plotW) * (xd[1] - xd[0]);
    return nearest(raw);
  }, [xd, plotW, nearest]);

  const valuesAt = useCallback((x: number) => toned.map((item) => {
    if (item.absent || hidden[item.id]) return null;
    const point = item.points.find(([px]) => px === x);
    return point ? { item, value: point[1] } : null;
  }).filter((entry): entry is { item: Series & { tone: Tone }; value: number | null } => entry != null), [toned, hidden]);

  const hasData = visible.some((item) => item.points.some(([, value]) => value != null));
  // A lone series takes no legend box — the title names it. Direct end-labels only
  // while there are few enough lines for them not to collide.
  const showLegend = toned.length > 1;
  const showDirect = visible.length <= 4;
  const straddles = yd[0] < 0 && yd[1] > 0;
  const ticks = yScale === "log" ? logTicks(yd[0], yd[1]) : niceTicks(yd[0], yd[1], 4);
  const xTicks = niceTicks(xd[0], xd[1], Math.max(2, Math.floor(plotW / 90)));
  const tipX = hoverX != null ? sx(hoverX) : 0;

  return <div className="deck-chart" data-empty={hasData ? undefined : ""} style={{ ["--deck-chart-h" as string]: `${height}px` }}>
    {(title || showLegend) && <div className="deck-chart-head">
      {title && <span className="deck-chart-title">{title}</span>}
      {showLegend && <div className="deck-chart-legend">
        {toned.map((item) => <button
          key={item.id}
          type="button"
          className="deck-chart-legend-item"
          data-tone={item.tone}
          data-absent={item.absent ? "" : undefined}
          aria-pressed={!item.absent && !hidden[item.id]}
          disabled={item.absent}
          title={item.absentReason}
          onClick={() => setHidden((current) => ({ ...current, [item.id]: !current[item.id] }))}
        ><i className="deck-chart-swatch" data-dash={item.dash ?? "solid"} /><span>{item.label}</span></button>)}
      </div>}
    </div>}

    <div className="deck-chart-plot-wrap" ref={wrapRef}>
      {width > 0 && <svg
        className="deck-chart-plot"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role={onSelect ? "slider" : "img"}
        tabIndex={onSelect ? 0 : undefined}
        aria-label={title ?? yLabel ?? "series"}
        aria-valuenow={onSelect && cursor != null ? cursor : undefined}
        aria-valuemin={onSelect ? xd[0] : undefined}
        aria-valuemax={onSelect ? xd[1] : undefined}
        onKeyDown={onSelect ? (event) => {
          if (!xs.length) return;
          const at = cursor == null ? 0 : xs.indexOf(nearest(cursor) ?? xs[0]);
          if (event.key === "ArrowLeft") { event.preventDefault(); onSelect(xs[Math.max(0, at - 1)]); }
          else if (event.key === "ArrowRight") { event.preventDefault(); onSelect(xs[Math.min(xs.length - 1, at + 1)]); }
          else if (event.key === "Home") { event.preventDefault(); onSelect(xs[0]); }
          else if (event.key === "End") { event.preventDefault(); onSelect(xs[xs.length - 1]); }
        } : undefined}
      >
        <g className="deck-chart-bands" aria-hidden="true">
          {bands.map((band, index) => <rect
            key={index} className="deck-chart-band"
            x={sx(band.x0)} y={MARGIN.top} width={Math.max(0, sx(band.x1) - sx(band.x0))} height={plotH}
          />)}
        </g>

        <g className="deck-chart-grid" aria-hidden="true">
          {ticks.map((tick) => <path key={tick} className="deck-chart-gridline" d={`M${MARGIN.left},${sy(tick).toFixed(2)}H${(MARGIN.left + plotW).toFixed(2)}`} />)}
        </g>
        {straddles && <line className="deck-chart-zero" x1={MARGIN.left} x2={MARGIN.left + plotW} y1={sy(0)} y2={sy(0)} />}

        <g className="deck-chart-series">
          {paths.map(({ item, line }) => <path
            key={item.id} className="deck-chart-line" data-tone={item.tone} data-dash={item.dash ?? "solid"} d={line}
          />)}
        </g>

        <g className="deck-chart-markers" aria-hidden="true">
          {markers.map((marker, index) => <g key={index}>
            <line className="deck-chart-marker" data-tone={marker.tone ?? "muted"} x1={sx(marker.x)} x2={sx(marker.x)} y1={MARGIN.top} y2={MARGIN.top + plotH} />
            <text className="deck-chart-marker-label" x={sx(marker.x) + 3} y={MARGIN.top + 8}>{marker.label}</text>
          </g>)}
        </g>

        {cursor != null && <line className="deck-chart-cursor" x1={sx(cursor)} x2={sx(cursor)} y1={MARGIN.top} y2={MARGIN.top + plotH} />}
        {cursor != null && paths.map(({ item, line }) => {
          const point = item.points.find(([x]) => x === cursor);
          if (!point || point[1] == null || !line) return null;
          return <circle key={item.id} className="deck-chart-point" data-tone={item.tone} cx={sx(cursor)} cy={sy(point[1])} r={4} />;
        })}

        {hoverX != null && <g className="deck-chart-crosshair" aria-hidden="true">
          <line x1={tipX} x2={tipX} y1={MARGIN.top} y2={MARGIN.top + plotH} />
          {valuesAt(hoverX).map((entry) => entry.value == null ? null : <circle
            key={entry.item.id} className="deck-chart-hover-dot" data-tone={entry.item.tone} cx={tipX} cy={sy(entry.value)} r={3.5}
          />)}
        </g>}

        {showDirect && paths.map(({ item, last }) => last && <text
          key={item.id} className="deck-chart-direct" x={last.x + 5} y={last.y} dominantBaseline="central"
        >{item.label}</text>)}

        <g className="deck-chart-axis deck-chart-axis-y" aria-hidden="true">
          {ticks.map((tick) => <text key={tick} className="deck-chart-tick" x={MARGIN.left - 6} y={sy(tick)} textAnchor="end" dominantBaseline="central">{format(tick)}</text>)}
        </g>
        <g className="deck-chart-axis deck-chart-axis-x" aria-hidden="true">
          {xTicks.map((tick) => <text key={tick} className="deck-chart-tick" x={sx(tick)} y={height - 8} textAnchor="middle">{format(tick)}</text>)}
        </g>
        {yLabel && <text className="deck-chart-axis-label" x={MARGIN.left - 6} y={MARGIN.top - 3} textAnchor="end">{yLabel}</text>}
        {xLabel && <text className="deck-chart-axis-label" x={MARGIN.left + plotW} y={height - 8} textAnchor="end">{xLabel}</text>}

        <rect
          className="deck-chart-hit"
          x={MARGIN.left} y={MARGIN.top} width={plotW} height={plotH}
          onPointerMove={(event) => setHoverX(fromClient(event.clientX))}
          onPointerLeave={() => setHoverX(null)}
          onClick={onSelect ? (event) => { const x = fromClient(event.clientX); if (x != null) onSelect(x); } : undefined}
          data-selectable={onSelect ? "" : undefined}
        />
      </svg>}

      {hoverX != null && width > 0 && <div className="deck-chart-tip" data-side={tipX > width / 2 ? "left" : "right"} style={{ left: `${tipX}px` }}>
        <div className="deck-chart-tip-x">{xLabel ?? "x"} {format(hoverX)}</div>
        {valuesAt(hoverX).map((entry) => <div key={entry.item.id} className="deck-chart-tip-row">
          <i className="deck-chart-swatch" data-tone={entry.item.tone} data-dash={entry.item.dash ?? "solid"} />
          <b>{entry.item.label}</b>
          <em>{entry.value == null ? "—" : format(entry.value)}</em>
        </div>)}
      </div>}

      {progress && progress.total > 0 && progress.done < progress.total && <div className="deck-chart-progress">
        <div className="deck-chart-progress-bar" style={{ ["--p" as string]: String(progress.done / progress.total) }} />
      </div>}
    </div>

    {!hasData && <div className="deck-chart-empty">{empty ?? "No measured values."}</div>}
  </div>;
}
