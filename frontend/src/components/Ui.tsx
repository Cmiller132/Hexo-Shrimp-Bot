import type { ReactNode } from "react";

export function Panel({ title, action, children }: {
  title: string; action?: ReactNode; children: ReactNode;
}) {
  return <section className="deck-panel">
    <header><span>{title}</span>{action}</header>
    <div className="deck-panel-body">{children}</div>
  </section>;
}

/** Red failure state for server, network, or missing-resource errors. */
export function ErrorBox({ message }: { message?: string }) {
  return message ? <div className="error-box">{message}</div> : null;
}

/** Mint informational notice or amber rejected-input/stale-result notice. */
export function Notice({ kind = "info", children, action }: {
  kind?: "info" | "warn"; children: ReactNode; action?: ReactNode;
}) {
  return <div className="deck-notice" data-kind={kind}><span>{children}</span>{action}</div>;
}

export function Metric({ label, value }: { label: string; value: ReactNode }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

export function Empty({ children = "No data yet." }: { children?: ReactNode }) {
  return <div className="empty-state">{children}</div>;
}

export interface SegmentedOption<T extends string> { value: T; label: ReactNode; title?: string }

export function Segmented<T extends string>({ options, value, onChange, label }: {
  options: Array<SegmentedOption<T>>; value: T; onChange: (value: T) => void; label?: string;
}) {
  return <div className="deck-segmented" role="group" aria-label={label}>
    {options.map((option) => <button
      key={option.value}
      type="button"
      aria-pressed={value === option.value}
      title={option.title}
      onClick={() => onChange(option.value)}
    >{option.label}</button>)}
  </div>;
}

/** Determinate whole-line progress with an explicit `done / total` readout. */
export function ProgressBar({ done, total, label }: { done: number; total: number; label?: string }) {
  return <div className="walk-status">
    <div className="walk-bar"><div className="walk-fill" style={{ ["--p" as string]: String(total > 0 ? Math.min(1, done / total) : 0) }} /></div>
    <span>{label ? `${label} · ` : ""}{done} / {total}</span>
  </div>;
}

/** Per-position heat-ramp key with its numeric domain.
 *  `seq` has seven logarithmic mint steps; `div` has nine symmetric violet↔amber steps. */
export function HeatLegend({ kind, max, note }: { kind: "seq" | "div"; max: number; note?: string }) {
  const steps = kind === "seq"
    ? [0, 1, 2, 3, 4, 5, 6].map((step) => `s${step}`)
    : [-4, -3, -2, -1, 0, 1, 2, 3, 4].map((step) => `d${step}`);
  return <div className="heat-legend" data-kind={kind}>
    <div className="heat-swatches">{steps.map((step) => <i key={step} className="heat-swatch" data-heat={step} />)}</div>
    <div className="heat-ticks">
      <span>{kind === "seq" ? "low" : `−${format(max)}`}</span>
      {kind === "div" && <span>0</span>}
      <span>{format(max)}</span>
    </div>
    {note && <span className="heat-note">{note}</span>}
  </div>;
}

export function format(value: unknown, digits = 3): string {
  if (typeof value !== "number") return value == null ? "—" : String(value);
  if (!Number.isFinite(value)) return String(value);
  if (value !== 0 && Math.abs(value) < 1e-3) return value.toExponential(1);
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}
