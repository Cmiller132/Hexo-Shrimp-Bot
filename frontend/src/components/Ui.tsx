import type { ReactNode } from "react";

export function Panel({ title, action, children, className = "" }: {
  title: string; action?: ReactNode; children: ReactNode; className?: string;
}) {
  return <section className={`deck-panel ${className}`}>
    <header><span>{title}</span>{action}</header>
    <div className="deck-panel-body">{children}</div>
  </section>;
}

export function ErrorBox({ message }: { message?: string }) {
  return message ? <div className="error-box">{message}</div> : null;
}

export function Metric({ label, value, unit }: { label: string; value: ReactNode; unit?: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong>{unit && <small>{unit}</small>}</div>;
}

export function Empty({ children = "No data yet." }: { children?: ReactNode }) {
  return <div className="empty-state">{children}</div>;
}

export function format(value: unknown, digits = 3): string {
  return typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : value == null ? "—" : String(value);
}
