export default function Chart({ rows, column, color = "#81d8b2" }: { rows: Record<string, number | null>[]; column: string; color?: string }) {
  const values = rows.map((row) => row[column]).filter((value): value is number => typeof value === "number");
  if (!values.length) return <div className="empty-state">No measured values.</div>;
  const min = Math.min(...values), max = Math.max(...values), span = max - min || 1;
  const points = rows.map((row, index) => {
    const value = typeof row[column] === "number" ? row[column]! : min;
    return `${(index / Math.max(1, rows.length - 1)) * 100},${92 - ((value - min) / span) * 80}`;
  }).join(" ");
  return <svg className="trend-chart" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label={`${column} over iteration`}>
    <polyline points={points} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
  </svg>;
}
