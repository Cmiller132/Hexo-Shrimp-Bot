import type { Inspect, Move } from "../types";

const moveKey = (move: Move) => `${move[0]},${move[1]}`;

export default function Board({ moves, legal = [], inspect, overlay = "policy", onMove }: {
  moves: Move[]; legal?: Move[]; inspect?: Inspect; overlay?: "policy" | "q" | "improved" | "rank"; onMove?: (move: Move) => void;
}) {
  const cells = new Map<string, { move: Move; player?: number; legal?: boolean; value?: number }>();
  moves.forEach((move, index) => cells.set(moveKey(move), { move, player: playerAt(index) }));
  legal.forEach((move) => {
    const row = inspect?.legal.find((candidate) => moveKey(candidate.move) === moveKey(move));
    cells.set(moveKey(move), { move, legal: true, value: row ? overlay === "rank" ? row.rank : row[overlay] : undefined });
  });
  const extent = Math.max(3, ...[...cells.values()].map(({ move: [q, r] }) => Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r))));
  const size = Math.max(10, 28 - extent);
  const points = [...cells.values()].map((cell) => {
    const [q, r] = cell.move;
    return { ...cell, x: q * size * 1.73 + r * size * .866, y: r * size * 1.5 };
  });
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const minX = Math.min(-100, ...xs) - size * 2, maxX = Math.max(100, ...xs) + size * 2;
  const minY = Math.min(-100, ...ys) - size * 2, maxY = Math.max(100, ...ys) + size * 2;
  return <svg className="hex-board" viewBox={`${minX} ${minY} ${maxX - minX} ${maxY - minY}`} role="img" aria-label={`Hexo position after ${moves.length} plies`}>
    {points.map((cell) => {
      const intensity = cell.value == null ? 0 : overlay === "q" ? (cell.value + 1) / 2 : overlay === "rank" ? 1 / (cell.value + 1) : cell.value;
      const fill = cell.player === 0 ? "#58a6ff" : cell.player === 1 ? "#ef6a73" : `rgba(129,216,178,${.08 + Math.max(0, Math.min(1, intensity)) * .7})`;
      const hex = Array.from({ length: 6 }, (_, i) => {
        const angle = Math.PI / 180 * (60 * i + 30);
        return `${cell.x + size * Math.cos(angle)},${cell.y + size * Math.sin(angle)}`;
      }).join(" ");
      return <g key={moveKey(cell.move)} className={cell.legal ? "legal-cell" : ""} onClick={() => cell.legal && onMove?.(cell.move)}>
        <polygon points={hex} fill={fill} stroke={cell.legal ? "#81d8b2" : "#27323e"} strokeWidth={cell.legal ? 1.5 : 1} />
        {cell.legal && cell.value != null && <text x={cell.x} y={cell.y + 3} textAnchor="middle">{overlay === "rank" ? `#${cell.value + 1}` : cell.value.toFixed(2)}</text>}
      </g>;
    })}
  </svg>;
}

function playerAt(ply: number) {
  if (ply === 0) return 0;
  return Math.floor((ply - 1) / 2) % 2 ? 0 : 1;
}
