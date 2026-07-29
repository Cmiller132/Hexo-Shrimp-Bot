import type { Move } from "../types";

/** The three axes a line can run along, in axial `(q, r)` steps. */
export const AXES: Move[] = [[1, 0], [0, 1], [1, -1]];

export const moveKey = (move: Move) => `${move[0]},${move[1]}`;
export const lineKey = (moves: Move[]) => moves.map(moveKey).join(";");

/** Player that placed ply `i`: P0 opens, then players alternate two-stone turns. */
export function playerAt(ply: number): 0 | 1 {
  if (ply === 0) return 0;
  return Math.floor((ply - 1) / 2) % 2 ? 0 : 1;
}

/**
 * Returns the six-or-more run through the last placed stone, or null.
 * Only that stone can complete a new winning run.
 */
export function findWinningLine(moves: Move[], cursor?: number): Move[] | null {
  const end = Math.min(cursor ?? moves.length, moves.length);
  if (end < 6) return null;
  const owner = new Map<string, number>();
  for (let i = 0; i < end; i++) owner.set(moveKey(moves[i]), playerAt(i));
  const last = moves[end - 1];
  const player = playerAt(end - 1);
  for (const [dq, dr] of AXES) {
    const run: Move[] = [last];
    for (const sign of [1, -1]) {
      let [q, r] = [last[0] + sign * dq, last[1] + sign * dr];
      while (owner.get(moveKey([q, r])) === player) {
        if (sign === 1) run.push([q, r]); else run.unshift([q, r]);
        q += sign * dq; r += sign * dr;
      }
    }
    if (run.length >= 6) return run;
  }
  return null;
}

/** P1's turns as chart bands over ply number, so the alternation is readable
 *  without counting stones. Half-ply edges centre each band on its placements. */
export function p1TurnBands(plies: number): Array<{ x0: number; x1: number }> {
  const out: Array<{ x0: number; x1: number }> = [];
  let start: number | null = null;
  for (let t = 0; t <= plies; t++) {
    const isP1 = t < plies && playerAt(t) === 1;
    if (isP1 && start === null) start = t;
    if (!isP1 && start !== null) { out.push({ x0: start - 0.5, x1: t - 0.5 }); start = null; }
  }
  return out;
}
