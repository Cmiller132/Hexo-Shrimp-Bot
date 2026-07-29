import { Crosshair, Hash, Minus, Plus, Waypoints } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AXES, findWinningLine, lineKey, moveKey as key, playerAt } from "../lib/hex";
import type { Candidate, Move } from "../types";
import { format, HeatLegend } from "./Ui";

export type Overlay = "none" | "policy" | "q" | "improved" | "delta";

/** Hex circumradius in user units; the viewBox owns board scaling. */
const HEX_R = 10;
const SQRT3 = Math.sqrt(3);
/** Label-count ladder shared by the slider and screen-level keyboard controls. */
export const TOP_K_STEPS = [0, 3, 6, 12, 24, 48];

function axialToPixel(q: number, r: number): { x: number; y: number } {
  return { x: HEX_R * SQRT3 * (q + r / 2), y: HEX_R * 1.5 * r };
}

function hexPoints(x: number, y: number, radius = HEX_R): string {
  let out = "";
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 180) * (60 * i + 30);
    out += `${i ? " " : ""}${(x + radius * Math.cos(angle)).toFixed(2)},${(y + radius * Math.sin(angle)).toFixed(2)}`;
  }
  return out;
}

/* ------------------------------------------------------------------- heat maps */

/** Logarithmic magnitude step over four decades below `max`; returns 0..6 or null. */
export function sequentialStep(value: number, max: number): number | null {
  if (!(value > 0) || !(max > 0)) return null;
  const u = Math.min(1, Math.max(0, (Math.log10(value) - Math.log10(max) + 4) / 4));
  return Math.floor(u * 6.999);
}

/** Diverging ramp step for a signed quantity over a symmetric domain ±max.
 *  Returns -4..4; 0 is the neutral midpoint. */
function divergingStep(value: number, max: number): number {
  if (!(max > 0) || value === 0) return 0;
  const arm = Math.round(Math.pow(Math.min(1, Math.abs(value) / max), 0.45) * 4);
  return value > 0 ? arm : -arm;
}

/* ------------------------------------------------------------------ the board */

export interface BoardProps {
  /** The whole line, never a pre-sliced prefix. */
  moves: Move[];
  /** Plies drawn: `moves.slice(0, cursor)`. Default `moves.length`. */
  cursor?: number;
  /** The legal cells at `cursor` *with a model read behind them*, carrying
   *  `policyRank`. Omit when nothing has read the position, when the read is
   *  stale, or when the position is terminal. */
  legal?: Candidate[];
  /** Unscored legal cells at `cursor`; pass this or `legal`, never both. */
  mask?: Move[];
  /** `inspect.played` — the move the line plays next from `cursor`. */
  played?: Move | null;
  overlay?: Overlay;
  /** Supply to control the overlay from outside; omit for the board's own state. */
  onOverlayChange?: (overlay: Overlay) => void;
  /** Cells that print a numeric label. Default 6. */
  topK?: number;
  onTopKChange?: (topK: number) => void;
  /** `auto` scales Q to ±max|Q| in this set; `unit` locks it to ±1. Default auto. */
  qDomain?: "auto" | "unit";
  /** Default `onSelect != null`. */
  interactive?: boolean;
  onSelect?: (move: Move) => void;
  /** A stone was clicked; `ply` is its index in `moves`. */
  onStone?: (ply: number) => void;
  /** Ring a cell chosen elsewhere, e.g. a candidate-table row. */
  selected?: Move | null;
  /** `pending` dims the heat: the candidate set on screen is being replaced. */
  status?: "idle" | "pending";
  /** Render the overlay / density toolbar. Default true when there are cells. */
  toolbar?: boolean;
  /** Stage height in px. Default 520. */
  height?: number;
  caption?: string;
}

interface Box { minX: number; minY: number; maxX: number; maxY: number }

/** Engine-known legality mask for the empty board: P0 opens at the origin. */
const ORIGIN: Move[] = [[0, 0]];

export default function Board({
  moves, cursor, legal, mask, played = null, overlay, onOverlayChange, topK, onTopKChange,
  qDomain = "auto", interactive, onSelect, onStone, selected = null, status = "idle", toolbar, height = 520, caption,
}: BoardProps) {
  if (legal && mask) throw new Error("board: `legal` and `mask` are the same slot — a legal set either carries a read or it does not");
  const drawn = Math.max(0, Math.min(cursor ?? moves.length, moves.length));
  const clickable = interactive ?? onSelect != null;
  const candidates = useMemo(() => legal ?? [], [legal]);
  /** Complete legal-cell set used for clicks, hit testing, and framing. */
  const cells = useMemo(
    () => legal?.map((candidate) => candidate.move) ?? mask ?? (clickable && moves.length === 0 ? ORIGIN : []),
    [legal, mask, clickable, moves.length],
  );

  const [ownOverlay, setOwnOverlay] = useState<Overlay>("policy");
  const [ownTopK, setOwnTopK] = useState(6);
  const paintable = candidates.length > 0;
  const activeOverlay = paintable ? (overlay ?? ownOverlay) : "none";
  const activeTopK = topK ?? ownTopK;
  const setOverlay = onOverlayChange ?? setOwnOverlay;
  const setTopK = onTopKChange ?? setOwnTopK;
  // The Δ overlay renders only when every candidate has a comparison value.
  const deltaReady = activeOverlay !== "delta" || candidates.every((candidate) => candidate.delta != null);
  // Move numbers default on below 60 drawn plies; the toolbar overrides the default.
  const [ownNumbers, setOwnNumbers] = useState<boolean | null>(null);
  const numbers = ownNumbers ?? drawn < 60;
  const [ghosts, setGhosts] = useState(true);
  const showToolbar = toolbar ?? cells.length > 0;

  const stones = useMemo(() => moves.slice(0, drawn), [moves, drawn]);
  const stoneOwner = useMemo(() => {
    const map = new Map<string, number>();
    stones.forEach((move, index) => map.set(key(move), index));
    return map;
  }, [stones]);

  // A legal-cell overlap with a stone is rejected as a stale-position invariant violation.
  const cellKeys = useMemo(() => {
    const set = new Set<string>();
    for (const move of cells) {
      const cellKey = key(move);
      if (stoneOwner.has(cellKey)) {
        throw new Error(`board: legal set overlaps stone at (${move.join(", ")}) — it describes a different ply`);
      }
      set.add(cellKey);
    }
    return set;
  }, [cells, stoneOwner]);

  const candidateAt = useMemo(() => {
    const map = new Map<string, Candidate>();
    candidates.forEach((candidate) => map.set(key(candidate.move), candidate));
    return map;
  }, [candidates]);

  /* ---- frame: memoised on the line, grown monotonically, never on the cursor -- */
  const lineId = useMemo(() => lineKey(moves), [moves]);
  const frameRef = useRef<{ id: string; box: Box } | null>(null);
  const frame = useMemo(() => {
    const points = [{ x: 0, y: 0 }];
    for (const move of moves) points.push(axialToPixel(move[0], move[1]));
    for (const move of cells) points.push(axialToPixel(move[0], move[1]));
    const pad = HEX_R * 2;
    const grid = HEX_R * 2;
    let box: Box = {
      minX: Math.floor((Math.min(...points.map((p) => p.x)) - pad) / grid) * grid,
      maxX: Math.ceil((Math.max(...points.map((p) => p.x)) + pad) / grid) * grid,
      minY: Math.floor((Math.min(...points.map((p) => p.y)) - pad) / grid) * grid,
      maxY: Math.ceil((Math.max(...points.map((p) => p.y)) + pad) / grid) * grid,
    };
    const previous = frameRef.current;
    if (previous && previous.id === lineId) {
      box = {
        minX: Math.min(previous.box.minX, box.minX), maxX: Math.max(previous.box.maxX, box.maxX),
        minY: Math.min(previous.box.minY, box.minY), maxY: Math.max(previous.box.maxY, box.maxY),
      };
    }
    frameRef.current = { id: lineId, box };
    return box;
  }, [lineId, moves, cells]);
  const frameW = frame.maxX - frame.minX;
  const frameH = frame.maxY - frame.minY;

  /* ------------------------------------------------------------ zoom and pan --- */
  const [view, setView] = useState({ scale: 1, tx: 0, ty: 0 });
  useEffect(() => { setView({ scale: 1, tx: 0, ty: 0 }); }, [lineId]);
  const svgRef = useRef<SVGSVGElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const [pixelWidth, setPixelWidth] = useState(0);

  useEffect(() => {
    const node = rootRef.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => setPixelWidth(entries[0].contentRect.width));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const pxPerUnit = pixelWidth > 0 && frameW > 0
    ? Math.min(pixelWidth / frameW, height / frameH) * view.scale
    : 1;
  const density = pxPerUnit < 1.15 ? "tiny" : pxPerUnit < 1.9 ? "small" : "normal";

  const toUser = useCallback((clientX: number, clientY: number) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || !rect.width || !rect.height) return { x: 0, y: 0 };
    // xMidYMid meet: one scale, the shorter axis letterboxed.
    const s = Math.min(rect.width / frameW, rect.height / frameH);
    const offX = (rect.width - frameW * s) / 2;
    const offY = (rect.height - frameH * s) / 2;
    return { x: frame.minX + (clientX - rect.left - offX) / s, y: frame.minY + (clientY - rect.top - offY) / s };
  }, [frame.minX, frame.minY, frameW, frameH]);

  const zoomAbout = useCallback((factor: number, clientX?: number, clientY?: number) => {
    setView((current) => {
      const next = Math.max(0.5, Math.min(12, current.scale * factor));
      if (next === current.scale) return current;
      const anchor = clientX != null && clientY != null
        ? toUser(clientX, clientY)
        : { x: frame.minX + frameW / 2, y: frame.minY + frameH / 2 };
      return {
        scale: next,
        tx: anchor.x - (next / current.scale) * (anchor.x - current.tx),
        ty: anchor.y - (next / current.scale) * (anchor.y - current.ty),
      };
    });
  }, [toUser, frame.minX, frame.minY, frameW, frameH]);

  useEffect(() => {
    const node = svgRef.current;
    if (!node) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      zoomAbout(event.deltaY < 0 ? 1.15 : 1 / 1.15, event.clientX, event.clientY);
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
  }, [zoomAbout]);

  const drag = useRef<{ id: number; x: number; y: number; moved: boolean } | null>(null);

  /* --------------------------------------------------------------- the layers -- */
  const heat = useMemo(() => {
    if (activeOverlay === "none" || !candidates.length || !deltaReady) return null;
    const diverging = activeOverlay === "q" || activeOverlay === "delta";
    const read = (candidate: Candidate) => {
      if (activeOverlay !== "delta") return candidate[activeOverlay];
      if (candidate.delta == null) throw new Error(`board: Δ overlay with no Δ at (${candidate.move.join(", ")})`);
      return candidate.delta;
    };
    let max = 0;
    for (const candidate of candidates) max = Math.max(max, diverging ? Math.abs(read(candidate)) : read(candidate));
    if (activeOverlay === "q" && qDomain === "unit") max = 1;
    const painted = candidates.map((candidate) => {
      const value = read(candidate);
      const step = diverging ? divergingStep(value, max) : sequentialStep(value, max);
      const { x, y } = axialToPixel(candidate.move[0], candidate.move[1]);
      return { candidate, value, x, y, step: step == null ? null : diverging ? `d${step}` : `s${step}` };
    });
    const labelled = new Set<string>();
    if (activeTopK > 0) {
      const ordered = painted.slice().sort(diverging
        ? (a, b) => Math.abs(b.value) - Math.abs(a.value)
        : (a, b) => a.candidate.policyRank - b.candidate.policyRank);
      ordered.slice(0, activeTopK).forEach((cell) => labelled.add(key(cell.candidate.move)));
    }
    return { diverging, max, painted, labelled };
  }, [activeOverlay, candidates, activeTopK, qDomain, deltaReady]);

  // Only scored candidates define a top move; an engine-order mask does not.
  const topCandidate = useMemo(
    () => candidates.find((candidate) => candidate.policyRank === 0) ?? null,
    [candidates],
  );
  const winLine = useMemo(() => findWinningLine(moves, drawn), [moves, drawn]);

  const backdrop = useMemo(() => {
    const rLo = Math.floor(frame.minY / (HEX_R * 1.5)) - 1;
    const rHi = Math.ceil(frame.maxY / (HEX_R * 1.5)) + 1;
    if ((rHi - rLo) > 200) return "";
    let path = "";
    let count = 0;
    for (let r = rLo; r <= rHi; r++) {
      const qLo = Math.floor(frame.minX / (HEX_R * SQRT3) - r / 2) - 1;
      const qHi = Math.ceil(frame.maxX / (HEX_R * SQRT3) - r / 2) + 1;
      if (qHi - qLo > 200) return "";
      for (let q = qLo; q <= qHi; q++) {
        if (++count > 1600) return "";
        const { x, y } = axialToPixel(q, r);
        path += `M${hexPoints(x, y).replace(/ /g, "L")}Z`;
      }
    }
    return path;
  }, [frame.minX, frame.maxX, frame.minY, frame.maxY]);

  /* ------------------------------------------------------- hover and selection - */
  const [hover, setHover] = useState<Move | null>(null);
  const [focusKey, setFocusKey] = useState<string | null>(null);
  // Board arrow navigation is active only after keyboard focus, leaving click focus to transport keys.
  const [keyNav, setKeyNav] = useState(false);
  const pointerHeld = useRef(false);

  /** What the read-out under the stage says about a cell. */
  const describe = useCallback((move: Move) => {
    const cellKey = key(move);
    const ply = stoneOwner.get(cellKey);
    return {
      move,
      player: ply === undefined ? null : playerAt(ply),
      ply: ply === undefined ? null : ply,
      legal: cellKeys.has(cellKey),
      candidate: candidateAt.get(cellKey) ?? null,
    };
  }, [stoneOwner, cellKeys, candidateAt]);

  const setHovered = useCallback((move: Move | null) => setHover(move), []);

  // Pointer-captured events fall back to point hit testing when the SVG is the target.
  const cellFromEvent = (event: { target: EventTarget | null; clientX: number; clientY: number }): Move | null => {
    let node = (event.target as Element | null)?.closest?.("[data-q]") as SVGGElement | null;
    if (!node) node = document.elementFromPoint(event.clientX, event.clientY)?.closest("[data-q]") as SVGGElement | null;
    if (!node) return null;
    return [Number(node.dataset.q), Number(node.dataset.r)];
  };

  const choose = useCallback((move: Move) => {
    const ply = stoneOwner.get(key(move));
    if (ply !== undefined) { onStone?.(ply); return; }
    if (clickable && cellKeys.has(key(move))) onSelect?.(move);
  }, [stoneOwner, cellKeys, clickable, onSelect, onStone]);

  /** Best policy first where there is a read, engine order where there is not. */
  const focusOrder = useMemo(
    () => (paintable ? candidates.slice().sort((a, b) => a.policyRank - b.policyRank).map((c) => c.move) : cells),
    [paintable, candidates, cells],
  );

  const walkFocus = useCallback((step: number) => {
    if (!focusOrder.length) return;
    const at = focusKey ? focusOrder.findIndex((move) => key(move) === focusKey) : -1;
    const next = focusOrder[Math.max(0, Math.min(focusOrder.length - 1, (at < 0 ? 0 : at + step)))];
    setFocusKey(key(next));
    setHovered(next);
  }, [focusOrder, focusKey, setHovered]);

  const hovered = hover ? describe(hover) : focusKey
    ? describe(focusKey.split(",").map(Number) as Move)
    : null;

  const transform = `translate(${view.tx.toFixed(3)} ${view.ty.toFixed(3)}) scale(${view.scale.toFixed(4)})`;

  return <div
    ref={rootRef}
    className="deck-board"
    data-overlay={activeOverlay}
    data-density={density}
    data-status={deltaReady ? status : "pending"}
    style={{ ["--deck-board-h" as string]: `${height}px` }}
  >
    {showToolbar && <div className="deck-board-toolbar">
      {paintable && <div className="deck-board-overlays" role="group" aria-label="board overlay">
        {(["none", "policy", "q", "improved", "delta"] as Overlay[]).map((option) => <button
          key={option}
          type="button"
          className="deck-board-tab"
          aria-pressed={activeOverlay === option}
          disabled={option === "delta" && !candidates.every((candidate) => candidate.delta != null)}
          title={option === "delta" ? "signed Δ against the compare checkpoint" : undefined}
          onClick={() => setOverlay(option)}
        >{option === "policy" ? "π" : option === "improved" ? "π′" : option === "q" ? "Q" : option === "delta" ? "Δ" : "off"}</button>)}
      </div>}
      <div className="deck-board-tools">
        {paintable && <label className="deck-board-topk">
          <span>top-k {activeTopK}</span>
          <input
            type="range" min={0} max={TOP_K_STEPS.length - 1} step={1}
            value={Math.max(0, TOP_K_STEPS.indexOf(activeTopK))}
            aria-label="labelled cells"
            onChange={(event) => setTopK(TOP_K_STEPS[Number(event.target.value)])}
          />
        </label>}
        <button type="button" className="deck-board-tool" title="zoom out" aria-label="zoom out" onClick={() => zoomAbout(1 / 1.3)}><Minus size={11} /></button>
        <button type="button" className="deck-board-tool" title="zoom in" aria-label="zoom in" onClick={() => zoomAbout(1.3)}><Plus size={11} /></button>
        <button type="button" className="deck-board-tool" title="fit" aria-label="fit" onClick={() => setView({ scale: 1, tx: 0, ty: 0 })}><Crosshair size={11} /></button>
        <button type="button" className="deck-board-tool" aria-pressed={ghosts} title="ghost the rest of the line" aria-label="ghost future" onClick={() => setGhosts(!ghosts)}><Waypoints size={11} /></button>
        <button type="button" className="deck-board-tool" aria-pressed={numbers} title="move numbers" aria-label="move numbers" onClick={() => setOwnNumbers(!numbers)}><Hash size={11} /></button>
      </div>
      {!deltaReady
        ? <span className="deck-board-status">Δ pending</span>
        : status !== "idle" && <span className="deck-board-status">{status}</span>}
    </div>}

    <div className="deck-board-stage">
      <svg
        ref={svgRef}
        className="hex-board"
        viewBox={`${frame.minX} ${frame.minY} ${frameW} ${frameH}`}
        tabIndex={0}
        role="grid"
        aria-label={caption ?? `Hexo position after ${drawn} of ${moves.length} plies`}
        onPointerDown={(event) => {
          // The flag distinguishes pointer focus from keyboard focus.
          pointerHeld.current = true;
          setKeyNav(false);
          drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
          (event.currentTarget as SVGSVGElement).setPointerCapture(event.pointerId);
        }}
        onFocus={() => { if (!pointerHeld.current) setKeyNav(true); }}
        onBlur={() => setKeyNav(false)}
        onPointerMove={(event) => {
          const state = drag.current;
          if (state && state.id === event.pointerId) {
            const dx = event.clientX - state.x;
            const dy = event.clientY - state.y;
            if (state.moved || Math.abs(dx) + Math.abs(dy) > 3) {
              state.moved = true;
              const from = toUser(state.x, state.y);
              const to = toUser(event.clientX, event.clientY);
              state.x = event.clientX; state.y = event.clientY;
              setView((current) => ({ ...current, tx: current.tx + to.x - from.x, ty: current.ty + to.y - from.y }));
              return;
            }
          }
          const cell = cellFromEvent(event);
          if (!cell) { if (hover) setHovered(null); return; }
          if (!hover || hover[0] !== cell[0] || hover[1] !== cell[1]) setHovered(cell);
        }}
        onPointerUp={(event) => {
          pointerHeld.current = false;
          const state = drag.current;
          drag.current = null;
          (event.currentTarget as SVGSVGElement).releasePointerCapture?.(event.pointerId);
          if (state?.moved) return;
          const cell = cellFromEvent(event);
          if (cell) { setFocusKey(key(cell)); choose(cell); }
        }}
        onPointerLeave={() => setHovered(null)}
        onKeyDown={(event) => {
          if (event.key === "Escape") { setKeyNav(false); setFocusKey(null); setHovered(null); (event.currentTarget as SVGSVGElement).blur(); return; }
          // Handled board keys prevent the document-level transport binding.
          if (!keyNav) return;
          if (event.key === "ArrowRight" || event.key === "ArrowDown") { event.preventDefault(); walkFocus(1); }
          else if (event.key === "ArrowLeft" || event.key === "ArrowUp") { event.preventDefault(); walkFocus(-1); }
          else if (event.key === "Enter" || event.key === " ") {
            if (!focusKey) return;
            event.preventDefault();
            choose(focusKey.split(",").map(Number) as Move);
          }
        }}
      >
        <g transform={transform}>
          <g className="deck-board-grid" aria-hidden="true">
            {backdrop && <path className="deck-board-grid-path" d={backdrop} />}
            {AXES.map(([dq, dr], index) => {
              const span = 400;
              const a = axialToPixel(dq * -span, dr * -span);
              const b = axialToPixel(dq * span, dr * span);
              return <line key={index} className="deck-board-axis" x1={a.x} y1={a.y} x2={b.x} y2={b.y} />;
            })}
            <circle className="deck-board-origin" cx={0} cy={0} r={2.4} />
          </g>

          {heat && <g className="deck-board-heat-layer">
            {heat.painted.map((cell) => cell.step && <polygon
              key={key(cell.candidate.move)}
              className="deck-board-heat"
              data-heat={cell.step}
              points={hexPoints(cell.x, cell.y)}
            />)}
          </g>}

          {ghosts && drawn < moves.length && <g className="deck-board-ghost-layer" aria-hidden="true">
            {moves.slice(drawn).map((move, index) => {
              const { x, y } = axialToPixel(move[0], move[1]);
              return <polygon key={`${key(move)}-${index}`} className="deck-board-ghost" data-player={playerAt(drawn + index)} points={hexPoints(x, y, HEX_R * 0.55)} />;
            })}
          </g>}

          {winLine && <path
            className="deck-board-win"
            d={winLine.map((move, index) => {
              const { x, y } = axialToPixel(move[0], move[1]);
              return `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
            }).join("")}
          />}

          <g className="deck-board-stones">
            {stones.map((move, index) => {
              const { x, y } = axialToPixel(move[0], move[1]);
              return <g key={`${key(move)}-${index}`} className="deck-board-stone" data-player={playerAt(index)} data-ply={index}>
                <polygon className="deck-board-stone-face" points={hexPoints(x, y, HEX_R * 0.86)} />
                {numbers && <text className="deck-board-number" x={x} y={y} textAnchor="middle" dominantBaseline="central">{index + 1}</text>}
              </g>;
            })}
          </g>

          <g className="deck-board-marks" aria-hidden="true">
            {drawn > 0 && (() => {
              const { x, y } = axialToPixel(moves[drawn - 1][0], moves[drawn - 1][1]);
              return <polygon className="deck-board-mark" data-mark="last" data-player={playerAt(drawn - 1)} points={hexPoints(x, y, HEX_R * 0.96)} />;
            })()}
            {(() => {
              const agree = played && topCandidate && key(played) === key(topCandidate.move);
              const marks: Array<{ move: Move; mark: string }> = [];
              if (agree && played) marks.push({ move: played, mark: "agree" });
              else {
                if (played) marks.push({ move: played, mark: "played" });
                if (topCandidate) marks.push({ move: topCandidate.move, mark: "top" });
              }
              return marks.map(({ move, mark }) => {
                const { x, y } = axialToPixel(move[0], move[1]);
                return <g key={mark}>
                  <polygon className="deck-board-mark" data-mark={mark} points={hexPoints(x, y, HEX_R * 0.78)} />
                  {(mark === "top" || mark === "agree") && <circle className="deck-board-mark-dot" cx={x} cy={y} r={HEX_R * 0.16} />}
                </g>;
              });
            })()}
            {selected && (() => {
              const { x, y } = axialToPixel(selected[0], selected[1]);
              return <polygon className="deck-board-mark" data-mark="focus" points={hexPoints(x, y, HEX_R * 0.9)} />;
            })()}
            {focusKey && (() => {
              const [q, r] = focusKey.split(",").map(Number);
              const { x, y } = axialToPixel(q, r);
              return <polygon className="deck-board-mark" data-mark="focus" points={hexPoints(x, y, HEX_R)} />;
            })()}
          </g>

          {heat && activeTopK > 0 && <g className="deck-board-labels" aria-hidden="true">
            {heat.painted.filter((cell) => heat.labelled.has(key(cell.candidate.move))).map((cell) => <text
              key={key(cell.candidate.move)} className="deck-board-label" x={cell.x} y={cell.y}
              textAnchor="middle" dominantBaseline="central"
            >{format(cell.value, 2)}</text>)}
          </g>}

          <g className="deck-board-hits">
            {cells.map((move) => {
              const { x, y } = axialToPixel(move[0], move[1]);
              const cellKey = key(move);
              const candidate = candidateAt.get(cellKey);
              return <g
                key={cellKey}
                className="deck-board-hit"
                data-q={move[0]}
                data-r={move[1]}
                role={clickable ? "button" : undefined}
                tabIndex={focusKey === cellKey ? 0 : -1}
                aria-label={candidate
                  ? `(${move.join(", ")}) policy rank ${candidate.policyRank + 1}`
                  : `(${move.join(", ")}) legal`}
              >
                <polygon className="deck-board-hit-face" points={hexPoints(x, y)} />
              </g>;
            })}
            {stones.map((move, index) => {
              const { x, y } = axialToPixel(move[0], move[1]);
              return <g
                key={`s${key(move)}-${index}`}
                className="deck-board-hit"
                data-q={move[0]}
                data-r={move[1]}
                data-stone=""
                aria-label={`ply ${index + 1} at (${move.join(", ")})`}
              >
                <polygon className="deck-board-hit-face" points={hexPoints(x, y, HEX_R * 0.9)} />
              </g>;
            })}
          </g>
        </g>
      </svg>
    </div>

    {heat && <HeatLegend
      kind={heat.diverging ? "div" : "seq"}
      max={heat.max}
      note={heat.diverging
        ? `${activeOverlay === "delta" ? "Δ" : "Q"} · ±${format(heat.max, 2)}${activeOverlay === "q" && qDomain === "unit" ? " locked" : ""}`
        : `${activeOverlay === "improved" ? "π′" : "π"}max ${format(heat.max, 2)} · four decades`}
    />}

    <div className="deck-board-readout">
      {hovered ? <>
        <span className="deck-board-readout-item"><b>cell</b><em>({hovered.move.join(", ")})</em></span>
        {hovered.ply != null && <span className="deck-board-readout-item"><b>ply</b><em>{hovered.ply + 1} · P{hovered.player}</em></span>}
        {hovered.legal && !hovered.candidate && <span className="deck-board-readout-item"><b>legal</b><em>no read</em></span>}
        {hovered.candidate && <>
          <span className="deck-board-readout-item"><b>π</b><em>{format(hovered.candidate.policy)}</em></span>
          <span className="deck-board-readout-item"><b>Q</b><em>{format(hovered.candidate.q)}</em></span>
          <span className="deck-board-readout-item"><b>π′</b><em>{format(hovered.candidate.improved)}</em></span>
          <span className="deck-board-readout-item"><b>rank</b><em>#{hovered.candidate.policyRank + 1}</em></span>
          {hovered.candidate.delta != null && <span className="deck-board-readout-item"><b>Δ</b><em>{format(hovered.candidate.delta)}</em></span>}
        </>}
      </> : <span className="deck-board-readout-item"><b>ply</b><em>{drawn} / {moves.length}</em></span>}
    </div>
  </div>;
}
