"use client";

import { useId, useMemo, useState, type CSSProperties, type ComponentType } from "react";
import {
  Activity,
  ArrowLeftRight,
  BarChart3,
  Bell,
  Bot,
  BrainCircuit,
  Check,
  ChevronDown,
  ChevronRight,
  CircleStop,
  Clock3,
  Cpu,
  Download,
  Eye,
  Filter,
  FlaskConical,
  Gauge,
  GitCompareArrows,
  History,
  Layers3,
  Menu,
  MoreHorizontal,
  Pause,
  Play,
  Plus,
  Radio,
  RotateCcw,
  Save,
  Search,
  SlidersHorizontal,
  Sparkles,
  Star,
  Terminal,
  UserRound,
  Users,
  X,
  Zap,
} from "lucide-react";

type Screen = "play" | "history" | "live" | "lab";
type Player = 0 | 1;
type Overlay = "off" | "policy" | "q" | "improved" | "rank" | "visits";
type LabModule = "graph" | "policy" | "q" | "improve" | "attention" | "value";
type LabContext = {
  source: string;
  checkpoint: string;
  prefix: number;
  module: LabModule;
  overlay: Overlay;
};

const DEFAULT_LAB_CONTEXT: LabContext = {
  source: "Saved legal prefix · game 482119",
  checkpoint: "live weights @ i240",
  prefix: 32,
  module: "graph",
  overlay: "off",
};

function playerAtPly(ply: number): Player {
  if (ply === 0) return 0;
  return Math.floor((ply - 1) / 2) % 2 === 0 ? 1 : 0;
}

function turnAtStoneCount(stones: number) {
  if (stones === 0) return { player: 0 as Player, movesRemaining: 1, phase: "opening" };
  const placementsAfterOpening = stones - 1;
  return {
    player: (Math.floor(placementsAfterOpening / 2) % 2 === 0 ? 1 : 0) as Player,
    movesRemaining: placementsAfterOpening % 2 === 0 ? 2 : 1,
    phase: placementsAfterOpening % 2 === 0 ? "first stone" : "second stone",
  };
}

const NAV_ITEMS: Array<{
  id: Screen;
  label: string;
  short: string;
  icon: ComponentType<{ size?: number; strokeWidth?: number }>;
}> = [
  { id: "play", label: "Play", short: "Play", icon: Play },
  { id: "history", label: "Game history", short: "History", icon: History },
  { id: "live", label: "Live run", short: "Live", icon: Activity },
  { id: "lab", label: "Model lab", short: "Lab", icon: FlaskConical },
];

const CHECKPOINTS = [
  { id: "mantis-042-i240", family: "mantis-042", step: "iter 240", artifact: "live weights @ i240", note: "live candidate", repr: "v1", heads: "πθ · Q", compatible: true },
  { id: "mantis-042-i225", family: "mantis-042", step: "iter 225", artifact: "checkpoint_000225.pt", note: "latest durable", repr: "v1", heads: "πθ · Q", compatible: true },
  { id: "mantis-042-i200", family: "mantis-042", step: "iter 200", artifact: "checkpoint_000200.pt", note: "measured", repr: "v1", heads: "πθ · Q", compatible: true },
  { id: "strix-eq-031-880k", family: "strix-eq-031", step: "880k", artifact: "strix checkpoint 880k", note: "legacy package", repr: "legacy", heads: "π", compatible: false },
  { id: "shrimp-v4-146", family: "shrimp-v4", step: "epoch 146", artifact: "shrimp epoch 146", note: "archive", repr: "legacy", heads: "π · V", compatible: false },
];

const OPPONENTS = [
  { id: "human", label: "You · local human", type: "human" },
  ...CHECKPOINTS.map((checkpoint) => ({
    id: checkpoint.id,
    label: `${checkpoint.family} · ${checkpoint.step}`,
    type: "checkpoint",
  })),
  { id: "mcts-1k", label: "Baseline · MCTS 1k", type: "baseline" },
  { id: "greedy", label: "Baseline · greedy policy", type: "baseline" },
  { id: "random", label: "Baseline · uniform random", type: "baseline" },
];

const GAMES = [
  { id: "482119", time: "—", index: 4095, kind: "selfplay", iteration: 240, blue: "mantis snapshot · p0", red: "mantis snapshot · p1", result: "blue", reason: "six-in-row", ply: 47, trace: true, capped: false, tags: ["high-swing", "favorite"] },
  { id: "483142", time: "—", index: 63, kind: "eval", iteration: 240, blue: "SealBot · depth 7.8", red: "mantis live · argmax πθ", result: "red", reason: "six-in-row", ply: 62, trace: false, capped: false, tags: ["anchor", "seat-p1"] },
  { id: "HX-8824", time: "12:21:55", index: 0, kind: "local", iteration: null, blue: "You", red: "mantis-042 · iter 225", result: "blue", reason: "resign", ply: 39, trace: true, capped: false, tags: ["human", "opening-suite"] },
  { id: "482118", time: "—", index: 4094, kind: "selfplay", iteration: 240, blue: "mantis snapshot · p0", red: "mantis snapshot · p1", result: "blue", reason: "six-in-row", ply: 43, trace: true, capped: false, tags: ["policy-disagreement"] },
  { id: "483141", time: "—", index: 62, kind: "eval", iteration: 240, blue: "mantis live · argmax πθ", red: "SealBot · depth 8.1", result: "draw", reason: "ply cap", ply: 512, trace: false, capped: true, tags: ["anchor", "capped", "seat-p0"] },
  { id: "482117", time: "—", index: 4093, kind: "selfplay", iteration: 240, blue: "mantis snapshot · p0", red: "mantis snapshot · p1", result: "red", reason: "six-in-row", ply: 58, trace: true, capped: false, tags: ["low-entropy"] },
];

const SEED_MOVES: Array<{ cell: number; player: Player }> =
  [44, 45, 34, 54, 35, 53, 43, 55, 24, 64, 33, 46, 25, 63, 65]
    .map((cell, ply) => ({ cell, player: playerAtPly(ply) }));

const HISTORY_MOVES: Array<{ cell: number; player: Player }> =
  [44, 34, 45, 35, 54, 43, 55, 24, 53, 64, 42, 33, 52, 23, 62, 46, 32, 56]
    .map((cell, ply) => ({ cell, player: playerAtPly(ply) }));

function mapMoves(moves: Array<{ cell: number; player: Player }>) {
  return Object.fromEntries(moves.map(({ cell, player }) => [cell, player])) as Record<number, Player>;
}

function StatusDot({ tone = "ok", pulse = false }: { tone?: "ok" | "warn" | "blue" | "red"; pulse?: boolean }) {
  return <span className={`status-dot ${tone} ${pulse ? "pulse" : ""}`} aria-hidden="true" />;
}

function SectionLabel({ children, action }: { children: React.ReactNode; action?: React.ReactNode }) {
  return <div className="section-label"><span>{children}</span>{action}</div>;
}

function IconButton({ label, children, active = false, onClick }: { label: string; children: React.ReactNode; active?: boolean; onClick?: () => void }) {
  return <button className={`icon-button ${active ? "active" : ""}`} aria-label={label} title={label} onClick={onClick}>{children}</button>;
}

function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T;
  options: Array<{ value: T; label: string; detail?: string }>;
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <button key={option.value} className={value === option.value ? "selected" : ""} onClick={() => onChange(option.value)} role="radio" aria-checked={value === option.value}>
          <span>{option.label}</span>{option.detail && <small>{option.detail}</small>}
        </button>
      ))}
    </div>
  );
}

function HexBoard({ moves, overlay = "off", selectedCell, onCellClick, compact = false, label = "Hexo position" }: {
  moves: Record<number, Player>;
  overlay?: Overlay;
  selectedCell?: number;
  onCellClick?: (cell: number) => void;
  compact?: boolean;
  label?: string;
}) {
  const gradientKey = useId().replaceAll(":", "");
  const stoneBlue = `stone-blue-${gradientKey}`;
  const stoneRed = `stone-red-${gradientKey}`;
  const S = 26;
  const draw = 0.965;
  const colWidth = S * Math.sqrt(3);
  const rowHeight = S * 1.5;
  const topPolicy = [36, 47, 37, 26, 57, 48];
  const tiles = [];

  function axialX(q: number, r: number) {
    return colWidth * (q + r / 2);
  }

  function axialY(r: number) {
    return rowHeight * r;
  }

  function hexPoints(q: number, r: number, radius: number) {
    const cx = axialX(q, r);
    const cy = axialY(r);
    return Array.from({ length: 6 }, (_, index) => {
      const angle = ((60 * index - 30) * Math.PI) / 180;
      return `${(cx + radius * Math.cos(angle)).toFixed(2)},${(cy + radius * Math.sin(angle)).toFixed(2)}`;
    }).join(" ");
  }

  function cellToAxial(cell: number) {
    return { q: (cell % 10) - 5, r: Math.floor(cell / 10) - 4 };
  }

  function axialToCell(q: number, r: number) {
    const column = q + 5;
    const row = r + 4;
    return column >= 0 && column < 10 && row >= 0 && row < 9 ? row * 10 + column : null;
  }

  function axialDistance(q: number, r: number, targetQ: number, targetR: number) {
    const dq = q - targetQ;
    const dr = r - targetR;
    return (Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2;
  }

  for (let r = -10; r <= 10; r += 1) {
    for (let q = -14; q <= 14; q += 1) {
      tiles.push({ q, r, cell: axialToCell(q, r) });
    }
  }

  const stoneEntries = Object.entries(moves).map(([cell, player]) => ({
    cell: Number(cell),
    player,
    ...cellToAxial(Number(cell)),
  }));
  const selectedAxial = selectedCell === undefined ? null : cellToAxial(selectedCell);
  const latestStones = stoneEntries.slice(-2);

  return (
    <svg
      className={`showcase-board ${compact ? "compact" : ""}`}
      viewBox={compact ? "-380 -333 760 666" : "-450 -395 900 790"}
      role="grid"
      aria-label={label}
      preserveAspectRatio="xMidYMid meet"
    >
      <title>{label}</title>
      <defs>
        <linearGradient id={stoneBlue} x1="0.22" y1="0" x2="0.78" y2="1">
          <stop offset="0" stopColor="#7db4ff" />
          <stop offset="0.45" stopColor="#4f93ff" />
          <stop offset="1" stopColor="#2f62c4" />
        </linearGradient>
        <linearGradient id={stoneRed} x1="0.22" y1="0" x2="0.78" y2="1">
          <stop offset="0" stopColor="#ff8a80" />
          <stop offset="0.45" stopColor="#f0524c" />
          <stop offset="1" stopColor="#b93732" />
        </linearGradient>
      </defs>

      <g className="showcase-grid-layer">
        {tiles.map(({ q, r, cell }) => {
          const player = cell === null ? undefined : moves[cell];
          const coordinate =
            cell === null ? `${q},${r}` : `${String.fromCharCode(65 + (cell % 10))}${Math.floor(cell / 10) + 1}`;
          return (
            <polygon
              key={`${q},${r}`}
              points={hexPoints(q, r, S * draw)}
              className={`showcase-cell ${player !== undefined ? "occupied" : ""}`}
              role={cell !== null ? "gridcell" : undefined}
              aria-label={cell !== null ? `Cell ${coordinate}${player !== undefined ? `, ${player === 0 ? "blue" : "red"} stone` : ""}` : undefined}
              tabIndex={cell !== null && onCellClick ? 0 : -1}
              onClick={() => cell !== null && onCellClick?.(cell)}
              onKeyDown={(event) => {
                if (cell !== null && onCellClick && (event.key === "Enter" || event.key === " ")) {
                  event.preventDefault();
                  onCellClick(cell);
                }
              }}
            />
          );
        })}
        <circle cx="0" cy="0" r="2.1" className="showcase-tengen" />
      </g>

      {overlay !== "off" && (
        <g className={`showcase-heat-layer overlay-${overlay}`}>
          {tiles.map(({ q, r, cell }) => {
            if (cell !== null && moves[cell] !== undefined) return null;
            const distance = Math.min(
              axialDistance(q, r, 1, -1),
              axialDistance(q, r, 0, 1) + 0.7,
            );
            const variation = 0.66 + (((q * 17 + r * 31 + 997) % 19) + 19) % 19 / 45;
            const heat = Math.max(0, 1 - distance / 6.2) * variation;
            if (heat < 0.13) return null;
            return (
              <polygon
                key={`heat-${q},${r}`}
                points={hexPoints(q, r, S * 0.91)}
                className="showcase-heat-cell"
                style={{ opacity: 0.03 + heat * 0.68 }}
              />
            );
          })}
          {topPolicy.map((cell, index) => {
            if (moves[cell] !== undefined) return null;
            const { q, r } = cellToAxial(cell);
            return overlay === "rank" || overlay === "visits" || overlay === "q" || overlay === "improved" ? (
              <text key={`label-${cell}`} x={axialX(q, r)} y={axialY(r) + 3} className="showcase-heat-label">
                {overlay === "rank"
                  ? index + 1
                  : overlay === "q"
                    ? ["+.42", "+.36", "+.31", "+.25", "+.19", "+.14"][index]
                    : overlay === "improved"
                      ? `${[31, 22, 16, 11, 8, 5][index]}%`
                      : Math.max(18, 186 - index * 27)}
              </text>
            ) : null;
          })}
          {(() => {
            const { q, r } = cellToAxial(topPolicy[0]);
            return <polygon points={hexPoints(q, r, S * 0.84)} className="showcase-heat-top" />;
          })()}
        </g>
      )}

      <g className="showcase-stones-layer">
        {stoneEntries.map(({ cell, player, q, r }) => (
          <polygon
            key={`stone-${cell}`}
            points={hexPoints(q, r, S * 0.8)}
            className={`showcase-stone player-${player}`}
            fill={`url(#${player === 0 ? stoneBlue : stoneRed})`}
          />
        ))}
      </g>

      <g className="showcase-marks-layer">
        {latestStones.map(({ cell, q, r }, index) => (
          <circle
            key={`last-${cell}`}
            cx={axialX(q, r)}
            cy={axialY(r)}
            r={index === latestStones.length - 1 ? 2.5 : 2}
            className={`showcase-last-dot ${index === latestStones.length - 1 ? "" : "previous"}`}
          />
        ))}
        {selectedAxial && moves[selectedCell as number] === undefined && (
          <polygon
            points={hexPoints(selectedAxial.q, selectedAxial.r, S * 0.79)}
            className="showcase-query"
          />
        )}
      </g>
    </svg>
  );
}

function TrendChart({ points, tone = "mint", label }: { points: number[]; tone?: "mint" | "blue" | "violet" | "amber"; label: string }) {
  const width = 100;
  const height = 34;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const coordinates = points.map((point, index) => ({ x: (index / (points.length - 1)) * width, y: height - ((point - min) / span) * (height - 7) - 3 }));

  return (
    <div className={`trend-chart ${tone}`} role="img" aria-label={label}>
      <div className="chart-grid" />
      {coordinates.slice(0, -1).map((point, index) => {
        const next = coordinates[index + 1];
        const dx = next.x - point.x;
        const dy = next.y - point.y;
        const length = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx) * (180 / Math.PI);
        return <span className="trend-line" key={index} style={{ left: `${point.x}%`, top: `${point.y}px`, width: `${length}%`, transform: `rotate(${angle}deg)` }} />;
      })}
      <span className="trend-end" style={{ left: `${coordinates.at(-1)?.x ?? 0}%`, top: `${coordinates.at(-1)?.y ?? 0}px` }} />
    </div>
  );
}

function PlayerSelect({ label, value, onChange, color }: { label: string; value: string; onChange: (value: string) => void; color: "blue" | "red" }) {
  const selected = OPPONENTS.find((opponent) => opponent.id === value);
  const checkpoint = CHECKPOINTS.find((item) => item.id === value);
  return (
    <label className="player-select">
      <span className="player-label"><i className={`player-mark ${color}`} />{label}</span>
      <span className="select-shell">
        {selected?.type === "human" ? <UserRound size={14} /> : <Bot size={14} />}
        <select value={value} onChange={(event) => onChange(event.target.value)}>
          {OPPONENTS.map((opponent) => <option key={opponent.id} value={opponent.id}>{opponent.label}</option>)}
        </select>
        <ChevronDown size={14} />
      </span>
      <small className={`seat-capability ${checkpoint?.compatible ? "compatible" : ""}`}>
        {checkpoint ? `${checkpoint.repr} · ${checkpoint.heads} · ${checkpoint.compatible ? "exact loader contract" : "package adapter only"}` : selected?.type === "baseline" ? "native baseline profile · not a Mantis checkpoint" : "local input · engine legality enforced"}
      </small>
    </label>
  );
}

function Panel({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`panel-card ${className}`}>{children}</section>;
}

function PlayScreen({ sendToLab }: { sendToLab: (context?: Partial<LabContext>) => void }) {
  const [matchMode, setMatchMode] = useState<"human" | "arena">("human");
  const [blue, setBlue] = useState("human");
  const [red, setRed] = useState("mantis-042-i240");
  const [moveMode, setMoveMode] = useState<"argmax" | "sample" | "improved">("argmax");
  const [analysisSeat, setAnalysisSeat] = useState<"blue" | "red">("red");
  const [overlay, setOverlay] = useState<Overlay>("off");
  const [status, setStatus] = useState<"ready" | "running" | "paused">("ready");
  const [moves, setMoves] = useState<Record<number, Player>>(() => mapMoves(SEED_MOVES));
  const [selectedCell, setSelectedCell] = useState(36);
  const turn = turnAtStoneCount(Object.keys(moves).length);
  const nextPlayer = turn.player;
  const currentSeat = nextPlayer === 0 ? blue : red;
  const requestedAnalysisSeat = analysisSeat === "blue" ? blue : red;
  const analysisCheckpoint =
    CHECKPOINTS.find((item) => item.id === requestedAnalysisSeat && item.compatible)
    ?? CHECKPOINTS.find((item) => (item.id === red || item.id === blue) && item.compatible)
    ?? CHECKPOINTS[0];

  function inspectInLab(module: LabModule = "policy", requestedOverlay: Overlay = "policy") {
    sendToLab({
      source: `Play console · legal prefix ${Object.keys(moves).length}`,
      checkpoint: analysisCheckpoint.artifact,
      prefix: Object.keys(moves).length,
      module,
      overlay: requestedOverlay,
    });
  }

  function setMode(mode: "human" | "arena") {
    setMatchMode(mode);
    if (mode === "human") { setBlue("human"); setRed("mantis-042-i240"); setAnalysisSeat("red"); }
    else { setBlue("mantis-042-i225"); setRed("mantis-042-i200"); setAnalysisSeat("blue"); }
  }

  function placeStone(cell: number) {
    setSelectedCell(cell);
    if (status !== "running" || moves[cell] !== undefined || matchMode === "arena" || currentSeat !== "human") return;
    setMoves((current) => ({ ...current, [cell]: nextPlayer }));
  }

  function resetBoard() { setMoves(mapMoves(SEED_MOVES)); setStatus("ready"); }

  return (
    <div className="screen play-screen">
      <div className="screen-heading">
        <div>
          <div className="eyebrow">MANTISNET MATCH CONSOLE / LOCAL / REPR V1</div>
          <h1>Play &amp; pit</h1>
          <p>Play exported Mantis checkpoints, compare past iterations, or seat them against native generic opponents.</p>
        </div>
        <div className="heading-actions">
          <button className="button ghost"><Save size={14} />Save preset</button>
          <button className="button ghost" onClick={() => inspectInLab("policy", overlay === "off" ? "policy" : overlay)}><FlaskConical size={14} />Open in lab</button>
        </div>
      </div>

      <div className="play-layout">
        <aside className="play-setup">
          <Panel>
            <SectionLabel>Match type</SectionLabel>
            <Segmented label="Match type" value={matchMode} onChange={setMode} options={[{ value: "human", label: "Human", detail: "play locally" }, { value: "arena", label: "Paired arena", detail: "bot vs bot" }]} />
            <div className="player-stack">
              <PlayerSelect label="Blue · first" color="blue" value={blue} onChange={setBlue} />
              <button className="swap-button" aria-label="Swap players" onClick={() => { setBlue(red); setRed(blue); }}><ArrowLeftRight size={13} />swap sides</button>
              <PlayerSelect label="Red · second" color="red" value={red} onChange={setRed} />
            </div>
            {matchMode === "arena" && <div className="arena-protocol"><span><b>40</b><small>games</small></span><span><b>2—6</b><small>opening plies</small></span><span><b>P0 / P1</b><small>seat balanced</small></span><span><b>95%</b><small>Wilson CI</small></span></div>}
          </Panel>

          <Panel>
            <SectionLabel action={<button className="tiny-action" onClick={() => inspectInLab("graph", "off")}>head manifest <ChevronRight size={11} /></button>}>Mantis seat policy</SectionLabel>
            <Segmented label="Mantis move selection" value={moveMode} onChange={setMoveMode} options={[{ value: "argmax", label: "Argmax πθ", detail: "deployed artifact" }, { value: "sample", label: "Sample πθ", detail: "raw policy" }, { value: "improved", label: "KLENT π′", detail: "training counterfactual" }]} />
            <dl className="definition-list">
              <div><dt>τ · reverse KL</dt><dd>0.10</dd></div><div><dt>λ · entropy</dt><dd>0.03</dd></div>
              <div><dt>heads</dt><dd>πθ + Q</dd></div><div><dt>Mantis search</dt><dd>none</dd></div>
            </dl>
            <div className={`mode-caveat ${moveMode === "improved" ? "warn" : ""}`}><StatusDot tone={moveMode === "improved" ? "warn" : "ok"} /><span>{moveMode === "improved" ? "π′ is reconstructed from πθ + Q for analysis; it is not the exported playing policy." : "Generic baselines keep their own native move-selection profile."}</span></div>
            <button className="inline-link" onClick={() => inspectInLab("improve", "improved")}><SlidersHorizontal size={13} />inspect τ, λ, πθ and Q</button>
          </Panel>

          <Panel>
            <SectionLabel>Quick suites</SectionLabel>
            <div className="suite-list">
              <button><span className="suite-icon"><Sparkles size={14} /></span><span><b>πθ / Q disagreements</b><small>18 saved high-KL positions</small></span><ChevronRight size={14} /></button>
              <button><span className="suite-icon"><Gauge size={14} /></span><span><b>D6 regression smoke</b><small>policy + Q · 12 transforms</small></span><ChevronRight size={14} /></button>
              <button><span className="suite-icon"><GitCompareArrows size={14} /></span><span><b>Checkpoint cross-play</b><small>seat-balanced · argmax πθ</small></span><ChevronRight size={14} /></button>
            </div>
          </Panel>
        </aside>

        <main className="board-workspace">
          <div className="board-toolbar">
            <div className="match-status">
              <StatusDot tone={status === "running" ? "ok" : status === "paused" ? "warn" : "blue"} pulse={status === "running"} />
              <span>{status === "running" ? `${currentSeat === "human" ? "your move" : "opponent thinking"} · ${nextPlayer === 0 ? "blue" : "red"} · ${turn.phase} · ${turn.movesRemaining} remaining` : status === "paused" ? "match paused" : "position ready"}</span>
              <span className="mono muted">ply {Object.keys(moves).length} · mover p{nextPlayer}</span>
            </div>
            <div className="overlay-switch">
              <span>overlay</span>
              {(["off", "policy", "q", "improved"] as Overlay[]).map((item) => <button key={item} className={overlay === item ? "active" : ""} onClick={() => setOverlay(item)}>{item === "improved" ? "π′" : item}</button>)}
            </div>
          </div>
          <div className={`board-frame side-${nextPlayer}`}>
            <div className="frame-corner top-left">FIELD · R9 · ACTION ORDER PINNED</div>
            <div className="frame-corner top-right">cursor <b>G4</b></div>
            <HexBoard moves={moves} overlay={overlay} selectedCell={selectedCell} onCellClick={placeStone} label="Current play position" />
          </div>
          <div className="game-commandbar">
            <div className="turn-card blue"><i /><span><small>BLUE · {CHECKPOINTS.some((item) => item.id === blue && item.compatible) ? "MANTIS REPR V1" : "NATIVE PROFILE"}</small><b>{OPPONENTS.find((item) => item.id === blue)?.label}</b></span><strong>{nextPlayer === 0 ? `${turn.movesRemaining}/2 LEFT` : CHECKPOINTS.some((item) => item.id === blue && item.compatible) ? "πθ" : "NATIVE"}</strong></div>
            <div className="match-controls">
              <IconButton label="Reset position" onClick={resetBoard}><RotateCcw size={16} /></IconButton>
              <button className={`play-command ${status === "running" ? "stop" : ""}`} onClick={() => setStatus(status === "running" ? "paused" : "running")}>
                {status === "running" ? <Pause size={16} /> : <Play size={16} />}{status === "running" ? "Pause" : status === "paused" ? "Resume" : matchMode === "arena" ? "Start paired set" : "Start match"}
              </button>
              <IconButton label="End match" onClick={() => setStatus("ready")}><CircleStop size={16} /></IconButton>
            </div>
            <div className="turn-card red"><i /><span><small>RED · {CHECKPOINTS.some((item) => item.id === red && item.compatible) ? "MANTIS REPR V1" : "NATIVE PROFILE"}</small><b>{OPPONENTS.find((item) => item.id === red)?.label}</b></span><strong>{nextPlayer === 1 ? `${turn.movesRemaining}/2 LEFT` : CHECKPOINTS.some((item) => item.id === red && item.compatible) ? "πθ" : "NATIVE"}</strong></div>
          </div>
        </main>

        <aside className="play-telemetry">
          <Panel>
            <SectionLabel action={<div className="head-toggle analyze-seat"><button className={analysisSeat === "blue" ? "active" : ""} disabled={!CHECKPOINTS.some((item) => item.id === blue && item.compatible)} onClick={() => setAnalysisSeat("blue")}>blue</button><button className={analysisSeat === "red" ? "active" : ""} disabled={!CHECKPOINTS.some((item) => item.id === red && item.compatible)} onClick={() => setAnalysisSeat("red")}>red</button></div>}>KLENT position read</SectionLabel>
            <div className="value-hero"><span className="value-number">v̂ +0.34</span><span className="value-caption">EXPECTED Q UNDER IMPROVED POLICY π′</span></div>
            <div className="split-meter klent-meter" aria-label="Top eight moves hold 74 percent of improved policy mass"><span style={{ width: "74%" }} /></div>
            <div className="mini-stats three"><div><strong>0.218</strong><span>KL π′‖πθ</span></div><div><strong>0.41</strong><span>H / log |A|</span></div><div><strong>1,284</strong><span>legal cells</span></div></div>
            <div className="analysis-source"><StatusDot tone={moveMode === "improved" ? "warn" : "ok"} /><span>{analysisCheckpoint.artifact}</span><em>{analysisSeat} seat · mover frame</em></div>
          </Panel>

          <Panel className="policy-panel">
            <SectionLabel action={<button className="tiny-action" onClick={() => setOverlay("improved")}>show π′ map</button>}>Policy · Q · improvement</SectionLabel>
            <ol className="candidate-list">
              {[["G4", "18.7", "+.42", "31"], ["F5", "14.2", "+.36", "22"], ["H3", "11.9", "+.31", "16"], ["G5", "9.1", "+.25", "11"], ["E6", "7.6", "+.19", "8"]].map(([move, policy, q, improved], index) => (
                <li key={move} className={selectedCell === 36 + index ? "active" : ""}>
                  <button onClick={() => setSelectedCell(36 + index)}><span className="rank">{index + 1}</span><b>{move}</b><span className="policy-bar"><i style={{ width: `${Number(policy) * 4.5}%` }} /></span><span>{policy}%</span><span className="mono">{q}</span><small>{improved}% π′</small></button>
                </li>
              ))}
            </ol>
          </Panel>

          <Panel>
            <SectionLabel>Decision recipe</SectionLabel>
            <div className="klent-formula">π′ ∝ exp[(Q + τ · log πθ) / (τ + λ)]</div>
            <div className="notice-row"><Zap size={13} />{moveMode === "improved" ? "analysis mode samples π′; results are not deployment-equivalent" : moveMode === "sample" ? "sampling raw πθ; evaluation remains deterministic argmax" : "deployment-equivalent · argmax πθ · one forward · no search"}</div>
          </Panel>

          <Panel className="queue-peek">
            <SectionLabel>After this match</SectionLabel>
            <div className="queue-row"><span className="queue-index">01</span><span><b>iter 225 vs iter 200</b><small>argmax πθ · swap seats · 40</small></span><span className="queue-count">40</span></div>
            <div className="queue-row"><span className="queue-index">02</span><span><b>iter 225 vs SealBot</b><small>paired openings · seat balanced</small></span><span className="queue-count">64</span></div>
            <button className="inline-link"><Plus size={13} />add another pairing</button>
          </Panel>
        </aside>
      </div>
    </div>
  );
}

function HistoryScreen({ sendToLab }: { sendToLab: (context?: Partial<LabContext>) => void }) {
  const [selected, setSelected] = useState(GAMES[0]);
  const [filter, setFilter] = useState<"all" | "selfplay" | "eval" | "local">("all");
  const [historyView, setHistoryView] = useState<"games" | "swings" | "calibration" | "openings" | "crossplay">("games");
  const [query, setQuery] = useState("");
  const [ply, setPly] = useState(32);
  const [recompute, setRecompute] = useState(false);
  const filteredGames = useMemo(() => GAMES.filter((game) => {
    const matchesFilter = filter === "all" || game.kind === filter;
    return matchesFilter && `${game.id} ${game.blue} ${game.red}`.toLowerCase().includes(query.toLowerCase());
  }), [filter, query]);
  const positionTurn = turnAtStoneCount(ply);

  function inspectHistoryPosition(module: LabModule = "improve", requestedOverlay: Overlay = "improved") {
    sendToLab({
      source: `${selected.id} · legal prefix ${ply}`,
      checkpoint: selected.kind === "eval" && selected.iteration === 240 ? "live weights @ i240" : "checkpoint_000225.pt",
      prefix: ply,
      module,
      overlay: requestedOverlay,
    });
  }

  return (
    <div className="screen">
      <div className="screen-heading">
        <div><div className="eyebrow">TELEMETRY.DB / SCHEMA V1 / 8,824 GAMES</div><h1>Games &amp; diagnostics</h1><p>Replay stored games, inspect acting-time KLENT scalars, find mover-frame swings, and compare checkpoint strength.</p></div>
        <div className="heading-actions"><button className="button ghost"><Download size={14} />Export query</button><button className="button ghost" onClick={() => inspectHistoryPosition("graph", "off")}><FlaskConical size={14} />Open replay editor</button></div>
      </div>

      <div className="history-overview">
        <Panel className="metric-card"><span>Finished self-play · i240</span><strong>4,096</strong><em>143.8k trainable plies</em><TrendChart points={[38,41,42,46,43,49,51,50,54,56]} label="Finished self-play games" /></Panel>
        <Panel className="metric-card"><span>Terminating fraction · f</span><strong>0.982</strong><em>starve streak 0 / 10</em><TrendChart points={[42,45,44,48,49,51,53,52,55,56]} tone="blue" label="Terminating fraction" /></Panel>
        <Panel className="metric-card"><span>v̂ calibration MAE</span><strong>0.214</strong><em className="neutral">finished self-play only</em><TrendChart points={[55,52,53,49,47,45,46,42,40,38]} tone="violet" label="Value estimate calibration error" /></Panel>
        <Panel className="metric-card"><span>Large mover-frame swings</span><strong>3</strong><em className="warn">|Δv̂| ≥ 0.50</em><TrendChart points={[1,1,2,1,2,4,3,5,4,3]} tone="amber" label="Large value swings" /></Panel>
      </div>

      <div className="archive-view-nav" role="tablist" aria-label="History diagnostics">
        {([["games", "Games", "8,824"], ["swings", "Value swings", "3"], ["calibration", "Calibration", "v̂"], ["openings", "Opening atlas", "D6"], ["crossplay", "Cross-play", "4×4"]] as const).map(([id, label, badge]) => (
          <button key={id} className={historyView === id ? "active" : ""} onClick={() => setHistoryView(id)}><span>{label}</span><b>{badge}</b></button>
        ))}
      </div>

      {historyView === "games" ? (
        <div className="history-layout">
          <Panel className="game-browser">
            <div className="browser-toolbar">
              <label className="search-box"><Search size={14} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search game, player, opponent…" /><kbd>/</kbd></label>
              <div className="filter-tabs">{(["all", "selfplay", "eval", "local"] as const).map((item) => <button className={filter === item ? "active" : ""} key={item} onClick={() => setFilter(item)}>{item}</button>)}</div>
              <IconButton label="Filter by iteration, winner, cap, seat, opening or length"><Filter size={15} /></IconButton>
            </div>
            <div className="active-query-row"><span>ITER 240</span><span>ANY WINNER</span><span>CAP: ANY</span><span>SORT: RECENT</span><button><X size={10} />clear</button></div>
            <div className="table-scroll">
              <table className="game-table telemetry-game-table">
                <thead><tr><th>Game / source</th><th>Blue / p0</th><th>Red / p1</th><th>Result</th><th>Iter</th><th>Plies</th><th>KLENT trace</th><th /></tr></thead>
                <tbody>
                  {filteredGames.map((game) => (
                    <tr key={game.id} className={selected.id === game.id ? "selected" : ""} onClick={() => { setSelected(game); setPly(Math.min(game.ply, 32)); setRecompute(false); }}>
                      <td><span className="game-id">{game.id}</span><small>{game.kind === "local" ? `local · ${game.time}` : `${game.kind} · index ${game.index}`}</small></td>
                      <td><i className="player-mark blue" />{game.blue}</td><td><i className="player-mark red" />{game.red}</td>
                      <td><span className={`result-chip ${game.result}`}>{game.result === "draw" ? "capped" : `${game.result} wins`}</span><small>{game.reason}</small></td>
                      <td className="mono">{game.iteration ?? "—"}</td><td className="mono">{game.ply}</td>
                      <td><span className={`trace-chip ${game.kind === "selfplay" ? "stored" : game.kind === "eval" ? "absent" : "local"}`}>{game.kind === "selfplay" ? "stored scalars" : game.kind === "eval" ? "game only" : "local trace"}</span></td>
                      <td><button className="row-menu" aria-label={`Actions for ${game.id}`}><MoreHorizontal size={15} /></button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="table-footer"><span>Showing {filteredGames.length} sample rows · 8,824 indexed</span><div><button disabled>Previous</button><b>1</b><button>2</button><button>3</button><span>…</span><button>148</button><button>Next</button></div></div>
          </Panel>

          <aside className="replay-panel">
            <Panel>
              <SectionLabel action={<span className={`source-badge ${selected.kind}`}>{selected.kind}</span>}>Replay · {selected.id}</SectionLabel>
              <div className="replay-matchup">
                <div><i className="player-mark blue" /><span><small>BLUE · P0</small><b>{selected.blue}</b></span></div><span className="versus">vs</span><div><i className="player-mark red" /><span><small>RED · P1</small><b>{selected.red}</b></span></div>
              </div>
              <div className="recompute-switch">
                <button className={!recompute ? "active" : ""} onClick={() => setRecompute(false)}>Stored trace</button>
                <button className={recompute ? "active" : ""} onClick={() => setRecompute(true)}>Recompute heads</button>
              </div>
              <div className="mini-board-frame"><HexBoard moves={mapMoves(HISTORY_MOVES.slice(0, Math.max(1, Math.round((ply / selected.ply) * HISTORY_MOVES.length))))} overlay={recompute ? "improved" : "off"} compact label={`Replay of ${selected.id}`} /><span className="replay-value">{selected.kind === "selfplay" && !recompute ? <>stored v̂ <b>+0.34</b></> : recompute ? <>counterfactual <b>π′</b></> : <><b>no ply trace</b></>}</span></div>
              <div className="replay-controls"><button aria-label="First move">|‹</button><button aria-label="Previous move">‹</button><button className="replay-play" aria-label="Play replay"><Play size={14} /></button><button aria-label="Next move">›</button><button aria-label="Last move">›|</button><span><b>{ply}</b> / {selected.ply}</span></div>
              <input className="timeline-range" type="range" min="1" max={selected.ply} value={ply} onChange={(event) => setPly(Number(event.target.value))} aria-label="Replay ply" />
              <div className="event-ticks"><i style={{ left: "22%" }} /><i style={{ left: "49%" }} /><i className="warn" style={{ left: "71%" }} /></div>
              {recompute && <div className="counterfactual-note">Uses checkpoint_000225.pt with its run coefficients. This is not guaranteed to be the collection-time actor view.</div>}
            </Panel>
            <Panel>
              <SectionLabel action={<span className="mono muted">mover frame</span>}>At ply {ply}</SectionLabel>
              {selected.kind === "eval" && !recompute ? (
                <div className="empty-trace"><Eye size={18} /><b>No acting-time ply rows</b><span>Evaluation used argmax πθ. Select a checkpoint and recompute to inspect its heads.</span></div>
              ) : (
                <div className="position-facts dense">
                  <div><span>mover / phase</span><strong>P{positionTurn.player} · {positionTurn.movesRemaining}/2</strong></div>
                  <div><span>legal cells</span><strong>1,284</strong></div>
                  <div><span>chosen rank</span><strong>2</strong></div>
                  <div><span>v̂</span><strong className="positive">+0.34</strong></div>
                  <div><span>KL(π′‖πθ)</span><strong>0.218</strong></div>
                  <div><span>norm entropy</span><strong>0.410</strong></div>
                  <div><span>π′ top-1</span><strong>31.2%</strong></div>
                  <div><span>π′ chosen</span><strong>22.4%</strong></div>
                </div>
              )}
              <button className="button primary full" onClick={() => inspectHistoryPosition()}><FlaskConical size={14} />Inspect prefix with checkpoint</button>
              <button className="button ghost full"><GitCompareArrows size={14} />Compare compatible checkpoints</button>
            </Panel>
            <Panel>
              <SectionLabel>Review labels &amp; notes</SectionLabel>
              <div className="tag-row">{selected.tags.map((tag) => <span key={tag}>{tag}<X size={10} /></span>)}<button><Plus size={11} />label</button></div>
              <textarea key={selected.id} defaultValue="Policy and Q disagree at this prefix. Inspect block 2, head 3 and the legal-cell decoder row." aria-label="Game review notes" />
            </Panel>
          </aside>
        </div>
      ) : (
        <div className="history-analysis-layout">
          <Panel className="history-analysis-main">
            {historyView === "swings" && (
              <>
                <SectionLabel action={<span className="mono muted">threshold |Δv̂| ≥ 0.50</span>}>Mover-frame value swing candidates</SectionLabel>
                <p className="analysis-intro">A large next-ply swing can be a real mistake or an unstable Q estimate. These are review candidates, not proven blunders.</p>
                <div className="swing-table">
                  <div className="table-head"><span>game</span><span>iter</span><span>ply</span><span>mover</span><span>v̂</span><span>next v̂</span><span>swing</span><span>rank</span></div>
                  {[["482119","240","32","P1 · 2/2","+0.61","−0.12","−0.49","2"],["482083","240","71","P0 · 1/2","+0.44","−0.18","−0.62","17"],["477201","239","18","P1 · 1/2","−0.28","+0.31","+0.59","9"],["471884","238","54","P0 · 2/2","+0.72","+0.09","−0.63","4"]].map((row, index) => <button key={row[0]} className={index === 0 ? "active" : ""}>{row.map((value) => <span key={value}>{value}</span>)}</button>)}
                </div>
              </>
            )}
            {historyView === "calibration" && (
              <>
                <SectionLabel action={<div className="head-toggle"><button className="active">by v̂</button><button>by ply</button><button>by length</button></div>}>v̂ reliability · finished self-play</SectionLabel>
                <p className="analysis-intro">Outcome is converted into the mover’s frame. Capped games are excluded because they have no realized winner.</p>
                <div className="calibration-chart">
                  <div className="perfect-line" />
                  {[[-.85,-.77,62],[-.65,-.58,184],[-.45,-.38,311],[-.25,-.19,508],[-.05,-.02,842],[.15,.12,694],[.35,.28,421],[.55,.47,236],[.75,.66,91]].map(([predicted, outcome, count]) => <i key={predicted} style={{ left: `${(predicted + 1) * 50}%`, bottom: `${(outcome + 1) * 50}%` }} title={`v̂ ${predicted} · outcome ${outcome} · ${count} plies`}><span>{count}</span></i>)}
                  <span className="cal-axis y">outcome</span><span className="cal-axis x">v̂ prediction</span>
                </div>
                <div className="calibration-summary"><span><small>PLIES</small><b>3,349</b></span><span><small>MAE</small><b>0.214</b></span><span><small>BIAS</small><b className="warn-text">+0.047</b></span><span><small>BUCKET</small><b>0.20</b></span></div>
              </>
            )}
            {historyView === "openings" && (
              <>
                <SectionLabel action={<span className="mono muted">first 4 plies · query-time canonical</span>}>D6 opening atlas</SectionLabel>
                <p className="analysis-intro">Rotations and reflections share one key. Stored games remain raw; canonicalization exists only in this query.</p>
                <div className="opening-list">
                  {[["(0,0) · (1,0) · (0,1) · (−1,1)","842","51.4%","48.1"],["(0,0) · (1,0) · (2,−1) · (0,1)","611","49.8%","52.7"],["(0,0) · (1,0) · (−1,1) · (1,−1)","489","53.2%","44.9"],["(0,0) · (1,0) · (0,−1) · (2,0)","374","47.1%","59.3"]].map(([opening,games,p0,length], index) => <button key={opening}><em>0{index + 1}</em><code>{opening}</code><span><small>games</small><b>{games}</b></span><span><small>p0 wins</small><b>{p0}</b></span><span><small>mean ply</small><b>{length}</b></span><ChevronRight size={13} /></button>)}
                </div>
              </>
            )}
            {historyView === "crossplay" && (
              <>
                <SectionLabel action={<span className="mono muted">offline argmax πθ · seat balanced</span>}>Checkpoint cross-play matrix</SectionLabel>
                <p className="analysis-intro">Each cell is the row checkpoint’s score fraction. The offline command replaces the matrix as a whole; it is not a promotion gate.</p>
                <div className="crossplay-matrix">
                  <div /><b>i175</b><b>i200</b><b>i225</b><b>i240</b>
                  {[
                    ["i175","—","0.43","0.38","0.35"],
                    ["i200","0.57","—","0.46","0.42"],
                    ["i225","0.62","0.54","—","0.48"],
                    ["i240","0.65","0.58","0.52","—"],
                  ].flatMap((row) => row.map((cell, index) => index === 0 ? <b key={`${row[0]}-label`}>{cell}</b> : <span key={`${row[0]}-${index}`} className={cell !== "—" && Number(cell) > .5 ? "positive-cell" : ""}>{cell}</span>))}
                </div>
              </>
            )}
          </Panel>
          <aside className="history-analysis-aside">
            <Panel>
              <SectionLabel>Query scope</SectionLabel>
              <dl className="definition-list">
                <div><dt>run</dt><dd>mantis-042</dd></div><div><dt>iterations</dt><dd>225—240</dd></div>
                <div><dt>kind</dt><dd>selfplay</dd></div><div><dt>capped</dt><dd>excluded</dd></div>
                <div><dt>telemetry</dt><dd>schema v1</dd></div><div><dt>repr</dt><dd>model v1</dd></div>
              </dl>
            </Panel>
            <Panel>
              <SectionLabel>Interpretation guardrail</SectionLabel>
              <div className="counterfactual-note">Stored self-play rows preserve acting-time scalars, not the dense π′ vector or actor checkpoint hash. Full head views are checkpoint-selected counterfactuals.</div>
              <button className="button primary full" onClick={() => inspectHistoryPosition()}><FlaskConical size={14} />Open selected case in lab</button>
            </Panel>
            <Panel>
              <SectionLabel>Related views</SectionLabel>
              <div className="suite-list">
                <button onClick={() => setHistoryView("swings")}><span className="suite-icon"><Activity size={14} /></span><span><b>Largest value swings</b><small>mover-aware transition query</small></span><ChevronRight size={14} /></button>
                <button onClick={() => setHistoryView("calibration")}><span className="suite-icon"><Gauge size={14} /></span><span><b>Calibration drift</b><small>v̂ vs realized outcome</small></span><ChevronRight size={14} /></button>
              </div>
            </Panel>
          </aside>
        </div>
      )}
    </div>
  );
}

function LiveRunScreen() {
  const [live, setLive] = useState(true);
  const [metric, setMetric] = useState<"eval" | "policy" | "q" | "kl" | "throughput">("eval");
  const metrics = {
    eval: { label: "SealBot score", value: "54.7%", delta: "+3.1 pp vs i230", unit: "periodic · seat balanced", y: ["60%", "55%", "50%", "45%"], points: [31,33,32,36,35,39,41,40,44,47,48,52] },
    policy: { label: "Policy CE", value: "1.284", delta: "−0.126 over 12 iters", unit: "πθ ← π′", y: ["1.7", "1.5", "1.3", "1.1"], points: [57,54,55,49,48,44,46,39,36,34,31,27] },
    q: { label: "Taken-action Q MSE", value: "0.483", delta: "−0.041 over 12 iters", unit: "Q(s,aₜ) ← Gₜ", y: [".62", ".54", ".46", ".38"], points: [52,50,51,46,43,45,39,37,34,31,29,27] },
    kl: { label: "Acting KL", value: "0.218", delta: "+0.012 over 12 iters", unit: "D_KL(π′‖πθ)", y: [".30", ".23", ".16", ".09"], points: [22,26,24,31,29,34,37,35,41,43,47,49] },
    throughput: { label: "Trainable samples / s", value: "8.16k", delta: "+8.1% over 12 iters", unit: "buffer_samples / iteration seconds", y: ["9k", "8k", "7k", "6k"], points: [22,25,31,29,36,39,37,41,47,45,53,56] },
  };
  const currentMetric = metrics[metric];

  return (
    <div className="screen live-run-screen">
      <div className="screen-heading run-heading">
        <div><div className="eyebrow">KLENT TRAINING / MANTIS-042 / MODEL REPR V1</div><h1>Live run <span className="run-live"><StatusDot tone={live ? "ok" : "warn"} pulse={live} />{live ? "LIVE" : "STREAM PAUSED"}</span></h1><p>Follow pipelined collection, one-epoch fitting, KLENT diagnostics, anchored evaluation, and persisted artifacts.</p></div>
        <div className="heading-actions"><label className="run-select"><span>RUN</span><select defaultValue="mantis-042"><option value="mantis-042">mantis-042 · active</option><option>mantis-041 · completed</option><option>mantis-ablate-q · stopped</option></select><ChevronDown size={14} /></label><button className="button ghost" onClick={() => setLive(!live)}>{live ? <Pause size={14} /> : <Play size={14} />}{live ? "Pause telemetry" : "Resume telemetry"}</button></div>
      </div>

      <div className="run-status-strip">
        <div><StatusDot tone="ok" /><span><small>FITTER</small><b>corpus i241</b></span><em>chunk 29 / 36 · epoch 1 / 1</em></div>
        <div><StatusDot tone="ok" /><span><small>COLLECTOR</small><b>corpus i242</b></span><em>3,872 / ≥4,096 finished</em></div>
        <div><StatusDot tone="blue" /><span><small>SNAPSHOT</small><b>pre-fit i241</b></span><em>one-fit-behind by design</em></div>
        <div><StatusDot tone="ok" /><span><small>LAST COMMIT</small><b>iteration 240</b></span><em>telemetry · 2s ago</em></div>
        <div><StatusDot tone="ok" /><span><small>STARVE GUARD</small><b>0 / 10</b></span><em>f 0.982 · healthy</em></div>
      </div>

      <div className="live-layout">
        <div className="live-main">
          <Panel className="hero-chart-panel">
            <div className="chart-header"><div><SectionLabel>{currentMetric.label}</SectionLabel><strong>{currentMetric.value}</strong><span className="chart-delta">{currentMetric.delta}</span><small className="metric-definition">{currentMetric.unit}</small></div><div className="filter-tabs metric-tabs">{(["eval", "policy", "q", "kl", "throughput"] as const).map((item) => <button key={item} className={metric === item ? "active" : ""} onClick={() => setMetric(item)}>{item}</button>)}</div></div>
            <div className="large-trend"><div className="y-labels">{currentMetric.y.map((label) => <span key={label}>{label}</span>)}</div><TrendChart points={currentMetric.points} label={`${currentMetric.label} over recent iterations`} /><div className="checkpoint-mark" style={{ left: "36%" }}><i /><span>ckpt i225</span></div><div className="checkpoint-mark active" style={{ left: "83%" }}><i /><span>eval i240</span></div></div>
            <div className="x-labels"><span>i230</span><span>i232</span><span>i234</span><span>i236</span><span>i238</span><span>i240</span></div>
          </Panel>

          <Panel className="iteration-pipeline-panel">
            <SectionLabel action={<span className="mono muted">GPU work serialized · CPU collection overlaps fit</span>}>Active KLENT iteration pipeline</SectionLabel>
            <div className="iteration-pipeline">
              <div className="pipeline-label"><span>MAIN</span><span>WORKER</span></div>
              <div className="pipeline-stage done"><span><Check size={13} /></span><div><small>CORPUS I241</small><b>143.8k on-policy samples</b><em>finished episodes only</em></div></div>
              <ChevronRight size={14} />
              <div className="pipeline-stage active"><span><BrainCircuit size={13} /></span><div><small>FIT I241</small><b>policy CE + Q MSE</b><em>one epoch · 29 / 36 chunks</em></div></div>
              <ChevronRight size={14} />
              <div className="pipeline-stage pending"><span><Save size={13} /></span><div><small>COMMIT</small><b>telemetry → checkpoint cadence</b><em>checkpoint next at i250</em></div></div>
              <div className="pipeline-snapshot"><span>weights snapshot</span><i /></div>
              <div className="pipeline-stage collecting"><span><Radio size={13} /></span><div><small>COLLECT I242</small><b>1,024 persistent slots</b><em>3,872 / ≥4,096 finished</em></div></div>
            </div>
            <div className="pipeline-footnote"><StatusDot tone="blue" />Collection i+1 acts through weights copied before fit i. Its one-fit staleness is explicit run provenance, not replay.</div>
          </Panel>

          <div className="live-grid-two">
            <Panel className="anchor-eval-panel">
              <SectionLabel action={<button className="tiny-action">strength curve <ChevronRight size={11} /></button>}>Last completed SealBot evaluation · i240</SectionLabel>
              <div className="eval-score-row"><div><small>SCORE</small><strong>54.7%</strong><span>95% CI · 42.5—66.4</span></div><div><small>WIN RATE</small><strong>50.0%</strong><span>32 W · 26 L · 6 capped</span></div><div><small>RELATIVE ELO</small><strong>+33</strong><span>CI · −53—+119</span></div></div>
              <div className="gate-matchup"><div><span className="checkpoint-glyph mint"><BrainCircuit size={17} /></span><span><b>Mantis live weights · i240</b><small>argmax πθ · no search · repr v1</small></span><strong>64 games</strong></div><div><span className="checkpoint-glyph blue"><Check size={17} /></span><span><b>SealBot · anchored</b><small>paired 2—6 ply openings · 0.10s · depth cap 12</small></span><strong>d̄ 7.9</strong></div></div>
              <div className="gate-facts"><span><b>56.3%</b> as P0</span><span><b>53.1%</b> as P1</span><span><b>61.8</b> avg ply</span><span><b>94s</b> elapsed</span></div>
            </Panel>
            <Panel className="klent-signal-panel">
              <SectionLabel action={<span className="mono muted">committed i240</span>}>KLENT losses &amp; acting signals</SectionLabel>
              <div className="loss-list compact">{[["policy CE","1.284",64,"mint"],["taken-action Q MSE","0.483",48,"blue"],["acting KL","0.218",44,"violet"],["normalized entropy","0.410",41,"amber"],["v̂ MAE","0.214",43,"blue"],["terminating fraction f","0.982",98,"mint"]].map(([name,value,width,tone]) => <div key={name}><span>{name}</span><b>{value}</b><i><em className={tone} style={{ width: `${width}%` }} /></i></div>)}</div>
            </Panel>
          </div>

          <Panel>
            <SectionLabel action={<span className="health-chip ok"><Check size={11} />no active alerts</span>}>Self-play health &amp; bias diagnostics</SectionLabel>
            <div className="run-diagnostic-grid">
              <div><span>won length mean</span><b>46.8 ply</b><em>stable</em></div>
              <div><span>P0 win rate</span><b>50.9%</b><em>+0.9 pp</em></div>
              <div><span>first-stone wins</span><b>48.6%</b><em>healthy</em></div>
              <div><span>v̂ · winners</span><b>+0.37</b><em>mover frame</em></div>
              <div><span>v̂ · losers</span><b>−0.35</b><em>mover frame</em></div>
              <div><span>capped episodes</span><b>74 / 4,170</b><em>drop whole</em></div>
              <div><span>fit steps</span><b>36</b><em>effective batch 4,096</em></div>
              <div><span>iteration time</span><b>17.62s</b><em>eval excluded</em></div>
            </div>
          </Panel>

          <Panel className="checkpoint-pipeline">
            <SectionLabel action={<button className="tiny-action"><Filter size={11} />artifacts &amp; evals</button>}>Artifact timeline · no automatic promotion</SectionLabel>
            <div className="pipeline-list">{[
              ["live","live weights","i241","not durable","12:43","fitting now"],
              ["measured","eval @ i240","54.7%","SealBot · 64","12:38","telemetry only"],
              ["scheduled","checkpoint_000250.pt","in 9 iters","periodic","—","next write"],
              ["written","checkpoint_000225.pt","versions exact","model + opt + RNG","10:48","latest durable"],
              ["written","checkpoint_000200.pt","measured","cross-play ready","08:31","retained"],
            ].map(([state,artifact,metric,detail,time,note]) => <div className="pipeline-row" key={artifact}><span className={`pipeline-state ${state}`}><i />{state}</span><b>{artifact}</b><span className="mono">{metric} <em>{detail}</em></span><span>{time}</span><span className="muted">{note}</span><button aria-label={`More actions for ${artifact}`}><MoreHorizontal size={15} /></button></div>)}</div>
          </Panel>
        </div>

        <aside className="live-aside">
          <Panel>
            <SectionLabel action={<span className="architecture-badge">EXACT RUN KNOBS</span>}>Run manifest</SectionLabel>
            <div className="run-manifest-head"><span className="checkpoint-glyph mint"><BrainCircuit size={17} /></span><span><b>mantis-042</b><small>fresh run · seed 84291 · CUDA:0</small></span><em>ACTIVE</em></div>
            <div className="manifest-version-row"><span>model repr <b>1</b></span><span>rules <b>exact</b></span><span>action order <b>exact</b></span><span>torch <b>exact</b></span></div>
            <dl className="definition-list manifest-list">
              <div><dt>τ / λ</dt><dd>0.10 / 0.03</dd></div><div><dt>λ-return</dt><dd>0.939</dd></div>
              <div><dt>env slots</dt><dd>1,024</dd></div><div><dt>game quota</dt><dd>≥4,096</dd></div>
              <div><dt>effective batch</dt><dd>4,096</dd></div><div><dt>learning rate</dt><dd>1e−3 Adam</dd></div>
              <div><dt>ply cap</dt><dd>512 · drop</dd></div><div><dt>autocast / compile</dt><dd>bf16 / on</dd></div>
              <div><dt>checkpoint</dt><dd>every 25</dd></div><div><dt>SealBot eval</dt><dd>every 10</dd></div>
              <div><dt>fit pair / cell</dt><dd>8.0M / 800k</dd></div><div><dt>collect pair / cell</dt><dd>24M / 2.4M</dd></div>
              <div><dt>starve limit</dt><dd>10 iters</dd></div><div><dt>provenance</dt><dd>fresh · no init</dd></div>
            </dl>
            <button className="inline-link"><Eye size={13} />open invocation record</button>
          </Panel>
          <Panel>
            <SectionLabel action={<span className="live-label"><StatusDot tone="ok" pulse />24 samples</span>}>Process &amp; device telemetry</SectionLabel>
            <div className="util-list">{[["CUDA:0 utilization",92,"mean 84%","max 99%"],["NVML memory",78,"mean 9.8 GB","max 11.2 GB"],["CPU process",64,"mean 61%","max 78%"],["Resident set",71,"mean 8.4 GB","max 9.1 GB"],["Host RAM used",58,"mean 36.9 GB","max 38.2 GB"]].map(([name,value,meta,meta2]) => <div key={name}><span><b>{name}</b><em>{value}%</em></span><i><em style={{ width: `${value}%` }} /></i><small><span>{meta}</span><span>{meta2}</span></small></div>)}</div>
            <div className="hardware-foot"><span>GPU 71°C</span><span>287W max</span><span>torch reserved 10.7 GB</span></div>
          </Panel>
          <Panel className="worker-panel">
            <SectionLabel action={<span className="mono muted">1,024 persistent · 1 tile = 16</span>}>Collector slot cohorts</SectionLabel>
            <div className="worker-grid slot-grid">{Array.from({ length: 64 }, (_, index) => <i key={index} className={index === 37 || index === 54 ? "slow" : index === 44 || index === 60 ? "warm" : ""} title={`slots ${index * 16 + 1}—${index * 16 + 16}`} />)}</div>
            <div className="worker-legend"><span><i />active 960</span><span><i className="warm" />near cap 32</span><span><i className="slow" />resetting 32</span></div>
            <p className="panel-note">Slots persist across iterations. A final lockstep can overshoot the finished-game quota; capped episodes contribute no samples.</p>
          </Panel>
          <Panel className="log-panel">
            <SectionLabel action={<button className="tiny-action"><Terminal size={11} />full logs</button>}>Event stream</SectionLabel>
            <div className="log-stream"><p><time>12:43:08</time><span className="ok">collect</span>i242 · 3,872 finished games</p><p><time>12:42:51</time><span>fit</span>i241 · chunk 29 / 36 complete</p><p><time>12:42:32</time><span>buffer</span>i241 corpus · 143,812 samples</p><p><time>12:42:10</time><span className="ok">db</span>iteration 240 committed atomically</p><p><time>12:41:58</time><span>eval</span>SealBot i240 · score 0.547 persisted</p><p><time>12:41:22</time><span>ckpt</span>next periodic write at iteration 250</p></div>
          </Panel>
          <Panel>
            <SectionLabel>Run controls</SectionLabel>
            <div className="danger-actions"><button><Save size={14} />checkpoint after fit</button><button><Pause size={14} />stop after iteration</button><button className="danger"><CircleStop size={14} />stop now</button></div>
            <p className="panel-note">Draft controls are inert. Resume restores model, optimizer, and learner RNG; in-flight collector slot state is not checkpointed.</p>
          </Panel>
        </aside>
      </div>
    </div>
  );
}

function MantisModelLabScreen({ context }: { context: LabContext }) {
  const [module, setModule] = useState<LabModule>(context.module);
  const [overlay, setOverlay] = useState<Overlay>(context.overlay);
  const [moves, setMoves] = useState<Record<number, Player>>(() => mapMoves(HISTORY_MOVES));
  const [selectedCell, setSelectedCell] = useState(36);
  const [editMode, setEditMode] = useState<"sequence" | "free">("sequence");
  const [compare, setCompare] = useState(true);
  const [opacity, setOpacity] = useState(82);
  const [tau, setTau] = useState(0.1);
  const [lam, setLam] = useState(0.03);
  const labTurn = turnAtStoneCount(Object.keys(moves).length);
  const comparisonArtifact = context.checkpoint.includes("000225") ? "checkpoint_000200.pt" : "checkpoint_000225.pt";

  const policyRows = [
    { move: "G4", policy: 18.7, q: 0.42, improved: 31.2, delta: "+12.5" },
    { move: "F5", policy: 14.2, q: 0.36, improved: 22.4, delta: "+8.2" },
    { move: "H3", policy: 11.9, q: 0.31, improved: 15.8, delta: "+3.9" },
    { move: "G5", policy: 9.1, q: 0.25, improved: 10.7, delta: "+1.6" },
    { move: "E6", policy: 7.6, q: 0.19, improved: 7.8, delta: "+0.2" },
    { move: "F4", policy: 6.4, q: 0.14, improved: 4.9, delta: "−1.5" },
  ];

  const moduleCopy: Record<LabModule, { title: string; detail: React.ReactNode }> = {
    graph: {
      title: "decoder path",
      detail: <>G4 is a legal decoder row touching <b>3 live windows</b>: 1 end, 1 near-end, and 1 centre slot.</>,
    },
    policy: {
      title: "policy decoder · πθ",
      detail: <>G4 has raw logit <b>+2.18</b> and segmented-softmax probability <b>18.7%</b> in engine legal order.</>,
    },
    q: {
      title: "action-value decoder · Q",
      detail: <>G4 has <b>Q(s,a) = +0.42</b>. The independent decoder ends in tanh, so every legal-cell value stays in (−1, 1).</>,
    },
    improve: {
      title: "closed-form improvement · π′",
      detail: <>With τ={tau.toFixed(2)} and λ={lam.toFixed(2)}, G4 rises from <b>18.7% πθ</b> to <b>31.2% π′</b>.</>,
    },
    attention: {
      title: "stone attention",
      detail: <>Block 2 · head 3 reads <b>[global token; stones]</b> with learned hex-distance bias. Windows do not attend.</>,
    },
    value: {
      title: "state-value readout · V65",
      detail: <>The 4-query, 65-bin state-value head is present in the base architecture but <b>outside the KLENT loss</b>.</>,
    },
  };

  function selectModule(next: LabModule) {
    setModule(next);
    setOverlay(
      next === "policy"
        ? "policy"
        : next === "q"
          ? "q"
          : next === "improve"
            ? "improved"
            : "off",
    );
  }

  function editCell(cell: number) {
    setSelectedCell(cell);
    setMoves((current) => {
      if (editMode === "free" && current[cell] !== undefined) {
        const next = { ...current };
        delete next[cell];
        return next;
      }
      if (current[cell] !== undefined) return current;
      return { ...current, [cell]: turnAtStoneCount(Object.keys(current).length).player };
    });
  }

  return (
    <div className="screen lab-screen mantis-lab">
      <div className="screen-heading">
        <div>
          <div className="eyebrow">MANTISNET / MODEL REPR V1 / KLENT</div>
          <h1>MantisNet model lab</h1>
          <p>Inspect the entity graph, policy decoder, action-value decoder, closed-form KLENT improvement, trunk attention, and optional state-value readout.</p>
        </div>
        <div className="heading-actions">
          <span className="architecture-badge"><Check size={12} /> H128 · B4 · A4 · 1.25M</span>
          <button className="button ghost"><Save size={14} />Save probe</button>
          <button className="button ghost"><GitCompareArrows size={14} />Compare <span className="count-badge">2</span></button>
        </div>
      </div>

      <div className="lab-context-strip"><span><small>POSITION SOURCE</small><b>{context.source}</b></span><ChevronRight size={12} /><span><small>MODEL ARTIFACT</small><b>{context.checkpoint}</b></span><ChevronRight size={12} /><span><small>OPENED AT</small><b>{moduleCopy[module].title}</b></span></div>

      <div className="lab-module-nav mantis-module-nav" role="tablist" aria-label="MantisNet lab modules">
        {([
          ["graph", Layers3, "Entity graph", "REPR"],
          ["policy", Eye, "Policy πθ", "HEAD"],
          ["q", Gauge, "Action Q", "HEAD"],
          ["improve", Sparkles, "KLENT π′", "DERIVED"],
          ["attention", BrainCircuit, "Attention", "TRUNK"],
          ["value", BarChart3, "State V", "UNTRAINED"],
        ] as Array<[LabModule, ComponentType<{ size?: number }>, string, string]>).map(([id, Icon, label, badge]) => (
          <button key={id} className={module === id ? "active" : ""} onClick={() => selectModule(id)} role="tab" aria-selected={module === id}>
            <Icon size={15} />
            {label}
            <span className={`beta ${id === "value" ? "warning" : ""}`}>{badge}</span>
          </button>
        ))}
      </div>

      <div className="lab-layout">
        <aside className="lab-tools">
          <Panel>
            <SectionLabel>Position</SectionLabel>
            <label className="field-label">
              <span>SOURCE</span>
              <span className="select-shell">
                <select defaultValue="hx-8824-32">
                  <option value="hx-8824-32">{context.source}</option>
                  <option value="empty">Empty legal position</option>
                  <option value="opening-7">Saved legal prefix · opening 7</option>
                  <option value="swing-71">Value-swing case · prefix 71</option>
                </select>
                <ChevronDown size={14} />
              </span>
            </label>
            <Segmented label="Position edit mode" value={editMode} onChange={setEditMode} options={[{ value: "sequence", label: "Legal replay" }, { value: "free", label: "Graph sketch" }]} />
            <div className="mini-stats three">
              <div><strong>{Object.keys(moves).length}</strong><span>stones</span></div>
              <div><strong className={labTurn.player === 0 ? "blue-text" : "negative"}>P{labTurn.player}</strong><span>{labTurn.phase}</span></div>
              <div><strong>{labTurn.movesRemaining}</strong><span>moves remain</span></div>
            </div>
            <div className="tool-button-row">
              <button onClick={() => setMoves(mapMoves(HISTORY_MOVES.slice(0, -1)))}><RotateCcw size={13} />undo</button>
              <button onClick={() => setMoves({})}><X size={13} />clear</button>
            </div>
            {editMode === "free" && <div className="counterfactual-note">Graph sketches may be unreachable. Head inference should stay disabled until the engine validates a legal prefix.</div>}
          </Panel>

          <Panel>
            <SectionLabel>Checkpoint</SectionLabel>
            <div className="checkpoint-select-card mint">
              <span className="checkpoint-glyph mint"><BrainCircuit size={17} /></span>
              <span><b>{context.checkpoint}</b><small>{context.checkpoint.startsWith("live") ? "in-memory snapshot · not yet durable" : "mantis-042 · version-checked checkpoint"}</small></span>
              <ChevronDown size={14} />
            </div>
            <div className="capability-row">
              <span className="ready">πθ</span><span className="ready">Q</span><span className="derived">π′</span><span className="idle">V65</span>
            </div>
            <label className="toggle-row">
              <span><b>Compare checkpoint</b><small>requires exact repr, rules, action order, torch and shape</small></span>
              <input type="checkbox" checked={compare} onChange={(event) => setCompare(event.target.checked)} /><i />
            </label>
            {compare && (
              <div className="checkpoint-select-card violet">
                <span className="checkpoint-glyph violet"><BrainCircuit size={17} /></span>
                <span><b>{comparisonArtifact}</b><small>same run · compatibility exact</small></span>
                <ChevronDown size={14} />
              </div>
            )}
          </Panel>

          <Panel>
            <SectionLabel action={<span className="mono muted">repr v1</span>}>Position graph</SectionLabel>
            <div className="entity-list">
              <div><i className="stone-entity" /><span><b>Stone nodes · S</b><small>own / opponent embedding</small></span><strong>{Object.keys(moves).length}</strong></div>
              <div><i className="window-entity" /><span><b>Live windows · W</b><small>colour × 34 canonical patterns</small></span><strong>184</strong></div>
              <div><i className="token-entity" /><span><b>Global token · g</b><small>base + moves_remaining</small></span><strong>1</strong></div>
              <div><i className="cell-entity" /><span><b>Legal decoder rows</b><small>not trunk nodes</small></span><strong>1,284</strong></div>
            </div>
            <div className="absence-note">No empty-cell grid, recency planes, or absolute-coordinate features.</div>
          </Panel>

          <Panel>
            <SectionLabel>Board overlay</SectionLabel>
            <div className="overlay-list">
              {([
                ["off", "None", "entity geometry"],
                ["policy", "Policy πθ", "segmented softmax"],
                ["q", "Action Q", "tanh · (−1, 1)"],
                ["improved", "Improved π′", "KLENT closed form"],
              ] as Array<[Overlay, string, string]>).map(([id, label, detail]) => (
                <button key={id} className={overlay === id ? "active" : ""} onClick={() => setOverlay(id)}>
                  <span className="radio-ring">{overlay === id && <i />}</span><span><b>{label}</b><small>{detail}</small></span>
                </button>
              ))}
            </div>
            <label className="range-field"><span>OPACITY <b>{opacity}%</b></span><input type="range" min="20" max="100" value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} /></label>
          </Panel>
        </aside>

        <main className="lab-board-area">
          <div className="board-toolbar">
            <div className="match-status"><StatusDot tone="blue" /><span>{moduleCopy[module].title}</span><span className="mono muted">query (1, −1) · engine rank 37</span></div>
            <div className="board-tool-icons"><IconButton label="Pan board"><Menu size={15} /></IconButton><IconButton label="Inspect legal cell" active><Search size={15} /></IconButton><IconButton label="Reset view"><RotateCcw size={15} /></IconButton></div>
          </div>
          <div className="board-frame lab-board-frame" style={{ "--overlay-opacity": opacity / 100 } as CSSProperties}>
            <div className="frame-corner top-left">MANTISNET · LEGAL CELL DECODER</div>
            <div className="frame-corner top-right">query <b>(1, −1)</b></div>
            <HexBoard moves={moves} overlay={overlay} selectedCell={selectedCell} onCellClick={editCell} label="Editable MantisNet model lab position" />
          </div>
          <div className="lab-readout mantis-readout">
            <div><span className="readout-icon"><Search size={14} /></span><span><small>{moduleCopy[module].title}</small><b>G4 · empty · legal</b></span></div>
            <p>{moduleCopy[module].detail}</p>
            <button aria-label="Pin readout"><Plus size={14} /></button>
          </div>

          <div className="mantis-architecture">
            <SectionLabel action={<span className="mono muted">H=128 · 4 blocks · final shared LN</span>}>Forward architecture</SectionLabel>
            <div className="architecture-flow">
              <div className="arch-inputs">
                <span><i className="stone-entity" />S · stones</span>
                <span><i className="window-entity" />W · live windows</span>
                <span><i className="token-entity" />g · token</span>
              </div>
              <ChevronRight size={14} />
              <div className="trunk-blocks">
                {[0, 1, 2, 3].map((block) => <span key={block} className={module === "attention" && block === 2 ? "active" : ""}><b>B{block}</b><small>W←S · S←W<br />attn + FFN</small></span>)}
              </div>
              <ChevronRight size={14} />
              <div className="arch-ln">LN<small>S · W · g</small></div>
              <ChevronRight size={14} />
              <div className="arch-heads">
                <span className={module === "policy" ? "active" : ""}><b>πθ</b><small>cell decoder</small></span>
                <span className={module === "q" ? "active" : ""}><b>Q</b><small>cell decoder</small></span>
                <span className={module === "value" ? "active warning" : ""}><b>V65</b><small>4 queries</small></span>
              </div>
            </div>
          </div>

          <div className="lab-bottom-grid">
            <Panel>
              <SectionLabel action={<span className="mono muted">shared incidence · separate weights</span>}>Cell decoder row · G4</SectionLabel>
              <div className="decoder-row">
                <div><span>Σ W</span><b>H=128</b><small>3 live-window embeddings</small></div>
                <Plus size={12} />
                <div><span>slot classes</span><b>[1, 1, 1]</b><small>end · near · centre</small></div>
                <Plus size={12} />
                <div><span>background</span><b>none</b><small>8 buckets when uncovered</small></div>
              </div>
            </Panel>
            <Panel>
              <SectionLabel>{module === "attention" ? "4 × 4 attention map" : "D6 invariance check"}</SectionLabel>
              {module === "attention" ? (
                <><div className="attention-4x4">{Array.from({ length: 16 }, (_, index) => <button key={index} className={index === 11 ? "selected" : ""} style={{ "--matrix-alpha": ((index * 31) % 97) / 100 } as CSSProperties} aria-label={`Block ${Math.floor(index / 4)}, head ${index % 4}`}><span>{Math.floor(index / 4)}.{index % 4}</span></button>)}</div><div className="matrix-axis"><span>block 0</span><span>block 3</span></div></>
              ) : (
                <div className="symmetry-check"><span><Check size={16} /></span><div><b>12 / 12 transforms agree</b><small>policy and Q permute · scalar readouts invariant</small></div></div>
              )}
            </Panel>
          </div>
        </main>

        <aside className="lab-inspector">
          <Panel className="head-status-panel">
            <SectionLabel>Output manifest</SectionLabel>
            <div className="head-manifest">
              <div className="ready"><span>πθ</span><p><b>Policy decoder</b><small>trained · one logit / legal cell</small></p><em>HEAD</em></div>
              <div className="ready"><span>Q</span><p><b>Action-value decoder</b><small>trained · tanh bounded</small></p><em>HEAD</em></div>
              <div className="derived"><span>π′</span><p><b>Improved policy</b><small>derived from πθ, Q, τ, λ</small></p><em>OPERATOR</em></div>
              <div className="warning"><span>V</span><p><b>State-value distribution</b><small>65 bins · outside KLENT loss</small></p><em>UNTRAINED</em></div>
            </div>
          </Panel>

          <Panel className="klent-readout-panel">
            <SectionLabel action={<span className="latency-badge"><Zap size={10} />23ms</span>}>Acting-time readout</SectionLabel>
            <div className="vhat-hero"><span>v̂</span><strong>+0.34</strong><small>E<sub>A~π′</sub>[Q(s,A)]</small></div>
            <div className="horizon-grid mantis-metrics">
              <div><span>KL(π′‖πθ)</span><b>0.218</b></div>
              <div><span>H(π′)/LOG|A|</span><b>0.410</b></div>
              <div><span>TOP-1 π′</span><b>31.2%</b></div>
              <div><span>|A LEGAL|</span><b>1,284</b></div>
            </div>
          </Panel>

          <Panel className="mantis-policy-panel">
            <SectionLabel action={<div className="head-toggle"><button className={overlay === "policy" ? "active" : ""} onClick={() => setOverlay("policy")}>πθ</button><button className={overlay === "q" ? "active" : ""} onClick={() => setOverlay("q")}>Q</button><button className={overlay === "improved" ? "active" : ""} onClick={() => setOverlay("improved")}>π′</button></div>}>Legal-cell outputs</SectionLabel>
            <div className="mantis-policy-columns"><span>move</span><span>πθ</span><span>Q</span><span>π′</span><span>Δ</span></div>
            {policyRows.map((row, index) => (
              <button key={row.move} className={selectedCell === 36 + index ? "active" : ""} onClick={() => setSelectedCell(36 + index)}>
                <span><em>{index + 1}</em><b>{row.move}</b></span>
                <span className="rank-bar mint"><i style={{ width: `${row.policy * 4}%` }} /><b>{row.policy}%</b></span>
                <strong className="positive">{row.q > 0 ? "+" : ""}{row.q.toFixed(2)}</strong>
                <span className="rank-bar amber"><i style={{ width: `${row.improved * 2.7}%` }} /><b>{row.improved}%</b></span>
                <strong className={row.delta.startsWith("+") ? "positive" : "negative"}>{row.delta}</strong>
              </button>
            ))}
          </Panel>

          {module === "graph" && (
            <Panel>
              <SectionLabel>Representation facts</SectionLabel>
              <div className="diagnostic-list">
                <div><span>Canonical window patterns</span><b>34</b><em>D6</em></div>
                <div><span>Slot classes</span><b>3</b><em>end / near / centre</em></div>
                <div><span>Background buckets</span><b>8</b><em>nearest stone</em></div>
                <div><span>Action ordering</span><b>engine</b><em className="ok">pinned</em></div>
              </div>
            </Panel>
          )}

          {module === "policy" && (
            <Panel>
              <SectionLabel>Policy decoder</SectionLabel>
              <p className="architecture-note">A gather-sum scores every legal cell exactly once. Covered cells read their live windows; uncovered cells use the nearest-stone background bucket. Softmax is segmented per position.</p>
              <div className="diagnostic-list"><div><span>Raw logit · G4</span><b>+2.18</b><em>rank 1</em></div><div><span>πθ probability</span><b>18.7%</b><em>engine index 37</em></div></div>
            </Panel>
          )}

          {module === "q" && (
            <Panel>
              <SectionLabel>Action-value decoder</SectionLabel>
              <p className="architecture-note">The Q head mirrors the policy decoder’s shape but owns every projection, embedding table, and MLP parameter. Only the incidence aggregation is shared.</p>
              <div className="q-scale"><span>−1</span><i><em style={{ left: "71%" }} /></i><span>+1</span></div>
              <div className="diagnostic-list"><div><span>Q(s,G4)</span><b>+0.42</b><em>tanh bounded</em></div><div><span>Taken-action target</span><b>G<sub>t</sub></b><em>λ-return</em></div></div>
            </Panel>
          )}

          {module === "improve" && (
            <Panel>
              <SectionLabel>KLENT operator · not a head</SectionLabel>
              <div className="klent-equation">π′(a|s) ∝ exp[(Q(s,a) + τ log πθ(a|s)) / (τ + λ)]</div>
              <label className="range-field"><span>τ · REVERSE KL <b>{tau.toFixed(2)}</b></span><input type="range" min="0" max="30" value={tau * 100} onChange={(event) => setTau(Number(event.target.value) / 100)} /></label>
              <label className="range-field"><span>λ · ENTROPY <b>{lam.toFixed(2)}</b></span><input type="range" min="0" max="20" value={lam * 100} onChange={(event) => setLam(Number(event.target.value) / 100)} /></label>
              <p className="panel-note">π′ exists during collection and training. Evaluation and the exported playing artifact use argmax πθ with no search.</p>
            </Panel>
          )}

          {module === "attention" && (
            <Panel>
              <SectionLabel>Stone attention + token</SectionLabel>
              <div className="block-head-controls"><label><span>BLOCK</span><select defaultValue="2"><option>0</option><option>1</option><option>2</option><option>3</option></select></label><label><span>HEAD</span><select defaultValue="3"><option>0</option><option>1</option><option>2</option><option>3</option></select></label></div>
              <div className="distance-buckets">{["d1","d2","d3","d4–6","d7–12","self","token"].map((bucket, index) => <div key={bucket}><span>{bucket}</span><i><em style={{ width: `${82 - index * 8}%` }} /></i><b>{(0.31 - index * 0.035).toFixed(2)}</b></div>)}</div>
              <p className="panel-note">Queries and keys are [global token; stones]. Windows receive global context through stone↔window message passing.</p>
            </Panel>
          )}

          {module === "value" && (
            <Panel className="value-warning-panel">
              <SectionLabel action={<span className="warning-chip">OUTSIDE KLENT LOSS</span>}>State-value head · V65</SectionLabel>
              <p className="architecture-note">Four learned queries attend over [token; live windows], then emit a 65-bin distribution on [−1, 1]. The scalar is its expectation. Do not confuse it with v̂ = E<sub>π′</sub>[Q].</p>
              <div className="distribution value-65">{Array.from({ length: 33 }, (_, index) => <i key={index} style={{ height: `${8 + Math.max(0, 32 - Math.abs(index - 21) * 3) + ((index * 7) % 8)}px` }} className={index > 16 ? "positive" : ""} />)}</div>
              <div className="distribution-axis"><span>−1</span><span>0</span><span>+1</span></div>
              <div className="query-row">{[0,1,2,3].map((query) => <span key={query}>query {query}<b>{[0.31,0.26,0.24,0.19][query]}</b></span>)}</div>
              <div className="value-caution"><StatusDot tone="warn" />This KLENT checkpoint does not train this head; values are diagnostic only.</div>
            </Panel>
          )}

          <Panel>
            <SectionLabel>Saved architecture probes</SectionLabel>
            <div className="probe-list">
              <button><span><Star size={13} /><b>Decoder disagreement · G4</b></span><small>πθ / Q / π′ · engine rank 37</small></button>
              <button><span><Clock3 size={13} /><b>Distance-bias check</b></span><small>block 2 · head 3 · token + stones</small></button>
              <button><span><Plus size={13} /><b>Save current probe</b></span><small>position + checkpoint + module state</small></button>
            </div>
          </Panel>
        </aside>
      </div>
    </div>
  );
}

function CommandPalette({ onClose, navigate }: { onClose: () => void; navigate: (screen: Screen) => void }) {
  return (
    <div className="command-backdrop" role="presentation" onMouseDown={onClose}>
      <div className="command-palette" role="dialog" aria-modal="true" aria-label="Command palette" onMouseDown={(event) => event.stopPropagation()}>
        <div className="command-search"><Search size={17} /><input autoFocus placeholder="Type a command or jump to…" /><kbd>ESC</kbd></div>
        <div className="command-group"><span>NAVIGATE</span>{NAV_ITEMS.map((item) => { const Icon = item.icon; return <button key={item.id} onClick={() => { navigate(item.id); onClose(); }}><Icon size={15} /><b>{item.label}</b><kbd>G {item.short[0]}</kbd></button>; })}</div>
        <div className="command-group"><span>QUICK ACTIONS</span><button><Plus size={15} /><b>Start paired evaluation</b><small>Seat-balanced · shared openings</small></button><button onClick={() => { navigate("lab"); onClose(); }}><FlaskConical size={15} /><b>Inspect current live weights</b><small>completed i240 · not yet durable</small></button><button onClick={() => { navigate("history"); onClose(); }}><Activity size={15} /><b>Open latest value swing</b><small>mover-aware v̂ diagnostic</small></button><button onClick={() => { navigate("live"); onClose(); }}><Radio size={15} /><b>View active iteration</b><small>fit i241 · collect i242</small></button></div>
        <div className="command-footer"><span><kbd>↑</kbd><kbd>↓</kbd> navigate</span><span><kbd>↵</kbd> open</span><span><kbd>esc</kbd> close</span></div>
      </div>
    </div>
  );
}

export default function ConsoleApp() {
  const [screen, setScreen] = useState<Screen>("play");
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [labContext, setLabContext] = useState<LabContext>(DEFAULT_LAB_CONTEXT);

  function navigate(next: Screen) {
    setScreen(next); setMobileNavOpen(false); window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function openLab(context: Partial<LabContext> = {}) {
    setLabContext((current) => ({ ...current, ...context }));
    navigate("lab");
  }

  const current = NAV_ITEMS.find((item) => item.id === screen);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark" aria-hidden="true"><i /><i /><i /></span><div><b>SHRIMP</b><small>CONTROL DECK</small></div></div>
        <nav className={mobileNavOpen ? "open" : ""} aria-label="Primary navigation">
          {NAV_ITEMS.map((item) => { const Icon = item.icon; return <button key={item.id} onClick={() => navigate(item.id)} className={screen === item.id ? "active" : ""}><Icon size={15} strokeWidth={1.8} /><span>{item.label}</span>{item.id === "live" && <i className="nav-live" />}</button>; })}
        </nav>
        <div className="topbar-actions">
          <div className="connection-chip"><StatusDot tone="ok" pulse /><span>LAN</span><b>12ms</b></div>
          <button className="command-trigger" onClick={() => setPaletteOpen(true)}><Search size={14} /><span>Jump or command</span><kbd>⌘ K</kbd></button>
          <IconButton label="Notifications"><Bell size={16} /><span className="notification-dot">2</span></IconButton>
          <button className="avatar" aria-label="Open user menu">CM</button>
          <button className="mobile-menu" aria-label="Toggle navigation" onClick={() => setMobileNavOpen(!mobileNavOpen)}>{mobileNavOpen ? <X size={18} /> : <Menu size={18} />}</button>
        </div>
      </header>

      <div className="contextbar">
        <div className="breadcrumbs"><span>HEXOSHRIMP</span><ChevronRight size={11} /><b>{current?.label.toUpperCase()}</b></div>
        <div className="system-summary"><span><Cpu size={12} />CUDA:0 <i>healthy</i></span><span><Radio size={12} />fit i241 · collect i242 <i>active</i></span><span><Users size={12} />1,024 slots</span><time>telemetry · 2s</time></div>
      </div>

      {screen === "play" && <PlayScreen sendToLab={openLab} />}
      {screen === "history" && <HistoryScreen sendToLab={openLab} />}
      {screen === "live" && <LiveRunScreen />}
      {screen === "lab" && <MantisModelLabScreen context={labContext} />}

      <div className="compare-tray">
        <button className="tray-handle"><GitCompareArrows size={14} /><span>Compare tray</span><b>2</b><ChevronDown size={13} /></button>
        <div className="tray-items"><span><i className="mint" />live weights · i240 · repr v1 <em>EXACT</em><button aria-label="Remove checkpoint"><X size={10} /></button></span><span><i className="violet" />checkpoint i225 · repr v1 <em>EXACT</em><button aria-label="Remove checkpoint"><X size={10} /></button></span></div>
        <button className="tray-open" onClick={() => openLab({ source: "Compare tray · saved legal prefix", checkpoint: "live weights @ i240", module: "policy", overlay: "policy" })}>Compare in lab <ChevronRight size={12} /></button>
      </div>

      {paletteOpen && <CommandPalette onClose={() => setPaletteOpen(false)} navigate={navigate} />}
    </div>
  );
}
