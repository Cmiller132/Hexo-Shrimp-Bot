import { ClipboardPaste, Copy, CornerUpLeft, Save, Search, Trash2, Undo2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, isAbortError, post, query, reasonText, useApi } from "../api";
import Board, { sequentialStep, TOP_K_STEPS, type Overlay } from "../components/Board";
import Chart, { type Series } from "../components/Chart";
import Transport, { useDeckKeys } from "../components/Replay";
import { Empty, ErrorBox, format, HeatLegend, Metric, Notice, Panel, ProgressBar, Segmented } from "../components/Ui";
import { lineKey, moveKey, p1TurnBands, playerAt } from "../lib/hex";
import { useLineInspect, type LineCursor, type Reading } from "../lib/inspect";
import type {
  AttentionResult, Candidate, D6Result, Game, GameRow, LabHandoff, ModelManifest, Move, Probe, Run, StoredPly,
} from "../types";

/* ------------------------------------------------------------------ the line */

/** Where the line on screen came from, and the line it came in as. Provenance
 *  decides which extra traces are available — only a self-play game carries the
 *  acting net's own per-ply read — and `moves` is what the line is measured
 *  against: where it departs from them is where the source stops describing it. */
type Source =
  | { kind: "blank" }
  | { kind: "paste"; moves: Move[] }
  | { kind: "probe"; probeId: number; name: string; moves: Move[] }
  | {
    kind: "game"; run: string; gameId: number; gameKind: GameRow["kind"]; iteration: number;
    winner: number | null; capped: boolean; moves: Move[]; plies: StoredPly[];
  };

const plies = (count: number) => `${count} ${count === 1 ? "ply" : "plies"}`;

/** How many plies two lines share from the start. */
function commonPrefix(a: Move[], b: Move[]): number {
  const limit = Math.min(a.length, b.length);
  let shared = 0;
  while (shared < limit && a[shared][0] === b[shared][0] && a[shared][1] === b[shared][1]) shared++;
  return shared;
}

interface RecentGame { run: string; gameId: number; length: number }
const RECENT_KEY = "deck-lab-recent";

/** The recently-opened list, or the reason it could not be read. A stored value of
 *  another shape is a problem to report and clear, not an empty list to present as
 *  if nothing had ever been opened. Pure: the clearing is the caller's effect. */
function readRecent(): { games: RecentGame[]; problem?: string } {
  const raw = localStorage.getItem(RECENT_KEY);
  if (raw == null) return { games: [] };
  let value: unknown;
  try { value = JSON.parse(raw); }
  catch (reason) { return { games: [], problem: `discarded an unreadable ${RECENT_KEY}: ${reasonText(reason)}` }; }
  const wellFormed = Array.isArray(value) && value.every((entry) =>
    entry != null && typeof entry === "object"
    && typeof (entry as RecentGame).run === "string"
    && Number.isInteger((entry as RecentGame).gameId)
    && Number.isInteger((entry as RecentGame).length));
  if (!wellFormed) return { games: [], problem: `discarded a ${RECENT_KEY} of another shape` };
  return { games: value as RecentGame[] };
}

/**
 * Parses a pasted move list. Three shapes are accepted — a JSON array of integer
 * pairs, parenthesised `(q,r)` pairs, and whitespace-separated `q,r` pairs — and
 * anything else is rejected by the index of the token that broke, because a paste
 * that silently drops half a line is worse than no paste at all.
 */
function parseLine(text: string): Move[] {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("nothing to apply: the field is empty");
  if (trimmed.startsWith("[")) {
    let value: unknown;
    try { value = JSON.parse(trimmed); }
    catch (reason) { throw new Error(`not valid JSON: ${reasonText(reason)}`); }
    if (!Array.isArray(value)) throw new Error("JSON must be an array of integer pairs");
    return value.map((entry, index) => {
      if (!Array.isArray(entry) || entry.length !== 2 || !entry.every((n) => Number.isInteger(n))) {
        throw new Error(`token ${index + 1} is not an integer pair: ${JSON.stringify(entry)}`);
      }
      return [entry[0], entry[1]] as Move;
    });
  }
  const parens = [...trimmed.matchAll(/\(([^()]*)\)/g)];
  let raw: string[];
  if (parens.length) {
    const leftover = trimmed.replace(/\([^()]*\)/g, "").trim();
    if (!/^[,;\s]*$/.test(leftover)) throw new Error(`unexpected text outside the pairs: "${leftover.slice(0, 32)}"`);
    raw = parens.map((match) => match[1]);
  } else {
    raw = trimmed.split(/[\s;]+/).filter(Boolean);
  }
  if (!raw.length) throw new Error("no move pairs found");
  return raw.map((token, index) => {
    const parts = token.split(",").map((part) => part.trim());
    const pair = parts.map(Number);
    if (parts.length !== 2 || parts.some((part) => !/^-?\d+$/.test(part)) || pair.some((n) => !Number.isInteger(n))) {
      throw new Error(`token ${index + 1} is not an integer pair: "${token.trim()}"`);
    }
    return [pair[0], pair[1]] as Move;
  });
}

const shortCheckpoint = (path: string) => path.split("/").slice(-2).join("/");

/* ---------------------------------------------------------- diagnostics: heat */

type Rgb = [number, number, number];
let rampCache: Rgb[] | null = null;

/** The sequential ramp, read back out of the stylesheet rather than restated here,
 *  so the canvas heatmap and the CSS `HeatLegend` beside it can never drift apart. */
function sequentialRamp(): Rgb[] {
  if (rampCache) return rampCache;
  const probe = document.createElement("div");
  probe.style.cssText = "position:absolute;left:-9999px;width:1px;height:1px";
  document.body.appendChild(probe);
  try {
    const steps: Rgb[] = [];
    for (let step = 0; step < 7; step++) {
      probe.setAttribute("data-heat", `s${step}`);
      const colour = getComputedStyle(probe).backgroundColor;
      const match = /rgba?\((\d+)[,\s]+(\d+)[,\s]+(\d+)/.exec(colour);
      if (!match) throw new Error(`heat ramp step s${step} is not defined in the stylesheet (got "${colour}")`);
      steps.push([Number(match[1]), Number(match[2]), Number(match[3])]);
    }
    rampCache = steps;
    return steps;
  } finally { probe.remove(); }
}

function AttentionMap({ data, block, head, stones }: {
  data: AttentionResult; block: number; head: number; stones: Move[];
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ q: number; k: number; weight: number } | null>(null);
  const layer = data.layers.find((entry) => entry.block === block);
  const matrix = layer?.heads[head] ?? null;
  const max = useMemo(() => {
    let out = 0;
    for (const row of matrix ?? []) for (const weight of row) if (weight > out) out = weight;
    return out;
  }, [matrix]);

  useEffect(() => {
    const node = canvas.current;
    if (!node || !matrix) return;
    const n = matrix.length;
    node.width = n; node.height = n;
    const context = node.getContext("2d");
    if (!context) throw new Error("attention: this browser gave no 2d canvas context");
    const ramp = sequentialRamp();
    const image = context.createImageData(n, n);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const step = sequentialStep(matrix[y][x], max);
        const [r, g, b] = step == null ? ramp[0] : ramp[step];
        const offset = (y * n + x) * 4;
        image.data[offset] = r; image.data[offset + 1] = g; image.data[offset + 2] = b; image.data[offset + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [matrix, max]);

  const label = (token: number) => token === 0
    ? "global"
    : `#${token} (${stones[token - 1] ? stones[token - 1].join(", ") : "?"})`;

  if (!matrix) return <Notice kind="warn">block {block} is not in this capture.</Notice>;
  return <>
    <canvas
      ref={canvas}
      className="attn-heatmap"
      role="img"
      aria-label={`attention weights, block ${block} head ${head}, ${matrix.length} tokens`}
      onPointerMove={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const n = matrix.length;
        const k = Math.min(n - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * n)));
        const q = Math.min(n - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * n)));
        setHover({ q, k, weight: matrix[q][k] });
      }}
      onPointerLeave={() => setHover(null)}
    />
    <div className="attn-axis"><span>rows: query token</span><span>columns: key token · token 0 is global, 1..n are stones in play order</span></div>
    <div className="attn-readout">{hover
      ? <><b>{label(hover.q)}</b><span>→</span><b>{label(hover.k)}</b><em>{format(hover.weight, 4)}</em></>
      : <span>hover the map for a query → key weight</span>}</div>
    <HeatLegend kind="seq" max={max} note={`max ${format(max, 4)} · four decades`} />
  </>;
}

/* ---------------------------------------------------------------- the screen */

/** Candidate rows the table prints before it cuts. A real mid-game position has
 *  450–550 legal moves; the whole set belongs on the board, not in a list. */
const CANDIDATE_ROWS = 48;

interface Capture<T> { state: "idle" | "loading" | "ready" | "error"; data?: T; error?: string; atPly: number; atCheckpoint: string; atLine: string }
const idleCapture = <T,>(): Capture<T> => ({ state: "idle", atPly: -1, atCheckpoint: "", atLine: "" });

export default function Lab({ runs, run, handoff }: { runs: Run[]; run?: Run; handoff: LabHandoff | null }) {
  /* ---- the line and where it came from ------------------------------------ */
  const [source, setSource] = useState<Source>({ kind: "blank" });
  const [mode, setMode] = useState<"blank" | "game" | "probe">("blank");
  const [line, setLine] = useState<Move[]>([]);
  const [cursor, setCursor] = useState(0);

  /* ---- checkpoints and the recipe ----------------------------------------- */
  const checkpoints = useMemo(
    () => runs.flatMap((entry) => entry.checkpoints.map((checkpoint) => ({ ...checkpoint, run: entry.name }))),
    [runs],
  );
  const [checkpoint, setCheckpoint] = useState("");
  const [compare, setCompare] = useState("");
  const [tauText, setTauText] = useState("0.1");
  const [lamText, setLamText] = useState("0.01");
  const [recipe, setRecipe] = useState<{ tau: number; lam: number } | null>(null);
  const [recipeError, setRecipeError] = useState<string>();

  /* ---- board and chart controls ------------------------------------------- */
  const [overlay, setOverlay] = useState<Overlay>("policy");
  const [topK, setTopK] = useState(6);
  const [qDomain, setQDomain] = useState<"auto" | "unit">("auto");
  const [valueView, setValueView] = useState<"p0" | "mover">("p0");
  const [chartGroup, setChartGroup] = useState<"value" | "policy" | "regret" | "entropy" | "kl" | "rank">("policy");
  const [selected, setSelected] = useState<Move | null>(null);

  /* ---- panels ------------------------------------------------------------- */
  const [paste, setPaste] = useState<{ open: boolean; text: string; error?: string }>({ open: false, text: "" });
  const [flash, setFlash] = useState<string | null>(null);
  const [gameId, setGameId] = useState("");
  const [gameError, setGameError] = useState<string>();
  const [gameLoading, setGameLoading] = useState(false);
  const [probeName, setProbeName] = useState("");
  const [probeError, setProbeError] = useState<string>();
  const [tab, setTab] = useState<"d6" | "attention" | "model" | null>(null);
  const [d6, setD6] = useState<Capture<D6Result>>(idleCapture<D6Result>());
  const [attention, setAttention] = useState<Capture<AttentionResult>>(idleCapture<AttentionResult>());
  const [attnBlock, setAttnBlock] = useState(0);
  const [attnHead, setAttnHead] = useState(0);
  const [recent, setRecent] = useState(readRecent);
  useEffect(() => { if (recent.problem) localStorage.removeItem(RECENT_KEY); }, [recent.problem]);

  const say = useCallback((message: string) => {
    setFlash(message);
    setTimeout(() => setFlash((current) => (current === message ? null : current)), 2600);
  }, []);

  /* ---- structural edits --------------------------------------------------- */
  // Every structural change goes through here: the inspect hook keys on the line's
  // contents, so a new line drops every reading and aborts a running walk by itself.
  // Dropping the readings is also what re-arms the auto-walk, which otherwise fires
  // once per game and checkpoint.
  const autoWalked = useRef("");
  const applyLine = useCallback((moves: Move[], nextCursor: number, nextSource: Source) => {
    setLine(moves);
    setCursor(Math.max(0, Math.min(moves.length, nextCursor)));
    setSource(nextSource);
    setSelected(null);
    autoWalked.current = "";
  }, []);

  /* ---- the inspect hooks -------------------------------------------------- */
  const primary = useLineInspect({ checkpoint, moves: line, cursor, recipe });
  const other = useLineInspect({ checkpoint: compare, moves: line, cursor, recipe, enabled: compare !== "" });

  useEffect(() => {
    if (checkpoint && checkpoints.some((entry) => entry.path === checkpoint)) return;
    const mine = run ? runs.find((entry) => entry.name === run.name)?.checkpoints ?? [] : [];
    const pick = mine.at(-1) ?? checkpoints.at(-1);
    if (pick) setCheckpoint(pick.path);
  }, [checkpoint, checkpoints, runs, run]);

  /* ---- hand-off from History ---------------------------------------------- */
  // One hand-off, one seed. The screen stays mounted for the life of the page, so
  // this fires on the token and never again — coming back from another screen
  // shows the line the user left here.
  const handoffToken = handoff?.token;
  useEffect(() => {
    if (!handoff) return;
    applyLine(handoff.moves, handoff.ply, {
      kind: "game", run: handoff.run, gameId: handoff.gameId, gameKind: handoff.kind,
      iteration: handoff.iteration, winner: handoff.winner, capped: handoff.capped,
      moves: handoff.moves, plies: handoff.plies,
    });
    setMode("game");
    setGameId(String(handoff.gameId));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handoffToken]);

  /* ---- candidates at the cursor ------------------------------------------- */
  const stoneKeys = useMemo(() => new Set(line.slice(0, cursor).map(moveKey)), [line, cursor]);
  // A reading is only usable while it still describes the line and ply on screen.
  // React renders before the hook's invalidation effect runs, so for one frame after
  // an edit the held reading belongs to the previous position; passing it to the
  // Board would put a legal cell on top of a stone, which the Board rejects.
  const freshLegal = useCallback((state: LineCursor): Candidate[] | null => {
    if (state.state !== "ready" || !state.legal || !state.read || state.read.terminal || state.read.t !== cursor) return null;
    for (const candidate of state.legal) if (stoneKeys.has(moveKey(candidate.move))) return null;
    return state.legal;
  }, [cursor, stoneKeys]);

  const baseLegal = freshLegal(primary.cursor);
  const otherLegal = freshLegal(other.cursor);
  const otherByMove = useMemo(() => {
    const map = new Map<string, Candidate>();
    otherLegal?.forEach((candidate) => map.set(moveKey(candidate.move), candidate));
    return map;
  }, [otherLegal]);

  // The Δ overlay is Δπ — policy is what the two checkpoints are actually being
  // compared on. The candidate table prints ΔQ beside it from the same pairing.
  const candidates = useMemo(() => {
    if (!baseLegal) return undefined;
    if (!otherByMove.size) return baseLegal;
    return baseLegal.map((candidate) => {
      const match = otherByMove.get(moveKey(candidate.move));
      return match ? { ...candidate, delta: candidate.policy - match.policy } : candidate;
    });
  }, [baseLegal, otherByMove]);

  const reading = primary.cursor.read && !primary.cursor.read.terminal ? primary.cursor.read : null;
  const terminal = primary.cursor.read?.terminal ? primary.cursor.read : null;
  // The hook reports an illegal line from either read — the cursor's own ply or
  // the walk — as a ply number, so this screen never re-parses a server message.
  const illegal = useMemo(() => {
    if (primary.cursor.illegalAtPly != null) return { ply: primary.cursor.illegalAtPly, message: primary.cursor.error! };
    if (primary.walk.illegalAtPly != null) return { ply: primary.walk.illegalAtPly, message: primary.walk.error! };
    return null;
  }, [primary.cursor.illegalAtPly, primary.cursor.error, primary.walk.illegalAtPly, primary.walk.error]);
  const hardError = primary.cursor.state === "error" && !illegal ? primary.cursor.error : undefined;

  const echoMismatch = recipe && reading && (reading.tau !== recipe.tau || reading.lam !== recipe.lam)
    ? `server used τ=${reading.tau} λ=${reading.lam}, not the requested τ=${recipe.tau} λ=${recipe.lam}`
    : null;

  /* ---- editing ------------------------------------------------------------ */
  // How the line on screen stands against the line its source arrived with. Derived
  // rather than recorded, so it survives any sequence of clicks, undos and second
  // branches — and "restore" always puts back exactly the line the label names.
  const sourceMoves = source.kind === "blank" ? null : source.moves;
  const sourceLabel = source.kind === "game" ? `game ${source.gameId}`
    : source.kind === "probe" ? `probe “${source.name}”`
      : source.kind === "paste" ? "the pasted line" : "";
  const branch = useMemo(() => {
    if (!sourceMoves || lineKey(sourceMoves) === lineKey(line)) return null;
    const at = commonPrefix(line, sourceMoves);
    return { moves: sourceMoves, at, discarded: sourceMoves.length - at };
  }, [line, sourceMoves]);

  const onCell = useCallback((move: Move) => {
    applyLine([...line.slice(0, cursor), move], cursor + 1, source);
  }, [applyLine, cursor, line, source]);

  const undo = useCallback(() => {
    if (!line.length) { say("nothing to undo"); return; }
    applyLine(line.slice(0, -1), Math.min(cursor, line.length - 1), source);
  }, [applyLine, cursor, line, say, source]);

  const clear = useCallback(() => {
    applyLine([], 0, { kind: "blank" });
    setMode("blank");
  }, [applyLine]);

  const restoreSource = useCallback(() => {
    if (!branch) { say("this is the source line"); return; }
    applyLine(branch.moves, Math.min(cursor, branch.moves.length), source);
  }, [applyLine, branch, cursor, say, source]);

  const copyLine = useCallback(() => {
    const text = JSON.stringify(line);
    navigator.clipboard?.writeText(text).then(
      () => say(`copied · ${line.length} plies`),
      (reason: unknown) => say(`clipboard refused: ${reasonText(reason)}`),
    );
  }, [line, say]);

  const applyPaste = useCallback(() => {
    try {
      const moves = parseLine(paste.text);
      applyLine(moves, moves.length, { kind: "paste", moves });
      setPaste({ open: false, text: "" });
      say(`applied · ${moves.length} plies`);
    } catch (reason) { setPaste((current) => ({ ...current, error: reasonText(reason) })); }
  }, [applyLine, paste.text, say]);

  /* ---- games -------------------------------------------------------------- */
  const rememberGame = useCallback((entry: RecentGame) => {
    setRecent((current) => {
      const games = [entry, ...current.games.filter((item) => !(item.run === entry.run && item.gameId === entry.gameId))].slice(0, 8);
      localStorage.setItem(RECENT_KEY, JSON.stringify(games));
      return { games };
    });
  }, []);

  // Two clicks in the listing race, and the payloads differ by two orders of
  // magnitude, so the second-asked game can easily answer first: the same abort
  // plus generation snapshot `useApi` uses keeps the last request asked the one
  // that lands.
  const loadAbort = useRef<AbortController | undefined>(undefined);
  const loadGeneration = useRef(0);
  const loadGame = useCallback(async (runName: string, id: number) => {
    const snapshot = ++loadGeneration.current;
    loadAbort.current?.abort();
    const controller = new AbortController();
    loadAbort.current = controller;
    setGameError(undefined);
    setGameLoading(true);
    try {
      const game = await api<Game>(`/api/runs/${runName}/games/${id}`, { signal: controller.signal });
      if (loadGeneration.current !== snapshot) return;
      applyLine(game.moves, game.moves.length, {
        kind: "game", run: runName, gameId: game.game_id, gameKind: game.kind, iteration: game.iteration,
        winner: game.winner, capped: !!game.capped, moves: game.moves, plies: game.plies,
      });
      setGameId(String(game.game_id));
      rememberGame({ run: runName, gameId: game.game_id, length: game.length });
    } catch (reason) {
      if (isAbortError(reason) || loadGeneration.current !== snapshot) return;
      setGameError(reasonText(reason));
    } finally {
      if (loadGeneration.current === snapshot) setGameLoading(false);
    }
  }, [applyLine, rememberGame]);

  const [browse, setBrowse] = useState(() => ({ run: run?.name ?? "", kind: "selfplay", from: "" }));
  useEffect(() => { setBrowse((current) => (current.run ? current : { ...current, run: run?.name ?? "" })); }, [run?.name]);
  const listingPath = browse.run
    ? `/api/runs/${browse.run}/games?${query({ kind: browse.kind, from_iteration: browse.from, limit: 25 })}`
    : null;
  const listing = useApi<GameRow[]>(listingPath, [listingPath], { manual: true });

  /* ---- probes ------------------------------------------------------------- */
  const probes = useApi<Probe[]>("/api/probes", []);
  const saveProbe = useCallback(async () => {
    setProbeError(undefined);
    if (!checkpoint) { setProbeError("select a checkpoint first"); return; }
    try {
      await post<Probe>("/api/probes", {
        name: probeName.trim() || `${source.kind === "game" ? `game ${source.gameId}` : "line"} · ${line.length} plies`,
        checkpoint, moves: line, module: tab ?? "position",
      });
      setProbeName("");
      void probes.refresh();
      say("probe saved");
    } catch (reason) { setProbeError(reasonText(reason)); }
  }, [checkpoint, line, probeName, probes, say, source, tab]);

  const deleteProbe = useCallback(async (id: number) => {
    setProbeError(undefined);
    try { await api(`/api/probes/${id}`, { method: "DELETE" }); void probes.refresh(); }
    catch (reason) { setProbeError(reasonText(reason)); }
  }, [probes]);

  /* ---- diagnostics -------------------------------------------------------- */
  const stamp = useMemo(() => lineKey(line.slice(0, cursor)), [line, cursor]);
  const runD6 = useCallback(async () => {
    setD6({ state: "loading", atPly: cursor, atCheckpoint: checkpoint, atLine: stamp });
    try {
      const data = await post<D6Result>("/api/inspect/d6", { checkpoint, moves: line.slice(0, cursor) });
      setD6({ state: "ready", data, atPly: cursor, atCheckpoint: checkpoint, atLine: stamp });
    } catch (reason) {
      setD6({ state: "error", error: reasonText(reason), atPly: cursor, atCheckpoint: checkpoint, atLine: stamp });
    }
  }, [checkpoint, cursor, line, stamp]);

  const runAttention = useCallback(async () => {
    setAttention({ state: "loading", atPly: cursor, atCheckpoint: checkpoint, atLine: stamp });
    try {
      const data = await post<AttentionResult>("/api/inspect/attention", { checkpoint, moves: line.slice(0, cursor) });
      setAttention({ state: "ready", data, atPly: cursor, atCheckpoint: checkpoint, atLine: stamp });
    } catch (reason) {
      setAttention({ state: "error", error: reasonText(reason), atPly: cursor, atCheckpoint: checkpoint, atLine: stamp });
    }
  }, [checkpoint, cursor, line, stamp]);

  const manifest = useApi<ModelManifest>("/api/model", [], { manual: true });
  const { requested: manifestRequested, refresh: refreshManifest } = manifest;
  useEffect(() => { if (tab === "model" && !manifestRequested) void refreshManifest(); }, [tab, manifestRequested, refreshManifest]);
  const isStale = (capture: Capture<unknown>) => capture.state === "ready"
    && (capture.atPly !== cursor || capture.atCheckpoint !== checkpoint || capture.atLine !== stamp);

  /* ---- the whole-line walk ------------------------------------------------ */
  const walkRunning = primary.walk.state === "running";
  const startWalk = useRef(primary.runWalk);
  startWalk.current = primary.runWalk;
  // Loading a game is a request to read the whole game, and the cost is bounded and
  // known: 0.5 s for a 17-ply game, 5.6 s for 127 plies at two in flight. Only the
  // game's own line walks itself. Once it has been edited the walk is explicit,
  // because an edit drops every reading — so a walk per click would re-read the
  // whole line from scratch each time, and only the last one would ever be shown.
  useEffect(() => {
    if (source.kind !== "game" || branch || !checkpoint || line.length > 300) return;
    const identity = `${checkpoint}|${lineKey(line)}`;
    if (autoWalked.current === identity) return;
    autoWalked.current = identity;
    startWalk.current();
  }, [source.kind, branch, checkpoint, line]);

  useDeckKeys({
    w: () => (walkRunning ? primary.cancelWalk() : primary.runWalk()),
    u: undo,
    m: restoreSource,
    Escape: () => {
      if (walkRunning) { primary.cancelWalk(); return; }
      if (paste.open) { setPaste({ open: false, text: "" }); return; }
      setSelected(null);
    },
    "1": () => setOverlay("policy"),
    "2": () => setOverlay("q"),
    "3": () => setOverlay("improved"),
    "4": () => (compare ? setOverlay("delta") : say("select a compare checkpoint first")),
    "[": () => setTopK((current) => TOP_K_STEPS[Math.max(0, TOP_K_STEPS.indexOf(current) - 1)]),
    "]": () => setTopK((current) => TOP_K_STEPS[Math.min(TOP_K_STEPS.length - 1, TOP_K_STEPS.indexOf(current) + 1)]),
  });

  /* ---- series ------------------------------------------------------------- */
  const byPly = useMemo(() => {
    const map = new Map<number, Reading>();
    primary.plies.forEach((entry) => map.set(entry.t, entry));
    return map;
  }, [primary.plies]);
  const otherByPly = useMemo(() => {
    const map = new Map<number, Reading>();
    other.plies.forEach((entry) => map.set(entry.t, entry));
    return map;
  }, [other.plies]);
  // The stored read at ply t is the opinion the acting net formed to choose move t,
  // so it describes this line only while this line still plays the game's moves
  // through t. Past where the two part it belongs to a move that is no longer here.
  const actingByPly = useMemo(() => {
    const map = new Map<number, StoredPly>();
    if (source.kind !== "game") return map;
    const shared = commonPrefix(line, source.moves);
    source.plies.forEach((entry) => { if (entry.t < shared) map.set(entry.t, entry); });
    return map;
  }, [source, line]);

  const xs = useMemo(() => Array.from({ length: line.length + 1 }, (_, index) => index), [line.length]);
  const sign = useCallback((mover: number) => (valueView === "p0" && mover === 1 ? -1 : 1), [valueView]);
  const now = useCallback((read: (entry: Reading) => number | null) =>
    xs.map((x) => [x, byPly.has(x) ? read(byPly.get(x)!) : null] as [number, number | null]), [xs, byPly]);
  const versus = useCallback((read: (entry: Reading) => number | null) =>
    xs.map((x) => [x, otherByPly.has(x) ? read(otherByPly.get(x)!) : null] as [number, number | null]), [xs, otherByPly]);
  const acted = useCallback((read: (entry: StoredPly) => number | null) =>
    xs.map((x) => [x, actingByPly.has(x) ? read(actingByPly.get(x)!) : null] as [number, number | null]), [xs, actingByPly]);

  const actingAbsent = source.kind !== "game"
    ? "only a loaded game carries the acting net's stored trace"
    : source.gameKind === "eval"
      ? "eval games store no per-ply trace — run the walk for this checkpoint's opinion"
      : undefined;

  const chart = useMemo((): { series: Series[]; yLabel: string; yDomain?: [number, number]; yScale?: "linear" | "log" } => {
    const absent = actingAbsent != null;
    switch (chartGroup) {
      case "value": return {
        yLabel: valueView === "p0" ? "v̂ · P0 perspective" : "v̂ · mover perspective",
        yDomain: [-1, 1],
        series: [
          { id: "v-now", label: "v̂ now", tone: "mint", dash: "solid", points: now((r) => r.vHat * sign(r.mover)) },
          { id: "v-act", label: "v̂ acting", tone: "blue", dash: "dashed", absent, absentReason: actingAbsent, points: acted((p) => p.v_hat * sign(p.mover)) },
          { id: "v-cmp", label: "v̂ compare", tone: "amber", dash: "dotted", absent: !compare, absentReason: "no compare checkpoint", points: versus((r) => r.vHat * sign(r.mover)) },
        ],
      };
      case "policy": return {
        yLabel: "π", yDomain: [0, 1],
        series: [
          { id: "p-played", label: "π played now", tone: "mint", dash: "solid", points: now((r) => r.piPlayed) },
          { id: "p-played-act", label: "π played acting", tone: "mint", dash: "dashed", absent, absentReason: actingAbsent, points: acted((p) => p.pi_chosen) },
          { id: "p-top", label: "π top now", tone: "blue", dash: "solid", points: now((r) => r.piTop) },
          { id: "p-top-act", label: "π top acting", tone: "blue", dash: "dashed", absent, absentReason: actingAbsent, points: acted((p) => p.pi_top1) },
        ],
      };
      case "regret": {
        const series: Series[] = [
          { id: "regret", label: "max Q − Q played", tone: "red", dash: "solid", points: now((r) => r.regret) },
          { id: "q-played", label: "Q played", tone: "blue", dash: "solid", points: now((r) => r.qPlayed) },
          { id: "q-max", label: "max Q", tone: "mint", dash: "solid", points: now((r) => r.qMax) },
        ];
        let lo = 0, hi = 0;
        for (const item of series) for (const [, value] of item.points) if (value != null) { lo = Math.min(lo, value); hi = Math.max(hi, value); }
        const pad = (hi - lo) * 0.08 || 0.1;
        return { yLabel: "Q", yDomain: [lo - pad, hi + pad], series };
      }
      case "entropy": return {
        yLabel: "H / log|A|", yDomain: [0, 1],
        series: [
          { id: "h-now", label: "H now", tone: "mint", dash: "solid", points: now((r) => r.normEntropy) },
          { id: "h-act", label: "H acting", tone: "mint", dash: "dashed", absent, absentReason: actingAbsent, points: acted((p) => p.norm_entropy) },
        ],
      };
      case "kl": return {
        yLabel: "KL(π′ ‖ π)",
        series: [
          { id: "kl-now", label: "KL now", tone: "mint", dash: "solid", points: now((r) => r.kl) },
          { id: "kl-act", label: "KL acting", tone: "mint", dash: "dashed", absent, absentReason: actingAbsent, points: acted((p) => p.kl) },
        ],
      };
      case "rank": default: return {
        yLabel: "policy rank of the played move", yScale: "log",
        series: [
          { id: "rank", label: "rank played", tone: "mint", dash: "solid", points: now((r) => r.policyRank) },
          { id: "legal", label: "legal moves", tone: "blue", dash: "solid", points: now((r) => r.legalCount) },
        ],
      };
    }
  }, [acted, actingAbsent, chartGroup, compare, now, sign, valueView, versus]);

  const bands = useMemo(() => p1TurnBands(line.length), [line.length]);

  /* ---- ply list scrolling ------------------------------------------------- */
  const lineList = useRef<HTMLDivElement>(null);
  // Keeps the cursor's row in view within the ply list, by moving that list's own
  // scrollTop and nothing else. `scrollIntoView` scrolls every scrollable
  // ancestor including the document, so on a narrow layout — where the rail sits
  // below the board — each ply change threw the page down to the list.
  useEffect(() => {
    const list = lineList.current;
    const row = list?.querySelector("[data-current]");
    if (!list || !row) return;
    const listBox = list.getBoundingClientRect();
    const rowBox = row.getBoundingClientRect();
    if (rowBox.top < listBox.top) list.scrollTop -= listBox.top - rowBox.top;
    else if (rowBox.bottom > listBox.bottom) list.scrollTop += rowBox.bottom - listBox.bottom;
  }, [cursor]);

  if (!checkpoints.length) {
    return <Empty>No checkpoints on disk. The lab reads a checkpoint for every position; launch or resume a run first.</Empty>;
  }

  const played = cursor < line.length ? line[cursor] : null;
  const agrees = played != null && reading != null && moveKey(played) === moveKey(reading.topMove);
  const walkCoverage = primary.readings.size;

  // The table is cut to a readable length, and the cut is printed. The played move
  // is pinned in even when it ranks past the cut — a move the model buried is
  // exactly what this screen exists to show.
  const ranked = candidates ? candidates.slice().sort((a, b) => a.policyRank - b.policyRank) : [];
  const listedCandidates = ranked.slice(0, CANDIDATE_ROWS);
  const playedRow = played && ranked.find((row) => moveKey(row.move) === moveKey(played));
  if (playedRow && !listedCandidates.includes(playedRow)) listedCandidates.push(playedRow);

  const checkpointOptions = (skip: string) => runs
    .filter((entry) => entry.checkpoints.length)
    .map((entry) => <optgroup key={entry.name} label={entry.name}>
      {entry.checkpoints.filter((item) => item.path !== skip).map((item) =>
        <option key={item.path} value={item.path}>{item.name}{item.iteration != null ? ` · iter ${item.iteration}` : ""}</option>)}
    </optgroup>);

  return <div className="explorer-page">
    {/* ------------------------------------------------------------- left rail */}
    <div className="explorer-rail">
      <Panel title="Source">
        <Segmented
          label="line source"
          value={mode}
          onChange={(next) => { setMode(next); if (next === "blank" && line.length) say("press clear to empty the board"); }}
          options={[{ value: "blank", label: "blank" }, { value: "game", label: "game" }, { value: "probe", label: "probe" }]}
        />

        {mode === "blank" && <div className="explorer-hint">
          Click legal cells to build a line. P0 opens at the origin; every later cell must be empty and within 8 hex steps of a stone.
        </div>}

        {mode === "game" && <div className="game-source">
          <label>Run<select value={browse.run} onChange={(event) => setBrowse({ ...browse, run: event.target.value })}>
            {!runs.length && <option value="">no runs</option>}
            {runs.map((entry) => <option key={entry.name} value={entry.name}>{entry.name}</option>)}
          </select></label>
          <div className="game-id-row">
            <input
              inputMode="numeric" placeholder="game id" value={gameId}
              onChange={(event) => setGameId(event.target.value.replace(/[^0-9]/g, ""))}
              onKeyDown={(event) => { if (event.key === "Enter" && gameId) void loadGame(browse.run, Number(gameId)); }}
            />
            <button type="button" disabled={!gameId || !browse.run || gameLoading} onClick={() => void loadGame(browse.run, Number(gameId))}>
              {gameLoading ? "loading…" : "Load"}
            </button>
          </div>
          {gameError && <Notice kind="warn">{gameError}</Notice>}

          <div className="explorer-sublabel">browse the listing</div>
          <div className="game-filter-row">
            <select value={browse.kind} onChange={(event) => setBrowse({ ...browse, kind: event.target.value })}>
              <option value="selfplay">selfplay</option><option value="eval">eval</option><option value="">any kind</option>
            </select>
            <input inputMode="numeric" placeholder="from iter" value={browse.from} onChange={(event) => setBrowse({ ...browse, from: event.target.value.replace(/[^0-9]/g, "") })} />
            <button type="button" disabled={!browse.run || listing.loading} onClick={() => void listing.refresh()}><Search size={11} /></button>
          </div>
          {listing.loading && <Notice kind="info">the listing sorts the whole games table — measured 40–46 s on a live run. Loading by id is 18 ms; everything else on this screen stays usable.</Notice>}
          {!listing.requested && <div className="explorer-hint">Not requested. The listing is slow on a live run, so it never fires on its own.</div>}
          <ErrorBox message={listing.error} />
          {listing.data && <div className="data-list">
            {listing.data.length
              ? listing.data.map((game) => <button key={game.game_id} type="button" disabled={gameLoading} onClick={() => void loadGame(browse.run, game.game_id)}>
                <span>#{game.game_id} · iter {game.iteration}</span>
                <b>{game.capped ? "cap" : `P${game.winner}`} · {game.length}p</b>
              </button>)
              : <Empty>no games matched</Empty>}
          </div>}

          {recent.problem && <Notice kind="warn">{recent.problem}</Notice>}
          {recent.games.length > 0 && <>
            <div className="explorer-sublabel">recently opened</div>
            <div className="data-list">{recent.games.map((entry) => <button key={`${entry.run}/${entry.gameId}`} type="button" disabled={gameLoading} onClick={() => void loadGame(entry.run, entry.gameId)}>
              <span>#{entry.gameId}</span><b>{entry.run} · {entry.length}p</b>
            </button>)}</div>
          </>}
        </div>}

        {mode === "probe" && <div className="probe-rows">
          <div className="game-id-row">
            <input placeholder="probe name" value={probeName} onChange={(event) => setProbeName(event.target.value)} />
            <button type="button" title="stores this checkpoint and the whole line; loading restores both with the cursor at the end" disabled={!line.length} onClick={() => void saveProbe()}><Save size={11} /></button>
          </div>
          {probeError && <Notice kind="warn">{probeError}</Notice>}
          <ErrorBox message={probes.error} />
          {probes.data?.length
            ? probes.data.map((probe) => <div key={probe.probe_id} className="probe-row">
              <button type="button" className="probe-open" onClick={() => {
                setCheckpoint(probe.checkpoint);
                applyLine(probe.moves, probe.moves.length, { kind: "probe", probeId: probe.probe_id, name: probe.name, moves: probe.moves });
              }}>
                <b>{probe.name}</b>
                <span>{probe.moves.length}p · {shortCheckpoint(probe.checkpoint)}</span>
              </button>
              <button type="button" className="probe-delete" aria-label={`delete ${probe.name}`} onClick={() => void deleteProbe(probe.probe_id)}><Trash2 size={11} /></button>
            </div>)
            : <Empty>No saved probes. Save the current line to return to it later.</Empty>}
        </div>}

        <label className="ckpt-select">Checkpoint
          <select value={checkpoint} data-invalid={hardError ? "" : undefined} onChange={(event) => setCheckpoint(event.target.value)}>
            {checkpointOptions("")}
          </select>
        </label>
        <label className="ckpt-select">Compare against
          <select value={compare} onChange={(event) => { setCompare(event.target.value); if (!event.target.value && overlay === "delta") setOverlay("policy"); }}>
            <option value="">None</option>{checkpointOptions(checkpoint)}
          </select>
        </label>

        <div className="explorer-sublabel">π′ recipe</div>
        <div className="recipe-row">
          <label>τ<input value={tauText} onChange={(event) => setTauText(event.target.value)} /></label>
          <label>λ<input value={lamText} onChange={(event) => setLamText(event.target.value)} /></label>
          <button type="button" onClick={() => {
            const tau = Number(tauText), lam = Number(lamText);
            if (!Number.isFinite(tau) || !Number.isFinite(lam)) { setRecipeError(`τ and λ must be numbers: got "${tauText}" and "${lamText}"`); return; }
            setRecipeError(undefined);
            // Both always travel: the server silently ignores a lone tau or lam.
            setRecipe({ tau, lam });
          }}>Apply</button>
          <button type="button" disabled={!recipe} onClick={() => { setRecipe(null); setTauText("0.1"); setLamText("0.01"); setRecipeError(undefined); }}>Reset</button>
        </div>
        {recipeError && <Notice kind="warn">{recipeError}</Notice>}
        {echoMismatch && <Notice kind="warn">{echoMismatch}</Notice>}
        {flash && <Notice kind="info">{flash}</Notice>}
      </Panel>

      <Panel title="Line" action={<span>{line.length} plies</span>}>
        <div className="line-actions">
          <button type="button" disabled={!line.length} title="undo (u)" onClick={undo}><Undo2 size={11} /></button>
          <button type="button" disabled={!line.length} title="clear" onClick={clear}><Trash2 size={11} /></button>
          <button type="button" disabled={!line.length} title="copy the move list" onClick={copyLine}><Copy size={11} /></button>
          <button type="button" title="paste a move list" aria-pressed={paste.open} onClick={() => setPaste({ open: !paste.open, text: "" })}><ClipboardPaste size={11} /></button>
          {branch && <button type="button" className="line-return" title={`restore ${sourceLabel} (m)`} onClick={restoreSource}><CornerUpLeft size={11} /> source</button>}
        </div>

        {paste.open && <div className="paste-field">
          <input
            autoFocus placeholder='[[0,0],[1,0]] or (0,0) (1,0) or 0,0 1,0'
            value={paste.text}
            onChange={(event) => setPaste({ open: true, text: event.target.value })}
            onKeyDown={(event) => { if (event.key === "Enter") applyPaste(); if (event.key === "Escape") setPaste({ open: false, text: "" }); }}
          />
          <button type="button" onClick={applyPaste}>Apply</button>
          <button type="button" onClick={() => setPaste({ open: false, text: "" })}><X size={11} /></button>
          {paste.error && <div className="paste-error">{paste.error}</div>}
        </div>}

        {branch && <Notice kind="info" action={<button type="button" onClick={restoreSource}>Restore the source line</button>}>
          {branch.discarded > 0
            ? `Branched at ply ${branch.at} of ${sourceLabel} — ${plies(branch.discarded)} discarded`
            : `Extended ${sourceLabel} by ${plies(line.length - branch.at)}`}
        </Notice>}

        {illegal && <Notice kind="warn" action={<button type="button" onClick={() => applyLine(line.slice(0, illegal.ply), Math.min(cursor, illegal.ply), source)}>Truncate to ply {illegal.ply}</button>}>
          {illegal.message}
        </Notice>}

        {source.kind === "game" && source.gameKind === "eval" && <Notice kind="info">
          eval games store no per-ply trace — the dashed “acting” series are unavailable; run the walk for this checkpoint's opinion.
        </Notice>}

        {source.kind === "game" && run && source.run !== run.name && <Notice kind="info">
          game {source.gameId} is from run {source.run}, not {run.name}
        </Notice>}

        <div className="line-list" ref={lineList}>
          {line.length === 0
            ? <Empty>Empty line. Click the board to place P0's opening stone.</Empty>
            : line.map((move, index) => {
              const entry = byPly.get(index);
              const unreachable = illegal != null && index > illegal.ply;
              return <button
                key={index}
                type="button"
                className="line-row"
                data-current={cursor === index + 1 ? "" : undefined}
                data-illegal={illegal?.ply === index ? "" : undefined}
                data-branch={branch && index === branch.at ? "" : undefined}
                data-unreachable={unreachable ? "" : undefined}
                onClick={() => setCursor(index + 1)}
              >
                <span className="line-index">{index + 1}</span>
                <i className="line-dot" data-player={playerAt(index)} />
                <span className="line-move">({move.join(", ")})</span>
                <span className="line-pi">{entry?.piPlayed != null ? format(entry.piPlayed) : ""}</span>
                {entry?.policyRank != null && <span
                  className="line-badge"
                  data-regret={entry.regret == null ? undefined : entry.regret > 0.25 ? "high" : entry.regret > 0.05 ? "mid" : "low"}
                >#{entry.policyRank}</span>}
              </button>;
            })}
        </div>
      </Panel>
    </div>

    {/* ---------------------------------------------------------------- centre */}
    <div className="explorer-stage">
      <div className="explorer-context">
        <span><b>source</b>{source.kind === "game"
          ? <em>game {source.gameId} · {source.gameKind} · iter {source.iteration} · {source.capped ? "capped" : `P${source.winner} won`}</em>
          : source.kind === "probe" ? <em>probe · {source.name}</em>
            : source.kind === "paste" ? <em>pasted line</em> : <em>blank board</em>}</span>
        <span><b>checkpoint</b><em>{checkpoint ? shortCheckpoint(checkpoint) : "none"}</em></span>
        {compare && <span><b>vs</b><em>{shortCheckpoint(compare)}</em></span>}
        <span><b>ply</b><em>{cursor} / {line.length}</em></span>
        <span><b>mover</b><em>{reading ? `P${reading.mover}` : terminal ? "—" : `P${playerAt(cursor)}`}</em></span>
        <span><b>legal</b><em>{reading ? reading.legalCount : "—"}</em></span>
        <span><b>walked</b><em>{walkCoverage} / {line.length + 1}</em></span>
      </div>

      <Panel title="Position" action={<Segmented
        label="Q domain"
        value={qDomain}
        onChange={setQDomain}
        options={[{ value: "auto", label: "Q auto" }, { value: "unit", label: "Q ±1" }]}
      />}>
        <ErrorBox message={hardError} />
        {terminal && <Notice kind="info">
          {terminal.message}
          {cursor >= 6 ? ` · P${playerAt(cursor - 1)} wins on the highlighted six.` : ""}
        </Notice>}
        <Board
          moves={line}
          cursor={cursor}
          legal={candidates}
          played={played}
          overlay={overlay}
          onOverlayChange={setOverlay}
          topK={topK}
          onTopKChange={setTopK}
          qDomain={qDomain}
          interactive
          onSelect={onCell}
          onStone={(ply) => setCursor(ply + 1)}
          selected={selected}
          toolbar
          status={primary.cursor.state === "loading" ? "pending" : "idle"}
          height={500}
          caption={line.length === 0
            ? "P0 opens at the origin — click it to start a line."
            : `ply ${cursor} of ${line.length}`}
        />
        {line.length === 0 && <div className="explorer-hint">P0 opens at the origin — click it to start a line.</div>}
        <Transport length={line.length} value={cursor} onChange={setCursor} />
        <div className="kbd-hint">click a legal cell to play it · click before the end to branch · click a stone to jump there · w walk · u undo · m restore the source line · 1-4 overlay · [ ] labels</div>
      </Panel>

      <Panel
        title="Policy across the line"
        action={<div className="explorer-walk">
          {walkRunning
            ? <button type="button" onClick={primary.cancelWalk}>Cancel</button>
            : <button type="button" disabled={!checkpoint} onClick={primary.runWalk}>
              {primary.walk.state === "cancelled" || (primary.walk.state === "done" && walkCoverage <= line.length) ? "Resume walk" : "Run walk"}
            </button>}
          {compare && <button type="button" disabled={other.walk.state === "running"} onClick={other.runWalk}>
            {other.walk.state === "running" ? "compare…" : "Walk compare"}
          </button>}
        </div>}
      >
        <div className="explorer-chart-controls">
          <Segmented
            label="series group"
            value={chartGroup}
            onChange={setChartGroup}
            options={[
              { value: "policy", label: "π" }, { value: "value", label: "v̂" }, { value: "regret", label: "regret" },
              { value: "entropy", label: "H" }, { value: "kl", label: "KL" }, { value: "rank", label: "rank" },
            ]}
          />
          {chartGroup === "value" && <Segmented
            label="value perspective"
            value={valueView}
            onChange={setValueView}
            options={[{ value: "p0", label: "P0 view" }, { value: "mover", label: "mover view" }]}
          />}
        </div>

        {walkRunning && <ProgressBar done={primary.walk.done} total={primary.walk.total} label="walking" />}
        {primary.walk.state === "cancelled" && <Notice kind="info" action={<button type="button" onClick={primary.runWalk}>Resume</button>}>
          partial · {primary.walk.done} / {primary.walk.total}
        </Notice>}
        {primary.walk.state === "failed" && <Notice kind="warn">
          walk stopped{primary.walk.illegalAtPly != null ? ` at ply ${primary.walk.illegalAtPly}` : ""}: {primary.walk.error}
        </Notice>}

        <Chart
          series={chart.series}
          xLabel="ply"
          yLabel={chart.yLabel}
          xDomain={[0, Math.max(1, line.length)]}
          yDomain={chart.yDomain}
          yScale={chart.yScale}
          cursor={cursor}
          onSelect={(x) => setCursor(Math.max(0, Math.min(line.length, Math.round(x))))}
          bands={bands}
          markers={primary.terminalPly != null ? [{ x: primary.terminalPly, label: "terminal" }] : []}
          progress={walkRunning ? { done: primary.walk.done, total: primary.walk.total } : null}
          height={230}
          empty="Run the walk to read this checkpoint's opinion at every ply — 1.5–3 s for a typical game. A game loaded from history walks itself."
        />
        <div className="kbd-hint">click the plot to move the cursor · solid = this checkpoint now · dashed = the acting net while the game was played · dotted = the compare checkpoint</div>
      </Panel>
    </div>

    {/* ------------------------------------------------------------ right rail */}
    <div className="explorer-read">
      <Panel title="Position read">
        {reading ? <>
          <div className="metric-row wrap">
            <Metric label="v̂" value={format(reading.vHat)} />
            <Metric label="KL" value={format(reading.kl)} />
            <Metric label="H/log|A|" value={format(reading.normEntropy)} />
            <Metric label="legal" value={reading.legalCount} />
          </div>
          {played
            ? agrees
              ? <Notice kind="info">the line plays the model's top move · π {format(reading.piPlayed)}</Notice>
              : <div className="explorer-verdict">
                <div><b>played</b><em>({played.join(", ")})</em><span>π {format(reading.piPlayed)} · rank #{reading.policyRank} · Q {format(reading.qPlayed)}</span></div>
                <div><b>model top</b><em>({reading.topMove.join(", ")})</em><span>π {format(reading.piTop)} · maxQ {format(reading.qMax)}</span></div>
                <div><b>regret</b><em>{format(reading.regret)}</em><span>Q given up here</span></div>
              </div>
            : <div className="explorer-hint">End of the line — nothing is played from this ply. The model's top move is ringed on the board.</div>}
          <div className="explorer-hint">τ {reading.tau} · λ {reading.lam} · {reading.movesRemaining} stone{reading.movesRemaining === 1 ? "" : "s"} left in this turn</div>
        </> : terminal
          ? <Empty>terminal position — no legal moves</Empty>
          : <Empty>{primary.cursor.state === "loading" ? "reading…" : "select a checkpoint"}</Empty>}
      </Panel>

      <Panel title="Candidates" action={<span>
        {ranked.length > listedCandidates.length ? `${listedCandidates.length} of ${ranked.length}` : ranked.length}
        {compare ? " · Δ vs compare" : ""}
      </span>}>
        {listedCandidates.length
          ? <div
            className="candidate-table"
            style={{ ["--cand-cols" as string]: "1fr 50px 50px 50px 26px" }}
            data-pending={primary.cursor.state === "loading" ? "" : undefined}
          >
            <div className="candidate-head">
              <b>move</b>{compare
                ? <><span>Δπ</span><span>ΔQ</span><span>π</span></>
                : <><span>π</span><span>Q</span><span>π′</span></>}<em>#</em>
            </div>
            {listedCandidates.map((row) => {
              const isSelected = selected != null && moveKey(selected) === moveKey(row.move);
              const versus = compare ? otherByMove.get(moveKey(row.move)) : undefined;
              return <button
                key={moveKey(row.move)}
                type="button"
                aria-pressed={isSelected}
                data-played={played && moveKey(played) === moveKey(row.move) ? "" : undefined}
                onClick={() => setSelected(isSelected ? null : row.move)}
              >
                <b>({row.move.join(", ")})</b>
                {compare
                  ? <>
                    <span>{row.delta == null ? "—" : format(row.delta)}</span>
                    <span>{versus ? format(row.q - versus.q) : "—"}</span>
                    <span>{format(row.policy)}</span>
                  </>
                  : <>
                    <span>{format(row.policy)}</span>
                    <span>{format(row.q)}</span>
                    <span>{format(row.improved)}</span>
                  </>}
                <em>#{row.policyRank + 1}</em>
              </button>;
            })}
          </div>
          : <Empty>{terminal ? "no legal moves" : primary.cursor.state === "loading" ? "reading…" : "no candidates"}</Empty>}
        {compare && !otherByMove.size && <Notice kind="info">compare checkpoint has not answered for this ply yet</Notice>}
      </Panel>
    </div>

    {/* -------------------------------------------------------------- bottom -- */}
    <div className="explorer-diag">
      <Panel title="Diagnostics" action={<Segmented
        label="diagnostic"
        value={tab ?? ""}
        onChange={(next) => setTab(next === "" ? null : next as "d6" | "attention" | "model")}
        options={[{ value: "", label: "off" }, { value: "d6", label: "D6" }, { value: "attention", label: "attention" }, { value: "model", label: "model" }]}
      />}>
        {tab === null && <div className="explorer-hint">D6 invariance, the SDPA attention capture and the model manifest. Nothing fires on its own — attention at 120 stones is a 5 MB response.</div>}

        {tab === "d6" && <>
          <div className="attn-controls">
            <button type="button" disabled={!checkpoint || d6.state === "loading"} onClick={() => void runD6()}>
              {d6.state === "loading" ? "running…" : `Run at ply ${cursor}`}
            </button>
            {isStale(d6) && <span className="diag-stale">STALE · captured at ply {d6.atPly} · recapture</span>}
          </div>
          <ErrorBox message={d6.error} />
          {d6.data ? (() => {
            const worst = Math.max(d6.data.policy_max, d6.data.q_max);
            const verdict = worst < 1e-4 ? "INVARIANT" : worst < 1e-3 ? "DRIFT" : "BROKEN";
            const bar = (value: number) => Math.max(0, Math.min(1, (Math.log10(Math.max(value, 1e-12)) + 8) / 4));
            return <div className="d6-list" data-stale={isStale(d6) ? "" : undefined}>
              <div className="d6-verdict" data-verdict={verdict}>
                <b>{verdict}</b>
                <span>worst deviation {format(worst, 4)} against a 1e-4 tolerance · 12 transforms of the D6 symmetry group</span>
              </div>
              {d6.data.transforms.map((entry) => <div key={entry.transform} className="d6-row">
                <span className="d6-index">t{entry.transform}</span>
                <span className="d6-value">π {format(entry.policy_max, 4)}</span>
                <div className="d6-bar"><i style={{ ["--p" as string]: String(bar(entry.policy_max)) }} /></div>
                <span className="d6-value">Q {format(entry.q_max, 4)}</span>
                <div className="d6-bar"><i style={{ ["--p" as string]: String(bar(entry.q_max)) }} /></div>
              </div>)}
              <div className="explorer-hint">bars are log-scaled: empty is 1e-8, full is the 1e-4 tolerance.</div>
            </div>;
          })() : d6.state !== "loading" && <Empty>Not captured. The 12 D6 transforms of the position at ply {cursor} must agree to 1e-4.</Empty>}
        </>}

        {tab === "attention" && <>
          <div className="attn-controls">
            <button type="button" disabled={!checkpoint || attention.state === "loading"} onClick={() => void runAttention()}>
              {attention.state === "loading" ? "capturing…" : `Capture at ply ${cursor}`}
            </button>
            <span className="explorer-hint">tokens = {cursor} stones + 1 = {cursor + 1} · ≈ {format((4 * 4 * (cursor + 1) ** 2 * 21) / 1e6, 2)} MB</span>
            {attention.data && <>
              <label className="attn-pick">block<select value={attnBlock} onChange={(event) => setAttnBlock(Number(event.target.value))}>
                {attention.data.layers.map((layer) => <option key={layer.block} value={layer.block}>{layer.block}</option>)}
              </select></label>
              <label className="attn-pick">head<select value={attnHead} onChange={(event) => setAttnHead(Number(event.target.value))}>
                {(attention.data.layers[0]?.heads ?? []).map((_, index) => <option key={index} value={index}>{index}</option>)}
              </select></label>
            </>}
            {isStale(attention) && <span className="diag-stale">STALE · captured at ply {attention.atPly} · recapture</span>}
          </div>
          <ErrorBox message={attention.error} />
          {attention.data
            ? <div className="attn-wrap" data-stale={isStale(attention) ? "" : undefined}>
              <AttentionMap data={attention.data} block={attnBlock} head={attnHead} stones={line.slice(0, attention.atPly)} />
            </div>
            : attention.state !== "loading" && <Empty>Not captured. {attention.state === "idle" ? `The response for ${cursor + 1} tokens is about ${format((4 * 4 * (cursor + 1) ** 2 * 21) / 1e6, 2)} MB.` : ""}</Empty>}
        </>}

        {tab === "model" && <>
          <ErrorBox message={manifest.error} />
          {manifest.data ? <div className="manifest-rows">
            <dl>
              <div className="manifest-head">config</div>
              {Object.entries(manifest.data.config).map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{String(value)}</dd></div>)}
            </dl>
            <dl>
              <div className="manifest-head">versions</div>
              {Object.entries(manifest.data.versions).map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{String(value)}</dd></div>)}
            </dl>
          </div> : <Empty>loading the model manifest…</Empty>}
        </>}
      </Panel>
    </div>
  </div>;
}
