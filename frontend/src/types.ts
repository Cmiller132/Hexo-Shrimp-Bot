export type Move = [number, number];
export type Screen = "play" | "history" | "live" | "lab";

export interface Checkpoint {
  name: string;
  path: string;
  iteration: number | null;
  bytes: number;
  modified: string;
}

export interface Run {
  name: string;
  state: "active" | "stopped" | "completed" | "starved";
  controlled: boolean;
  pid: number | null;
  iterations: number;
  iteration: number;
  working: boolean;
  heartbeat: null | {
    updated: string;
    iteration: number;
    collect: null | { iteration: number; finished: number; quota: number; steps: number; slot_plies: number[] };
    fit: null | { iteration: number; chunk: number; chunks: number };
    eval: null | { iteration: number };
  };
  checkpoints: Checkpoint[];
}

/** One legal move as `/api/inspect` returns it, in the engine's legal ordering.
 *  `rank` is that ordering's index — it carries no quality information. */
export interface InspectLegal {
  move: Move;
  rank: number;
  logit: number;
  policy: number;
  q: number;
  improved: number;
}

/** An `InspectLegal` carrying the derived quality rank, and optionally the signed
 *  difference against a compare checkpoint. Produced by `withPolicyRank`. */
export interface Candidate extends InspectLegal {
  /** 0-based index of this move when the legal set is sorted by `policy` descending.
   *  This is the real quality rank; `rank` is engine order. */
  policyRank: number;
  /** Signed Δ against the compare checkpoint for the active overlay quantity.
   *  Required when a Board is rendered with `overlay="delta"`. */
  delta?: number;
}

export interface Inspect {
  /** The prefix `moves[:t]` of the line that was posted, not the whole line. */
  moves: Move[];
  t: number;
  mover: number;
  moves_remaining: number;
  stone_count: number;
  legal_count: number;
  /** Echoed π′ recipe. The server ignores `tau` or `lam` sent alone, so a request
   *  that overrides either must send both, and the echo must be checked. */
  tau: number;
  lam: number;
  v_hat: number;
  kl: number;
  norm_entropy: number;
  /** The move the posted line plays from `t`; null when `t === moves.length`. */
  played: Move | null;
  legal: InspectLegal[];
}

/** The columns both game endpoints return. `capped` is 0/1, not a boolean. */
interface GameFacts {
  game_id: number;
  kind: "selfplay" | "eval";
  iteration: number;
  match: number | null;
  game_index: number;
  winner: number | null;
  length: number;
  capped: number;
  model_seat: number | null;
  opening_len: number | null;
  opponent_depth_mean: number | null;
}

/** A telemetry game as the listing returns it. The canonical `opening` is
 *  computed by the listing query alone — the detail endpoint does not carry it,
 *  so it lives here and not on `Game`. */
export interface GameRow extends GameFacts {
  opening: Move[];
}

/** The acting net's own read, recorded while the game was collected. `rank` is
 *  engine order. Empty for `kind: "eval"` games — they store no per-ply trace. */
export interface StoredPly {
  game_id: number;
  t: number;
  mover: number;
  moves_remaining: number;
  legal_count: number;
  rank: number;
  v_hat: number;
  kl: number;
  norm_entropy: number;
  pi_top1: number;
  pi_chosen: number;
}

export interface GameReview {
  run: string;
  game_id: number;
  tags: string[];
  note: string;
}

/** One game as `/api/runs/{run}/games/{id}` returns it: the same facts plus the
 *  move list, the acting net's per-ply trace and the stored review. */
export interface Game extends GameFacts {
  moves: Move[];
  plies: StoredPly[];
  review: GameReview;
}

/** History → Lab hand-off. The whole game travels, never a truncated prefix;
 *  `token` is bumped on every hand-off so re-sending the same game re-seeds. */
export interface LabHandoff {
  token: number;
  run: string;
  gameId: number;
  kind: GameRow["kind"];
  iteration: number;
  winner: number | null;
  capped: boolean;
  moves: Move[];
  plies: StoredPly[];
  ply: number;
}

export interface D6Transform {
  transform: number;
  policy_max: number;
  q_max: number;
}

export interface D6Result {
  transforms: D6Transform[];
  policy_max: number;
  q_max: number;
}

/** `layers[block].heads[head][query][key]`; `tokens` is stones + 1 global token. */
export interface AttentionResult {
  tokens: number;
  layers: Array<{ block: number; heads: number[][][] }>;
}

export interface ModelManifest {
  config: Record<string, number>;
  versions: Record<string, string | number>;
}

export interface Probe {
  probe_id: number;
  name: string;
  checkpoint: string;
  moves: Move[];
  module: string;
}
