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

export interface HorizonBucket {
  k_min: number;
  k_max: number | null;
  bucket: string;
  outcome: "won" | "lost";
  count: number;
  sign_accuracy: number | null;
  mean_abs_v_hat: number | null;
}

export interface HorizonReadout {
  lo: number | null;
  hi: number | null;
  buckets: HorizonBucket[];
}

export interface EvalReadout {
  match_id: number;
  opponent: number;
  opponent_name: string;
  opponent_config: string;
  family: "sealbot" | "seat" | "h2h" | "other";
  iteration: number | null;
  checkpoint: string | null;
  games: number;
  win_rate: number;
  ci_lo: number | null;
  ci_hi: number | null;
  elo: number | null;
  elo_lo: number | null;
  elo_hi: number | null;
  score_as_p0: number | null;
  score_as_p1: number | null;
  opponent_depth_mean: number | null;
  sign_test_p?: number;
  decisive_pairs?: number;
  pair_counts?: {
    model_both: number;
    split: number;
    reference_both: number;
    capped: number;
  };
}

/** One legal move from `/api/inspect`; `rank` is its engine-order index. */
export interface InspectLegal {
  move: Move;
  rank: number;
  logit: number;
  policy: number;
  q: number;
  improved: number;
}

/** An `InspectLegal` with policy-derived rank and an optional checkpoint difference. */
export interface Candidate extends InspectLegal {
  /** 0-based index of this move when the legal set is sorted by `policy` descending.
   *  `rank` remains the engine-order index. */
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

/** A telemetry listing row; only the listing endpoint supplies canonical `opening`. */
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

/** History → Lab handoff containing the complete game; each handoff gets a new token. */
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
