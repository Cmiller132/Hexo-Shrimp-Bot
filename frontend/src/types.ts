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

export interface Inspect {
  moves: Move[];
  t: number;
  mover: number;
  moves_remaining: number;
  stone_count: number;
  legal_count: number;
  v_hat: number;
  kl: number;
  norm_entropy: number;
  legal: Array<{ move: Move; rank: number; policy: number; q: number; improved: number; logit: number }>;
}
