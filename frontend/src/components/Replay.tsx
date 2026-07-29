import { ChevronLeft, ChevronRight, Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { usePaneActive } from "./Pane";

/**
 * Excludes handled events, modifier combinations, modal input, editable fields,
 * and Space on a focused button from global bindings.
 */
function shouldIgnoreKey(event: KeyboardEvent): boolean {
  if (event.defaultPrevented) return true;
  if (event.ctrlKey || event.metaKey || event.altKey) return true;
  if (document.body.dataset.modal === "open") return true;
  const element = event.target as HTMLElement | null;
  if (!element) return false;
  if (element.isContentEditable) return true;
  if (element.tagName === "INPUT" || element.tagName === "TEXTAREA" || element.tagName === "SELECT") return true;
  if (element.tagName === "BUTTON" && event.key === " ") return true;
  return false;
}

export type KeyHandlers = Record<string, (event: KeyboardEvent) => void>;

/** Binds guarded document-level keys only for the visible pane.
 *  Names use `KeyboardEvent.key` with optional `Shift+`; matches call `preventDefault()`. */
export function useDeckKeys(handlers: KeyHandlers): void {
  const active = usePaneActive();
  const ref = useRef(handlers);
  ref.current = handlers;
  useEffect(() => {
    if (!active) return;
    const listener = (event: KeyboardEvent) => {
      if (shouldIgnoreKey(event)) return;
      const handler = ref.current[`${event.shiftKey ? "Shift+" : ""}${event.key}`]
        ?? (event.shiftKey ? undefined : ref.current[event.key]);
      if (!handler) return;
      event.preventDefault();
      handler(event);
    };
    document.addEventListener("keydown", listener);
    return () => document.removeEventListener("keydown", listener);
  }, [active]);
}

const SPEEDS = [0.25, 0.5, 1, 2, 4];
/** Milliseconds per step at 1×. */
const STEP_MS = 600;

export interface TransportProps {
  /** Highest reachable cursor. The track spans `0..length` inclusive. */
  length: number;
  value: number;
  onChange: (value: number) => void;
}

/**
 * Provides cursor track, step, endpoint, playback-speed, and keyboard controls.
 * A hidden pane stops playback and does not bind the keyboard.
 */
export default function Transport({ length, value, onChange }: TransportProps) {
  const active = usePaneActive();
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const valueRef = useRef(value);
  valueRef.current = value;
  // A ref keeps inline `onChange` callbacks from restarting the playback timer.
  const changeRef = useRef(onChange);
  changeRef.current = onChange;

  const go = useCallback((next: number) => {
    changeRef.current(Math.max(0, Math.min(length, next)));
  }, [length]);

  // A length change stops playback and clamps the cursor to the new endpoint.
  useEffect(() => {
    setPlaying(false);
    if (valueRef.current > length) changeRef.current(length);
  }, [length]);

  useEffect(() => { if (!active) setPlaying(false); }, [active]);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      if (valueRef.current >= length) { setPlaying(false); return; }
      changeRef.current(valueRef.current + 1);
    }, Math.max(40, STEP_MS / speed));
    return () => clearInterval(timer);
  }, [playing, length, speed]);

  const toggle = useCallback(() => {
    setPlaying((current) => {
      if (current) return false;
      if (valueRef.current >= length) changeRef.current(0);
      return true;
    });
  }, [length]);

  useDeckKeys({
    ArrowLeft: () => go(valueRef.current - 1),
    ArrowRight: () => go(valueRef.current + 1),
    "Shift+ArrowLeft": () => go(valueRef.current - 10),
    "Shift+ArrowRight": () => go(valueRef.current + 10),
    Home: () => go(0),
    End: () => go(length),
    " ": toggle,
  });

  return <div className="transport" data-playing={playing ? "" : undefined}>
    <div className="transport-buttons">
      <button type="button" title="first (Home)" aria-label="first" disabled={value === 0} onClick={() => go(0)}><SkipBack size={12} /></button>
      <button type="button" title="previous (←)" aria-label="previous" disabled={value === 0} onClick={() => go(value - 1)}><ChevronLeft size={12} /></button>
      <button type="button" title={playing ? "pause (space)" : "play (space)"} aria-label={playing ? "pause" : "play"} data-primary="" disabled={length === 0} onClick={toggle}>{playing ? <Pause size={12} /> : <Play size={12} />}</button>
      <button type="button" title="next (→)" aria-label="next" disabled={value >= length} onClick={() => go(value + 1)}><ChevronRight size={12} /></button>
      <button type="button" title="last (End)" aria-label="last" disabled={value >= length} onClick={() => go(length)}><SkipForward size={12} /></button>
      <select className="transport-speed" aria-label="playback speed" value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
        {SPEEDS.map((rate) => <option key={rate} value={rate}>{rate}×</option>)}
      </select>
    </div>
    <input
      className="transport-track"
      type="range"
      min={0}
      max={length}
      step={1}
      value={value}
      disabled={length === 0}
      aria-label="ply cursor"
      aria-valuetext={`ply ${value} of ${length}`}
      onChange={(event) => go(Number(event.target.value))}
    />
    <div className="transport-readout"><b>ply {value}</b><span>/ {length}</span></div>
    <div className="transport-keys">← → step · shift ±10 · home/end ends · space play</div>
  </div>;
}
