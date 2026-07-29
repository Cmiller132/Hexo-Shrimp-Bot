import { useCallback, useEffect, useRef, useState } from "react";

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

/** True for the rejection `fetch` produces when its `AbortSignal` fires. Aborted
 *  work is discarded, never surfaced — it is a cancellation, not a failure. */
export function isAbortError(reason: unknown): boolean {
  return reason instanceof DOMException ? reason.name === "AbortError"
    : reason instanceof Error && reason.name === "AbortError";
}

export function reasonText(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

/** The one JSON client. `init.signal` is honoured, so every caller can cancel. */
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new ApiError(payload?.error?.message ?? `${response.status} ${response.statusText}`, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export function post<T>(path: string, body: unknown, init?: RequestInit): Promise<T> {
  return api<T>(path, { ...init, method: "POST", body: JSON.stringify(body) });
}

export interface UseApiOptions {
  /** Do not fetch on mount or on a path change; only `refresh()` fetches. Required
   *  for the aggregates that cost 60–110 s on a live run. */
  manual?: boolean;
}

export interface UseApiResult<T> {
  data: T | undefined;
  error: string | undefined;
  loading: boolean;
  /** Whether a fetch for the current path has been asked for at all. Lets a manual
   *  panel distinguish "not requested" from "requested and empty". */
  requested: boolean;
  refresh: () => Promise<void>;
}

/**
 * One request per path, with two guards: the in-flight request is aborted when the
 * path changes or the component unmounts, and every response is checked against a
 * monotone generation snapshot so an already-resolved stale response can never land
 * on top of a fresher one.
 */
export function useApi<T>(path: string | null, deps: unknown[] = [], options: UseApiOptions = {}): UseApiResult<T> {
  const manual = options.manual ?? false;
  const [data, setData] = useState<T>();
  const [error, setError] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [requested, setRequested] = useState(false);
  const generation = useRef(0);
  const inFlight = useRef<AbortController | undefined>(undefined);

  const refresh = useCallback(async () => {
    if (!path) return;
    const snapshot = ++generation.current;
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setRequested(true);
    setLoading(true);
    try {
      const value = await api<T>(path, { signal: controller.signal });
      if (generation.current !== snapshot) return;
      setData(value);
      setError(undefined);
    } catch (reason) {
      if (isAbortError(reason) || generation.current !== snapshot) return;
      setError(reasonText(reason));
    } finally {
      if (generation.current === snapshot) setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);

  useEffect(() => {
    if (manual) {
      // A new path means the held result describes something else. Drop it rather
      // than showing last run's numbers under this run's heading.
      generation.current++;
      inFlight.current?.abort();
      setData(undefined);
      setError(undefined);
      setLoading(false);
      setRequested(false);
      return;
    }
    void refresh();
  }, [refresh, manual]);

  useEffect(() => () => { generation.current++; inFlight.current?.abort(); }, []);

  return { data, error, loading, requested, refresh };
}

export function query(values: Record<string, unknown>) {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") params.set(key, String(value));
  });
  return params.toString();
}
