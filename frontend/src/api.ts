import { useCallback, useEffect, useRef, useState } from "react";

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

/** Identifies `AbortSignal` rejections so callers can discard cancellations. */
export function isAbortError(reason: unknown): boolean {
  return reason instanceof DOMException ? reason.name === "AbortError"
    : reason instanceof Error && reason.name === "AbortError";
}

export function reasonText(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

/** Shared JSON client; `init.signal` is honored and non-success responses throw `ApiError`. */
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
  /** Do not fetch on mount or path change; only `refresh()` fetches. */
  manual?: boolean;
}

export interface UseApiResult<T> {
  data: T | undefined;
  error: string | undefined;
  loading: boolean;
  /** Whether the current path has been requested, including an empty response. */
  requested: boolean;
  refresh: () => Promise<void>;
}

/**
 * Fetches the current path and accepts only its latest response.
 * Path changes and unmounts abort in-flight work; generation checks reject stale results.
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
      // Manual hooks clear held state when the path changes because it describes the prior path.
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
