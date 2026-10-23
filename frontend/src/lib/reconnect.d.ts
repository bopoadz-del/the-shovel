export declare const RETRYABLE_STATUS: number[]
export declare const DEFAULT_MAX_MS: number
export declare const BACKOFF_MS: number[]
export declare class ReconnectGaveUp extends Error {}
export interface ReconnectOptions {
  maxMs?: number
  onReconnecting?: (attempt: number) => void
  signal?: AbortSignal
  now?: () => number
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>
}
export declare function sendWithReconnect(
  send: () => Promise<Response>,
  serverIsUp: () => Promise<boolean>,
  opts?: ReconnectOptions,
): Promise<Response>
