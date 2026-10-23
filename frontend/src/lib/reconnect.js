/* Send a chat turn through a deploy without losing it — or sending it twice.
 *
 * Every deploy swaps the single instance that holds the documents disk, so the
 * site answers 502 for 13 s or more (measured 2026-09-19). A message sent in
 * that window failed with a raw error and the user had to retype it.
 *
 * What is safe to retry, and what is not:
 *   - 502 / 503 / 504: the PROXY answered; the app never saw the request. Safe.
 *   - fetch() threw (network error): the request MAY have reached the app. It
 *     is retried only when /livez shows the server really was down. If the
 *     server is up, the turn may already be running — retrying would run it
 *     twice, so the error is reported instead.
 *   - Any other response (200, 4xx, 500): returned as-is. A stream that has
 *     started is never restarted.
 *
 * Plain JavaScript on purpose: CI's Node 20 runs `node --test` on it directly,
 * with no test framework to install. Types are in reconnect.d.ts.
 */

export const RETRYABLE_STATUS = [502, 503, 504]
export const DEFAULT_MAX_MS = 90_000
export const BACKOFF_MS = [2_000, 3_000, 5_000, 8_000]

export class ReconnectGaveUp extends Error {
  constructor() {
    super(
      'The assistant is restarting and did not come back in time. Your message ' +
        'was not sent — please send it again in a minute.',
    )
    this.name = 'ReconnectGaveUp'
  }
}

const defaultSleep = (ms, signal) =>
  new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms)
    signal?.addEventListener?.('abort', () => {
      clearTimeout(t)
      reject(new DOMException('Aborted', 'AbortError'))
    })
  })

/**
 * @param {() => Promise<Response>} send       performs the POST
 * @param {() => Promise<boolean>} serverIsUp  GET /livez → true when 200
 * @param {object} [opts]
 */
export async function sendWithReconnect(send, serverIsUp, opts = {}) {
  const {
    maxMs = DEFAULT_MAX_MS,
    onReconnecting = () => {},
    signal,
    now = () => Date.now(),
    sleep = defaultSleep,
  } = opts
  const started = now()
  let attempt = 0
  let sawServerDown = false

  for (;;) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
    let retry = false
    let lastError = null
    try {
      const res = await send()
      if (!RETRYABLE_STATUS.includes(res.status)) return res
      sawServerDown = true
      retry = true
    } catch (err) {
      if (err?.name === 'AbortError') throw err
      lastError = err
      // Did the server go away, or did only the reply get lost?
      let up = true
      try {
        up = await serverIsUp()
      } catch {
        up = false
      }
      if (up && !sawServerDown) throw err // it may be running: never send twice
      if (!up) sawServerDown = true
      retry = true
    }
    if (!retry) throw lastError
    const wait = BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)]
    if (now() - started + wait > maxMs) throw new ReconnectGaveUp()
    attempt += 1
    onReconnecting(attempt)
    await sleep(wait, signal)
    // Wait until the server answers before re-sending the turn.
    for (;;) {
      let up = false
      try {
        up = await serverIsUp()
      } catch {
        up = false
      }
      if (up) break
      if (now() - started + 2_000 > maxMs) throw new ReconnectGaveUp()
      await sleep(2_000, signal)
    }
  }
}
