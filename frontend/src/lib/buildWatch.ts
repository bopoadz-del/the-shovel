import { createBuildWatcher, installFetchObserver } from './buildCheck.js'

// The build this bundle was made from. Stamped at build time (Dockerfile
// frontend stage: VITE_BUILD_ID=$GIT_SHA). Empty in `vite dev` and in a bare
// build, and then the check stays silent: unknown is not "differs".
const CLIENT_BUILD = (import.meta.env.VITE_BUILD_ID as string | undefined) ?? ''
const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000'

const listeners = new Set<() => void>()
let mismatch = false

const watcher = createBuildWatcher(CLIENT_BUILD, () => {
  mismatch = true
  listeners.forEach((fn) => fn())
})

export function subscribeBuildMismatch(fn: () => void): () => void {
  listeners.add(fn)
  return () => {
    listeners.delete(fn)
  }
}

export function hasBuildMismatch(): boolean {
  return mismatch
}

/** Called once, before the app renders: every response from every caller is
 *  checked, and a tab that comes back to the front asks the server once. */
export function startBuildCheck(): void {
  installFetchObserver(window, watcher)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible' || mismatch || !CLIENT_BUILD) return
    // The response header is what the observer reads; the body is not needed.
    fetch(`${API_BASE}/livez`, { cache: 'no-store' }).catch(() => undefined)
  })
}
