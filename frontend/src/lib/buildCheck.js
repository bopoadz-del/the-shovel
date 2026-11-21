// A browser tab keeps the script it loaded until it is reloaded. After a
// deploy, that tab still runs the old script against the new server, and a
// control can send what the old script sent (17 Nov: a Word press reached the
// server without the scope the new script sends). Nothing told the user.
//
// Rule: the bundle knows the build it was made from (VITE_BUILD_ID, stamped at
// build time); the server names its build on every response (X-Build-Id).
// When both are known and differ, the page is told once. When either is
// unknown we cannot tell, so we say nothing rather than guess.

export const BUILD_HEADER = 'X-Build-Id'

export const MISMATCH_MESSAGE =
  'This page is running a different version of The Shovel than the server. ' +
  'Some buttons may not do what they say until you reload.'

function norm(value) {
  return typeof value === 'string' ? value.trim().toLowerCase() : ''
}

/** 'same' | 'differs' | 'unknown'. Direction does not matter: an older page
 *  on a newer server and a newer page on an older server both differ. */
export function compareBuilds(clientBuild, serverBuild) {
  const c = norm(clientBuild)
  const s = norm(serverBuild)
  if (!c || !s) return 'unknown'
  return c === s ? 'same' : 'differs'
}

/** Watch every response the page receives. `onMismatch(serverBuild)` fires at
 *  most once per tab. Returns { observe(response), check(serverBuild) }. */
export function createBuildWatcher(clientBuild, onMismatch) {
  let told = false
  function check(serverBuild) {
    if (told) return
    if (compareBuilds(clientBuild, serverBuild) === 'differs') {
      told = true
      onMismatch(norm(serverBuild))
    }
  }
  function observe(response) {
    const headers = response && response.headers
    if (!headers || typeof headers.get !== 'function') return
    check(headers.get(BUILD_HEADER))
  }
  return { observe, check }
}

/** Wrap `target.fetch` so every response, from every caller, is observed.
 *  The caller gets the very same Response object back; a failed fetch is
 *  rethrown unchanged. Installing twice is a no-op. */
export function installFetchObserver(target, watcher) {
  const original = target.fetch
  if (typeof original !== 'function' || original.__buildObserved) return
  const wrapped = async function (...args) {
    const response = await original.apply(this, args)
    try {
      watcher.observe(response)
    } catch (err) {
      // Observing must never break the request it observes.
      console.warn('build check failed', err)
    }
    return response
  }
  wrapped.__buildObserved = true
  target.fetch = wrapped
}

/** When the tab comes back to the front, ask the server once so a tab left
 *  open across a deploy learns the server's build without waiting for the
 *  user's next press. The probe goes through `target.fetch`, so the observer
 *  installed above reads its header: there is one reading path, not two.
 *  `isQuiet()` true (already told, or this bundle has no build id) sends
 *  nothing. A failed probe is swallowed: it must never surface as an error.
 *  Returns a function that removes the listener. */
export function installVisibilityCheck(doc, target, url, isQuiet) {
  const onChange = () => {
    if (doc.visibilityState !== 'visible' || isQuiet()) return
    let pending
    try {
      pending = target.fetch(url, { cache: 'no-store' })
    } catch {
      return
    }
    if (pending && typeof pending.catch === 'function') pending.catch(() => undefined)
  }
  doc.addEventListener('visibilitychange', onChange)
  return () => doc.removeEventListener('visibilitychange', onChange)
}
