// Session token storage, with "Remember me" deciding WHERE it lives.
//
// localStorage   survives closing the browser  -> "Remember me" ticked
// sessionStorage dies with the tab             -> "Remember me" unticked
//
// Before this, the token always went to localStorage, so every sign-in was
// permanent and there was no way to opt out. On a shared or site machine that
// is the wrong default to force: the next person to open the browser is signed
// in as you.
//
// The PASSWORD is never stored, in either place. A password in web storage is
// readable by any script on the page and persists on disk. The login form
// carries the correct autoComplete attributes, so the browser's own password
// manager handles that properly and safely. Only the email is remembered, and
// only as a convenience to pre-fill the field.

const TOKEN_KEY = 'tf_token'
const EMAIL_KEY = 'tf_remembered_email'

// Storage can throw, not just return null: Safari private mode and browsers
// with site data blocked raise on access. Never let that break sign-in.
function safe(fn: () => void): void {
  try {
    fn()
  } catch {
    /* storage unavailable — degrade to an in-memory session */
  }
}

function read(store: Storage, key: string): string | null {
  try {
    return store.getItem(key)
  } catch {
    return null
  }
}

export function getToken(): string | null {
  // sessionStorage first: if both somehow hold a token, the one belonging to
  // this tab is the current intent.
  return read(sessionStorage, TOKEN_KEY) ?? read(localStorage, TOKEN_KEY)
}

export function setToken(token: string, remember = true): void {
  safe(() => {
    // Always clear the other store, or an old token outlives the new choice
    // and getToken() can hand back a stale one.
    if (remember) {
      sessionStorage.removeItem(TOKEN_KEY)
      localStorage.setItem(TOKEN_KEY, token)
    } else {
      localStorage.removeItem(TOKEN_KEY)
      sessionStorage.setItem(TOKEN_KEY, token)
    }
  })
}

export function clearToken(): void {
  safe(() => {
    localStorage.removeItem(TOKEN_KEY)
    sessionStorage.removeItem(TOKEN_KEY)
  })
}

/** True when the current session was stored to survive a browser restart. */
export function isRemembered(): boolean {
  return read(localStorage, TOKEN_KEY) !== null
}

// ─── remembered email (convenience only, never the password) ────────────────

export function getRememberedEmail(): string | null {
  return read(localStorage, EMAIL_KEY)
}

export function setRememberedEmail(email: string): void {
  safe(() => localStorage.setItem(EMAIL_KEY, email))
}

export function clearRememberedEmail(): void {
  safe(() => localStorage.removeItem(EMAIL_KEY))
}
