/* eslint-disable react-refresh/only-export-components -- provider + useAuth share this module */
import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { ApiError, apiGet, apiPost } from '../lib/api'
import {
  clearRememberedEmail,
  clearToken,
  getToken,
  setRememberedEmail,
  setToken,
} from '../lib/token'

// ─── Types ────────────────────────────────────────────────────────────────────
// /me returns user_id; /login returns user.id — normalise to a single shape.
export interface AuthUser {
  id: string
  email: string
  role: string
  display_name?: string
}

interface LoginResponse {
  token: string
  token_type: string
  user: { id: string; email: string; role: string }
}

interface RegisterResponse {
  id: string
  email: string
  role: string
  display_name?: string
  created_at: string
  email_verified: boolean
  verification_email_sent: boolean
}

/** What registration produced. When the address still needs confirming there
 *  is deliberately no session: the account exists but cannot log in yet. */
export interface RegisterResult {
  verificationRequired: boolean
  verificationEmailSent: boolean
}

interface MeResponse {
  user_id: string
  email: string
  role: string
  display_name?: string
}

// ─── Context shape ────────────────────────────────────────────────────────────
interface AuthContextValue {
  user: AuthUser | null
  loading: boolean
  /** ``remember`` decides whether the session survives closing the browser. */
  login: (email: string, password: string, remember?: boolean) => Promise<void>
  register: (
    email: string,
    password: string,
    displayName?: string,
  ) => Promise<RegisterResult>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

// ─── Provider ─────────────────────────────────────────────────────────────────
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [loading, setLoading] = useState(() => Boolean(getToken()))
  // Guard against StrictMode double-invocation
  const bootstrapped = useRef(false)

  useEffect(() => {
    if (bootstrapped.current) return
    bootstrapped.current = true

    const token = getToken()
    if (!token) {
      return
    }

    apiGet<MeResponse>('/v1/users/me')
      .then((me) => {
        setUser({
          id: me.user_id,
          email: me.email,
          role: me.role,
          display_name: me.display_name,
        })
      })
      .catch((err: unknown) => {
        // Only discard the token on a real auth failure (401).
        // Network errors (TypeError, no status) leave the token intact so a
        // transient blip during bootstrap does not silently log the user out.
        if (err instanceof ApiError && err.status === 401) {
          clearToken()
        }
      })
      .finally(() => {
        setLoading(false)
      })
  }, [])

  const login = async (
    email: string,
    password: string,
    remember = true,
  ): Promise<void> => {
    const data = await apiPost<LoginResponse>('/v1/users/login', { email, password })
    // remember=true  -> localStorage, survives a browser restart
    // remember=false -> sessionStorage, gone when the tab closes
    setToken(data.token, remember)
    // The email is a convenience for the next visit. The password is never
    // stored -- the browser's password manager owns that.
    if (remember) {
      setRememberedEmail(email)
    } else {
      clearRememberedEmail()
    }
    setUser({
      id: data.user.id,
      email: data.user.email,
      role: data.user.role,
    })
  }

  const register = async (
    email: string,
    password: string,
    displayName?: string,
  ): Promise<RegisterResult> => {
    const data = await apiPost<RegisterResponse>('/v1/users/register', {
      email,
      password,
      ...(displayName ? { display_name: displayName } : {}),
    })
    // Only auto-login when the address needed no confirmation (local/dev
    // builds with no mail provider). Attempting it while verification is
    // pending would surface the 403 gate as an error on a registration that
    // actually SUCCEEDED.
    if (!data.email_verified) {
      return {
        verificationRequired: true,
        verificationEmailSent: Boolean(data.verification_email_sent),
      }
    }
    await login(email, password)
    return { verificationRequired: false, verificationEmailSent: false }
  }

  const logout = (): void => {
    clearToken()
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

// ─── Hook ─────────────────────────────────────────────────────────────────────
export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return ctx
}
