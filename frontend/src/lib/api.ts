import { clearToken, getToken } from './token'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

// ─── ApiError — carries HTTP status so callers can distinguish 401 vs network ──
export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function authHeaders(): Record<string, string> {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (res.ok) {
    return res.json() as Promise<T>
  }

  // Attempt to parse backend error body
  let message = `HTTP ${res.status}`
  try {
    const body = await res.json() as {
      detail?: string | { msg?: string }[]
      error?: { code?: string; message?: string }
    }
    if (typeof body.detail === 'string') {
      message = body.detail
    } else if (Array.isArray(body.detail) && body.detail.length > 0) {
      const first = body.detail[0]
      message = typeof first === 'object' && first.msg ? first.msg : String(first)
    } else if (body.error?.message) {
      message = body.error.message
    }
  } catch {
    // body wasn't JSON — keep the HTTP status message
  }

  // Global session-expiry handling: a 401 on any non-auth call means the token
  // is missing/expired/invalid. Clear it and bounce to login instead of
  // stranding the user on a panel whose Retry just keeps 401-ing. The auth
  // bootstrap endpoints (login/register/me) are excluded — a bad-credentials
  // 401 must surface on the Login page, and AuthContext handles /me itself.
  const isAuthBootstrap = /\/v1\/users\/(login|register|me)\b/.test(res.url)
  if (res.status === 401 && !isAuthBootstrap) {
    clearToken()
    if (window.location.pathname !== '/login') {
      window.location.assign('/login')
    }
  }

  throw new ApiError(message, res.status)
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
    },
  })
  return handleResponse<T>(res)
}

export async function apiPatch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
    },
    body: JSON.stringify(body),
  })
  return handleResponse<T>(res)
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
    },
    body: JSON.stringify(body),
  })
  return handleResponse<T>(res)
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
    },
  })
  return handleResponse<T>(res)
}

export async function apiPostForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: {
      // Do NOT set Content-Type — browser sets it with boundary for multipart
      ...authHeaders(),
    },
    body: form,
  })
  return handleResponse<T>(res)
}
