import { useEffect, useState } from 'react'
import { apiGet } from './api'

let cached: string | null = null
let inFlight: Promise<string> | null = null

/** The support contact address, resolved from the single backend config
 *  source (GET /v1/support). Cached for the session; degrades to '' when the
 *  endpoint is unavailable so the UI omits the address instead of guessing. */
export function supportEmail(): Promise<string> {
  if (cached !== null) return Promise.resolve(cached)
  if (!inFlight) {
    inFlight = apiGet<{ support_email?: string }>('/v1/support')
      .then((data) => {
        cached = typeof data.support_email === 'string' ? data.support_email : ''
        return cached
      })
      .catch(() => {
        cached = ''
        return cached
      })
      .finally(() => {
        inFlight = null
      })
  }
  return inFlight
}

/** Hook form of supportEmail for function components. */
export function useSupportEmail(): string {
  const [email, setEmail] = useState('')
  useEffect(() => {
    let alive = true
    void supportEmail().then((value) => {
      if (alive) setEmail(value)
    })
    return () => {
      alive = false
    }
  }, [])
  return email
}
