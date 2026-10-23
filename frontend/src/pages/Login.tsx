import { useState, useEffect, useRef, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { apiPost } from '../lib/api'
import { getRememberedEmail } from '../lib/token'
import BrandMark from '../components/BrandMark'
import './pages.css'
import './auth.css'

type Mode = 'signin' | 'register' | 'reset'

export default function Login() {
  const { user, loading, login, register } = useAuth()
  const navigate = useNavigate()

  // A password-reset link lands here as /?reset=<token>: the page opens on
  // the new-password form, with the token held only in memory.
  const [resetToken] = useState<string | null>(
    () => new URLSearchParams(window.location.search).get('reset'),
  )
  // The front page's "Create an account" opens the form on registration.
  const [mode, setMode] = useState<Mode>(() => {
    if (resetToken) return 'reset'
    return new URLSearchParams(window.location.search).get('mode') === 'register' ? 'register' : 'signin'
  })
  // Pre-fill from the last "Remember me" sign-in. Only ever the address --
  // the password is the browser password manager's job, never ours.
  const [email, setEmail] = useState(() => getRememberedEmail() ?? '')
  const [password, setPassword] = useState('')
  // Ticked by default only for someone we already remember; a first-time or
  // opted-out visitor has to choose persistence deliberately.
  const [remember, setRemember] = useState(() => getRememberedEmail() !== null)
  const [displayName, setDisplayName] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Browser password managers autofill the field's DOM value WITHOUT firing
  // React's onChange, so the controlled `password` state stays '' — which
  // disabled the Sign-in button and let the next re-render's value="" wipe the
  // autofilled password (the "autofill clears on refresh" bug). Refs read the
  // real DOM value: we sync it into state after mount so the controlled value
  // matches the field, and read it as the source of truth at submit.
  const emailRef = useRef<HTMLInputElement>(null)
  const passwordRef = useRef<HTMLInputElement>(null)
  // The verification link lands back here as /?verified=success|invalid.
  // Computed in the initializer rather than an effect: it is derived from the
  // URL that rendered this page, so deriving it during render avoids both a
  // flash of the un-noticed form and a setState-inside-effect.
  const [notice, setNotice] = useState<string | null>(() => {
    const verified = new URLSearchParams(window.location.search).get('verified')
    if (!verified) return null
    return verified === 'success'
      ? 'Email confirmed. You can sign in now.'
      : 'That confirmation link is invalid or has expired. Request a new one below.'
  })

  // Strip the flags so a refresh does not re-announce a stale outcome, and so
  // the reset token does not stay in the address bar or the history.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    if (!params.has('verified') && !params.has('reset')) return
    params.delete('verified')
    params.delete('reset')
    const rest = params.toString()
    window.history.replaceState({}, '', window.location.pathname + (rest ? `?${rest}` : ''))
  }, [])

  // Pull autofilled values into state once the browser has written them (it can
  // land a few frames after mount), so the controlled value matches the field
  // and the Sign-in button enables. Functional updates keep this a mount effect.
  useEffect(() => {
    const sync = () => {
      const e = emailRef.current?.value
      const p = passwordRef.current?.value
      if (e) setEmail((cur) => (cur === e ? cur : e))
      if (p) setPassword((cur) => (cur === p ? cur : p))
    }
    const t1 = setTimeout(sync, 80)
    const t2 = setTimeout(sync, 400)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [])

  // Redirect already-authenticated users away from /login
  useEffect(() => {
    if (!loading && user) {
      navigate('/', { replace: true })
    }
  }, [loading, user, navigate])

  function switchMode(next: Mode) {
    setMode(next)
    setError(null)
    setNotice(null)
    setEmail('')
    setPassword('')
    setDisplayName('')
  }

  async function handleResend() {
    setError(null)
    setNotice(null)
    setSubmitting(true)
    try {
      await apiPost('/v1/users/resend-verification', { email })
      // The endpoint answers 202 whether or not the address exists, so this
      // wording must not imply the account was found -- that would rebuild
      // the account oracle the endpoint deliberately avoids.
      setNotice(`If ${email} needs confirming, a new link is on its way.`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not request a new link.')
    } finally {
      setSubmitting(false)
    }
  }

  async function handleForgot() {
    setError(null)
    setNotice(null)
    setSubmitting(true)
    try {
      await apiPost('/v1/users/forgot-password', { email })
      // Same 202 whether or not the address has an account: the notice must
      // not imply one was found.
      setNotice(`If ${email} has an account, a reset link is on its way. It works once, for 1 hour.`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not request a reset link.')
    } finally {
      setSubmitting(false)
    }
  }

  async function handleReset(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    setError(null)
    setNotice(null)
    const newPassword = passwordRef.current?.value ?? password
    if (newPassword.length < 8) {
      setError('Choose a password of at least 8 characters.')
      return
    }
    setSubmitting(true)
    try {
      await apiPost('/v1/users/reset-password', { token: resetToken, new_password: newPassword })
      setMode('signin')
      setPassword('')
      setNotice('Password changed. Sign in with your new password.')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not change the password.')
    } finally {
      setSubmitting(false)
    }
  }

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    setError(null)
    setNotice(null)

    // Read the live DOM values — an autofilled credential may not have reached
    // state yet — and sync them back so the UI reflects what we submit.
    const emailVal = (emailRef.current?.value ?? email).trim()
    const passwordVal = passwordRef.current?.value ?? password
    if (emailVal !== email) setEmail(emailVal)
    if (passwordVal !== password) setPassword(passwordVal)
    if (!emailVal || !passwordVal) {
      setError('Enter your email and password to sign in.')
      return
    }

    setSubmitting(true)

    try {
      if (mode === 'signin') {
        await login(emailVal, passwordVal, remember)
      } else {
        const result = await register(emailVal, passwordVal, displayName.trim() || undefined)
        if (result.verificationRequired) {
          // The account exists but has no session yet. Navigating to the app
          // would bounce off ProtectedRoute; tell them to check their mail.
          // The reply is the same for a new and an existing address, so it
          // says what happens next without claiming an account was made.
          setNotice(
            `If ${emailVal} can be registered, a confirmation link is on its way - follow it to finish signing in. ` +
              `If none arrives, use "Resend confirmation email" below.`,
          )
          setMode('signin')
          return
        }
      }
      navigate('/', { replace: true })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred.')
    } finally {
      setSubmitting(false)
    }
  }

  // While bootstrapping, render nothing (ProtectedRoute handles the spinner;
  // here we just avoid a flash of the login form for already-logged-in users).
  if (loading) return null

  return (
    <div className="page-center">
      <div className="auth-card">
        {/* Logo mark — shovel from the Quarry standalone design */}
        <div className="auth-card__logo">
          <span className="auth-card__mark">
            <BrandMark size={26} />
          </span>
        </div>

        <h1 className="auth-card__title">
          {mode === 'signin' ? 'Sign in' : mode === 'register' ? 'Create account' : 'Choose a new password'}
        </h1>
        <p className="auth-card__subtitle">Construction Intelligence Platform</p>

        {mode === 'reset' && (
          <form className="auth-form" onSubmit={handleReset} noValidate>
            <div className="auth-field">
              <label className="auth-label" htmlFor="newPassword">New password</label>
              <input
                ref={passwordRef}
                id="newPassword"
                className="auth-input"
                type="password"
                autoComplete="new-password"
                placeholder="Minimum 8 characters"
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={submitting}
              />
            </div>
            {error && (
              <div className="auth-error" role="alert">
                {error}
              </div>
            )}
            <button className="auth-submit" type="submit" disabled={submitting}>
              {submitting ? 'Saving…' : 'Set new password'}
            </button>
          </form>
        )}

        {/* Form */}
        {mode !== 'reset' && (
        <form className="auth-form" onSubmit={handleSubmit} noValidate>
          {mode === 'register' && (
            <div className="auth-field">
              <label className="auth-label" htmlFor="displayName">
                Display name <span className="auth-label__optional">(optional)</span>
              </label>
              <input
                id="displayName"
                className="auth-input"
                type="text"
                autoComplete="name"
                placeholder="e.g. Alex Chen"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                disabled={submitting}
              />
            </div>
          )}

          <div className="auth-field">
            <label className="auth-label" htmlFor="email">Email</label>
            <input
              ref={emailRef}
              id="email"
              className="auth-input"
              type="email"
              autoComplete="email"
              placeholder="you@example.com"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={submitting}
            />
          </div>

          <div className="auth-field">
            <label className="auth-label" htmlFor="password">Password</label>
            <input
              ref={passwordRef}
              id="password"
              className="auth-input"
              type="password"
              autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
              placeholder={mode === 'register' ? 'Minimum 8 characters' : '••••••••'}
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={submitting}
            />
          </div>

          {/* Sign-in only. On registration the session is always kept, so
              offering the choice there would be a control that does nothing. */}
          {mode === 'signin' && (
            <div className="auth-remember">
              <input
                id="remember"
                className="auth-checkbox"
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
                disabled={submitting}
              />
              <label className="auth-remember__label" htmlFor="remember">
                Remember me
                <span className="auth-remember__hint">
                  Stay signed in on this device
                </span>
              </label>
            </div>
          )}

          {error && (
            <div className="auth-error" role="alert">
              {error}
            </div>
          )}

          {notice && (
            <div className="auth-notice" role="status">
              {notice}
            </div>
          )}

          <button
            className="auth-submit"
            type="submit"
            disabled={submitting}
          >
            {submitting
              ? (mode === 'signin' ? 'Signing in…' : 'Creating account…')
              : (mode === 'signin' ? 'Sign in' : 'Create account')}
          </button>
        </form>
        )}

        {mode === 'signin' && (
          <div className="auth-toggle">
            <button
              className="auth-toggle__link"
              type="button"
              onClick={handleForgot}
              disabled={submitting || !email}
            >
              Forgot password?
            </button>
            {' · '}
            <button
              className="auth-toggle__link"
              type="button"
              onClick={handleResend}
              disabled={submitting || !email}
            >
              Resend confirmation email
            </button>
          </div>
        )}

        {/* Toggle */}
        {mode !== 'reset' && (
        <div className="auth-toggle">
          {mode === 'signin' ? (
            <>
              Don&apos;t have an account?{' '}
              <button
                className="auth-toggle__link"
                type="button"
                onClick={() => switchMode('register')}
              >
                Create one
              </button>
            </>
          ) : (
            <>
              Already have an account?{' '}
              <button
                className="auth-toggle__link"
                type="button"
                onClick={() => switchMode('signin')}
              >
                Sign in
              </button>
            </>
          )}
        </div>
        )}
      </div>
    </div>
  )
}
