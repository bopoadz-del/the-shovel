import { Navigate } from 'react-router-dom'
import { useAuth } from './AuthContext'

interface ProtectedRouteProps {
  children: React.ReactNode
}

export default function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { user, loading } = useAuth()

  if (loading) {
    return (
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'var(--bg)',
        }}
      >
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '12px',
            color: 'var(--text-muted)',
            letterSpacing: '0.05em',
          }}
        >
          …
        </span>
      </div>
    )
  }

  if (!user) {
    // A link from an email (/?verified=..., /?reset=...) goes to the sign-in
    // page with its query string: bouncing to a bare page would swallow the
    // one piece of feedback that confirms the click worked. Any other
    // signed-out visitor sees the public front page.
    const params = new URLSearchParams(window.location.search)
    if (params.has('verified') || params.has('reset')) {
      return (
        <Navigate
          to={{ pathname: '/login', search: window.location.search }}
          replace
        />
      )
    }
    return <Navigate to="/welcome" replace />
  }

  return <>{children}</>
}
