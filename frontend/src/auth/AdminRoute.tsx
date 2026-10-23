import { Navigate } from 'react-router-dom'
import { useAuth } from './AuthContext'
import ProtectedRoute from './ProtectedRoute'

interface AdminRouteProps {
  children: React.ReactNode
}

/** Authenticated route limited to users with role admin. */
export default function AdminRoute({ children }: AdminRouteProps) {
  const { user, loading } = useAuth()

  return (
    <ProtectedRoute>
      {loading ? null : user?.role === 'admin' ? (
        children
      ) : (
        <Navigate to="/" replace />
      )}
    </ProtectedRoute>
  )
}
