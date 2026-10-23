import { useAuth } from '../auth/AuthContext'
import ThemeToggle from '../theme/ThemeToggle'
import BrandMark from './BrandMark'
import './AppHeader.css'

interface AppHeaderProps {
  /** Extra breadcrumb segments rendered after the brand */
  breadcrumb?: React.ReactNode
}

export default function AppHeader({ breadcrumb }: AppHeaderProps) {
  const { user, logout } = useAuth()

  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div className="app-header__brand">
          <span className="brand-mark" aria-hidden="true">
            <BrandMark />
          </span>
          <span className="brand-name">The Shovel</span>
        </div>

        {breadcrumb && (
          <nav className="app-header__nav">
            {breadcrumb}
          </nav>
        )}

        <div className="app-header__user">
          <ThemeToggle />
          {user && (
            <>
              <span className="app-header__email">{user.email}</span>
              <button
                className="app-header__logout"
                onClick={logout}
                type="button"
              >
                Sign out
              </button>
            </>
          )}
        </div>
      </div>
    </header>
  )
}
