import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import BrandMark from '../components/BrandMark'
import './pages.css'
import './auth.css'
import './landing.css'

// What the platform does, in the words a site team uses. Every line names a
// capability the product has today; no client, project or figure is claimed.
const CAPABILITIES: { title: string; body: string }[] = [
  {
    title: 'Ask your project documents',
    body: 'Specifications, contracts, BOQs and drawings answer in plain words, with the passage they came from.',
  },
  {
    title: 'Calculations you can check',
    body: 'Quantities, rates, durations and design checks run on stated formulas, and every assumed input is named.',
  },
  {
    title: 'Deliverables, not just answers',
    body: 'Schedules, cost build-ups, procurement lists and look-ahead programmes, ready to download.',
  },
]

export default function Landing() {
  const { user, loading } = useAuth()
  const navigate = useNavigate()

  // A signed-in visitor has no use for the front page: straight to the app.
  useEffect(() => {
    if (!loading && user) navigate('/', { replace: true })
  }, [loading, user, navigate])

  if (loading) return null

  return (
    <div className="page-center">
      <main className="landing">
        <header className="landing__head">
          <span className="auth-card__mark">
            <BrandMark size={26} />
          </span>
          <h1 className="landing__title">The Shovel</h1>
          <p className="landing__lede">
            Construction intelligence for the people who build: your project's documents, its
            calculations and its deliverables, in one place.
          </p>
        </header>

        <ul className="landing__list">
          {CAPABILITIES.map((c) => (
            <li key={c.title} className="landing__item">
              <h2 className="landing__item-title">{c.title}</h2>
              <p className="landing__item-body">{c.body}</p>
            </li>
          ))}
        </ul>

        <div className="landing__actions">
          <button className="auth-submit" type="button" onClick={() => navigate('/login')}>
            Sign in
          </button>
          <button
            className="auth-toggle__link landing__secondary"
            type="button"
            onClick={() => navigate('/login?mode=register')}
          >
            Create an account
          </button>
        </div>
      </main>
    </div>
  )
}
