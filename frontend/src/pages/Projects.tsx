import { useEffect, useState } from 'react'
import AppHeader from '../components/AppHeader'
import { apiGet } from '../lib/api'
import ProjectCard, { type Project } from './ProjectCard'
import NewProjectModal from './NewProjectModal'
import './pages.css'
import './projects.css'

interface ProjectsResponse {
  projects: Project[]
}

type PageState =
  | { tag: 'loading' }
  | { tag: 'error'; message: string }
  | { tag: 'loaded'; projects: Project[] }

export default function Projects() {
  const [state, setState] = useState<PageState>({ tag: 'loading' })
  const [showModal, setShowModal] = useState(false)

  async function loadProjects() {
    setState({ tag: 'loading' })
    try {
      const data = await apiGet<ProjectsResponse>('/v1/projects')
      setState({ tag: 'loaded', projects: data.projects ?? [] })
    } catch (err: unknown) {
      setState({
        tag: 'error',
        message: err instanceof Error ? err.message : 'Failed to load projects.',
      })
    }
  }

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const data = await apiGet<ProjectsResponse>('/v1/projects')
        if (!cancelled) setState({ tag: 'loaded', projects: data.projects ?? [] })
      } catch (err: unknown) {
        if (!cancelled) {
          setState({
            tag: 'error',
            message: err instanceof Error ? err.message : 'Failed to load projects.',
          })
        }
      }
    })()
    return () => { cancelled = true }
  }, [])

  function handleCreated(project: Project) {
    setShowModal(false)
    setState((prev) => {
      if (prev.tag === 'loaded') {
        return { tag: 'loaded', projects: [project, ...prev.projects] }
      }
      return { tag: 'loaded', projects: [project] }
    })
  }

  return (
    <div className="page-shell">
      <AppHeader
        breadcrumb={
          <span className="nav-item nav-item--active">Projects</span>
        }
      />

      <main className="page-main">
        <div className="projects-header">
          <div className="projects-header__text">
            <h1>Projects</h1>
            <p className="page-description">
              All construction projects in your workspace.
            </p>
          </div>
          <button
            className="btn btn--primary"
            type="button"
            onClick={() => setShowModal(true)}
          >
            + New project
          </button>
        </div>

        {state.tag === 'loading' && (
          <div className="projects-state">
            <span className="projects-state__label">Loading…</span>
          </div>
        )}

        {state.tag === 'error' && (
          <div className="projects-state">
            <span className="projects-state__label">Error</span>
            <div className="projects-state__error-text">{state.message}</div>
            <button
              className="btn btn--ghost"
              type="button"
              onClick={() => void loadProjects()}
            >
              Retry
            </button>
          </div>
        )}

        {state.tag === 'loaded' && state.projects.length === 0 && (
          <div className="projects-state">
            <div className="empty-mark" aria-hidden="true">◫</div>
            <p className="projects-state__message">
              No projects yet. Create your first project to start tracking
              documents, readiness, and progress.
            </p>
            <button
              className="btn btn--primary"
              type="button"
              onClick={() => setShowModal(true)}
            >
              + Create first project
            </button>
          </div>
        )}

        {state.tag === 'loaded' && state.projects.length > 0 && (
          <>
            {state.projects.some((p) => p.is_master_corpus) && (
              <div className="pilot-banner" role="note">
                <strong>Pilot tip:</strong> Ask questions inside{' '}
                <em>Master Corpus</em> and filter by package in
                your prompt, e.g. “Answer only from the client project documents.”
                Individual project shells are still being indexed.
              </div>
            )}
            <div className="projects-grid">
              {state.projects.map((project) => (
                <ProjectCard
                  key={project.id}
                  project={project}
                  onDelete={() => void loadProjects()}
                />
              ))}
            </div>
          </>
        )}
      </main>

      {showModal && (
        <NewProjectModal
          onClose={() => setShowModal(false)}
          onCreated={handleCreated}
        />
      )}
    </div>
  )
}
