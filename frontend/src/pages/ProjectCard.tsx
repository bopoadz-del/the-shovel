import { Link } from 'react-router-dom'
import { apiDelete } from '../lib/api'

// readiness shape may vary — render defensively
export interface Project {
  id: string
  name: string
  client?: string
  status: string
  aconex_connected?: boolean
  user_id?: string
  created_at: string
  readiness?: unknown
  documents?: unknown
  origin?: string
  document_count?: number
  is_master_corpus?: boolean
}

interface ProjectCardProps {
  project: Project
  onDelete?: (id: string) => void
}

function statusClass(status: string): string {
  const s = status.toLowerCase()
  if (s === 'active') return 'project-card__status--active'
  if (s === 'draft') return 'project-card__status--draft'
  return 'project-card__status--other'
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  } catch {
    return iso
  }
}

// The `readiness` object comes from the backend PROGRESS-TRACKING gate
// (compute_readiness, Roadmap V2 · 0.2). A project is "ready" ONLY once it has
// a baseline schedule + >=1 daily report + >=1 weekly report AND Aconex is
// connected — i.e. it is set up for live progress tracking (S-curve / EVM /
// daily-weekly progress). It is NOT a RAG/indexing status: a corpus can retrieve
// and answer perfectly and still be "Not ready" by this gate.
//
// We HIDE this badge for document corpora (master corpus + Drive-approved packs)
// because progress-readiness does not apply to them — they are knowledge packs,
// not tracked projects — and because Aconex is not wired yet (aconex_connected
// is never true), so "Ready" is currently unreachable for any project. Showing
// "Not ready" on a working corpus reads as "broken" when it is not. See
// isDocumentCorpus() below and the gate in the card footer.
function readinessHint(readiness: unknown): string | null {
  if (!readiness) return null
  if (typeof readiness !== 'object') return null
  const r = readiness as Record<string, unknown>
  // Try common keys: score, percent, label, status, ready
  if (typeof r.label === 'string' && r.label) return r.label
  if (typeof r.status === 'string' && r.status) return r.status
  if (typeof r.score === 'number') return `Score: ${r.score}`
  if (typeof r.percent === 'number') return `${r.percent}%`
  if (typeof r.ready === 'boolean') return r.ready ? 'Ready' : 'Not ready'
  // Readiness object exists but we can't distill a hint — just acknowledge it
  return 'Readiness available'
}

function isIncompleteShell(project: Project): boolean {
  return (
    project.origin === 'admin_drive_approved' &&
    !project.is_master_corpus &&
    (project.document_count ?? 0) <= 1
  )
}

// A knowledge / document corpus is not a progress-tracked project, so the
// readiness gate does not apply. This covers every way a corpus enters the
// system: the master corpus, an admin Drive-approved pack
// (origin='admin_drive_approved'), a USER Drive import from the New Project
// modal (origin='user_drive_import'), and seeded knowledge bases
// (origin='system_seed'). Missing user_drive_import here left those packs
// showing the same misleading "Not ready" badge this change removes.
const CORPUS_ORIGINS = ['admin_drive_approved', 'user_drive_import', 'system_seed']
function isDocumentCorpus(project: Project): boolean {
  return !!project.is_master_corpus || CORPUS_ORIGINS.includes(project.origin ?? '')
}

export default function ProjectCard({ project, onDelete }: ProjectCardProps) {
  const master = project.is_master_corpus
  const incomplete = isIncompleteShell(project)
  // Progress-readiness is meaningless for document corpora — hide the badge
  // there (see readinessHint docblock). Keep it for real tracked projects.
  const hint = isDocumentCorpus(project) ? null : readinessHint(project.readiness)

  async function handleDelete(e: React.MouseEvent) {
    e.preventDefault()
    e.stopPropagation()
    if (!window.confirm(`Delete project "${project.name}"? This cannot be undone.`)) {
      return
    }
    try {
      await apiDelete(`/v1/projects/${project.id}`)
      onDelete?.(project.id)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Delete failed'
      alert(message)
    }
  }

  return (
    <Link
      to={`/projects/${project.id}`}
      className="project-card-link"
      aria-label={`Open project ${project.name}`}
    >
      <div className={`project-card ${master ? 'project-card--master' : ''} ${incomplete ? 'project-card--incomplete' : ''}`}>
        <div className="project-card__top">
          <span className="project-card__id">{project.id}</span>
          <span className={`project-card__status ${statusClass(project.status)}`}>
            {project.status}
          </span>
        </div>

        <div className="project-card__name">
          {project.name}
          {master && (
            <span className="project-card__badge project-card__badge--master">
              Master Corpus
            </span>
          )}
          {incomplete && (
            <span className="project-card__badge project-card__badge--warning">
              Incomplete shell
            </span>
          )}
        </div>

        {project.client && (
          <div className="project-card__client">{project.client}</div>
        )}

        <div className="project-card__footer">
          <span className="project-card__date">
            {formatDate(project.created_at)}
          </span>
          {hint && (
            <span className="project-card__readiness">{hint}</span>
          )}
        </div>

        <button
          type="button"
          className="project-card__delete"
          onClick={handleDelete}
          aria-label={`Delete project ${project.name}`}
        >
          Delete
        </button>
      </div>
    </Link>
  )
}
