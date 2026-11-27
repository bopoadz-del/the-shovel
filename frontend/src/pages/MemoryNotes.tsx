/* MemoryNotes — the owner reads what the model taught itself and calibrates it.
 *
 * One admin section: pick a project, see each of the model's own notes
 * (name, date written, turn, whole content, the owner's mark) with Keep and
 * Discard. A discarded note stays in the table, marked, and the model's
 * read_memory / list_memory no longer return it. Below, every decision ever
 * made, with the content it was made on. The screen never edits a note.
 *
 *   GET  /v1/projects                                   → project picker
 *   GET  /v1/admin/projects/{id}/memory                 → notes + marks
 *   POST /v1/admin/projects/{id}/memory/{name}/review   → keep / discard
 *   GET  /v1/admin/projects/{id}/memory/reviews         → decision record
 */
import { useCallback, useEffect, useState } from 'react'
import { apiGet, apiPost, ApiError } from '../lib/api'

export interface MemoryNote {
  name: string
  data: unknown
  bytes: number
  written_at: string
  written_by_turn: string
  review: string
  reviewed_at: string
  reviewed_by: string
}

export interface MemoryDecision {
  name: string
  decision: string
  data: unknown
  note_written_at: string
  decided_at: string
  decided_by: string
}

interface ProjectOption {
  id: string
  name: string
}

const MARK_WORDS: Record<string, string> = {
  '': 'Not reviewed',
  kept: 'Kept',
  discarded: 'Discarded — the model no longer sees it',
}

function day(iso: string): string {
  return iso ? iso.replace('T', ' ').slice(0, 16) : ''
}

export default function MemoryNotes({ initialProjectId = '' }: { initialProjectId?: string }) {
  const [projects, setProjects] = useState<ProjectOption[]>([])
  const [projectId, setProjectId] = useState(initialProjectId)
  const [notes, setNotes] = useState<MemoryNote[] | null>(null)
  const [decisions, setDecisions] = useState<MemoryDecision[]>([])
  const [message, setMessage] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  useEffect(() => {
    apiGet<{ projects: ProjectOption[] }>('/v1/projects')
      .then((body) => setProjects(body.projects ?? []))
      .catch(() => setProjects([]))
  }, [])

  const load = useCallback(async (pid: string) => {
    if (!pid) return
    setMessage(null)
    try {
      const enc = encodeURIComponent(pid)
      const [n, r] = await Promise.all([
        apiGet<{ notes: MemoryNote[] }>(`/v1/admin/projects/${enc}/memory`),
        apiGet<{ reviews: MemoryDecision[] }>(`/v1/admin/projects/${enc}/memory/reviews`),
      ])
      setNotes(n.notes)
      setDecisions(r.reviews)
    } catch (err) {
      setNotes(null)
      setMessage(err instanceof ApiError ? err.message : 'Could not load the notes.')
    }
  }, [])

  useEffect(() => { void load(projectId) }, [load, projectId])

  const decide = async (name: string, decision: 'keep' | 'discard') => {
    setBusy(name)
    setMessage(null)
    try {
      await apiPost(
        `/v1/admin/projects/${encodeURIComponent(projectId)}/memory/${encodeURIComponent(name)}/review`,
        { decision },
      )
      await load(projectId)
    } catch (err) {
      setMessage(err instanceof ApiError ? err.message : 'Could not record the decision.')
    } finally {
      setBusy(null)
    }
  }

  return (
    <section className="admin-section" aria-label="The model's notes">
      <header className="admin-section__head">
        <h2 className="admin-section__title">The model's notes</h2>
      </header>
      <p className="admin-empty">
        What the model wrote down for itself about how to work in a project. Keep a note, or
        discard it so the model no longer reads it. Figures always come from the tools, never from a note.
      </p>
      <div className="admin-cta">
        <label htmlFor="memory-project">Project</label>
        <select id="memory-project" value={projectId} onChange={(e) => setProjectId(e.target.value)}>
          <option value="">Choose a project</option>
          {projectId && !projects.some((p) => p.id === projectId) && (
            <option value={projectId}>{projectId}</option>
          )}
          {projects.map((p) => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
      </div>
      {message && <p className="admin-alert" role="alert">{message}</p>}
      {notes && notes.length === 0 && <p className="admin-empty">The model has written no notes in this project.</p>}
      {notes && notes.length > 0 && (
        <ul className="admin-list memory-notes">
          {notes.map((n) => (
            <li key={n.name} className="memory-note" data-note={n.name} data-review={n.review}>
              <div className="memory-note__head">
                <strong>{n.name}</strong>
                <span> · written {day(n.written_at)}</span>
                {n.written_by_turn && <span> · turn {n.written_by_turn}</span>}
                <span className="memory-note__mark"> · {MARK_WORDS[n.review] ?? n.review}</span>
              </div>
              <pre className="memory-note__data">{JSON.stringify(n.data, null, 2)}</pre>
              <div className="memory-note__actions">
                <button type="button" className="admin-btn" disabled={busy !== null || n.review === 'kept'}
                        onClick={() => void decide(n.name, 'keep')}>Keep</button>
                <button type="button" className="admin-btn" disabled={busy !== null || n.review === 'discarded'}
                        onClick={() => void decide(n.name, 'discard')}>Discard</button>
              </div>
            </li>
          ))}
        </ul>
      )}
      {decisions.length > 0 && (
        <details className="memory-decisions">
          <summary>Every decision ({decisions.length})</summary>
          <ol>
            {decisions.map((d, i) => (
              <li key={`${d.name}-${i}`}>
                {day(d.decided_at)} · {d.decision === 'discard' ? 'Discarded' : 'Kept'} {d.name}
                {d.decided_by && <> by {d.decided_by}</>} (version written {day(d.note_written_at)})
                <pre className="memory-note__data">{JSON.stringify(d.data, null, 2)}</pre>
              </li>
            ))}
          </ol>
        </details>
      )}
    </section>
  )
}
