/* MemoryNotes — the owner reads what the model taught itself and calibrates it.
 *
 * One admin section: pick a project, see each of the model's own notes
 * (name, date written, turn, whole content, the owner's mark) with Keep and
 * Discard. A discarded note stays in the table, marked, and the model's
 * read_memory / list_memory no longer return it. Below, every decision ever
 * made, with the content it was made on. The screen never edits a note.
 *
 * A decision names the version of the note on screen; when the model wrote
 * the note again since, the server refuses (409) and the screen reloads. A
 * list only shows for the project it was loaded for: a slower answer for a
 * project no longer chosen is dropped.
 *
 *   GET  /v1/projects                                   → project picker
 *   GET  /v1/admin/projects/{id}/memory                 → notes + marks
 *   POST /v1/admin/projects/{id}/memory/{name}/review   → keep / discard
 *   GET  /v1/admin/projects/{id}/memory/reviews         → decision record
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { apiGet, apiPost, ApiError } from '../lib/api'
import { turnWords, type TurnLabel } from './memoryTurn'

export interface MemoryNote {
  name: string
  data: unknown
  bytes: number
  written_at: string
  written_by_turn: string
  turn?: TurnLabel
  review: string
  reviewed_at: string
  reviewed_by: string
  version: string
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
  // The list and the record, with the project they were loaded for.
  const [loaded, setLoaded] = useState<{ projectId: string, notes: MemoryNote[] | null,
    decisions: MemoryDecision[] }>({ projectId: '', notes: null, decisions: [] })
  const wanted = useRef(initialProjectId)
  const [message, setMessage] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  useEffect(() => {
    apiGet<{ projects: ProjectOption[] }>('/v1/projects')
      .then((body) => setProjects(body.projects ?? []))
      .catch(() => setProjects([]))
  }, [])

  const load = useCallback(async (pid: string, keepMessage = false) => {
    if (!pid) return
    if (!keepMessage) setMessage(null)
    try {
      const enc = encodeURIComponent(pid)
      const [n, r] = await Promise.all([
        apiGet<{ notes: MemoryNote[] }>(`/v1/admin/projects/${enc}/memory`),
        apiGet<{ reviews: MemoryDecision[] }>(`/v1/admin/projects/${enc}/memory/reviews`),
      ])
      if (wanted.current !== pid) return
      setLoaded({ projectId: pid, notes: n.notes, decisions: r.reviews })
    } catch (err) {
      if (wanted.current !== pid) return
      setLoaded({ projectId: pid, notes: null, decisions: [] })
      setMessage(err instanceof ApiError ? err.message : 'Could not load the notes.')
    }
  }, [])

  useEffect(() => {
    wanted.current = projectId
    void load(projectId)
  }, [load, projectId])

  const shown = loaded.projectId === projectId ? loaded : { projectId, notes: null, decisions: [] }
  const notes = shown.notes
  const decisions = shown.decisions

  const decide = async (note: MemoryNote, decision: 'keep' | 'discard') => {
    const pid = shown.projectId
    setBusy(note.name)
    setMessage(null)
    try {
      await apiPost(
        `/v1/admin/projects/${encodeURIComponent(pid)}/memory/${encodeURIComponent(note.name)}/review`,
        { decision, version: note.version },
      )
      await load(pid)
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setMessage(`The model wrote ${note.name} again since this list was loaded; `
          + 'the version shown now is the new one. Nothing was recorded.')
        await load(pid, true)
      } else {
        setMessage(err instanceof ApiError ? err.message : 'Could not record the decision.')
      }
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
                {turnWords(n.turn) && <span> · {turnWords(n.turn)}</span>}
                <span className="memory-note__mark"> · {MARK_WORDS[n.review] ?? n.review}</span>
              </div>
              <pre className="memory-note__data">{JSON.stringify(n.data, null, 2)}</pre>
              <div className="memory-note__actions">
                <button type="button" className="admin-btn" disabled={busy !== null || n.review === 'kept'}
                        onClick={() => void decide(n, 'keep')}>Keep</button>
                <button type="button" className="admin-btn" disabled={busy !== null || n.review === 'discarded'}
                        onClick={() => void decide(n, 'discard')}>Discard</button>
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
                {' '}(version written {day(d.note_written_at)})
                <pre className="memory-note__data">{JSON.stringify(d.data, null, 2)}</pre>
              </li>
            ))}
          </ol>
        </details>
      )}
    </section>
  )
}
