/* LeftPanel — Quarry design, wired to real backend (PR #104).
 *
 * Sections, top-to-bottom:
 *   • Brand: "The Shovel"
 *   • PROJECTS — real list from /v1/projects, active row highlighted.
 *   • DOCUMENTS — the active project's uploaded files. Slot-rendered
 *     so the existing DocumentsPanel (in ProjectWorkspace) provides
 *     upload + delete + status. Hidden when no project is active.
 *   • CONVERSATION — message count plus Export and Clear for the
 *     session open in the composer.
 *   • CHAT HISTORY — this project's sessions for the signed-in user,
 *     newest first, each renameable by its owner.
 *   • Sign out — bottom of rail.
 */
import { useEffect, useState, type ReactNode } from 'react'
import { Plus, LogOut, Download, RotateCcw, Settings, MessageSquare, Pencil } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { apiGet, ApiError } from '../lib/api'
import { renameTitleError, sessionListTitle } from '../chat/session'
import './LeftPanel.css'

interface ProjectRow {
  id: string
  name: string
  /** Origin of the project. Only admin-approved Drive projects (and the
   *  injected master corpus) appear in the sidebar. */
  origin?: string
  is_master_corpus?: boolean
}

/** Sidebar rows: the Master Corpus, the user's own projects, and always the
 *  project currently open. Hide Drive-approved backing shells so they do not
 *  duplicate Master Corpus. A leftover live workspace showed "No projects yet"
 *  while Leftover Hat Battery was open because this filter kept only the corpus. */
function isSidebarVisible(p: ProjectRow, activeProjectId?: string): boolean {
  if (activeProjectId && p.id === activeProjectId) return true
  if (p.is_master_corpus === true) return true
  if (p.origin === 'admin_drive_approved') return false
  return true
}

interface ProjectsResponse {
  projects: ProjectRow[]
}

interface Props {
  /** Active project id — drives Projects highlight + visibility of Documents
   *  and Conversation sections. */
  activeProjectId?: string
  /** Active project name — used as the fallback active row while the
   *  /v1/projects fetch is in flight. */
  activeProjectName?: string
  /** DocumentsPanel rendered by the caller (ProjectWorkspace) so the
   *  existing upload + delete wiring is reused. Optional: render the
   *  Documents section only when both this and activeProjectId are set. */
  documents?: ReactNode
  /** Number of messages in the active conversation. Drives the "X messages"
   *  label and gates the Export + Clear actions. */
  messageCount?: number
  /** Export the active conversation as a docx. */
  onExportConversation?: () => void
  /** Clear the active conversation server-side + reset the UI. */
  onClearConversation?: () => void
  /** Chat sessions for the CHAT HISTORY section (Quarry parity). */
  conversations?: Array<{
    id: string
    title: string | null
    updated_at: string | null
  }>
  /** Currently open session id. */
  activeConversationId?: string | null
  /** Open a past session. */
  onSelectConversation?: (id: string) => void
  /** Start a fresh session. */
  onNewConversation?: () => void
  /** Persist a new name for a session the caller owns. */
  onRenameConversation?: (id: string, title: string) => Promise<void>
}

/** "Today" / "Yesterday" / "N days ago" — matches the standalone's history
 *  labels without pulling in a date library. */
function relativeDay(iso: string | null): string {
  if (!iso) return ''
  const then = new Date(iso)
  if (Number.isNaN(then.getTime())) return ''
  const days = Math.floor((Date.now() - then.getTime()) / 86_400_000)
  if (days <= 0) return 'Today'
  if (days === 1) return 'Yesterday'
  if (days < 7) return `${days} days ago`
  return then.toLocaleDateString()
}

type LoadState =
  | { tag: 'loading' }
  | { tag: 'error'; message: string }
  | { tag: 'loaded'; projects: ProjectRow[] }

function withActiveProject(
  rows: ProjectRow[],
  activeProjectId?: string,
  activeProjectName?: string,
): ProjectRow[] {
  if (
    activeProjectId
    && activeProjectName
    && !rows.some((p) => p.id === activeProjectId)
  ) {
    return [{ id: activeProjectId, name: activeProjectName }, ...rows]
  }
  return rows
}

export default function LeftPanel({
  activeProjectId,
  activeProjectName,
  documents,
  messageCount,
  onExportConversation,
  onClearConversation,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewConversation,
  onRenameConversation,
}: Props) {
  const { logout, user, loading: authLoading } = useAuth()
  const [state, setState] = useState<LoadState>({ tag: 'loading' })
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameDraft, setRenameDraft] = useState('')
  const [renameError, setRenameError] = useState<string | null>(null)

  useEffect(() => {
    if (authLoading || !user) return
    let cancelled = false
    async function load() {
      try {
        const data = await apiGet<ProjectsResponse>('/v1/projects')
        if (cancelled) return
        const visible = (data.projects ?? []).filter((p) =>
          isSidebarVisible(p, activeProjectId),
        )
        setState({ tag: 'loaded', projects: visible })
      } catch (err: unknown) {
        if (cancelled) return
        const msg = err instanceof ApiError ? err.message
          : err instanceof Error ? err.message
          : 'Failed to load projects.'
        setState({ tag: 'error', message: msg })
      }
    }
    void load()
    return () => { cancelled = true }
  }, [activeProjectId, user, authLoading])

  function renderProjectsBody() {
    if (!authLoading && !user) {
      return <p className="left-panel__empty">No projects yet.</p>
    }
    const rows = withActiveProject(
      state.tag === 'loaded' ? state.projects : [],
      activeProjectId,
      activeProjectName,
    )
    // Never show "No projects yet" while a project workspace is open.
    if (rows.length > 0) {
      return (
        <ul className="left-panel__list">
          {rows.map((p) => {
            const isActive = p.id === activeProjectId
            return (
              <li key={p.id}>
                <Link
                  to={`/projects/${p.id}`}
                  className={
                    'left-panel__list-item' +
                    (isActive ? ' left-panel__list-item--active' : '')
                  }
                  aria-current={isActive ? 'page' : undefined}
                >
                  {p.name}
                </Link>
              </li>
            )
          })}
        </ul>
      )
    }
    if (state.tag === 'loading' || authLoading) {
      return <p className="left-panel__empty">Loading…</p>
    }
    if (state.tag === 'error') {
      return <p className="left-panel__empty">Couldn't load projects.</p>
    }
    return <p className="left-panel__empty">No projects yet.</p>
  }

  const showDocsSection = !!activeProjectId && !!documents
  const showConvoSection = !!activeProjectId

  return (
    <div className="left-panel">
      <div className="left-panel__brand">
        <span className="left-panel__brand-text">The Shovel</span>
      </div>

      <section className="left-panel__section">
        <header className="left-panel__section-head">Projects</header>
        {renderProjectsBody()}
        <Link to="/" className="left-panel__new-project">
          <Plus size={14} />
          <span>New project</span>
        </Link>
      </section>

      {showDocsSection && (
        <section className="left-panel__section">
          <header className="left-panel__section-head">Documents</header>
          <div className="left-panel__slot">{documents}</div>
        </section>
      )}

      {showConvoSection && (
        <section className="left-panel__section">
          <header className="left-panel__section-head">Conversation</header>
          {messageCount && messageCount > 0 ? (
            <>
              <p className="left-panel__convo-summary">
                {messageCount} message{messageCount === 1 ? '' : 's'} in the
                current thread.
              </p>
              <div className="left-panel__convo-actions">
                {onExportConversation && (
                  <button
                    type="button"
                    className="left-panel__convo-btn"
                    onClick={onExportConversation}
                    title="Export this conversation as a .docx file"
                  >
                    <Download size={13} />
                    <span>Export</span>
                  </button>
                )}
                {onClearConversation && (
                  <button
                    type="button"
                    className="left-panel__convo-btn left-panel__convo-btn--danger"
                    onClick={onClearConversation}
                    title="Clear server-side history (cannot be undone)"
                  >
                    <RotateCcw size={13} />
                    <span>Clear</span>
                  </button>
                )}
              </div>
            </>
          ) : (
            <p className="left-panel__empty">No messages yet. Start the chat.</p>
          )}
        </section>
      )}

      {activeProjectId && onSelectConversation && (
        <section className="left-panel__section">
          <header className="left-panel__section-head">Chat history</header>
          {onNewConversation && (
            <button
              type="button"
              className="left-panel__convo-btn left-panel__history-new"
              onClick={onNewConversation}
              title="Start a fresh chat session"
            >
              <Plus size={13} />
              <span>New chat</span>
            </button>
          )}
          {conversations && conversations.length > 0 ? (
            <ul className="left-panel__history">
              {conversations.map((c) => (
                <li key={c.id}>
                  {renamingId === c.id ? (
                    <form
                      className="left-panel__history-rename"
                      onSubmit={(event) => {
                        event.preventDefault()
                        if (!onRenameConversation) return
                        const problem = renameTitleError(renameDraft)
                        if (problem) {
                          setRenameError(problem)
                          return
                        }
                        void onRenameConversation(c.id, renameDraft)
                          .then(() => {
                            setRenamingId(null)
                            setRenameError(null)
                          })
                          .catch((err: unknown) => {
                            setRenameError(err instanceof Error ? err.message : 'Could not rename')
                          })
                      }}
                    >
                      <input
                        className="left-panel__history-rename-input"
                        value={renameDraft}
                        maxLength={80}
                        aria-label="Session name"
                        autoFocus
                        onChange={(event) => setRenameDraft(event.target.value)}
                      />
                      <button type="submit" className="left-panel__convo-btn">Save</button>
                      <button
                        type="button"
                        className="left-panel__convo-btn"
                        onClick={() => {
                          setRenamingId(null)
                          setRenameError(null)
                        }}
                      >
                        Cancel
                      </button>
                      {renameError ? (
                        <p className="left-panel__empty">{renameError}</p>
                      ) : null}
                    </form>
                  ) : (
                    <div className="left-panel__history-line">
                      <button
                        type="button"
                        className={`left-panel__history-item${
                          c.id === activeConversationId ? ' left-panel__history-item--active' : ''
                        }`}
                        onClick={() => onSelectConversation(c.id)}
                        title={sessionListTitle(c.title)}
                      >
                        <MessageSquare size={12} />
                        <span className="left-panel__history-title">
                          {sessionListTitle(c.title)}
                        </span>
                        <span className="left-panel__history-when">
                          {relativeDay(c.updated_at)}
                        </span>
                      </button>
                      {onRenameConversation ? (
                        <button
                          type="button"
                          className="left-panel__history-rename-btn"
                          aria-label="Rename session"
                          title="Rename session"
                          onClick={() => {
                            setRenamingId(c.id)
                            setRenameDraft(String(c.title ?? '').replace(/\s+/g, ' ').trim())
                            setRenameError(null)
                          }}
                        >
                          <Pencil size={12} />
                        </button>
                      ) : null}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          ) : (
            <p className="left-panel__empty">No past sessions yet.</p>
          )}
        </section>
      )}

      <div className="left-panel__footer">
        {user?.role === 'admin' ? (
          <Link to="/admin" className="left-panel__admin">
            <Settings size={14} />
            <span>Admin</span>
          </Link>
        ) : null}
        <button
          type="button"
          className="left-panel__signout"
          onClick={() => logout()}
        >
          <LogOut size={14} />
          <span>Sign out</span>
        </button>
      </div>
    </div>
  )
}
