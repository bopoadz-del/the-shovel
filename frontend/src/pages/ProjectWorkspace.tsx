/* ProjectWorkspace — project detail + streaming chat + documents/Drive.
 *
 * Post-redesign coordinator: state, SSE wiring, and inline DocumentsPanel +
 * DrivePanel definitions live here. The visual shell (3-column layout, panels,
 * chat bubbles/composer/sources) come from layout/, chat/, and documents/.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import AppHeader from '../components/AppHeader'
import { type Project } from './ProjectCard'
import { apiGet, apiPatch, apiPost, apiPostForm, ApiError } from '../lib/api'
import { getToken } from '../lib/token'
import { exportEndpointForWorkspace } from '../lib/exportEndpoint'
import { ReconnectGaveUp, sendWithReconnect } from '../lib/reconnect'
import WorkspaceShell from '../layout/WorkspaceShell'
import LeftPanel from '../layout/LeftPanel'
import RightPanel from '../layout/RightPanel'
import ChatList from '../chat/ChatList'
import ChatComposer, { type AgentOption } from '../chat/ChatComposer'
import SourcesList from '../chat/SourcesList'
import { sanitizeAssistantContent } from '../chat/toolJsonGuard'
import {
  freshSessionEntropy,
  historyBelongsToSession,
  mintConversationId,
  orderSessionsNewestFirst,
  renameTitleError,
} from '../chat/session'
import { activityForTool, plainMetadata } from '../chat/userFacingText'
import { INTERRUPTED_MESSAGE, isInterruptedStream } from '../lib/streamOutcome'
import DocumentGraph from '../documents/DocumentGraph'
import './pages.css'
import './workspace.css'

// ── Types ──────────────────────────────────────────────────────────────────

interface ProjectDetail extends Project {
  documents?: DocumentRecord[]
  readiness?: ProjectReadiness
}

interface DocumentRecord {
  id: string
  project_id?: string
  original_name: string
  doc_type?: string
  doc_role?: string
  size?: number
  /** True when bytes exist locally or R2/Drive can supply them. */
  has_file?: boolean
  has_remote_source?: boolean
  uploaded_at?: string
  /** Chunks the indexer stored. Zero means the file is not in project search. */
  chunk_count?: number
  /**
   * Why a stored file is absent from project search, taken from the row.
   * Absent when the file is searchable. The panel shows this and does not
   * invent an extraction failure.
   */
  panel_status?: { label: string; detail: string; status?: string } | null
}

interface DriveFile {
  id: string
  name: string
  /** Backend returns `mime_type` (snake_case). Used to render folder
   * rows differently. */
  mime_type?: string
  /** Convenience boolean populated by the backend listing endpoint —
   * true when mime_type === application/vnd.google-apps.folder. */
  is_folder?: boolean
  modified?: string
}

interface FolderCrumb {
  /** '' represents the top of the user's Drive (My Drive root). */
  id: string
  name: string
}

interface DriveStatus {
  connected: boolean
  email?: string | null
  configured: boolean
}

interface ProjectReadiness {
  ready: boolean
  missing?: string[]
  label?: string
  status?: string
  score?: number
  percent?: number
}

type MessageRole = 'user' | 'assistant'

export interface ConversationSummary {
  id: string
  title: string | null
  created_at: string | null
  updated_at: string | null
}

interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  streaming?: boolean
  error?: boolean
  /** Transient tool-activity label shown while the agent is calling tools. Cleared on first token. */
  toolStatus?: string
  /** Top retrieved sources, populated from the SSE 'end' event when RAG injection fired. */
  sources?: Array<{
    doc_id: string
    doc_name: string
    page_or_section: string
    score: number
    confidence: 'High' | 'Medium' | 'Low'
    project_id?: string
    layer?: string
    layer_label?: string
    source_class?: string
    source_class_label?: string
  }>
  /** Data-backed download offers from the SSE 'end' event (e.g. cost BOQ). */
  exports?: Array<{
    label: string
    format: string
    method: string
    endpoint: string
    payload: Record<string, unknown>
  }>
}

// ── Constants ──────────────────────────────────────────────────────────────

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000'

// ── Helpers ────────────────────────────────────────────────────────────────

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

function formatRelativeDate(iso: string): string {
  try {
    const ts = new Date(iso).getTime()
    if (Number.isNaN(ts)) return iso
    const diffSec = Math.round((Date.now() - ts) / 1000)
    if (diffSec < 60) return 'just now'
    const diffMin = Math.round(diffSec / 60)
    if (diffMin < 60) return `${diffMin} min ago`
    const diffHr = Math.round(diffMin / 60)
    if (diffHr < 24) return `${diffHr} hour${diffHr === 1 ? '' : 's'} ago`
    const diffDay = Math.round(diffHr / 24)
    if (diffDay < 7) return `${diffDay} day${diffDay === 1 ? '' : 's'} ago`
    if (diffDay < 30) {
      const w = Math.round(diffDay / 7)
      return `${w} week${w === 1 ? '' : 's'} ago`
    }
    return formatDate(iso)
  } catch {
    return iso
  }
}

/**
 * Classify a filename into a coarse type for the document badge.
 * Returns null for unknown — caller should fall back to ``doc_type`` from the
 * server or render no badge.
 */
function fileTypeBadge(filename: string): { label: string; kind: 'pdf' | 'docx' | 'xlsx' | 'image' | 'text' } | null {
  const ext = filename.toLowerCase().split('.').pop() || ''
  if (ext === 'pdf') return { label: 'PDF', kind: 'pdf' }
  if (ext === 'docx' || ext === 'doc') return { label: 'DOCX', kind: 'docx' }
  if (ext === 'xlsx' || ext === 'xls' || ext === 'csv') return { label: ext.toUpperCase(), kind: 'xlsx' }
  if (['png', 'jpg', 'jpeg', 'webp', 'gif', 'tif', 'tiff', 'bmp'].includes(ext)) return { label: ext.toUpperCase(), kind: 'image' }
  if (['txt', 'md', 'json', 'xml'].includes(ext)) return { label: ext.toUpperCase(), kind: 'text' }
  return null
}

function formatSize(bytes: number, hasFile?: boolean): string {
  if (!bytes || bytes <= 0) return hasFile ? 'stored remotely' : 'no file'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/**
 * Combined readiness + LLM-availability state for the rail badge and the
 * composer send-button. Three mutually-exclusive modes:
 *   setting-up     — backend reports project-execution gates unsatisfied
 *                    (baseline schedule, daily reports, Aconex). Badge only;
 *                    does NOT block chat — the chat surface is functional as
 *                    soon as the project exists.
 *   ai-unavailable — last LLM call errored. Blocks send.
 *   ready          — green light, send enabled.
 */
type ReadinessMode = 'setting-up' | 'ai-unavailable' | 'ready'

function readinessMode(
  readiness: ProjectReadiness | null | undefined,
  llmAvailable: boolean,
): ReadinessMode {
  if (!llmAvailable) return 'ai-unavailable'
  if (!readiness || !readiness.ready) return 'setting-up'
  return 'ready'
}

// readinessModeLabel removed during redesign — the rail badge no longer
// surfaces this label. Tooltip still in use for the composer disabled state.

function readinessModeTooltip(mode: ReadinessMode): string {
  switch (mode) {
    case 'setting-up': return 'Project execution gates (baseline schedule, daily reports, connectors) are not yet satisfied. Chat is available.'
    case 'ai-unavailable': return 'The assistant is temporarily unreachable. Sending will retry once it recovers.'
    case 'ready': return ''
  }
}

/** Send button is gated only on streaming + actual LLM failure, not on
 *  project-execution readiness — chat must work even when the broader
 *  project setup is incomplete. */
function composerBlocked(mode: ReadinessMode): boolean {
  return mode === 'ai-unavailable'
}

function msgId(): string {
  return Math.random().toString(36).slice(2)
}

/**
 * Map raw exception text (Python tracebacks, fetch errors, HTTP status strings)
 * to a single user-safe sentence. We never want to show "Errno -2", stack
 * traces, or "HTTP 502" verbatim — the operator sees that in logs; users see
 * the friendly version.
 */

function friendlyErrorMessage(raw: string): string {
  const r = raw.toLowerCase()
  // OpenRouter in-flight 402 is transient — not a dead account.
  if (r.includes('in-flight') || r.includes('in_flight_budget')) {
    return 'The language model is busy with other requests. Please wait a moment and try again.'
  }
  // Billing 429s include both "429" and "insufficient_quota" / "balance".
  // OpenRouter 402 says "can only afford" / "requires more credits" / "prompt
  // tokens limit" — none of those contain "insufficient", so the old matcher
  // fell through to the generic Wave1 banner.
  // Match credit failures FIRST so they are not mislabeled as rate-limit.
  if (
    r.includes('http 402') ||
    r.includes('payment required') ||
    r.includes('can only afford') ||
    r.includes('fewer max_tokens') ||
    r.includes('prompt tokens limit') ||
    (r.includes('requires more') && r.includes('credit')) ||
    (r.includes('insufficient') &&
      (r.includes('balance') || r.includes('quota') || r.includes('credit') || r.includes('billing'))) ||
    (r.includes('org') && (r.includes('suspend') || r.includes('disabled')))
  ) {
    return 'The language-model account cannot reserve enough tokens for this request. Wait a moment and try again, or top up the provider (OpenRouter / Kimi / Groq).'
  }
  if (r.includes('offline mode') || r.includes('no cloud or local language model')) {
    return 'Chat is running in offline mode. No language model is reachable right now.'
  }
  if (r.includes('invalid temperature') || (r.includes('temperature') && r.includes('only 1'))) {
    return 'The assistant is misconfigured for this model. Please try again in a moment.'
  }
  if (
    r.includes('temporarily unavailable') ||
    r.includes('problem on our side') ||
    r.includes('errno -2') ||
    r.includes('errno -3') ||
    r.includes('name or service not known') ||
    r.includes('getaddrinfo') ||
    r.includes('connection refused') ||
    r.includes('connection error') ||
    r.includes('502') ||
    r.includes('503') ||
    r.includes('504') ||
    r.includes('failed to fetch') ||
    r.includes('llm call failed')
  ) {
    return 'The assistant is temporarily unavailable. Please try again in a moment.'
  }
  if (r.includes('timeout') || r.includes('timed out') || r.includes('etimedout')) {
    return 'The assistant took too long to respond. Please try again.'
  }
  if (r.includes('429') || r.includes('rate limit')) {
    return 'Too many requests right now. Please wait a moment and try again.'
  }
  if (r.includes('401') || r.includes('unauthorized') || r.includes('403')) {
    return 'Your session has expired. Please refresh the page and sign in again.'
  }
  if (r.includes('conversation not found') || r.includes('http 404')) {
    return 'This chat session could not be found. Start a new chat and try again.'
  }
  if (r.includes('aborterror') || r.includes('aborted')) {
    return 'Request was cancelled.'
  }
  return 'Something went wrong. Please try again.'
}

// ── DocumentsPanel ─────────────────────────────────────────────────────────

interface DocumentsPanelProps {
  projectId: string
  documents: DocumentRecord[]
  onDocumentAdded: (doc: DocumentRecord) => void
  onDocumentRemoved: (docId: string) => void
}

// Cap how many document rows we render at once. The master-corpus alias backs
// thousands of documents; rendering them all in one synchronous commit froze
// the workspace on open (a ~2,700-node render blocks the main thread). 200 is
// ample for the pilot; the remainder are summarised in a footer.
const MAX_VISIBLE_DOCS = 200

function DocumentsPanel({
  projectId,
  documents,
  onDocumentAdded,
  onDocumentRemoved,
}: DocumentsPanelProps) {
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    setUploadError(null)
    setUploading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const resp = await apiPostForm<{ document: DocumentRecord }>(
        `/v1/projects/${projectId}/documents`,
        form,
      )
      onDocumentAdded(resp.document)
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed.')
    } finally {
      setUploading(false)
      // Reset the input so the same file can be re-uploaded if needed
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  async function handleDeleteDirect(doc: DocumentRecord) {
    if (!window.confirm(`Delete "${doc.original_name}"? This cannot be undone.`)) return
    setDeletingId(doc.id)
    setDeleteError(null)
    try {
      await deleteDocument(projectId, doc.id)
      onDocumentRemoved(doc.id)
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Delete failed.')
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="docs-panel">
      {documents.length === 0 ? (
        <div className="docs-empty">
          <span className="docs-empty__icon" aria-hidden="true">◫</span>
          <span className="docs-empty__label">No documents yet</span>
        </div>
      ) : (
        <ul className="doc-list" aria-label="Project documents">
          {documents.slice(0, MAX_VISIBLE_DOCS).map((doc) => {
            const typeBadge = fileTypeBadge(doc.original_name)
            const indexNote = doc.panel_status
            return (
              <li key={doc.id} className="doc-row">
                <div className="doc-row__main">
                  <span className="doc-row__name" title={doc.original_name}>
                    {doc.original_name}
                  </span>
                  <span className="doc-row__meta">
                    {indexNote ? (
                      <span
                        className="doc-tag doc-tag--not-indexed"
                        title={indexNote.detail || undefined}
                      >
                        {indexNote.label || 'Not indexed'}
                      </span>
                    ) : typeBadge ? (
                      <span className={`doc-tag doc-tag--type doc-tag--type-${typeBadge.kind}`}>
                        {typeBadge.label}
                      </span>
                    ) : doc.doc_type ? (
                      <span className="doc-tag">{doc.doc_type}</span>
                    ) : null}
                    {doc.doc_role && doc.doc_role !== 'other' && (
                      <span className="doc-tag doc-tag--role">{doc.doc_role.replace('_', ' ')}</span>
                    )}
                  </span>
                </div>
                <div className="doc-row__data">
                  {doc.size != null && (
                    <span className="mono doc-row__size">
                      {formatSize(doc.size ?? 0, doc.has_file || doc.has_remote_source)}
                    </span>
                  )}
                  {doc.uploaded_at && (
                    <span
                      className="doc-row__date"
                      title={formatDate(doc.uploaded_at)}
                    >
                      {formatRelativeDate(doc.uploaded_at)}
                    </span>
                  )}
                  <button
                    type="button"
                    className="doc-row__delete"
                    aria-label={`Delete ${doc.original_name}`}
                    disabled={deletingId === doc.id}
                    onClick={() => void handleDeleteDirect(doc)}
                  >
                    {deletingId === doc.id ? '…' : '×'}
                  </button>
                </div>
              </li>
            )
          })}
          {documents.length > MAX_VISIBLE_DOCS && (
            <li className="doc-row doc-row--overflow">
              Showing {MAX_VISIBLE_DOCS} of {documents.length} documents.
            </li>
          )}
        </ul>
      )}

      <div className="docs-upload">
        <input
          ref={fileInputRef}
          type="file"
          id="doc-file-input"
          className="docs-upload__input"
          onChange={(e) => void handleFileChange(e)}
          disabled={uploading}
          aria-label="Choose file to upload"
        />
        <label
          htmlFor="doc-file-input"
          className={`docs-upload__btn btn btn--ghost${uploading ? ' btn--disabled' : ''}`}
          aria-busy={uploading}
        >
          {uploading ? 'Uploading…' : '+ Upload file'}
        </label>
        {uploadError && (
          <p className="docs-upload__error" role="alert">{uploadError}</p>
        )}
        {deleteError && (
          <p className="docs-upload__error" role="alert">{deleteError}</p>
        )}
      </div>
    </div>
  )
}

// ── deleteDocument helper (DELETE verb not in api.ts) ──────────────────────

async function deleteDocument(projectId: string, documentId: string): Promise<void> {
  const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000'
  const token = getToken()
  const res = await fetch(`${API_BASE}/v1/projects/${projectId}/documents/${documentId}`, {
    method: 'DELETE',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  if (!res.ok) {
    let msg = `HTTP ${res.status}`
    try {
      const body = await res.json() as { detail?: string }
      if (typeof body.detail === 'string') msg = body.detail
    } catch { /* not JSON */ }
    throw new Error(msg)
  }
}

// ── DrivePanel ─────────────────────────────────────────────────────────────

interface DrivePanelProps {
  projectId: string
  onDocumentAdded: (doc: DocumentRecord) => void
}

const DRIVE_OAUTH_ERRORS: Record<string, string> = {
  denied: 'Google consent was declined. Try connecting again.',
  expired_state: 'The connection attempt expired. Try connecting again.',
  invalid_state: 'The connection attempt could not be verified. Try connecting again.',
  missing_code: 'Google returned no authorization code. Try connecting again.',
  exchange_failed: 'Google rejected the connection. Try again or check the Drive configuration.',
}

function driveOauthErrorFromWindow(): string | null {
  if (typeof window === 'undefined') return null
  const params = new URLSearchParams(window.location.search)
  if (params.get('drive') !== 'error') return null
  return DRIVE_OAUTH_ERRORS[params.get('reason') ?? '']
    ?? 'Google Drive connection failed. Try again.'
}

function stripDriveOauthParams(): void {
  if (typeof window === 'undefined') return
  const params = new URLSearchParams(window.location.search)
  if (!params.has('drive') && !params.has('reason')) return
  params.delete('drive')
  params.delete('reason')
  const rest = params.toString()
  window.history.replaceState(null, '', window.location.pathname + (rest ? `?${rest}` : ''))
}

function DrivePanel({ projectId, onDocumentAdded }: DrivePanelProps) {
  const [status, setStatus] = useState<DriveStatus | null>(null)
  const [statusError, setStatusError] = useState<string | null>(driveOauthErrorFromWindow)
  const [connecting, setConnecting] = useState(false)
  const [query, setQuery] = useState('')
  const [driveFiles, setDriveFiles] = useState<DriveFile[]>([])
  const [searching, setSearching] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [importingId, setImportingId] = useState<string | null>(null)
  const [importErrors, setImportErrors] = useState<Record<string, string>>({})
  /** Breadcrumb stack — last entry is the currently-open folder. Empty
   * means the picker isn't in browse mode (search-results view or
   * idle). */
  const [folderStack, setFolderStack] = useState<FolderCrumb[]>([])

  /** Open a Drive folder (or My Drive root when id=''), refresh the
   * file list, and update the breadcrumb stack. `push=true` means we're
   * descending; `push=false` means we're clicking a crumb to go back. */
  async function openFolder(folder: FolderCrumb, push: boolean) {
    setSearchError(null)
    setDriveFiles([])
    setSearching(true)
    try {
      const url = folder.id
        ? `/v1/drive/files?folder_id=${encodeURIComponent(folder.id)}`
        : '/v1/drive/files'
      const resp = await apiGet<{ files: DriveFile[] }>(url)
      setDriveFiles(resp.files)
      setHasSearched(true)
      setFolderStack((prev) =>
        push
          ? [...prev, folder]
          : prev.slice(0, prev.findIndex((c) => c.id === folder.id) + 1),
      )
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Drive browse failed.')
    } finally {
      setSearching(false)
    }
  }

  // Load Drive status on mount
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const s = await apiGet<DriveStatus>('/v1/drive/status')
        if (!cancelled) setStatus(s)
      } catch (err) {
        if (!cancelled) setStatusError(err instanceof Error ? err.message : 'Failed to load Drive status.')
      }
    })()
    return () => { cancelled = true }
  }, [])

  // Surface the OAuth round-trip outcome (?drive=connected / ?drive=error)
  // the callback redirect appends, then strip it from the URL. The error
  // string is read during initial state so this effect only touches history.
  useEffect(() => {
    stripDriveOauthParams()
  }, [])

  async function handleConnect() {
    setConnecting(true)
    try {
      // return_to: the OAuth callback 302s the browser back to THIS page
      // (not the app root), so the user never loses their place.
      const resp = await apiGet<{ auth_url: string }>(
        `/v1/drive/connect?return_to=${encodeURIComponent(window.location.pathname)}`,
      )
      window.location.href = resp.auth_url
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : 'Failed to start Drive connection.')
      setConnecting(false)
    }
  }

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    setSearchError(null)
    setDriveFiles([])
    setHasSearched(false)
    setSearching(true)
    try {
      const resp = await apiGet<{ files: DriveFile[] }>(`/v1/drive/files?q=${encodeURIComponent(query)}`)
      setDriveFiles(resp.files)
      setHasSearched(true)
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Drive search failed.')
    } finally {
      setSearching(false)
    }
  }

  async function handleImport(file: DriveFile) {
    setImportingId(file.id)
    setImportErrors((prev) => { const next = { ...prev }; delete next[file.id]; return next })
    try {
      const resp = await apiPost<{ document: DocumentRecord }>(
        `/v1/projects/${projectId}/drive/import`,
        { file_id: file.id, name: file.name },
      )
      onDocumentAdded(resp.document)
      // Remove from search results to avoid double-import
      setDriveFiles((prev) => prev.filter((f) => f.id !== file.id))
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Import failed.'
      setImportErrors((prev) => ({ ...prev, [file.id]: msg }))
    } finally {
      setImportingId(null)
    }
  }

  if (statusError && !status) {
    return (
      <p className="drive-status-error" role="alert">{statusError}</p>
    )
  }

  if (!status) {
    return <p className="drive-loading">Checking Drive…</p>
  }

  if (!status.configured) {
    return (
      <p className="drive-not-configured">
        Google Drive not configured on this server.
      </p>
    )
  }

  if (!status.connected) {
    return (
      <div className="drive-connect">
        {statusError && <p className="drive-status-error" role="alert">{statusError}</p>}
        <button
          type="button"
          className="btn btn--ghost drive-connect__btn"
          onClick={() => void handleConnect()}
          disabled={connecting}
        >
          {connecting ? 'Redirecting…' : 'Connect Google Drive'}
        </button>
      </div>
    )
  }

  return (
    <div className="drive-connected">
      {status.email && (
        <p className="drive-email">
          Connected as <span className="mono">{status.email}</span>
        </p>
      )}
      <form className="drive-search" onSubmit={(e) => void handleSearch(e)}>
        <input
          type="search"
          className="drive-search__input"
          placeholder="Search Drive files…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search Google Drive"
          disabled={searching}
        />
        <button
          type="submit"
          className="btn btn--ghost drive-search__btn"
          disabled={searching}
          aria-label="Search"
        >
          {searching ? '…' : 'Search'}
        </button>
        <button
          type="button"
          className="btn btn--ghost drive-search__browse"
          disabled={searching}
          onClick={() => void openFolder({ id: '', name: 'My Drive' }, true)}
          aria-label="Browse my Drive"
        >
          Browse
        </button>
      </form>
      {searchError && <p className="drive-search__error" role="alert">{searchError}</p>}
      {folderStack.length > 0 && (
        <nav className="drive-breadcrumbs" aria-label="Drive folder navigation">
          <button
            type="button"
            className="drive-breadcrumb"
            onClick={() => { setFolderStack([]); setDriveFiles([]); setHasSearched(false) }}
          >
            ↑ Back to search
          </button>
          {folderStack.map((crumb, i) => (
            <span key={crumb.id || `crumb-${i}`}>
              <span className="drive-breadcrumb-sep"> / </span>
              <button
                type="button"
                className="drive-breadcrumb"
                onClick={() => void openFolder(crumb, false)}
                disabled={i === folderStack.length - 1 || searching}
              >
                {crumb.name}
              </button>
            </span>
          ))}
        </nav>
      )}
      {driveFiles.length > 0 && (
        <ul className="drive-file-list" aria-label="Google Drive files">
          {driveFiles.slice(0, 300).map((file) => (
            <li
              key={file.id}
              className={`drive-file-row${file.is_folder ? ' drive-file-row--folder' : ''}`}
            >
              <span className="drive-file-row__name" title={file.name}>
                {file.is_folder && (
                  <span className="drive-file-row__folder-marker">[folder]</span>
                )}
                {file.is_folder ? ' ' : ''}{file.name}
              </span>
              <div className="drive-file-row__actions">
                {importErrors[file.id] && (
                  <span className="drive-file-row__error" role="alert">
                    {importErrors[file.id]}
                  </span>
                )}
                {file.is_folder ? (
                  <button
                    type="button"
                    className="btn btn--ghost drive-file-row__open"
                    disabled={searching}
                    onClick={() => void openFolder({ id: file.id, name: file.name }, true)}
                  >
                    Open
                  </button>
                ) : (
                  <button
                    type="button"
                    className="btn btn--ghost drive-file-row__add"
                    disabled={importingId === file.id}
                    onClick={() => void handleImport(file)}
                  >
                    {importingId === file.id ? '…' : 'Add'}
                  </button>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
      {!searching && hasSearched && driveFiles.length === 0 && !searchError && (
        <p className="drive-no-results">No files found.</p>
      )}
    </div>
  )
}

// ── ProjectWorkspace ────────────────────────────────────────────────────────

type WorkspaceState =
  | { tag: 'loading' }
  | { tag: 'not-found' }
  | { tag: 'error'; message: string }
  | { tag: 'ready'; project: ProjectDetail }

export default function ProjectWorkspace() {
  const { id } = useParams<{ id: string }>()
  return <ProjectWorkspaceInner key={id ?? 'none'} id={id} />
}

function ProjectWorkspaceInner({ id }: { id: string | undefined }) {

  const [wsState, setWsState] = useState<WorkspaceState>(
    () => (id ? { tag: 'loading' } : { tag: 'not-found' }),
  )
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [streaming, setStreaming] = useState(false)
  // LLM availability — flipped false when a stream errors, true again once a
  // stream completes cleanly. Drives the rail badge's "AI unavailable" state.
  const [llmAvailable, setLlmAvailable] = useState(true)

  // PR #101 / Quarry redesign:
  //   rightExpanded — right panel covers main + left as a full-width overlay
  //   driveModalOpen — DrivePanel rendered as a modal (opened from + popover)
  const [rightExpanded, setRightExpanded] = useState(false)
  const [driveModalOpen, setDriveModalOpen] = useState(false)
  const [previewRequest, setPreviewRequest] = useState<{ docId: string; nonce: number } | null>(null)

  // ── / agent-picker: available agents + the user's pinned choice ───────────
  // A pinned agent is sent as `agent` on /v1/chat/stream and runs directly
  // (bypassing the auto-router). null = automatic routing (today's behavior).
  const [agents, setAgents] = useState<AgentOption[]>([])
  const [pinnedAgent, setPinnedAgent] = useState<string | null>(null)
  useEffect(() => {
    let cancelled = false
    apiGet<{ agents: AgentOption[] }>('/v1/agents')
      .then((r) => { if (!cancelled) setAgents(r.agents ?? []) })
      .catch(() => { /* picker is optional — silent on failure */ })
    return () => { cancelled = true }
  }, [])

  // Opening a project and "New chat" each mint a new id. The legacy
  // stable id is never the composer thread, so the first send cannot
  // append to an older conversation. Past sessions are reopened from
  // the list. The list is GET /v1/projects/{id}/conversations.
  const [activeConversationId, setActiveConversationId] = useState<string | null>(() =>
    id ? mintConversationId(id, freshSessionEntropy()) : null,
  )
  const [conversationList, setConversationList] = useState<ConversationSummary[]>([])
  const conversationId = activeConversationId
  const activeIdRef = useRef<string | null>(activeConversationId)
  activeIdRef.current = activeConversationId

  const refreshConversations = useCallback(() => {
    if (!id) return
    apiGet<{ conversations: ConversationSummary[] }>(`/v1/projects/${id}/conversations`)
      .then((r) => setConversationList(orderSessionsNewestFirst(r.conversations ?? [])))
      .catch(() => { /* history list is progressive enhancement */ })
  }, [id])

  // Mirror messages into a ref so handleSend can read current history without
  // being a stale closure or needing messages in its dependency array.
  const messagesRef = useRef<ChatMessage[]>([])
  useEffect(() => { messagesRef.current = messages }, [messages])

  // Abort controller — cancel in-flight stream on unmount or re-send
  const abortRef = useRef<AbortController | null>(null)

  // Synchronous in-flight guard. The `streaming` STATE lags a render behind the
  // send, so a second message fired in that window slipped past `if (streaming)`
  // and ran handleSend again — whose abortRef.abort() KILLED the first turn,
  // which then vanished with no answer (the "rapid-send drop"). A ref flips
  // synchronously, so a racing second send is rejected before it can touch the
  // in-flight stream.
  const streamingRef = useRef(false)

  // S4: document_ids attached via the composer since the last send. The
  // upload response's real doc id is kept here and shipped alongside the
  // inline [attached: name] marker (which the backend also parses as a
  // fallback), then cleared once the send goes out.
  const pendingAttachedDocIdsRef = useRef<string[]>([])

  // ── Load project ──────────────────────────────────────────────────────────

  useEffect(() => {
    if (!id) return
    refreshConversations()
    let cancelled = false
    void (async () => {
      try {
        const project = await apiGet<ProjectDetail>(`/v1/projects/${id}`)
        if (cancelled) return
        setWsState({ tag: 'ready', project })
        setDocuments(project.documents ?? [])
      } catch (err) {
        if (cancelled) return
        if (err instanceof ApiError && err.status === 404) {
          setWsState({ tag: 'not-found' })
        } else {
          setWsState({
            tag: 'error',
            message: err instanceof Error ? err.message : 'Failed to load project.',
          })
        }
      }
    })()
    return () => { cancelled = true }
  }, [id, refreshConversations])

  // ── Chat session switching (Quarry CHAT HISTORY) ──────────────────────────

  const handleSelectConversation = useCallback((convId: string) => {
    abortRef.current?.abort()
    activeIdRef.current = convId
    setActiveConversationId(convId)
    setMessages([])
    apiGet<{ messages: Array<{ role: string; content: string }> }>(
      `/v1/agents/conversations/${convId}/messages`,
    )
      .then((hist) => {
        const loaded = (hist.messages ?? []).map((m) => {
          const role = (m.role === 'user' ? 'user' : 'assistant') as MessageRole
          return {
            id: msgId(),
            role,
            content: role === 'assistant' ? sanitizeAssistantContent(m.content) : m.content,
          }
        })
        const kept = historyBelongsToSession(convId, activeIdRef.current, loaded)
        if (!kept) return
        setMessages(kept)
      })
      .catch(() => { /* empty thread on failure — same as project load */ })
  }, [])

  const handleNewConversation = useCallback(() => {
    if (!id) return
    abortRef.current?.abort()
    const next = mintConversationId(id, freshSessionEntropy())
    activeIdRef.current = next
    setActiveConversationId(next)
    setMessages([])
  }, [id])

  const handleRenameConversation = useCallback(async (convId: string, title: string) => {
    const problem = renameTitleError(title)
    if (problem) throw new Error(problem)
    if (!id) return
    const updated = await apiPatch<{ title: string }>(
      `/v1/projects/${id}/conversations/${encodeURIComponent(convId)}`,
      { title },
    )
    setConversationList((prev) =>
      prev.map((row) => (row.id === convId ? { ...row, title: updated.title } : row)),
    )
  }, [id])

  // ── Document mutation callbacks ───────────────────────────────────────────

  const handleDocumentAdded = useCallback((doc: DocumentRecord) => {
    setDocuments((prev) => [...prev, doc])
  }, [])

  // PR #104: DocumentsPanel is back in the LeftPanel — handler used again.
  const handleDocumentRemoved = useCallback((docId: string) => {
    setDocuments((prev) => prev.filter((d) => d.id !== docId))
  }, [])

  const handlePreviewHydrated = useCallback((docId: string, size: number) => {
    if (!docId || size <= 0) return
    setDocuments((prev) =>
      prev.map((d) =>
        d.id === docId ? { ...d, size, has_file: true } : d,
      ),
    )
  }, [])

  // Cleanup on unmount
  useEffect(() => () => { abortRef.current?.abort() }, [])

  // ── Send + stream ─────────────────────────────────────────────────────────

  const handleSend = useCallback(async (userText: string) => {
    if (streamingRef.current || streaming) return
    streamingRef.current = true

    // Snapshot complete prior turns for history (before this user turn)
    const priorMessages = messagesRef.current.filter((m) => !m.streaming && !m.error)
    const historyForRequest = priorMessages.map((m) => ({
      role: m.role,
      content: m.content,
    }))

    // Append user message + empty streaming assistant bubble.
    // Drop any stale error bubbles from a prior turn so they don't linger.
    const userMsgId = msgId()
    const assistantMsgId = msgId()

    setMessages((prev) => [
      ...prev.filter((m) => !m.error),
      { id: userMsgId, role: 'user', content: userText },
      { id: assistantMsgId, role: 'assistant', content: '', streaming: true },
    ])
    setStreaming(true)

    // Cancel any previous in-flight stream
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    // Reader-side silence deadline (FOLLOW-UP #92). Resets on every
    // chunk read — token, heartbeat, or tool event all count as proof of
    // life. If 95s pass with no bytes from the server, abort the fetch and
    // surface a friendly timeout banner instead of an indefinite spinner.
    // This is NOT the server wall-clock (CHAT_STREAM_TIMEOUT_SECONDS, 240s
    // default). Heartbeats every CHAT_STREAM_HEARTBEAT_SECONDS keep this
    // timer reset for the whole Kimi reasoning burst.
    const READER_TIMEOUT_MS = 95_000
    let didTimeout = false
    let readerTimer: ReturnType<typeof setTimeout> | null = null
    const resetReaderDeadline = () => {
      if (readerTimer !== null) clearTimeout(readerTimer)
      readerTimer = setTimeout(() => {
        didTimeout = true
        controller.abort()
      }, READER_TIMEOUT_MS)
    }
    const clearReaderDeadline = () => {
      if (readerTimer !== null) {
        clearTimeout(readerTimer)
        readerTimer = null
      }
    }

    /** Build a human-readable status label for an agent tool_call event. */
    function toolStatusLabel(toolName: string, argsPreview: string): string {
      let args: Record<string, unknown> = {}
      try {
        args = JSON.parse(argsPreview) as Record<string, unknown>
      } catch { /* truncated JSON — ignore */ }

      switch (toolName) {
        case 'search_project_documents': {
          const q = typeof args['query'] === 'string' ? args['query'] : ''
          return q ? `Searching documents for "${q}"…` : 'Searching documents…'
        }
        case 'delegate_to_agent': {
          const agent = typeof args['agent_name'] === 'string' ? args['agent_name'] : ''
          const shown = plainMetadata(agent)
          return shown ? `Delegating to ${shown}…` : 'Delegating to specialist…'
        }
        case 'remember_fact':
          return 'Saving fact to memory…'
        default:
          return activityForTool(toolName)
      }
    }

    try {
      // Repointed to the ORCHESTRATOR front door (not the agent directly). The
      // orchestrator decides: predefined reasoning for a known workflow, else it
      // dispatches to the project-assistant agent (which self-routes generative
      // turns to heavy-reasoning). Everything flows through one entry point.
      // A deploy swaps the single instance and the site answers 502 for a
      // while. Ride through it: see lib/reconnect.js for what is safe to retry.
      const res = await sendWithReconnect(() => fetch(`${API_BASE}/v1/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(getToken() ? { Authorization: `Bearer ${getToken()}` } : {}),
        },
        body: JSON.stringify({
          message: userText,
          project_id: id ?? null,
          conversation_id: conversationId,
          session_id: conversationId ?? undefined,
          history: historyForRequest,
          // / agent-picker: omit (or "auto") = today's automatic routing.
          agent: pinnedAgent ?? undefined,
          // S4: real ids for files attached this turn (marker is the fallback).
          document_ids: pendingAttachedDocIdsRef.current.length
            ? [...pendingAttachedDocIdsRef.current]
            : undefined,
        }),
        signal: controller.signal,
      }),
      async () => (await fetch(`${API_BASE}/livez`, { signal: controller.signal })).ok,
      {
        signal: controller.signal,
        onReconnecting: () =>
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, toolStatus: 'Reconnecting… your message will be sent as soon as the assistant is back.' }
                : m
            )
          ),
      })
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantMsgId ? { ...m, toolStatus: undefined } : m))
      )
      // Sent (ok or not) — the server-side marker parser covers any retry.
      pendingAttachedDocIdsRef.current = []

      if (!res.ok || !res.body) {
        let errMsg = `HTTP ${res.status}`
        try {
          const body = await res.json() as { detail?: string }
          if (typeof body.detail === 'string') errMsg = body.detail
        } catch { /* not JSON */ }
        throw new Error(errMsg)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let sseBuffer = ''
      let accumulatedContent = ''
      let firstTokenReceived = false
      // 'end' is terminal — set on receipt so we stop reading immediately
      // instead of waiting for the server to close the connection (which left
      // the global "typing" spinner + a placeholder bubble hanging after the
      // answer had already rendered).
      let streamEnded = false

      // Arm the wall-clock deadline now that the response has started.
      resetReaderDeadline()

      // Read the stream chunk by chunk
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break

        // Do not reset on raw bytes (partial SSE). Reset per parsed event
        // below, including heartbeats — those fire every ~15s while Kimi is
        // still reasoning. Ignoring them made the UI abort at 95s with
        // "took too long" on multi-tool RFP turns that the server (240s)
        // was still producing.
        sseBuffer += decoder.decode(value, { stream: true })

        // SSE events are separated by \n\n
        const events = sseBuffer.split('\n\n')
        // Last element may be a partial event — keep it in the buffer
        sseBuffer = events.pop() ?? ''

        for (const eventBlock of events) {
          for (const line of eventBlock.split('\n')) {
            if (!line.startsWith('data:')) continue
            const jsonStr = line.slice('data:'.length).trim()
            if (!jsonStr) continue

            let evt: Record<string, unknown>
            try {
              evt = JSON.parse(jsonStr) as Record<string, unknown>
            } catch {
              continue // malformed line — skip
            }

            const evtType = evt['type'] as string | undefined

            // Any parsed event is proof of life, including heartbeat.
            // A truly stalled TCP stream (no events for READER_TIMEOUT_MS)
            // still aborts. A heartbeat-only reasoning burst must not.
            if (evtType) resetReaderDeadline()

            if (evtType === 'start') {
              // Agent stream start — {type, agent}. No session_id to echo back.
            } else if (evtType === 'heartbeat') {
              // Server is alive but the LLM hasn't produced a token yet.
              // Deadline already reset above; render nothing.
            } else if (evtType === 'status') {
              // Turn progress (accepted, searching, writing): shown in the
              // bubble until the first token, so it is never blank.
              const label = typeof evt['label'] === 'string' ? evt['label'] : ''
              if (label) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsgId && !m.content
                      ? { ...m, toolStatus: label }
                      : m
                  )
                )
              }
            } else if (evtType === 'tool_call') {
              // Show ephemeral status inside the assistant bubble while tools run.
              const toolName = typeof evt['tool'] === 'string' ? evt['tool'] : 'tool'
              const argsPreview = typeof evt['args_preview'] === 'string' ? evt['args_preview'] : ''
              const status = toolStatusLabel(toolName, argsPreview)
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId
                    ? { ...m, toolStatus: status }
                    : m
                )
              )
            } else if (evtType === 'tool_result') {
              // Tool finished — clear the status; real tokens follow.
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId
                    ? { ...m, toolStatus: undefined }
                    : m
                )
              )
            } else if (evtType === 'token') {
              const token = typeof evt['content'] === 'string' ? evt['content'] : ''
              // Clear toolStatus on the first real token so no ghost label appears.
              if (!firstTokenReceived) {
                firstTokenReceived = true
                accumulatedContent += token
                const snap = sanitizeAssistantContent(accumulatedContent)
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsgId
                      ? { ...m, content: snap, streaming: true, toolStatus: undefined }
                      : m
                  )
                )
              } else {
                accumulatedContent += token
                const snap = sanitizeAssistantContent(accumulatedContent)
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsgId
                      ? { ...m, content: snap, streaming: true }
                      : m
                  )
                )
              }
            } else if (evtType === 'end') {
              streamEnded = true
              // Prefer the server's authoritative sanitized content when present
              // (covers SYNTHESIS_STREAMING races where raw tool JSON was flushed
              // into accumulated tokens before sanitize).
              const serverContent =
                typeof evt['content'] === 'string' ? evt['content'] : ''
              const finalContent = sanitizeAssistantContent(
                serverContent || accumulatedContent,
              )
              const rawSources = Array.isArray(evt['sources']) ? (evt['sources'] as ChatMessage['sources']) : []
              const rawExports = Array.isArray(evt['exports']) ? (evt['exports'] as ChatMessage['exports']) : []
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId
                    ? { ...m, content: finalContent, streaming: false, toolStatus: undefined, sources: rawSources, exports: rawExports }
                    : m
                )
              )
              setLlmAvailable(true)
            } else if (evtType === 'error') {
              const rawMsg =
                typeof evt['message'] === 'string'
                  ? evt['message']
                  : 'stream error'
              const friendly = friendlyErrorMessage(rawMsg)
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId
                    ? { ...m, content: friendly, streaming: false, error: true, toolStatus: undefined }
                    : m
                )
              )
              // Error bubble persists until the user sends a new message or
              // clears the conversation — no 8s self-delete, which made a
              // failure look like a silent hang. Unlock the composer so the
              // user can retry immediately.
              setLlmAvailable(true)
            }
          }
        }
        // Terminal 'end' event seen — stop reading now (don't wait for the
        // server to close the socket, which left the spinner hanging).
        if (streamEnded) {
          try { await reader.cancel() } catch { /* already closing */ }
          clearReaderDeadline()
          break
        }
      }

      // If no explicit 'end' event arrived, finalise anyway. A stream that
      // closed with nothing in it (the server restarted mid-turn, the
      // connection dropped) would otherwise leave an empty answer with no
      // word to the user -- say what happened so they can send it again.
      const interrupted = isInterruptedStream(streamEnded, accumulatedContent)
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId && m.streaming
            ? interrupted
              ? { ...m, content: INTERRUPTED_MESSAGE, streaming: false, error: true, toolStatus: undefined }
              : { ...m, streaming: false, toolStatus: undefined }
            : m
        )
      )
      if (interrupted) setLlmAvailable(true)
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        if (didTimeout) {
          // OUR timeout fired (95s of reader silence) — surface a friendly
          // error rather than the silent-cancel UX, otherwise the user is
          // left with an indefinite spinner that just vanished.
          const friendly = friendlyErrorMessage('timeout')
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, content: friendly, streaming: false, error: true, toolStatus: undefined }
                : m
            )
          )
          // Error bubble persists (no 8s self-delete). Unlock composer to retry.
          setLlmAvailable(true)
          return
        }
        // Intentional cancel — mark assistant message done, keep content, clear tool status
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId && m.streaming
              ? { ...m, streaming: false, toolStatus: undefined }
              : m
          )
        )
        return
      }
      const rawMsg = err instanceof Error ? err.message : 'stream failed'
      const friendly = err instanceof ReconnectGaveUp ? err.message : friendlyErrorMessage(rawMsg)
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? { ...m, content: friendly, streaming: false, error: true, toolStatus: undefined }
            : m
        )
      )
      // Error bubble persists (no 8s self-delete). Unlock composer to retry.
      setLlmAvailable(true)
    } finally {
      clearReaderDeadline()
      streamingRef.current = false
      setStreaming(false)
      // First message stamps the session title server-side — refresh the
      // sidebar CHAT HISTORY list so the new session appears named.
      refreshConversations()
    }
  }, [id, conversationId, streaming, refreshConversations, pinnedAgent])

  // ── Render ────────────────────────────────────────────────────────────────

  const projectName = wsState.tag === 'ready' ? wsState.project.name : (id ?? '—')

  const breadcrumb = (
    <>
      <Link to="/" className="nav-item">Projects</Link>
      <span className="nav-sep">/</span>
      <span className="nav-item nav-item--active mono">{projectName}</span>
    </>
  )

  if (wsState.tag === 'loading') {
    return (
      <div className="workspace-shell page-shell">
        <AppHeader breadcrumb={breadcrumb} />
        <div className="workspace-loading">
          <span className="workspace-loading__label">Loading project…</span>
        </div>
      </div>
    )
  }

  if (wsState.tag === 'not-found') {
    return (
      <div className="workspace-shell page-shell">
        <AppHeader breadcrumb={breadcrumb} />
        <div className="workspace-notfound">
          <span className="workspace-notfound__code">404</span>
          <h2>Project not found</h2>
          <p className="workspace-notfound__message">
            This project doesn&apos;t exist or you don&apos;t have access to it.
          </p>
          <Link to="/" className="btn btn--primary" style={{ textDecoration: 'none' }}>
            ← Back to projects
          </Link>
        </div>
      </div>
    )
  }

  if (wsState.tag === 'error') {
    return (
      <div className="workspace-shell page-shell">
        <AppHeader breadcrumb={breadcrumb} />
        <div className="workspace-notfound">
          <span className="workspace-notfound__code">Error</span>
          <h2>Could not load project</h2>
          <p className="workspace-notfound__message">{wsState.message}</p>
          <Link to="/" className="btn btn--ghost" style={{ textDecoration: 'none' }}>
            ← Back to projects
          </Link>
        </div>
      </div>
    )
  }

  const { project } = wsState
  const mode = readinessMode(project.readiness, llmAvailable)
  const composerIsBlocked = composerBlocked(mode)
  const composerBlockedReason = composerIsBlocked ? readinessModeTooltip(mode) : undefined

  // Latest assistant message's sources drive the right-panel SourcesList and
  // the DocumentGraph "cited" highlight. Falls back to undefined when the
  // last message is the user, an error bubble, or has no sources yet.
  const latestAssistant = [...messages].reverse().find(
    (m) => m.role === 'assistant' && !m.error,
  )
  const latestSources = latestAssistant?.sources ?? []
  const citedDocIds = latestSources.map((s) => s.doc_id)

  // Sources the answer cited that this project does not own — Master-Corpus
  // fallback hits. Deduplicated by doc_id (one document usually contributes
  // several chunks, and the panel lists documents, not chunks). Feeds the
  // DocumentGraph so a project with zero own-documents still shows what the
  // answer was actually grounded in, instead of a bare "No documents yet"
  // beside an answer citing three files.
  // NOT a hook. This sits BELOW the wsState early returns (loading /
  // not-found / error), so a useMemo here is skipped on the loading render
  // and called on the next one — React error #310, "Rendered more hooks than
  // during the previous render", which crashed the whole workspace into the
  // error boundary on load. Shipped that way in #304; caught live.
  //
  // The work is a single pass over at most a handful of sources, so it does
  // not need memoising. Keeping it as a plain computation makes it immune to
  // its position in the component.
  const citedExternalDocs = (() => {
    const owned = new Set(documents.map((d) => d.id))
    const seen = new Set<string>()
    const out: { id: string; original_name: string }[] = []
    for (const s of latestSources) {
      if (!s.doc_id || owned.has(s.doc_id) || seen.has(s.doc_id)) continue
      seen.add(s.doc_id)
      out.push({ id: s.doc_id, original_name: s.doc_name || s.doc_id })
    }
    return out
  })()
  const citedPreviewDocs = (() => {
    const seen = new Set<string>()
    const out: { id: string; original_name: string }[] = []
    for (const s of latestSources) {
      if (!s.doc_id || seen.has(s.doc_id)) continue
      seen.add(s.doc_id)
      out.push({ id: s.doc_id, original_name: s.doc_name || s.doc_id })
    }
    return out
  })()
  void mode  // mode used inside left panel via DocumentsPanel/onClear hooks below

  // PR #104: DocumentsPanel is back, rendered as a slot inside LeftPanel
  // (between Projects and Conversation). Conversation Clear + Export
  // handlers are also lifted out so the LeftPanel can drive them directly
  // without duplicating fetch wiring. Drive still opens as a modal from
  // the ChatComposer's + popover.

  async function exportConversation() {
    if (!id || !conversationId) return
    const token = getToken() || ''
    try {
      const res = await fetch(
        `${API_BASE}/v1/projects/${id}/conversations/${conversationId}/export?format=docx`,
        { method: 'POST', headers: { Authorization: `Bearer ${token}` } },
      )
      if (!res.ok) {
        const detail = await res.text().catch(() => '')
        alert(`Export failed (${res.status}): ${detail.slice(0, 200)}`)
        return
      }
      const blob = await res.blob()
      const a = document.createElement('a')
      const objUrl = URL.createObjectURL(blob)
      a.href = objUrl
      const cd = res.headers.get('Content-Disposition') || ''
      const m = /filename="?([^";]+)"?/.exec(cd)
      a.download = m?.[1] || `the-fork-${conversationId.slice(0, 8)}.docx`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(objUrl)
    } catch (err) {
      alert(`Export error: ${(err as Error).message}`)
    }
  }

  async function clearConversation() {
    if (!id || !conversationId) return
    if (!window.confirm(
      'Clear this conversation? The chat history on the server will be wiped. This cannot be undone.',
    )) return
    try {
      const token = getToken() || ''
      const res = await fetch(
        `${API_BASE}/v1/projects/${id}/conversations/${conversationId}/clear`,
        {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
          body: '{}',
        },
      )
      if (!res.ok) {
        alert(`Clear failed (${res.status})`)
        return
      }
      setMessages([])
    } catch (err) {
      alert(`Clear failed: ${(err as Error).message}`)
    }
  }

  return (
    <>
    <WorkspaceShell
      header={<AppHeader breadcrumb={breadcrumb} />}
      rightExpanded={rightExpanded}
      onRightOpen={() => setRightExpanded(true)}
      left={
        <LeftPanel
          activeProjectId={id}
          activeProjectName={projectName}
          messageCount={messages.length}
          onExportConversation={exportConversation}
          conversations={conversationList}
          activeConversationId={conversationId}
          onSelectConversation={handleSelectConversation}
          onNewConversation={handleNewConversation}
          onRenameConversation={handleRenameConversation}
          onClearConversation={clearConversation}
          documents={
            <DocumentsPanel
              projectId={id ?? ''}
              documents={documents}
              onDocumentAdded={handleDocumentAdded}
              onDocumentRemoved={handleDocumentRemoved}
            />
          }
        />
      }
      main={
        <div className="workspace-main">
          <ChatList
            messages={messages}
            documentCount={documents.length}
            onSuggestion={(text) => void handleSend(text)}
            suggestionsDisabled={streaming || composerIsBlocked}
            onDownloadMessage={(assistantIndex, format) => {
              if (!id || !conversationId) return
              const token = getToken() || ''
              const url = `${API_BASE}/v1/projects/${id}/conversations/${conversationId}/export?format=${format}&message_index=${assistantIndex}`
              void fetch(url, {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
              })
                .then(async (res) => {
                  if (!res.ok) {
                    const detail = await res.text().catch(() => '')
                    alert(`Download failed (${res.status}): ${detail.slice(0, 200)}`)
                    return
                  }
                  const blob = await res.blob()
                  const a = document.createElement('a')
                  const objUrl = URL.createObjectURL(blob)
                  a.href = objUrl
                  const cd = res.headers.get('Content-Disposition') || ''
                  const m = /filename="?([^";]+)"?/.exec(cd)
                  a.download = m?.[1] || `the-fork-${conversationId.slice(0, 8)}.${format}`
                  document.body.appendChild(a)
                  a.click()
                  a.remove()
                  URL.revokeObjectURL(objUrl)
                })
                .catch((e) => alert(`Download error: ${(e as Error).message}`))
            }}
            onExport={(descriptor) => {
              // Data-backed download offer (e.g. cost BOQ / A1–A9 Word). The
              // descriptor's endpoint is server-relative; generation happens
              // server-side on click (bounded), not in the chat hot path.
              // Rewrite a remapped RAG corpus id (drive_archive) to the
              // workspace the user is actually in.
              const token = getToken() || ''
              const endpoint = exportEndpointForWorkspace(descriptor.endpoint, id)
              void fetch(`${API_BASE}${endpoint}`, {
                method: descriptor.method || 'POST',
                headers: {
                  Authorization: `Bearer ${token}`,
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify(descriptor.payload ?? {}),
              })
                .then(async (res) => {
                  if (!res.ok) {
                    const detail = await res.text().catch(() => '')
                    alert(`Export failed (${res.status}): ${detail.slice(0, 200)}`)
                    return
                  }
                  const blob = await res.blob()
                  const a = document.createElement('a')
                  const objUrl = URL.createObjectURL(blob)
                  a.href = objUrl
                  const cd = res.headers.get('Content-Disposition') || ''
                  const m = /filename="?([^";]+)"?/.exec(cd)
                  a.download = m?.[1] || `the-fork-export.${descriptor.format || 'xlsx'}`
                  document.body.appendChild(a)
                  a.click()
                  a.remove()
                  URL.revokeObjectURL(objUrl)
                })
                .catch((e) => alert(`Export error: ${(e as Error).message}`))
            }}
          />
          <ChatComposer
            onSend={(text) => {
              // Reject synchronously while a stream is in flight: the composer
              // then KEEPS the typed text (it clears only on accept) and the
              // in-flight turn is never aborted by a racing second send.
              if (streamingRef.current) return false
              void handleSend(text)
              return true
            }}
            disabled={streaming || composerIsBlocked}
            disabledReason={composerBlockedReason}
            projectId={id ?? ''}
            hasHistory={messages.length > 0}
            onOpenDrive={() => setDriveModalOpen(true)}
            onClear={clearConversation}
            agents={agents}
            pinnedAgent={pinnedAgent}
            onPinAgent={setPinnedAgent}
            onAttached={(_name, docId) => {
              // S4: keep the real document_id so the next send carries it as
              // document_ids — the [attached: name] marker alone is lossy.
              if (docId) pendingAttachedDocIdsRef.current.push(docId)
            }}
          />
        </div>
      }
      right={
        <RightPanel
          title="Sources"
          expanded={rightExpanded}
          onToggleExpand={() => setRightExpanded((v) => !v)}
          projectId={id ?? ''}
          documents={documents.map((d) => ({
            id: d.id,
            original_name: d.original_name,
            doc_type: d.doc_type,
          }))}
          citedDocuments={citedPreviewDocs}
          previewRequest={previewRequest}
          onPreviewHydrated={handlePreviewHydrated}
          sources={
            <SourcesList
              sources={latestSources}
              streaming={latestAssistant?.streaming}
              activeDocId={previewRequest?.docId}
              onOpenSource={(s) => {
                if (!s.doc_id) return
                setPreviewRequest({ docId: s.doc_id, nonce: Date.now() })
              }}
            />
          }
          graph={
            <DocumentGraph
              documents={documents.map((d) => ({
                id: d.id,
                original_name: d.original_name,
                doc_type: d.doc_type,
              }))}
              citedDocIds={citedDocIds}
              citedExternal={citedExternalDocs}
            />
          }
        />
      }
    />
    {driveModalOpen && (
      <div
        className="ws-modal-backdrop"
        role="dialog"
        aria-modal="true"
        aria-label="Google Drive picker"
        onClick={(ev) => {
          if (ev.target === ev.currentTarget) setDriveModalOpen(false)
        }}
        onKeyDown={(ev) => {
          if (ev.key === 'Escape') setDriveModalOpen(false)
        }}
      >
        <div className="ws-modal">
          <div className="ws-modal__head">
            <span className="ws-modal__title">Google Drive</span>
            <button
              type="button"
              className="ws-modal__close"
              onClick={() => setDriveModalOpen(false)}
              aria-label="Close"
            >
              ×
            </button>
          </div>
          <div className="ws-modal__body">
            <DrivePanel
              projectId={id ?? ''}
              onDocumentAdded={(doc) => {
                handleDocumentAdded(doc)
                setDriveModalOpen(false)
              }}
            />
          </div>
        </div>
      </div>
    )}
    </>
  )
}
