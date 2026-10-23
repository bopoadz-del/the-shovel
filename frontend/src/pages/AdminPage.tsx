/* AdminPage — operator-supplied requirement 2026-06-21 + 2026-06-22.
 *
 * Single page mounted at /admin. Four first-class sections:
 *
 *   1. Google Drive — the full Drive integration (connected account,
 *      browse, search, folder navigation, import). Drive is the pilot's
 *      project server; this is where it lives.
 *
 *   2. Detected from Drive — cascaded folder tree from /v1/admin/drive/scan
 *      (PR A). Each candidate folder gets an "Approve as project" action
 *      that POSTs /v1/admin/projects/approve-from-drive and creates a
 *      row with origin='admin_drive_approved'.
 *
 *   3. Approved projects — filtered to origin='admin_drive_approved'
 *      (so personal-account-style user-created rows do NOT appear here). Each
 *      row carries documents + chunks counts plus Re-index and Delete.
 *
 *   4. Header — "Connected as: <email>" up top (not in the LeftPanel).
 *
 * No stubs, no "coming soon" placeholders. Wired to live backend
 * endpoints:
 *   GET  /v1/drive/status                         → connection + email
 *   GET  /v1/drive/connect                        → OAuth redirect
 *   POST /v1/drive/disconnect                     → unlink
 *   GET  /v1/drive/files                          → search / browse
 *   POST /v1/projects/{id}/drive/import           → import file into project
 *   GET  /v1/projects                             → all project rows
 *   GET  /v1/admin/drive/scan                     → cascaded folder tree (PR A)
 *   POST /v1/admin/projects/approve-from-drive    → approve folder (PR A)
 *   GET  /v1/admin/corpus/collections             → chunk/doc counts
 *   POST /v1/admin/debug/project-reindex          → re-index a project
 *   DELETE /v1/projects/{id}                      → delete a project
 */
import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  RefreshCw, FolderTree, Search, Plug, LogOut, ArrowLeft,
  FolderSearch, CheckCircle2, Trash2, ChevronRight, ChevronDown,
} from 'lucide-react'
import AppHeader from '../components/AppHeader'
import { apiGet, apiPost, apiDelete, ApiError } from '../lib/api'
import { getToken } from '../lib/token'
import { useAuth } from '../auth/AuthContext'
import './admin.css'

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000'

interface DriveStatus {
  configured: boolean
  connected: boolean
  email?: string
}

interface DriveFile {
  id: string
  name: string
  is_folder?: boolean
  mime_type?: string
}

interface FolderCrumb {
  id: string
  name: string
}

interface ProjectRow {
  id: string
  name: string
  status?: string
  user_id?: string
  is_approved?: boolean
  origin?: string
  is_master_corpus?: boolean
}

interface CorpusCollection {
  project_id: string
  documents: number
  chunks: number
}

interface CorpusResponse {
  collections: CorpusCollection[]
  total_project_ids: number
  total_documents: number
  total_chunks: number
}

interface ScanFolder {
  folder_id: string
  name: string
  direct_file_count: number
  subfolder_count: number
  is_candidate: boolean
  children: ScanFolder[]
}

interface ScanResponse {
  max_depth: number
  root_file_count: number
  candidates_total: number
  tree: ScanFolder[]
}

export default function AdminPage() {
  const navigate = useNavigate()
  const { logout } = useAuth()
  // Bumped every time a Drive folder is approved. ApprovedProjectsSection
  // re-fetches when this changes so the new row appears without a manual
  // Refresh click — fix for "I approved 2 and nothing showed" (PR F).
  const [refreshKey, setRefreshKey] = useState(0)

  return (
    <div className="admin-page">
      <AppHeader breadcrumb={
        <nav className="admin-breadcrumb" aria-label="Admin breadcrumb">
          <Link to="/" className="admin-breadcrumb__link">
            <ArrowLeft size={14} /> <span>Projects</span>
          </Link>
          <span className="admin-breadcrumb__sep">/</span>
          <span className="admin-breadcrumb__current">Admin</span>
        </nav>
      } />

      <main className="admin-main">
        <header className="admin-main__header">
          <h1 className="admin-main__title">Admin</h1>
          <p className="admin-main__subtitle">
            Data management layer. Google Drive integration + project corpus status.
          </p>
        </header>

        <DriveSection />

        <DetectedFromDriveSection onApproved={() => setRefreshKey((k) => k + 1)} />

        <ApprovedProjectsSection
          refreshKey={refreshKey}
          onPickProject={(pid) => navigate(`/projects/${pid}`)}
        />

        <DriverModeSection />

        <footer className="admin-main__footer">
          <button
            type="button"
            className="admin-signout"
            onClick={() => logout()}
          >
            <LogOut size={14} />
            <span>Sign out</span>
          </button>
        </footer>
      </main>
    </div>
  )
}

// ─── Google Drive section ────────────────────────────────────────────────

function DriveSection() {
  const [status, setStatus] = useState<DriveStatus | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [connecting, setConnecting] = useState(false)
  const [disconnecting, setDisconnecting] = useState(false)
  const [query, setQuery] = useState('')
  const [files, setFiles] = useState<DriveFile[]>([])
  const [searching, setSearching] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [folderStack, setFolderStack] = useState<FolderCrumb[]>([])
  const [projects, setProjects] = useState<ProjectRow[]>([])
  const [importTarget, setImportTarget] = useState<string>('')
  const [importingId, setImportingId] = useState<string | null>(null)
  const [importErrors, setImportErrors] = useState<Record<string, string>>({})
  const [importedFlash, setImportedFlash] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const [s, p] = await Promise.all([
          apiGet<DriveStatus>('/v1/drive/status'),
          apiGet<{ projects: ProjectRow[] }>('/v1/projects'),
        ])
        if (cancelled) return
        setStatus(s)
        setProjects(p.projects ?? [])
        if (p.projects && p.projects.length > 0 && !importTarget) {
          setImportTarget(p.projects[0].id)
        }
      } catch (err) {
        if (cancelled) return
        setStatusError(err instanceof Error ? err.message : 'Failed to load Drive status.')
      }
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function openFolder(folder: FolderCrumb, push: boolean) {
    setSearchError(null); setFiles([]); setSearching(true)
    try {
      const url = folder.id
        ? `/v1/drive/files?folder_id=${encodeURIComponent(folder.id)}`
        : '/v1/drive/files'
      const resp = await apiGet<{ files: DriveFile[] }>(url)
      setFiles(resp.files)
      setHasSearched(true)
      setFolderStack((prev) =>
        push ? [...prev, folder]
             : prev.slice(0, prev.findIndex((c) => c.id === folder.id) + 1),
      )
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Drive browse failed.')
    } finally { setSearching(false) }
  }

  async function handleConnect() {
    setConnecting(true)
    try {
      // return_to keeps the user on the admin page across the OAuth round-trip.
      const resp = await apiGet<{ auth_url: string }>(
        `/v1/drive/connect?return_to=${encodeURIComponent(window.location.pathname)}`,
      )
      window.location.href = resp.auth_url
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : 'Connect failed.')
      setConnecting(false)
    }
  }

  async function handleDisconnect() {
    if (!window.confirm('Disconnect Google Drive? You will need to re-authorize to import again.')) return
    setDisconnecting(true)
    try {
      await apiPost('/v1/drive/disconnect', {})
      setStatus({ configured: status?.configured ?? true, connected: false })
      setFiles([]); setHasSearched(false); setFolderStack([])
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : 'Disconnect failed.')
    } finally { setDisconnecting(false) }
  }

  async function handleSearch(e: FormEvent) {
    e.preventDefault()
    setSearchError(null); setFiles([]); setHasSearched(false); setSearching(true)
    setFolderStack([])
    try {
      const resp = await apiGet<{ files: DriveFile[] }>(
        `/v1/drive/files?q=${encodeURIComponent(query)}`,
      )
      setFiles(resp.files); setHasSearched(true)
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Drive search failed.')
    } finally { setSearching(false) }
  }

  async function handleImport(file: DriveFile) {
    if (!importTarget) {
      setImportErrors((p) => ({ ...p, [file.id]: 'Pick a project first.' }))
      return
    }
    setImportingId(file.id)
    setImportErrors((p) => { const n = { ...p }; delete n[file.id]; return n })
    try {
      await apiPost(`/v1/projects/${importTarget}/drive/import`,
        { file_id: file.id, name: file.name })
      setFiles((p) => p.filter((f) => f.id !== file.id))
      setImportedFlash(`Imported "${file.name}" into ${projects.find(p => p.id === importTarget)?.name ?? importTarget}.`)
      setTimeout(() => setImportedFlash(null), 4000)
    } catch (err) {
      setImportErrors((p) => ({ ...p, [file.id]: err instanceof Error ? err.message : 'Import failed.' }))
    } finally { setImportingId(null) }
  }

  return (
    <section className="admin-section">
      <header className="admin-section__head">
        <Plug size={16} />
        <h2 className="admin-section__title">Google Drive</h2>
      </header>

      {!status && !statusError && <p className="admin-loading">Checking Drive…</p>}
      {statusError && <p className="admin-alert">{statusError}</p>}

      {status && !status.configured && (
        <p className="admin-empty">
          Google Drive is not configured on this server. Add the OAuth client + redirect
          URI to enable. See docs for setup.
        </p>
      )}

      {status && status.configured && !status.connected && (
        <div className="admin-cta">
          <p>Connect a Google account to browse and import Drive files into projects.</p>
          <button
            type="button"
            className="admin-btn admin-btn--primary"
            onClick={() => void handleConnect()}
            disabled={connecting}
          >
            {connecting ? 'Redirecting…' : 'Connect Google Drive'}
          </button>
        </div>
      )}

      {status && status.connected && (
        <>
          <div className="admin-drive-meta">
            <div>
              <span className="admin-drive-meta__label">Connected as</span>{' '}
              <span className="admin-drive-meta__email">{status.email ?? '(unknown)'}</span>
            </div>
            <button
              type="button"
              className="admin-btn admin-btn--ghost"
              onClick={() => void handleDisconnect()}
              disabled={disconnecting}
            >
              {disconnecting ? 'Disconnecting…' : 'Disconnect'}
            </button>
          </div>

          <div className="admin-drive-importer">
            <label className="admin-drive-importer__label">
              Import target
              <select
                value={importTarget}
                onChange={(e) => setImportTarget(e.target.value)}
                className="admin-drive-importer__select"
                disabled={projects.length === 0}
              >
                {projects.length === 0 && <option value="">No projects yet</option>}
                {projects.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
            </label>
          </div>

          <form className="admin-drive-search" onSubmit={(e) => void handleSearch(e)}>
            <Search size={14} />
            <input
              type="search"
              className="admin-drive-search__input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search Drive files…"
              disabled={searching}
            />
            <button type="submit" className="admin-btn admin-btn--ghost" disabled={searching}>
              {searching ? '…' : 'Search'}
            </button>
            <button
              type="button"
              className="admin-btn admin-btn--ghost"
              disabled={searching}
              onClick={() => void openFolder({ id: '', name: 'My Drive' }, true)}
            >
              <FolderTree size={13} />
              <span>Browse</span>
            </button>
          </form>

          {searchError && <p className="admin-alert">{searchError}</p>}
          {importedFlash && <p className="admin-flash">{importedFlash}</p>}

          {folderStack.length > 0 && (
            <nav className="admin-crumbs" aria-label="Drive folder">
              <button
                type="button"
                className="admin-crumbs__back"
                onClick={() => { setFolderStack([]); setFiles([]); setHasSearched(false) }}
              >
                ↑ Back to search
              </button>
              {folderStack.map((c, i) => (
                <span key={c.id || `c-${i}`}>
                  <span className="admin-crumbs__sep">/</span>
                  <button
                    type="button"
                    className="admin-crumbs__item"
                    onClick={() => void openFolder(c, false)}
                    disabled={i === folderStack.length - 1 || searching}
                  >
                    {c.name}
                  </button>
                </span>
              ))}
            </nav>
          )}

          {files.length > 0 && (
            <ul className="admin-files">
              {files.slice(0, 300).map((f) => (
                <li key={f.id} className={'admin-file' + (f.is_folder ? ' admin-file--folder' : '')}>
                  <span className="admin-file__name">
                    {f.is_folder && <span className="admin-file__folder-mark">[folder]</span>}
                    {f.is_folder ? ' ' : ''}{f.name}
                  </span>
                  <div className="admin-file__actions">
                    {importErrors[f.id] && (
                      <span className="admin-file__error">{importErrors[f.id]}</span>
                    )}
                    {f.is_folder ? (
                      <button
                        type="button"
                        className="admin-btn admin-btn--ghost admin-btn--small"
                        disabled={searching}
                        onClick={() => void openFolder({ id: f.id, name: f.name }, true)}
                      >Open</button>
                    ) : (
                      <button
                        type="button"
                        className="admin-btn admin-btn--primary admin-btn--small"
                        disabled={importingId === f.id || !importTarget}
                        onClick={() => void handleImport(f)}
                      >{importingId === f.id ? '…' : 'Add'}</button>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}

          {!searching && hasSearched && files.length === 0 && !searchError && (
            <p className="admin-empty">No files found.</p>
          )}
        </>
      )}
    </section>
  )
}

// ─── Detected from Drive section ────────────────────────────────────────
//
// Renders /v1/admin/drive/scan as a collapsible cascaded tree. Folders
// flagged is_candidate (direct_file_count > 0) get an Approve button
// that POSTs /v1/admin/projects/approve-from-drive — the backend slugs
// the folder name into a project id, creates the row with
// origin='admin_drive_approved', and queues a recursive Drive import
// in the background. The Approved projects section below then shows
// the new row once it's created.

function DetectedFromDriveSection({ onApproved }: { onApproved?: () => void }) {
  const [scan, setScan] = useState<ScanResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [approvingId, setApprovingId] = useState<string | null>(null)
  const [approvedFlash, setApprovedFlash] = useState<string | null>(null)
  const [approveErrors, setApproveErrors] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  // Type-to-confirm gate on Approve. Approving a folder creates a project AND
  // kicks off a full recursive ingest of every file inside it — heavy enough
  // to OOM the box on a large folder (e.g. Master Folder, ~196 files). Arm an
  // inline input and only fire once the admin types the folder's exact name.
  // Throttle-proof (no native dialog) and no shared secret in the bundle;
  // mirrors the Re-index gate in ApprovedProjectsSection. Replaces the old
  // 5-second "click again to confirm" two-step, which was too easy to trip.
  const [approveArmId, setApproveArmId] = useState<string | null>(null)
  const [approveConfirmText, setApproveConfirmText] = useState('')
  const [approvedIds, setApprovedIds] = useState<Set<string>>(new Set())

  async function load() {
    setLoading(true); setError(null)
    try {
      const resp = await apiGet<ScanResponse>('/v1/admin/drive/scan?max_depth=2')
      setScan(resp)
      // Auto-expand top-level by default; depth-2 stays collapsed unless clicked.
      const next: Record<string, boolean> = {}
      for (const f of resp.tree) next[f.folder_id] = true
      setExpanded(next)
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        setError('Admin role required to scan Drive.')
      } else if (err instanceof ApiError && err.status === 409) {
        setError('Connect Google Drive in the section above before scanning.')
      } else {
        setError(err instanceof Error ? err.message : 'Drive scan failed.')
      }
    } finally { setLoading(false) }
  }

  useEffect(() => { void load() }, [])

  async function handleApprove(folder: ScanFolder) {
    setApprovingId(folder.folder_id)
    setApproveErrors((p) => { const n = { ...p }; delete n[folder.folder_id]; return n })
    setApprovedFlash(null)
    try {
      const body = await apiPost<{ project: ProjectRow }>(
        '/v1/admin/projects/approve-from-drive',
        { folder_id: folder.folder_id, name: folder.name },
      )
      setApprovedIds((p) => new Set(p).add(folder.folder_id))
      setApprovedFlash(
        `Approved "${folder.name}" → project "${body.project.id}". `
        + `Indexing runs in the background; the row appears in Approved projects below.`,
      )
      // Tell the parent so the Approved Projects table re-fetches and
      // the new row appears without a manual Refresh click.
      onApproved?.()
    } catch (err) {
      setApproveErrors((p) => ({
        ...p, [folder.folder_id]: err instanceof Error ? err.message : 'Approve failed.',
      }))
    } finally { setApprovingId(null) }
  }

  function toggle(folderId: string) {
    setExpanded((p) => ({ ...p, [folderId]: !p[folderId] }))
  }

  function renderFolder(folder: ScanFolder, depth: number) {
    const open = !!expanded[folder.folder_id]
    const hasChildren = folder.children.length > 0
    return (
      <li key={folder.folder_id} className="admin-tree__node" style={{ paddingLeft: `${depth * 16}px` }}>
        <div className="admin-tree__row">
          <button
            type="button"
            className="admin-tree__toggle"
            onClick={() => toggle(folder.folder_id)}
            disabled={!hasChildren}
            aria-label={open ? 'Collapse' : 'Expand'}
          >
            {hasChildren ? (open ? <ChevronDown size={14} /> : <ChevronRight size={14} />)
                         : <span className="admin-tree__leaf-mark" />}
          </button>
          <span className="admin-tree__name">{folder.name}</span>
          <span className="admin-tree__counts">
            {folder.direct_file_count} file{folder.direct_file_count === 1 ? '' : 's'}
            {folder.subfolder_count > 0 && (
              <>{' · '}{folder.subfolder_count} folder{folder.subfolder_count === 1 ? '' : 's'}</>
            )}
          </span>
          <div className="admin-tree__actions">
            {approveErrors[folder.folder_id] && (
              <span className="admin-file__error">{approveErrors[folder.folder_id]}</span>
            )}
            {folder.is_candidate ? (
              approvedIds.has(folder.folder_id) ? (
                <span className="admin-tree__approved-flag">
                  <CheckCircle2 size={13} /> Approved
                </span>
              ) : approveArmId === folder.folder_id ? (
                <span className="admin-corpus__reindex-confirm">
                  <input
                    type="text"
                    className="admin-corpus__confirm-input"
                    style={{ minWidth: 150, padding: '2px 6px' }}
                    placeholder={`Type "${folder.name}" to confirm`}
                    value={approveConfirmText}
                    onChange={(e) => setApproveConfirmText(e.target.value)}
                    autoFocus
                    aria-label={`Type the folder name to confirm approving ${folder.name}`}
                  />
                  <button
                    type="button"
                    className="admin-btn admin-btn--small admin-btn--danger"
                    onClick={() => {
                      setApproveArmId(null); setApproveConfirmText(''); void handleApprove(folder)
                    }}
                    disabled={approveConfirmText.trim() !== folder.name || approvingId !== null}
                  >
                    Confirm approve
                  </button>
                  <button
                    type="button"
                    className="admin-btn admin-btn--ghost admin-btn--small"
                    onClick={() => { setApproveArmId(null); setApproveConfirmText('') }}
                  >
                    Cancel
                  </button>
                </span>
              ) : (
                <button
                  type="button"
                  className="admin-btn admin-btn--small admin-btn--primary"
                  onClick={() => { setApproveArmId(folder.folder_id); setApproveConfirmText('') }}
                  disabled={approvingId !== null || approveArmId !== null}
                  title="Creates a project and ingests every file in this folder (heavy on large folders)."
                >
                  {approvingId === folder.folder_id
                    ? 'Approving…'
                    : <><CheckCircle2 size={13} /><span>Approve</span></>}
                </button>
              )
            ) : (
              <span className="admin-tree__nocand">no direct files</span>
            )}
          </div>
        </div>
        {open && hasChildren && (
          <ul className="admin-tree__children">
            {folder.children.map((c) => renderFolder(c, depth + 1))}
          </ul>
        )}
      </li>
    )
  }

  return (
    <section className="admin-section">
      <header className="admin-section__head">
        <FolderSearch size={16} />
        <h2 className="admin-section__title">Detected from Drive</h2>
        <button
          type="button"
          className="admin-btn admin-btn--ghost admin-btn--small"
          onClick={() => void load()}
          disabled={loading}
          style={{ marginLeft: 'auto' }}
        >
          {loading ? 'Scanning…' : 'Rescan'}
        </button>
      </header>

      <p className="admin-section__hint">
        Top-level Drive folders the platform sees. Approve a folder to turn it into a
        project — only its documents will be indexed and visible to users.
      </p>

      {error && <p className="admin-alert">{error}</p>}
      {approvedFlash && <p className="admin-flash">{approvedFlash}</p>}

      {!error && scan && scan.tree.length === 0 && !loading && (
        <p className="admin-empty">No folders at Drive root.</p>
      )}

      {scan && scan.tree.length > 0 && (
        <ul className="admin-tree">
          {scan.tree.map((f) => renderFolder(f, 0))}
        </ul>
      )}
    </section>
  )
}

// ─── Approved projects section ──────────────────────────────────────────
//
// Filtered to origin='admin_drive_approved'. Operator requirement:
// personal-account-style user-created rows MUST NOT appear here — admin owns
// the platform-canonical project list. Each row exposes Re-index and
// Delete actions.

function ApprovedProjectsSection({
  onPickProject,
  refreshKey = 0,
}: {
  onPickProject: (id: string) => void
  refreshKey?: number
}) {
  const [projects, setProjects] = useState<ProjectRow[]>([])
  const [corpus, setCorpus] = useState<Record<string, CorpusCollection>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [reindexingId, setReindexingId] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [flash, setFlash] = useState<string | null>(null)
  // Type-to-confirm gate on the Re-index action. Re-index re-extracts + re-embeds
  // EVERY document in a project; on a large corpus (e.g. drive_archive, ~3k docs)
  // that is heavy enough to OOM the box, so a stray click is dangerous. Native
  // window.confirm can be silently throttled away by the browser (see the
  // Detected-from-Drive note above), so this guard lives in the DOM: Re-index
  // arms an inline input, and the run only fires once the operator types the
  // project's exact name — no shared password to leak into the bundle, and
  // immune to dialog throttling.
  const [reindexArmId, setReindexArmId] = useState<string | null>(null)
  const [reindexConfirmText, setReindexConfirmText] = useState('')

  async function load() {
    setLoading(true); setError(null)
    try {
      // Projects load unconditionally. Corpus counts are best-effort: the
      // admin-only COUNT over the full (~70GB) corpus can 403 for a non-admin
      // or time out on a large corpus — that must NOT blank the projects table
      // and its per-row Hide controls (the "Hide button not visible" bug).
      const pResp = await apiGet<{ projects: ProjectRow[] }>('/v1/projects')
      setProjects(pResp.projects ?? [])
      try {
        const cResp = await apiGet<CorpusResponse>('/v1/admin/corpus/collections?folder_breakdown=false')
        const map: Record<string, CorpusCollection> = {}
        for (const c of cResp.collections ?? []) map[c.project_id] = c
        setCorpus(map)
      } catch { /* counts optional; rows + Hide still render */ }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to load approved projects.'
      setError(err instanceof ApiError && err.status === 403
        ? 'Admin role required to view approved projects. You are logged in but not as an admin.'
        : msg)
    } finally { setLoading(false) }
  }

  // Re-fetch when the parent bumps refreshKey (a Drive approve just
  // landed) so the new row appears without a manual click.
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const pResp = await apiGet<{ projects: ProjectRow[] }>('/v1/projects')
        if (cancelled) return
        setProjects(pResp.projects ?? [])
        try {
          const cResp = await apiGet<CorpusResponse>('/v1/admin/corpus/collections?folder_breakdown=false')
          if (cancelled) return
          const map: Record<string, CorpusCollection> = {}
          for (const c of cResp.collections ?? []) map[c.project_id] = c
          setCorpus(map)
        } catch { /* counts optional; rows + Hide still render */ }
      } catch (err) {
        if (cancelled) return
        const msg = err instanceof Error ? err.message : 'Failed to load approved projects.'
        setError(err instanceof ApiError && err.status === 403
          ? 'Admin role required to view approved projects. You are logged in but not as an admin.'
          : msg)
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [refreshKey])

  async function runReindex(p: ProjectRow) {
    setReindexArmId(null)
    setReindexConfirmText('')
    setReindexingId(p.id)
    setFlash(null)
    try {
      const token = getToken() || ''
      const res = await fetch(
        `${API_BASE}/v1/admin/debug/project-reindex?project_id=${encodeURIComponent(p.id)}`,
        { method: 'POST', headers: { Authorization: `Bearer ${token}` } },
      )
      if (!res.ok) {
        const detail = await res.text().catch(() => '')
        setFlash(`Re-index failed (${res.status}): ${detail.slice(0, 200)}`)
        return
      }
      const body = await res.json().catch(() => ({}))
      setFlash(
        `Re-indexed "${p.name}": ${body.indexed ?? '?'} documents, ` +
        `${body.skipped_unsupported ?? 0} skipped (unsupported), ` +
        `${body.total_chunks ?? '?'} chunks.`,
      )
      await load()
    } catch (err) {
      setFlash(`Re-index failed: ${(err as Error).message}`)
    } finally { setReindexingId(null) }
  }

  async function handleDelete(p: ProjectRow) {
    if (!window.confirm(
      `Hide project "${p.name}"? It disappears from the app and from retrieval, ` +
      `but its documents and RAG chunks are preserved on the server — this is a ` +
      `reversible archive, not a permanent delete.`,
    )) return
    setDeletingId(p.id)
    setFlash(null)
    try {
      await apiDelete(`/v1/projects/${encodeURIComponent(p.id)}`)
      setFlash(`Archived "${p.name}" (hidden; RAG preserved).`)
      await load()
    } catch (err) {
      setFlash(`Delete failed: ${(err as Error).message}`)
    } finally { setDeletingId(null) }
  }

  // The filter that solves the operator's "no personal accounts" requirement.
  // Only rows the admin explicitly approved via /v1/admin/projects/approve-from-drive
  // make it into this table.
  const approved = projects
    .filter((p) => (p.origin ?? 'user_create') === 'admin_drive_approved')
    // Hide genuinely-empty approved rows (0 documents AND 0 chunks) so junk
    // like an abandoned "the client project Bills of Quantities" (0/0) doesn't clutter the
    // list. Only hide once the corpus counts for that project_id have loaded —
    // never hide a row whose counts we simply haven't fetched yet (that would
    // make a mid-ingest or slow-count project vanish).
    .filter((p) => {
      const c = corpus[p.id]
      if (!c) return true
      return (c.documents ?? 0) > 0 || (c.chunks ?? 0) > 0
    })

  return (
    <section className="admin-section">
      <header className="admin-section__head">
        <RefreshCw size={16} />
        <h2 className="admin-section__title">Approved projects</h2>
        <button
          type="button"
          className="admin-btn admin-btn--ghost admin-btn--small"
          onClick={() => void load()}
          disabled={loading}
          style={{ marginLeft: 'auto' }}
        >
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </header>

      <p className="admin-section__hint">
        Projects approved from a Drive folder. User-created personal projects
        live on their owners' profiles and do not appear here.
      </p>

      {error && <p className="admin-alert">{error}</p>}
      {flash && <p className="admin-flash">{flash}</p>}

      {!error && approved.length === 0 && !loading && (
        <p className="admin-empty">No approved projects yet. Approve a folder above to create one.</p>
      )}

      {approved.length > 0 && (
        <table className="admin-corpus">
          <thead>
            <tr>
              <th>Project</th>
              <th className="num">Documents</th>
              <th className="num">Chunks</th>
              <th className="num">Actions</th>
            </tr>
          </thead>
          <tbody>
            {approved.map((p) => {
              const c = corpus[p.id]
              return (
                <tr key={p.id}>
                  <td>
                    <button
                      type="button"
                      className="admin-corpus__namelink"
                      onClick={() => onPickProject(p.id)}
                    >
                      {p.name}
                    </button>
                    <span className="admin-corpus__pid mono">{p.id}</span>
                  </td>
                  <td className="num">{c?.documents ?? 0}</td>
                  <td className="num">{c?.chunks ?? 0}</td>
                  <td className="num admin-corpus__actions">
                    {reindexArmId === p.id ? (
                      <span className="admin-corpus__reindex-confirm">
                        <input
                          type="text"
                          className="admin-corpus__confirm-input"
                          style={{ minWidth: 150, padding: '2px 6px' }}
                          placeholder={`Type "${p.name}" to confirm`}
                          value={reindexConfirmText}
                          onChange={(e) => setReindexConfirmText(e.target.value)}
                          autoFocus
                          aria-label={`Type the project name to confirm re-indexing ${p.name}`}
                        />
                        <button
                          type="button"
                          className="admin-btn admin-btn--small admin-btn--danger"
                          onClick={() => void runReindex(p)}
                          disabled={reindexConfirmText.trim() !== p.name}
                        >
                          Confirm re-index
                        </button>
                        <button
                          type="button"
                          className="admin-btn admin-btn--ghost admin-btn--small"
                          onClick={() => { setReindexArmId(null); setReindexConfirmText('') }}
                        >
                          Cancel
                        </button>
                      </span>
                    ) : (
                      <button
                        type="button"
                        className="admin-btn admin-btn--ghost admin-btn--small"
                        onClick={() => { setReindexArmId(p.id); setReindexConfirmText('') }}
                        disabled={reindexingId !== null || deletingId !== null || reindexArmId !== null}
                        title="Re-extracts + re-embeds every document in this project (heavy on large corpora)."
                      >
                        {reindexingId === p.id ? '…' : 'Re-index'}
                      </button>
                    )}
                    {!p.is_master_corpus && (
                      <button
                        type="button"
                        className="admin-btn admin-btn--ghost admin-btn--small admin-btn--danger"
                        onClick={() => void handleDelete(p)}
                        disabled={reindexingId !== null || deletingId !== null || reindexArmId !== null}
                        aria-label={`Hide ${p.name}`}
                      >
                        {deletingId === p.id ? '…' : <><Trash2 size={13} /><span>Hide</span></>}
                      </button>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </section>
  )
}


// ─── Driver mode (per-user switch) ───────────────────────────────────────
// Lets an admin try F-DRIVER's driver mode for one account from the real
// UI. Honoured only when the server runs DRIVER_MODE=request.

interface DriverModeStatus {
  mode: string
  users: Array<{ email: string; id: string }>
}

function DriverModeSection() {
  const [status, setStatus] = useState<DriverModeStatus | null>(null)
  const [email, setEmail] = useState('')
  const [message, setMessage] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    try {
      setStatus(await apiGet<DriverModeStatus>('/v1/admin/driver-mode'))
    } catch (err) {
      setMessage(err instanceof ApiError ? err.message : 'Could not load driver mode status.')
    }
  }, [])

  useEffect(() => { void load() }, [load])

  const setFor = async (target: string, on: boolean) => {
    if (!target.trim()) return
    setBusy(true)
    setMessage(null)
    try {
      await apiPost('/v1/admin/driver-mode', { email: target.trim(), on })
      setMessage(`Driver mode ${on ? 'on' : 'off'} for ${target.trim()}.`)
      await load()
    } catch (err) {
      setMessage(err instanceof ApiError ? err.message : 'Could not change driver mode.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="admin-section">
      <header className="admin-section__head">
        <h2 className="admin-section__title">Driver mode</h2>
      </header>
      {status && status.mode !== 'request' && (
        <p className="admin-empty">
          The server runs DRIVER_MODE={status.mode}; the per-user switch takes effect only with DRIVER_MODE=request.
        </p>
      )}
      <div className="admin-cta">
        <label htmlFor="driver-mode-email">User email</label>
        <input
          id="driver-mode-email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="user@example.com"
        />
        <button type="button" className="admin-btn admin-btn--primary" disabled={busy}
                onClick={() => void setFor(email, true)}>Turn on</button>
        <button type="button" className="admin-btn" disabled={busy}
                onClick={() => void setFor(email, false)}>Turn off</button>
      </div>
      {message && <p className="admin-alert">{message}</p>}
      {status && (
        <ul className="admin-list">
          {status.users.length === 0 && <li>No user has driver mode on.</li>}
          {status.users.map((u) => (
            <li key={u.id}>
              {u.email}{' '}
              <button type="button" className="admin-btn" disabled={busy}
                      onClick={() => void setFor(u.email, false)}>Turn off</button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
