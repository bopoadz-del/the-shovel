/* NewProjectModal — Blank or From-Drive-folder creation.
 *
 * PR C: adds a "From Drive folder" mode beside the blank-create form.
 * The user picks a Drive folder (search or browse) and the modal POSTs
 * /v1/projects/from-drive, which creates a user-owned project with
 * origin='user_drive_import' and queues the recursive import in the
 * background. The owner can then refresh the project page to see
 * documents land as they index.
 */
import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { apiGet, apiPost, ApiError } from '../lib/api'
import type { Project } from './ProjectCard'

interface NewProjectModalProps {
  onClose: () => void
  onCreated: (project: Project) => void
}

interface CreateProjectResponse {
  id: string
  name: string
  client?: string
  status: string
  aconex_connected?: boolean
  user_id?: string
  created_at: string
  readiness?: unknown
  documents?: unknown
}

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

type Mode = 'blank' | 'drive'

export default function NewProjectModal({ onClose, onCreated }: NewProjectModalProps) {
  const [mode, setMode] = useState<Mode>('blank')

  // Shared
  const [name, setName] = useState('')
  const [client, setClient] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Drive picker state
  const [driveStatus, setDriveStatus] = useState<DriveStatus | null>(null)
  const [driveStatusError, setDriveStatusError] = useState<string | null>(null)
  const [folderQuery, setFolderQuery] = useState('')
  const [folderResults, setFolderResults] = useState<DriveFile[]>([])
  const [browsing, setBrowsing] = useState(false)
  const [pickedFolder, setPickedFolder] = useState<DriveFile | null>(null)

  const overlayRef = useRef<HTMLDivElement>(null)
  const nameRef = useRef<HTMLInputElement>(null)

  useEffect(() => { nameRef.current?.focus() }, [mode])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  // Load Drive status lazily when the user switches to the Drive tab.
  // Loading is derived (null status + no error + drive mode) so the
  // effect does not setState before the fetch.
  const driveStatusLoading = mode === 'drive' && driveStatus === null && driveStatusError === null

  useEffect(() => {
    if (mode !== 'drive') return
    if (driveStatus !== null || driveStatusError !== null) return
    let cancelled = false
    apiGet<DriveStatus>('/v1/drive/status')
      .then((s) => { if (!cancelled) setDriveStatus(s) })
      .catch((err) => {
        if (!cancelled) {
          setDriveStatusError(
            err instanceof Error ? err.message : 'Drive status check failed.',
          )
        }
      })
    return () => { cancelled = true }
  }, [mode, driveStatus, driveStatusError])

  function retryDriveStatus() {
    setDriveStatusError(null)
    setDriveStatus(null)  // resets the gate so the effect fires again
  }

  function handleBackdropClick(e: React.MouseEvent<HTMLDivElement>) {
    if (e.target === overlayRef.current) onClose()
  }

  async function searchFolders(e?: FormEvent) {
    e?.preventDefault()
    setBrowsing(true); setError(null); setFolderResults([])
    try {
      // Filter to folders only — mimeType filter happens server-side in
      // /v1/drive/files when the q param includes the folder mime.
      const q = folderQuery.trim()
      const url = q
        ? `/v1/drive/files?q=${encodeURIComponent(q)}&folders_only=true`
        : '/v1/drive/files?folders_only=true'
      const resp = await apiGet<{ files: DriveFile[] }>(url)
      // Client-side guard in case the backend ignores folders_only.
      const folders = resp.files.filter((f) => f.is_folder)
      setFolderResults(folders)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Drive folder search failed.')
    } finally { setBrowsing(false) }
  }

  // Auto-load top-level folders when entering the Drive tab + connected.
  useEffect(() => {
    if (mode !== 'drive' || !driveStatus?.connected) return
    let cancelled = false
    void (async () => {
      try {
        const resp = await apiGet<{ files: DriveFile[] }>('/v1/drive/files?folders_only=true')
        if (cancelled) return
        setFolderResults(resp.files.filter((f) => f.is_folder))
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Drive folder search failed.')
        }
      }
    })()
    return () => { cancelled = true }
  }, [mode, driveStatus?.connected])

  const canSubmit = useMemo(() => {
    if (submitting) return false
    if (!name.trim()) return false
    if (mode === 'drive' && !pickedFolder) return false
    return true
  }, [submitting, name, mode, pickedFolder])

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!canSubmit) return
    setSubmitting(true); setError(null)
    try {
      if (mode === 'blank') {
        const body: { name: string; client?: string } = { name: name.trim() }
        if (client.trim()) body.client = client.trim()
        const project = await apiPost<CreateProjectResponse>('/v1/projects', body)
        onCreated(project as Project)
      } else {
        if (!pickedFolder) throw new Error('Pick a Drive folder first.')
        const body = {
          folder_id: pickedFolder.id,
          name: name.trim(),
          ...(client.trim() ? { client: client.trim() } : {}),
        }
        const resp = await apiPost<{ project: CreateProjectResponse }>(
          '/v1/projects/from-drive', body,
        )
        onCreated(resp.project as Project)
      }
    } catch (err: unknown) {
      if (err instanceof ApiError && err.status === 409) {
        setError('Google Drive is not connected for your account. Connect it on the Admin page first.')
      } else {
        setError(err instanceof Error ? err.message : 'An unexpected error occurred.')
      }
    } finally { setSubmitting(false) }
  }

  return (
    <div
      className="modal-overlay"
      ref={overlayRef}
      onClick={handleBackdropClick}
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
    >
      <div className="modal">
        <div className="modal__header">
          <h2 className="modal__title" id="modal-title">New project</h2>
          <button
            className="modal__close"
            type="button"
            onClick={onClose}
            aria-label="Close"
            disabled={submitting}
          >×</button>
        </div>

        <div className="modal__tabs" role="tablist" aria-label="Project source">
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'blank'}
            className={'modal__tab' + (mode === 'blank' ? ' modal__tab--active' : '')}
            onClick={() => setMode('blank')}
            disabled={submitting}
          >Blank</button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'drive'}
            className={'modal__tab' + (mode === 'drive' ? ' modal__tab--active' : '')}
            onClick={() => setMode('drive')}
            disabled={submitting}
          >From Drive folder</button>
        </div>

        <form onSubmit={handleSubmit} noValidate>
          <div className="form-field">
            <label className="form-label" htmlFor="proj-name">Project name</label>
            <input
              id="proj-name"
              ref={nameRef}
              className="form-input"
              type="text"
              placeholder={mode === 'drive' ? 'Defaults from picked folder' : 'e.g. North Tower Fit-out'}
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              autoComplete="off"
              disabled={submitting}
            />
          </div>

          <div className="form-field">
            <label className="form-label" htmlFor="proj-client">
              Client <span className="form-label--optional">(optional)</span>
            </label>
            <input
              id="proj-client"
              className="form-input"
              type="text"
              placeholder="e.g. Acme Corp"
              value={client}
              onChange={(e) => setClient(e.target.value)}
              autoComplete="off"
              disabled={submitting}
            />
          </div>

          {mode === 'drive' && (
            <div className="form-field">
              <label className="form-label">Drive folder</label>

              {driveStatusLoading && (
                <p className="form-hint">Checking Drive connection…</p>
              )}

              {!driveStatusLoading && driveStatusError && (
                <p className="form-hint form-hint--alert">
                  Could not check Drive status: {driveStatusError}{' '}
                  <button
                    type="button"
                    className="btn btn--ghost btn--small"
                    onClick={retryDriveStatus}
                    disabled={submitting}
                  >Retry</button>
                </p>
              )}

              {driveStatus && !driveStatus.connected && (
                <p className="form-hint form-hint--alert">
                  Google Drive is not connected. Connect it from <a href="/admin">/admin</a>,
                  then come back.
                </p>
              )}

              {driveStatus?.connected && (
                <>
                  <div className="modal__drive-search">
                    <input
                      type="search"
                      className="form-input"
                      placeholder="Search folder name…"
                      value={folderQuery}
                      onChange={(e) => setFolderQuery(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); void searchFolders() } }}
                      disabled={browsing || submitting}
                    />
                    <button
                      type="button"
                      className="btn btn--ghost"
                      onClick={() => void searchFolders()}
                      disabled={browsing || submitting}
                    >{browsing ? '…' : 'Search'}</button>
                  </div>

                  {pickedFolder && (
                    <div className="modal__drive-picked">
                      Picked: <strong>{pickedFolder.name}</strong>
                      <button
                        type="button"
                        className="btn btn--ghost btn--small"
                        onClick={() => setPickedFolder(null)}
                        disabled={submitting}
                      >Change</button>
                    </div>
                  )}

                  {!pickedFolder && folderResults.length > 0 && (
                    <ul className="modal__drive-results">
                      {folderResults.slice(0, 300).map((f) => (
                        <li key={f.id}>
                          <button
                            type="button"
                            className="modal__drive-result"
                            onClick={() => {
                              setPickedFolder(f)
                              if (!name.trim()) setName(f.name)
                            }}
                            disabled={submitting}
                          >
                            <span className="modal__drive-result-mark">[folder]</span>
                            <span>{f.name}</span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}

                  {!pickedFolder && !browsing && folderResults.length === 0 && (
                    <p className="form-hint">No folders found yet — try a search above.</p>
                  )}
                </>
              )}
            </div>
          )}

          {error && (
            <div className="form-error" role="alert">{error}</div>
          )}

          <div className="form-actions">
            <button
              type="button"
              className="btn btn--ghost"
              onClick={onClose}
              disabled={submitting}
            >Cancel</button>
            <button
              type="submit"
              className="btn btn--primary"
              disabled={!canSubmit}
            >
              {submitting
                ? (mode === 'drive' ? 'Creating + queueing import…' : 'Creating…')
                : (mode === 'drive' ? 'Create from Drive folder' : 'Create project')}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
