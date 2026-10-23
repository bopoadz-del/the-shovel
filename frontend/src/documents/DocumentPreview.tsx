/* DocumentPreview — inline preview of a stored project artifact.
 *
 * Fetches the render-friendly JSON from the backend preview endpoint and
 * renders it inline in the right panel instead of forcing a download:
 *   kind table       → scrollable HTML table (sheet tabs when multi-sheet)
 *   kind pdf          → <object> over a blob object URL (auth stays in the
 *                       Authorization header — the raw route is bearer-gated,
 *                       so we fetch the bytes and embed a blob URL rather than
 *                       pointing the element at a URL an iframe can't authorize)
 *   kind text         → formatted <pre> block; ``indexed_only`` marks the
 *                       text the platform indexed when no original is stored
 *   kind unsupported  → functional message naming the file type
 *
 * Every state offers Download (GET .../download): the original bytes, or the
 * indexed text when no original is stored.
 */
import { useEffect, useState } from 'react'
import { apiGet, ApiError } from '../lib/api'
import { getToken } from '../lib/token'
import './DocumentPreview.css'

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000'

interface DocRef {
  id: string
  original_name: string
}

interface Props {
  projectId: string
  document: DocRef | null
  /** Functional empty-state copy shown when no document is selected. */
  emptyLabel?: string
  /** Called with repaired size after a successful preview hydrate. */
  onHydrated?: (docId: string, size: number) => void
}

type PreviewData =
  | { kind: 'table'; sheets: Array<{ name: string; rows: string[][] }>; truncated?: boolean; size?: number; has_file?: boolean }
  | { kind: 'pdf'; size?: number; has_file?: boolean }
  | { kind: 'text'; text: string; truncated?: boolean; size?: number; has_file?: boolean; indexed_only?: boolean; note?: string }
  | { kind: 'unsupported'; ext: string; size?: number; has_file?: boolean }

export default function DocumentPreview({ projectId, document: doc, emptyLabel, onHydrated }: Props) {
  if (!doc || !projectId) {
    return (
      <div className="doc-preview doc-preview--empty">
        {emptyLabel ?? 'Select a document to preview'}
      </div>
    )
  }
  return (
    <DocumentPreviewLoaded
      key={doc.id}
      projectId={projectId}
      document={doc}
      onHydrated={onHydrated}
    />
  )
}

function DocumentPreviewLoaded({
  projectId, document: doc, onHydrated,
}: { projectId: string; document: DocRef; onHydrated?: (docId: string, size: number) => void }) {
  const [data, setData] = useState<PreviewData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeSheet, setActiveSheet] = useState(0)

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const resp = await apiGet<PreviewData>(
          `/v1/projects/${projectId}/documents/${doc.id}/preview`,
        )
        if (cancelled) return
        setData(resp)
        if (typeof resp.size === 'number' && resp.size > 0) {
          onHydrated?.(doc.id, resp.size)
        }
      } catch (err) {
        if (cancelled) return
        const msg =
          err instanceof ApiError && err.status === 422
            ? 'This file could not be rendered for preview.'
            : err instanceof Error
              ? err.message
              : 'Could not load the preview.'
        setError(msg)
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [projectId, doc.id, onHydrated])

  if (loading) {
    return (
      <div className="doc-preview doc-preview--status">
        Loading preview of {doc.original_name}…
      </div>
    )
  }

  if (error) {
    return (
      <div className="doc-preview doc-preview--status" role="alert">
        {error}
        <DownloadButton projectId={projectId} docId={doc.id} name={doc.original_name} />
      </div>
    )
  }

  if (!data) {
    return (
      <div className="doc-preview doc-preview--status">
        No preview available for {doc.original_name}.
      </div>
    )
  }

  return (
    <div className="doc-preview">
      <div className="doc-preview__header">
        <div className="doc-preview__title" title={doc.original_name}>
          {doc.original_name}
        </div>
        <DownloadButton projectId={projectId} docId={doc.id} name={doc.original_name} />
      </div>
      {data.kind === 'table' && (
        <TablePreview
          sheets={data.sheets}
          truncated={data.truncated}
          activeSheet={activeSheet}
          onSelectSheet={setActiveSheet}
        />
      )}
      {data.kind === 'pdf' && (
        <PdfPreview key={doc.id} projectId={projectId} docId={doc.id} name={doc.original_name} />
      )}
      {data.kind === 'text' && (
        <div className="doc-preview__scroll">
          {data.indexed_only && (
            <p className="doc-preview__note" role="note">{data.note}</p>
          )}
          <pre className="doc-preview__text">{data.text}</pre>
          {data.truncated && (
            <p className="doc-preview__note">Preview truncated to the first part of the document.</p>
          )}
        </div>
      )}
      {data.kind === 'unsupported' && (
        <div className="doc-preview--status">
          Inline preview is not available for {data.ext || 'this'} files. Use Download to save it.
        </div>
      )}
    </div>
  )
}

// ── Download ─────────────────────────────────────────────────────────────────

/** Filename from a Content-Disposition header (RFC 6266 ``filename*`` first). */
function dispositionFilename(header: string | null, fallback: string): string {
  if (!header) return fallback
  const star = /filename\*=UTF-8''([^;]+)/i.exec(header)
  if (star) {
    try {
      return decodeURIComponent(star[1].trim())
    } catch {
      // fall through to the plain filename
    }
  }
  const plain = /filename="([^"]+)"/i.exec(header)
  return plain ? plain[1] : fallback
}

function DownloadButton({ projectId, docId, name }: { projectId: string; docId: string; name: string }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const download = async () => {
    setBusy(true)
    setError(null)
    try {
      const token = getToken()
      // Bearer-gated, so fetch the bytes and save a blob URL rather than
      // pointing a link at a URL the browser cannot authorize.
      const res = await fetch(
        `${API_BASE}/v1/projects/${projectId}/documents/${docId}/download`,
        { headers: token ? { Authorization: `Bearer ${token}` } : {} },
      )
      if (!res.ok) {
        let detail = `Could not download the document (HTTP ${res.status}).`
        try {
          const body = (await res.json()) as { detail?: string }
          if (typeof body.detail === 'string' && body.detail.trim()) detail = body.detail
        } catch {
          // keep the HTTP status fallback
        }
        throw new Error(detail)
      }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = window.document.createElement('a')
      a.href = url
      a.download = dispositionFilename(res.headers.get('Content-Disposition'), name)
      window.document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not download the document.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <span className="doc-preview__download">
      <button type="button" className="doc-preview__download-btn" onClick={() => void download()} disabled={busy}>
        {busy ? 'Downloading…' : 'Download'}
      </button>
      {error && <span className="doc-preview__download-error" role="alert">{error}</span>}
    </span>
  )
}

// ── Table ────────────────────────────────────────────────────────────────────

interface TablePreviewProps {
  sheets: Array<{ name: string; rows: string[][] }>
  truncated?: boolean
  activeSheet: number
  onSelectSheet: (i: number) => void
}

function TablePreview({ sheets, truncated, activeSheet, onSelectSheet }: TablePreviewProps) {
  if (sheets.length === 0) {
    return <div className="doc-preview--status">This spreadsheet has no sheets to show.</div>
  }
  const idx = Math.min(activeSheet, sheets.length - 1)
  const sheet = sheets[idx]
  const rows = sheet.rows
  const header = rows[0] ?? []
  const bodyRows = rows.slice(1)

  return (
    <>
      {sheets.length > 1 && (
        <div className="doc-preview__sheets" role="tablist">
          {sheets.map((s, i) => (
            <button
              key={`${s.name}-${i}`}
              type="button"
              role="tab"
              aria-selected={i === idx}
              className={
                'doc-preview__sheet-tab' +
                (i === idx ? ' doc-preview__sheet-tab--active' : '')
              }
              onClick={() => onSelectSheet(i)}
            >
              {s.name}
            </button>
          ))}
        </div>
      )}
      <div className="doc-preview__scroll">
        {rows.length === 0 ? (
          <div className="doc-preview--status">This sheet is empty.</div>
        ) : (
          <table className="doc-preview__table">
            <thead>
              <tr>
                {header.map((cell, c) => (
                  <th key={c}>{cell}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {bodyRows.map((row, r) => (
                <tr key={r}>
                  {header.map((_, c) => (
                    <td key={c}>{row[c] ?? ''}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {truncated && (
        <p className="doc-preview__note">
          Large sheet — showing the first rows and columns only.
        </p>
      )}
    </>
  )
}

// ── PDF ──────────────────────────────────────────────────────────────────────

interface PdfPreviewProps {
  projectId: string
  docId: string
  name: string
}

function PdfPreview({ projectId, docId, name }: PdfPreviewProps) {
  const [url, setUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    let objectUrl: string | null = null
    void (async () => {
      try {
        const token = getToken()
        const res = await fetch(
          `${API_BASE}/v1/projects/${projectId}/documents/${docId}/preview/raw`,
          { headers: token ? { Authorization: `Bearer ${token}` } : {} },
        )
        if (!res.ok) {
          let detail = `Could not load the PDF (HTTP ${res.status}).`
          try {
            const body = (await res.json()) as { detail?: string }
            if (typeof body.detail === 'string' && body.detail.trim()) {
              detail = body.detail
            }
          } catch {
            // keep the HTTP status fallback
          }
          throw new Error(detail)
        }
        const blob = await res.blob()
        if (cancelled) return
        objectUrl = URL.createObjectURL(blob)
        setUrl(objectUrl)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Could not load the PDF.')
      }
    })()
    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [projectId, docId])

  if (error) {
    return <div className="doc-preview--status" role="alert">{error}</div>
  }
  if (!url) {
    return <div className="doc-preview--status">Loading document…</div>
  }
  return (
    <object className="doc-preview__pdf" data={url} type="application/pdf" aria-label={name}>
      <p className="doc-preview--status">
        This PDF cannot be displayed inline in your browser.
      </p>
    </object>
  )
}
