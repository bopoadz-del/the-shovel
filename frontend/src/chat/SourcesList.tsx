/* SourcesList — RAG citations from the latest assistant message.
 *
 * Lives in the RightPanel. Data shape comes verbatim from the SSE `end`
 * event populated by ProjectWorkspace; no fetching here.
 *
 * Empty states:
 *   • streaming  — soft "Sources will appear here once the answer
 *                  completes." message
 *   • no sources — short hint
 *
 * Each row shows: confidence chip + filename + chunk reference + the
 * #468 source class (this contract / master corpus / knowledge base /
 * template). Clicking a row opens that document in the preview pane.
 *
 * W3 — citations in the side panel only. When the answer carries numbered
 * markers, each marker's entry is listed here by its number: document, page
 * or section, the passage, and its tier (a calculation shows its inputs).
 * The coverage line (N of M project documents indexed) is shown here too;
 * neither ever appears in the answer text.
 */
import { FileText } from 'lucide-react'
import { visibleSourceClassLabel } from './sourceClassLabels'
import { plainMetadata, textLeaksInternalCode } from './userFacingText'
import type { AnswerCitation, AnswerCoverage } from './types'
import './SourcesList.css'

export interface CitedSource {
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
}

interface Props {
  /** Latest assistant message's sources, or undefined while streaming. */
  sources?: CitedSource[]
  /** True while the assistant message is still streaming. */
  streaming?: boolean
  /** Highlight the source currently open in the preview pane. */
  activeDocId?: string | null
  /** Open this cited document in the right-panel preview. */
  onOpenSource?: (source: CitedSource) => void
  /** The latest answer's marker entries, in marker order. */
  citations?: AnswerCitation[]
  /** The latest answer's coverage: a panel line, never answer text. */
  coverage?: AnswerCoverage
  /** The marker the user last opened from the answer. */
  activeMarker?: number | null
}

export default function SourcesList({
  sources, streaming, activeDocId, onOpenSource, citations, coverage, activeMarker,
}: Props) {
  const entries = [...(citations ?? [])]
    .filter((c) => Number.isInteger(Number(c.marker)) && (c.document || '').trim())
    .sort((a, b) => Number(a.marker) - Number(b.marker))
  const count = entries.length || sources?.length || 0
  return (
    <div className="sources-list">
      <header className="sources-list__head">
        <FileText size={14} />
        <span>Sources cited</span>
        {count > 0 && <span className="sources-list__count">{count}</span>}
      </header>

      {coverage && Number.isFinite(coverage.indexed) && Number.isFinite(coverage.total) && coverage.total > 0 ? (
        <p className="sources-list__coverage">
          {coverage.indexed} of {coverage.total} project documents indexed
        </p>
      ) : null}

      {entries.length > 0 ? (
        <ol className="sources-list__items sources-list__citations">
          {entries.map((c) => {
            const row = (sources ?? []).find((s) => s.doc_id && s.doc_name === c.document)
            const isActive = Number(c.marker) === activeMarker
            const body = <CitationRow citation={c} />
            return (
              <li
                key={`cite-${c.marker}`}
                id={`citation-entry-${c.marker}`}
                data-citation-marker={c.marker}
                className={'sources-list__item' + (isActive ? ' sources-list__item--active' : '')}
              >
                {row && onOpenSource ? (
                  <button
                    type="button"
                    className="sources-list__open"
                    onClick={() => onOpenSource(row)}
                    aria-label={`Preview ${visibleDocumentName(row)}`}
                  >
                    {body}
                  </button>
                ) : (
                  <div className="sources-list__static">{body}</div>
                )}
              </li>
            )
          })}
        </ol>
      ) : streaming && !sources?.length ? (
        <p className="sources-list__empty">Sources appear once the answer completes.</p>
      ) : !sources?.length ? (
        <p className="sources-list__empty">No citations for the current answer.</p>
      ) : (
        <ul className="sources-list__items">
          {sources.map((s, i) => {
            const canOpen = Boolean(s.doc_id && onOpenSource)
            const isActive = Boolean(s.doc_id && s.doc_id === activeDocId)
            return (
              <li key={`${s.doc_id}-${i}`} className="sources-list__item">
                {canOpen ? (
                  <button
                    type="button"
                    className={
                      'sources-list__open' +
                      (isActive ? ' sources-list__open--active' : '')
                    }
                    onClick={() => onOpenSource?.(s)}
                    aria-pressed={isActive}
                    aria-label={`Preview ${visibleDocumentName(s)}`}
                  >
                    <SourceRow source={s} />
                  </button>
                ) : (
                  <div className="sources-list__static">
                    <SourceRow source={s} />
                  </div>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

function visibleDocumentName(s: CitedSource): string {
  const name = (s.doc_name || '').trim()
  if (name) return name
  const id = (s.doc_id || '').trim()
  if (id && !textLeaksInternalCode(id)) return id
  return 'Document'
}

function CitationRow({ citation: c }: { citation: AnswerCitation }) {
  const isCalc = (c.kind || '').toLowerCase() === 'calculation'
  const tier = c.tier ? plainMetadata(String(c.tier)) : ''
  const section = c.section ? plainMetadata(c.section) : ''
  const passage = (c.passage || '').trim()
  return (
    <>
      <span className="sources-list__marker" aria-hidden="true">{c.marker}</span>
      <div className="sources-list__body">
        <div className="sources-list__doc" title={c.document}>{c.document}</div>
        {section || tier || isCalc ? (
          <div className="sources-list__ref">
            {[section, isCalc ? 'Calculation' : '', tier].filter(Boolean).join(' · ')}
          </div>
        ) : null}
        {passage ? (
          <blockquote className="sources-list__passage">
            {isCalc ? `Inputs: ${passage}` : passage}
          </blockquote>
        ) : null}
      </div>
    </>
  )
}

function SourceRow({ source: s }: { source: CitedSource }) {
  const sourceClass = (s.source_class || '').trim()
  const classLabel = visibleSourceClassLabel(sourceClass, s.source_class_label)
  const docName = visibleDocumentName(s)
  const page = plainMetadata(s.page_or_section || '')
  const layer = s.layer_label ? plainMetadata(s.layer_label) : ''
  return (
    <>
      <span className={`sources-list__chip sources-list__chip--${s.confidence.toLowerCase()}`}>
        {s.confidence}
      </span>
      <div className="sources-list__body">
        <div className="sources-list__doc" title={docName}>
          {docName}
        </div>
        <div className="sources-list__ref">
          {page}
          {layer ? ` · ${layer}` : ''}
        </div>
        {sourceClass && classLabel ? (
          <div
            className="sources-list__class"
            data-source-class={sourceClass}
            data-testid="source-class"
          >
            <span className="sources-list__class-label">{classLabel}</span>
          </div>
        ) : null}
      </div>
    </>
  )
}
