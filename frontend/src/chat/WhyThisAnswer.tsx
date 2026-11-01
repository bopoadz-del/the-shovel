/* WhyThisAnswer — per-figure provenance for the latest assistant answer.
 *
 * Lives in the RightPanel's Sources tab, beneath SourcesList. Data shape comes
 * verbatim from the SSE `end` event's `provenance` list (S-12a), which the
 * backend builds in app.agents.citation_provenance.figure_provenance. Each
 * entry says WHERE a number in the answer came from:
 *   • a calculator run — with its envelope badge (certified kernel / reference
 *     data / estimated), the standard it states, and the inputs it used, each
 *     input carrying WHERE its value came from (S-12b);
 *   • the operator's own words;
 *   • a retrieved document chunk.
 *
 * No fetching here, and no internal identifiers reach the screen: the badge
 * renders a fixed vocabulary, the formula name is humanised + guarded, and an
 * input's source is a plain word ("your words" / "project data" / "document" /
 * "unresolved"). A chunk id is never shown.
 */
import { useState } from 'react'
import { Calculator, FileText, MessageSquare } from 'lucide-react'
import type { CitedSource } from './SourcesList'
import { textLeaksInternalCode } from './userFacingText'
import './WhyThisAnswer.css'

/** One calculator input and where its value came from (S-12a/S-12b). */
export interface ProvenanceInput {
  name: string
  value: unknown
  unit?: string
  /** user_text | project_data | document:<chunk_id> | unresolved | absent. */
  source?: string
}

export interface ProvenanceEntry {
  figure: string
  source: string
  /** calculator entries */
  formula?: string
  inputs?: ProvenanceInput[]
  provenance?: 'kernel' | 'reference_table' | 'improvised' | string
  standard?: string
  certified?: boolean
  defaults?: string[]
  /** retrieval entries */
  doc_name?: string
  page_or_section?: string
  layer_label?: string
}

interface Props {
  /** Latest assistant message's provenance, or undefined while streaming. */
  entries?: ProvenanceEntry[]
  /** The same cited sources SourcesList shows, so a document-sourced input can
   *  link to its row in the preview pane. */
  sources?: CitedSource[]
  /** True while the assistant message is still streaming. */
  streaming?: boolean
  /** Open a cited document in the preview pane (shared with SourcesList). */
  onOpenSource?: (source: CitedSource) => void
}

/** Match a ``document:<chunk_id>`` input source to the cited source it came
 *  from. The backend stamps ``document:<doc_id>`` or ``document:<doc_id>#<n>``
 *  (citation_provenance), so the doc id is the part before any ``#``. Returns
 *  null when nothing matches — the word then renders plainly. */
export function resolveDocumentSource(
  source: string | undefined,
  sources: CitedSource[] | undefined,
): CitedSource | null {
  if (!source || !source.startsWith('document:') || !sources) return null
  const docId = source.slice('document:'.length).split('#')[0]
  if (!docId) return null
  return sources.find((s) => s.doc_id === docId) ?? null
}

const BADGE: Record<string, { label: string; tone: string }> = {
  kernel: { label: 'Certified calculation', tone: 'ok' },
  reference_table: { label: 'Reference data', tone: 'info' },
  improvised: { label: 'Estimated', tone: 'warn' },
}

/** A plain word for where one input's value came from. An unknown value is
 *  shown raw so a backend regression is visible, not hidden; an absent source
 *  shows nothing. "derived" is never emitted by the backend (ruling A). */
function sourceWord(source?: string): string {
  if (!source) return ''
  if (source === 'user_text') return 'your words'
  if (source === 'project_data') return 'project data'
  if (source.startsWith('document:')) return 'document' // chunk id never shown
  if (source === 'unresolved') return 'unresolved'
  return source
}

function humanName(name: string): string {
  return (name || '').replace(/_/g, ' ').trim()
}

/** A registry name ("dewatering_uplift_check") shown as words, never if it
 *  still looks like an internal code after humanising. */
function humaniseFormula(name?: string): string | null {
  if (!name) return null
  const words = name.replace(/_/g, ' ').trim()
  if (!words || textLeaksInternalCode(words)) return null
  return words.charAt(0).toUpperCase() + words.slice(1)
}

function CalcRow({
  e,
  sources,
  onOpenSource,
}: {
  e: ProvenanceEntry
  sources?: CitedSource[]
  onOpenSource?: (source: CitedSource) => void
}) {
  const [open, setOpen] = useState(false)
  const badge = (e.provenance && BADGE[e.provenance]) || null
  const formula = humaniseFormula(e.formula)
  const standard = e.standard && e.standard !== 'none declared' ? e.standard : null
  const inputs = Array.isArray(e.inputs) ? e.inputs : []
  return (
    <li className="why-row" data-source="calculator">
      <Calculator className="why-row__icon" aria-hidden />
      <div className="why-row__body">
        <div className="why-row__head">
          <span className="why-row__figure">{e.figure}</span>
          {badge && <span className={`why-badge why-badge--${badge.tone}`}>{badge.label}</span>}
        </div>
        {formula && <div className="why-row__formula">{formula}</div>}
        {inputs.length > 0 && (
          <>
            <button
              type="button"
              className="why-row__toggle"
              aria-expanded={open}
              onClick={() => setOpen((o) => !o)}
            >
              {open ? 'Hide inputs' : `Inputs (${inputs.length})`}
            </button>
            <ul className="why-row__inputs" hidden={!open}>
              {inputs.map((inp, i) => {
                const word = sourceWord(inp.source)
                const cited = onOpenSource ? resolveDocumentSource(inp.source, sources) : null
                return (
                  <li key={i}>
                    <span className="why-row__k">{humanName(inp.name)}</span>:{' '}
                    {String(inp.value)}
                    {inp.unit ? ` ${inp.unit}` : ''}
                    {word &&
                      (cited ? (
                        <>
                          {' · '}
                          <button
                            type="button"
                            className="why-row__srclink"
                            onClick={() => onOpenSource?.(cited)}
                          >
                            {word}
                          </button>
                        </>
                      ) : (
                        <span className="why-row__src"> · {word}</span>
                      ))}
                  </li>
                )
              })}
            </ul>
          </>
        )}
        {standard && <div className="why-row__standard">per {standard}</div>}
        {e.defaults && e.defaults.length > 0 && (
          <ul className="why-row__defaults">
            {e.defaults.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        )}
      </div>
    </li>
  )
}

function SimpleRow({ e }: { e: ProvenanceEntry }) {
  const isUser = e.source === 'user_input'
  const Icon = isUser ? MessageSquare : FileText
  const where = isUser
    ? 'From your message'
    : [e.doc_name, e.layer_label, e.page_or_section].filter(Boolean).join(' · ') || 'From a retrieved document'
  return (
    <li className="why-row" data-source={isUser ? 'user' : 'document'}>
      <Icon className="why-row__icon" aria-hidden />
      <div className="why-row__body">
        <div className="why-row__head">
          <span className="why-row__figure">{e.figure}</span>
        </div>
        <div className="why-row__where">{where}</div>
      </div>
    </li>
  )
}

export default function WhyThisAnswer({ entries, sources, streaming, onOpenSource }: Props) {
  // Only figures the turn could actually attribute; a null-source figure never
  // reaches the client, but guard anyway.
  const rows = (entries ?? []).filter((e) => e && e.source)
  if (streaming && rows.length === 0) {
    return (
      <section className="why-panel" aria-label="Why this answer">
        <h3 className="why-panel__title">Why this answer</h3>
        <p className="why-panel__empty">The breakdown appears once the answer completes.</p>
      </section>
    )
  }
  if (rows.length === 0) return null
  return (
    <section className="why-panel" aria-label="Why this answer">
      <h3 className="why-panel__title">Why this answer</h3>
      <ul className="why-panel__list">
        {rows.map((e, i) =>
          e.source === 'calculator' ? (
            <CalcRow key={i} e={e} sources={sources} onOpenSource={onOpenSource} />
          ) : (
            <SimpleRow key={i} e={e} />
          ),
        )}
      </ul>
    </section>
  )
}
