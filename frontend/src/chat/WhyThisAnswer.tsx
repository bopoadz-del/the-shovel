/* WhyThisAnswer — per-figure provenance for the latest assistant answer.
 *
 * Lives in the RightPanel's Sources tab, beneath SourcesList. Data shape comes
 * verbatim from the SSE `end` event's `provenance` list (S-12a), which the
 * backend builds in app.agents.citation_provenance.figure_provenance. Each
 * entry says WHERE a number in the answer came from:
 *   • a calculator run — with its envelope badge (certified kernel / reference
 *     data / estimated), the standard it states, and any inputs it defaulted;
 *   • the operator's own words;
 *   • a retrieved document chunk.
 *
 * No fetching here, and no internal identifiers reach the screen: the badge
 * renders a fixed vocabulary and the formula name is humanised + guarded.
 */
import { Calculator, FileText, MessageSquare } from 'lucide-react'
import { textLeaksInternalCode } from './userFacingText'
import './WhyThisAnswer.css'

export interface ProvenanceEntry {
  figure: string
  source: string
  /** calculator entries */
  formula?: string
  inputs?: Record<string, unknown>
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
  /** True while the assistant message is still streaming. */
  streaming?: boolean
}

const BADGE: Record<string, { label: string; tone: string }> = {
  kernel: { label: 'Certified calculation', tone: 'ok' },
  reference_table: { label: 'Reference data', tone: 'info' },
  improvised: { label: 'Estimated', tone: 'warn' },
}

/** A registry name ("dewatering_uplift_check") shown as words, never if it
 *  still looks like an internal code after humanising. */
function humaniseFormula(name?: string): string | null {
  if (!name) return null
  const words = name.replace(/_/g, ' ').trim()
  if (!words || textLeaksInternalCode(words)) return null
  return words.charAt(0).toUpperCase() + words.slice(1)
}

function CalcRow({ e }: { e: ProvenanceEntry }) {
  const badge = (e.provenance && BADGE[e.provenance]) || null
  const formula = humaniseFormula(e.formula)
  const standard = e.standard && e.standard !== 'none declared' ? e.standard : null
  const inputs = e.inputs ? Object.entries(e.inputs) : []
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
          <ul className="why-row__inputs">
            {inputs.map(([k, v]) => (
              <li key={k}>
                <span className="why-row__k">{k.replace(/_/g, ' ')}</span>: {String(v)}
              </li>
            ))}
          </ul>
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

export default function WhyThisAnswer({ entries, streaming }: Props) {
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
          e.source === 'calculator' ? <CalcRow key={i} e={e} /> : <SimpleRow key={i} e={e} />,
        )}
      </ul>
    </section>
  )
}
