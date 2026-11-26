/* "How this was worked out": the working sheet of a catalogue-path turn,
 * folded under the answer. Display only -- the answer text above it is the
 * reply as sent and is never rewritten, re-ordered or added to here.
 *
 * One renderer for any sheet (workingSheet.js builds the view): the givens
 * with the source code matched for each, every plan step with its tool, its
 * inputs (value and unit, resolved from the sheet), its result or why it has
 * none, and the inputs still missing. No figure is shown for a step that did
 * not run to a result. */
import { endingWords, showNumber, workingView, type SheetInputView } from './workingSheet'
import './ShowWorking.css'

interface Props {
  sheet: unknown
  provenance?: unknown
  /** trace/1 `ended`: answer | asked_input | ceiling. */
  ended?: string
}

function unitWords(unit: unknown): string {
  const u = String(unit ?? '').trim()
  return u === '-' ? '' : u
}

function quantity(value: unknown, unit: unknown): string {
  return [showNumber(value), unitWords(unit)].filter(Boolean).join(' ')
}

function inputLine(input: SheetInputView): string {
  const from = input.ref ? ` (from ${input.ref})` : ''
  if (input.missing) return `${input.name}: not available yet${from}`
  return `${input.name}: ${quantity(input.value, input.unit)}${from}`
}

const STATUS_WORDS: Record<string, string> = {
  done: 'calculated',
  failed: 'not calculated',
  waiting_input: 'waiting for an input',
  not_run: 'not run',
}

export default function ShowWorking({ sheet, provenance, ended }: Props) {
  const view = workingView(sheet, provenance)
  if (!view) return null
  const ending = endingWords(ended)
  return (
    <details className="show-working" data-testid="show-working">
      <summary className="show-working__summary">
        How this was worked out: {view.steps.length} {view.steps.length === 1 ? 'step' : 'steps'},{' '}
        {view.givens.length} {view.givens.length === 1 ? 'given' : 'givens'}
      </summary>

      {view.givens.length > 0 && (
        <section className="show-working__section">
          <div className="show-working__heading">Givens</div>
          <ul className="show-working__list">
            {view.givens.map((g) => (
              <li key={g.id} className="show-working__item" data-testid="sw-given">
                <strong>{g.id}</strong> {g.what}: {quantity(g.value, g.unit)}
                {g.source ? (
                  <span className="show-working__chip">{g.source}</span>
                ) : (
                  <span className="show-working__chip show-working__chip--none">no source found</span>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      {view.steps.length > 0 && (
        <section className="show-working__section">
          <div className="show-working__heading">Steps</div>
          <ol className="show-working__list">
            {view.steps.map((s) => (
              <li key={s.id} className={`show-working__item show-working__item--${s.status}`} data-testid="sw-step" data-status={s.status}>
                <strong>{s.id}</strong> {s.tool}
                <span className="show-working__chip">{STATUS_WORDS[s.status] ?? s.status}</span>
                {s.inputs.length > 0 && (
                  <ul className="show-working__inputs">
                    {s.inputs.map((i) => <li key={i.name}>{inputLine(i)}</li>)}
                  </ul>
                )}
                {s.result && (
                  <div data-testid="sw-result">
                    Result: {quantity(s.result.value, s.result.unit)} <span className="show-working__chip">calculator</span>
                  </div>
                )}
                {s.error && <div data-testid="sw-error">Why not: {s.error}</div>}
                {s.needs.length > 0 && <div data-testid="sw-needs">Waiting for: {s.needs.join(', ')}</div>}
              </li>
            ))}
          </ol>
        </section>
      )}

      {view.missing.length > 0 && (
        <section className="show-working__section">
          <div className="show-working__heading">Missing</div>
          <ul className="show-working__list">
            {view.missing.map((m) => (
              <li key={`${m.for}:${m.label}`} className="show-working__item" data-testid="sw-missing">
                {m.label}{m.for ? ` (for ${m.for})` : ''}
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="show-working__ending" data-testid="sw-ending">
        {[ending, view.unsourced === 0
          ? 'every given has a source'
          : `${view.unsourced} ${view.unsourced === 1 ? 'given has' : 'givens have'} no source found`]
          .filter(Boolean).join(' · ')}
      </div>
    </details>
  )
}
