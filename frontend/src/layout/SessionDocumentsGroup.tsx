// W2 / G-1: the "This session" group in the sidebar. It renders the view
// computed by sessionDocumentsView: a failed read is an error with a retry,
// never "No files in this session yet" (that line is said only after a read
// succeeded and returned nothing).
import type { SessionDocumentsView } from '../chat/sessionDocuments'

interface Props {
  view: SessionDocumentsView
  onRetry?: () => void
}

export default function SessionDocumentsGroup({ view, onRetry }: Props) {
  return (
    <section className="left-panel__section" aria-label="This session's documents">
      <header className="left-panel__section-head">This session</header>
      {view.kind === 'error' ? (
        <p className="left-panel__empty left-panel__session-docs-error" role="alert">
          Could not load this session's files.{' '}
          {onRetry ? (
            <button type="button" className="left-panel__session-docs-retry" onClick={onRetry}>
              Try again
            </button>
          ) : null}
        </p>
      ) : null}
      {view.rows.length > 0 ? (
        <ul className="left-panel__session-docs">
          {view.rows.map((d) => (
            <li key={d.id} className="left-panel__session-doc" title={d.name}>
              <span className="left-panel__session-doc-name">{d.name}</span>
              <span
                className={`left-panel__session-doc-tag left-panel__session-doc-tag--${d.tag.toLowerCase()}`}
              >
                {d.tag}
              </span>
            </li>
          ))}
        </ul>
      ) : null}
      {view.kind === 'loading' ? <p className="left-panel__empty">Loading this session's files…</p> : null}
      {view.kind === 'empty' ? <p className="left-panel__empty">No files in this session yet.</p> : null}
    </section>
  )
}
