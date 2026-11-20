/* AnswerFeedback — the user's verdict on one settled answer (ML-2).
 *
 * Four choices under the bubble. "Wrong figure" takes an optional value and
 * unit; "Wrong document" and "Missing" take an optional note. The control
 * says "Thanks, recorded" only after the server's echo matches the answer
 * on screen; anything else is an inline error. It never changes the answer.
 */
import { useState } from 'react'
import {
  FEEDBACK_KINDS,
  NOTE_MAX,
  UNIT_MAX,
  buildFeedbackBody,
  echoMatches,
  type FeedbackBody,
  type FeedbackEcho,
  type FeedbackKind,
} from './feedbackPayload'
import './AnswerFeedback.css'

interface Props {
  content: string
  projectId: string
  conversationId: string
  /** Position among the conversation's assistant turns (as for download). */
  assistantIndex: number
  post: (body: FeedbackBody) => Promise<FeedbackEcho>
}

type Phase = 'idle' | 'sending' | 'recorded'

export default function AnswerFeedback({ content, projectId, conversationId, assistantIndex, post }: Props) {
  const [kind, setKind] = useState<FeedbackKind | null>(null)
  const [value, setValue] = useState('')
  const [unit, setUnit] = useState('')
  const [note, setNote] = useState('')
  const [phase, setPhase] = useState<Phase>('idle')
  const [error, setError] = useState<string | null>(null)

  if (phase === 'recorded') {
    return <p className="answer-feedback__done" role="status">Thanks, recorded</p>
  }

  const send = (chosen: FeedbackKind) => {
    const built = buildFeedbackBody({
      projectId, conversationId, assistantIndex, kind: chosen, value, unit, note, content,
    })
    if (!built.body) {
      setError(built.error)
      return
    }
    setPhase('sending')
    setError(null)
    post(built.body)
      .then((echo) => {
        if (echoMatches(echo, content)) {
          setPhase('recorded')
        } else {
          setPhase('idle')
          setError('Could not record: this answer no longer matches the saved one.')
        }
      })
      .catch((e: unknown) => {
        setPhase('idle')
        setError(`Could not record: ${(e as Error)?.message || 'request failed'}`)
      })
  }

  const needsDetail = kind === 'wrong_figure' || kind === 'wrong_document' || kind === 'missing'
  const busy = phase === 'sending'

  return (
    <div className="answer-feedback" role="group" aria-label="Rate this answer">
      <div className="answer-feedback__choices">
        {FEEDBACK_KINDS.map(({ kind: k, label }) => (
          <button
            key={k}
            type="button"
            className={`answer-feedback__choice${kind === k ? ' is-active' : ''}`}
            disabled={busy}
            aria-pressed={kind === k}
            onClick={() => {
              setError(null)
              if (k === 'correct') {
                setKind(k)
                send(k)
              } else {
                setKind(k)
              }
            }}
          >
            {label}
          </button>
        ))}
      </div>
      {needsDetail && (
        <div className="answer-feedback__detail">
          {kind === 'wrong_figure' ? (
            <>
              <input
                className="answer-feedback__input"
                inputMode="decimal"
                placeholder="Correct value (optional)"
                aria-label="Correct value"
                value={value}
                disabled={busy}
                onChange={(e) => setValue(e.target.value)}
              />
              <input
                className="answer-feedback__input answer-feedback__input--unit"
                placeholder="Unit"
                aria-label="Unit"
                maxLength={UNIT_MAX}
                value={unit}
                disabled={busy}
                onChange={(e) => setUnit(e.target.value)}
              />
            </>
          ) : (
            <>
              <input
                className="answer-feedback__input"
                placeholder="What was wrong or missing? (optional)"
                aria-label="Note"
                maxLength={NOTE_MAX}
                value={note}
                disabled={busy}
                onChange={(e) => setNote(e.target.value)}
              />
              <span className="answer-feedback__count">{note.length}/{NOTE_MAX}</span>
            </>
          )}
          <button
            type="button"
            className="answer-feedback__send"
            disabled={busy}
            onClick={() => kind && send(kind)}
          >
            {busy ? 'Sending…' : 'Send'}
          </button>
        </div>
      )}
      {error && <p className="answer-feedback__error" role="alert">{error}</p>}
    </div>
  )
}
