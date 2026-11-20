// Answer feedback (ML-2). The control names an answer by its assistant index
// (the same index the download uses); the server resolves the stored message.
// Before the control says "recorded" it checks the server's echo against the
// answer on screen, so a drifted index never reads as a recorded verdict.

export const FEEDBACK_KINDS = [
  { kind: 'correct', label: 'Correct' },
  { kind: 'wrong_figure', label: 'Wrong figure' },
  { kind: 'wrong_document', label: 'Wrong document' },
  { kind: 'missing', label: 'Missing' },
]

export const NOTE_MAX = 200
export const UNIT_MAX = 32
export const HEAD_CHARS = 80

export function collapsed(text) {
  return String(text ?? '').split(/\s+/).filter(Boolean).join(' ')
}

export function answerHead(content) {
  return collapsed(content).slice(0, HEAD_CHARS)
}

// The request body for one verdict, or an error string when an input is
// not acceptable. Only the fields the kind accepts are sent.
export function buildFeedbackBody({ projectId, conversationId, assistantIndex, kind, value, unit, note, content }) {
  const body = {
    project_id: projectId,
    conversation_id: conversationId,
    message_index: assistantIndex,
    kind,
    answer_head: answerHead(content),
  }
  if (kind === 'wrong_figure') {
    const v = String(value ?? '').trim()
    if (v !== '') {
      const n = Number(v)
      if (!Number.isFinite(n)) return { error: 'The value must be a number.' }
      body.correct_value = n
    }
    const u = String(unit ?? '').trim()
    if (u.length > UNIT_MAX) return { error: `The unit is longer than ${UNIT_MAX} characters.` }
    if (u) body.unit = u
  } else if (kind === 'wrong_document' || kind === 'missing') {
    const t = String(note ?? '').trim()
    if (t.length > NOTE_MAX) return { error: `The note is longer than ${NOTE_MAX} characters.` }
    if (t) body.note = t
  }
  return { body }
}

// True only when the server recorded the verdict on the answer shown.
export function echoMatches(response, content) {
  if (!response || response.status !== 'recorded' || !response.message_id) return false
  const echoed = collapsed(response.answer_head)
  return echoed.length > 0 && collapsed(content).startsWith(echoed)
}
