/** The turn a note was written in, as the admin list labels it. */
export interface TurnLabel {
  conversation: string | null
  message: number | null
}

/** Where a note was written, in words; never the stored turn id. */
export function turnWords(turn?: TurnLabel): string {
  if (!turn) return ''
  const where = turn.conversation ? `“${turn.conversation}”` : 'a conversation'
  return turn.message ? `message ${turn.message} of ${where}` : turn.conversation ? `in ${where}` : ''
}
