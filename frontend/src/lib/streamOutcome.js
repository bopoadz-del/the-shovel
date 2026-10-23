// How a chat answer's stream finished, for the user. Plain JS + .d.ts (like
// reconnect.js) so node --test runs it without a build step.

/** Shown when an answer's stream closed before any of the answer arrived. */
export const INTERRUPTED_MESSAGE =
  'The answer was interrupted before it arrived (the connection closed). Please send your question again.'

/** True when the stream closed without its end event and with no answer text. */
export function isInterruptedStream(streamEnded, content) {
  return !streamEnded && !String(content || '').trim()
}
