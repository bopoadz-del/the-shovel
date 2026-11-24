/**
 * What a person may read in source, tool, and calculator metadata.
 *
 * Machine identifiers (snake_case ids, key=value assignments) stay in
 * data attributes and JSON fields. Every chat surface that renders that
 * metadata runs the text through here first, so the rule does not depend
 * on which class, tool, document, or project produced the value.
 */

const SNAKE = /\b[a-z]+_[a-z_]+\b/
const KEY_VALUE = /\w+=\w+/

export function textLeaksInternalCode(text: string): boolean {
  return SNAKE.test(text) || KEY_VALUE.test(text)
}

/** ``alpha_beta`` → "alpha beta". Already-plain text is unchanged. */
export function wordsForIdentifier(value: string): string {
  return (value || '').replace(/_+/g, ' ').replace(/\s+/g, ' ').trim()
}

/**
 * Drop key=value assignments and read any leftover snake_case token as
 * words. Document titles are not metadata and must not be passed through.
 */
export function plainMetadata(text: string): string {
  return (text || '')
    .replace(/\b\w+=[^\s),]+/g, ' ')
    .replace(/\b[a-z]+_[a-z_]+\b/g, (token) => wordsForIdentifier(token))
    .replace(/\(\s*(?:,\s*)*\)/g, '')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\s+([,.;:])/g, '$1')
    .trim()
}

/** The line shown while a stream event's work runs. */
export const WORKING_LINE = 'Working through your request…'

/**
 * The activity line for a ``tool_call`` stream event: the label the server
 * gives the KIND of work (searching, calculating, working), never the
 * tool's name, its arguments or the model's query. An event without a
 * label is shown as plain work.
 */
export function activityLine(evt: Record<string, unknown>): string {
  const label = typeof evt['label'] === 'string' ? evt['label'].trim() : ''
  return label || WORKING_LINE
}
