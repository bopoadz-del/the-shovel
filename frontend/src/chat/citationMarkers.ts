/* Citation markers in an answer — W3 (citations in the side panel only).
 *
 * The answer contract leaves a numbered marker ``[n]`` in the text for each
 * cited passage and sends the passage itself beside the answer. Here a marker
 * that HAS an entry becomes a link to that entry; a bracketed number with no
 * entry (a figure reference, a list index) is left exactly as written, so no
 * control ever points at nothing. */
import type { AnswerCitation, AnswerCoverage } from './types'

export const CITATION_HREF_PREFIX = '#citation-'

// ``[n]`` not already a link label (``[n](``), a reference definition
// (``[n]:``) or the first half of a reference link (``[n][``).
const MARKER_RE = /\[(\d{1,3})\](?![(:[])/g

/** The answer with each marker that has an entry rewritten as a link. */
export function linkCitationMarkers(content: string, citations?: AnswerCitation[]): string {
  if (!content || !citations?.length) return content
  const known = new Set(citations.map((c) => Number(c.marker)).filter((n) => Number.isInteger(n)))
  if (!known.size) return content
  return content.replace(MARKER_RE, (whole, digits: string) =>
    known.has(Number(digits)) ? `[${digits}](${CITATION_HREF_PREFIX}${digits})` : whole,
  )
}

/** The marker a link points at, or null when it is an ordinary link. */
export function citationMarkerFromHref(href?: string): number | null {
  if (!href || !href.startsWith(CITATION_HREF_PREFIX)) return null
  const n = Number(href.slice(CITATION_HREF_PREFIX.length))
  return Number.isInteger(n) ? n : null
}

/** The SSE 'end' event's ``coverage`` field, or undefined when absent or malformed. */
export function readCoverage(value: unknown): AnswerCoverage | undefined {
  if (!value || typeof value !== 'object') return undefined
  const { indexed, total } = value as Record<string, unknown>
  if (typeof indexed !== 'number' || typeof total !== 'number') return undefined
  if (!Number.isFinite(indexed) || !Number.isFinite(total) || total <= 0) return undefined
  return { indexed, total }
}

/** The SSE 'end' event's ``details`` field (rule 5 of the answer contract:
 *  supporting text moved out of the lead), or undefined when absent, blank
 *  or not text. */
export function readDetails(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined
}

/** A ``citations`` field (the SSE 'end' event, or a stored answer read back
 *  when a session is reopened), or undefined when absent or not a list. */
export function readCitations(value: unknown): AnswerCitation[] | undefined {
  return Array.isArray(value) ? (value as AnswerCitation[]) : undefined
}
