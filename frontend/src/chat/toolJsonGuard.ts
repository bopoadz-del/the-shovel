/**
 * Client-side guard against leaked internal tool-call JSON in assistant
 * bubbles (UI-PHYS A5). The server must never emit these; this is a
 * last-line defence for stream races and older persisted history.
 */

const TOOL_JSON_FALLBACK =
  'I hit an internal search formatting issue before I could produce a grounded answer. Please retry or narrow the question.'

function isSearchArgsObject(obj: Record<string, unknown>): boolean {
  const keys = Object.keys(obj)
  if ('arguments' in obj || 'parameters' in obj) {
    if ('name' in obj || 'tool' in obj || 'function' in obj) return true
  }
  if (typeof obj['query'] !== 'string') return false
  if ('answer' in obj || 'result' in obj || 'results' in obj || 'items' in obj) {
    return false
  }
  const allowed = new Set([
    'query',
    'top_k',
    'project_id',
    'corpus',
    'filters',
    'source',
    'tool',
    'function',
  ])
  return keys.every((k) => allowed.has(k))
}

function isToolCallObject(obj: unknown): boolean {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return false
  const rec = obj as Record<string, unknown>
  if ('name' in rec && ('arguments' in rec || 'parameters' in rec)) return true
  if (rec['type'] === 'function' && typeof rec['function'] === 'object') return true
  return isSearchArgsObject(rec)
}

/** True when `text` is (or is only) a leaked tool-call / search-args envelope. */
export function looksLikeToolCallJson(text: string): boolean {
  const trimmed = (text || '').trim()
  if (!trimmed) return false
  const unfenced = trimmed.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim()
  if (!unfenced.startsWith('{') && !unfenced.startsWith('[')) return false
  try {
    const parsed: unknown = JSON.parse(unfenced)
    if (Array.isArray(parsed)) {
      return parsed.length > 0 && parsed.every(isToolCallObject)
    }
    return isToolCallObject(parsed)
  } catch {
    return false
  }
}

/** Replace a leaked tool-call envelope with a safe user-facing fallback. */
export function sanitizeAssistantContent(text: string): string {
  if (looksLikeToolCallJson(text)) return TOOL_JSON_FALLBACK
  return text
}
