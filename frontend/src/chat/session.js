// Workspace chat sessions. Opening a project and "New chat" both mint a
// fresh id. The legacy stable id (ws-{projectId}) is never reused, so a
// send cannot land on an older thread. History for a different id is
// dropped. Titles shown in the list collapse whitespace; an empty stored
// title displays as "New chat".

export function legacyConversationId(projectId) {
  if (!projectId) return null
  return `ws-${projectId}`
}

let entropySeq = 0

export function freshSessionEntropy() {
  entropySeq += 1
  const stamp = Date.now().toString(36)
  const seq = entropySeq.toString(36)
  const rand = Math.random().toString(36).slice(2)
  return `${stamp}${seq}${rand}`.replace(/[^a-z0-9]/g, '')
}

export function mintConversationId(projectId, entropy) {
  if (!projectId || entropy == null || String(entropy) === '') return null
  const id = `ws-${projectId}-${entropy}`
  if (id === legacyConversationId(projectId)) return null
  return id
}

export function openConversationId(projectId) {
  return mintConversationId(projectId, freshSessionEntropy())
}

export function historyBelongsToSession(fetchedId, activeId, messages) {
  if (!fetchedId || fetchedId !== activeId) return null
  return Array.isArray(messages) ? messages : null
}

export function orderSessionsNewestFirst(rows) {
  return [...(rows || [])].sort((a, b) =>
    String(b?.updated_at || '').localeCompare(String(a?.updated_at || '')),
  )
}

export function sessionListTitle(title) {
  const collapsed = String(title ?? '').replace(/\s+/g, ' ').trim()
  return collapsed || 'New chat'
}

const TITLE_MAX = 80

export function renameTitleError(title) {
  const text = String(title ?? '')
  if (text.includes('<') || text.includes('>')) return 'Title cannot contain HTML'
  const collapsed = text.replace(/\s+/g, ' ').trim()
  if (!collapsed) return 'Title cannot be empty'
  if (collapsed.length > TITLE_MAX) return 'Title is too long'
  return null
}

// The open session is remembered per project in the browser, so a reload or
// leaving the workspace and coming back reopens the same thread. Only an id
// minted for this project is restored; anything else (the legacy stable
// thread, another project's id, junk) starts a fresh session instead.

export function sessionMemoryKey(projectId) {
  if (!projectId) return null
  return `shovel.session.${projectId}`
}

export function restorableSessionId(projectId, stored) {
  if (!projectId || typeof stored !== 'string') return null
  const prefix = `${legacyConversationId(projectId)}-`
  if (!stored.startsWith(prefix) || stored.length <= prefix.length) return null
  if (/[^A-Za-z0-9_-]/.test(stored.slice(prefix.length))) return null
  return stored
}

export function readRememberedSession(projectId, storage) {
  const key = sessionMemoryKey(projectId)
  if (!key || !storage) return null
  try {
    return restorableSessionId(projectId, storage.getItem(key))
  } catch {
    return null
  }
}

export function rememberSession(projectId, conversationId, storage) {
  const key = sessionMemoryKey(projectId)
  if (!key || !storage) return
  try {
    if (restorableSessionId(projectId, conversationId)) storage.setItem(key, conversationId)
    else storage.removeItem(key)
  } catch {
    /* storage full or blocked: the session still works, it is just not remembered */
  }
}

// After a session is deleted: the list without it, and whether the open
// thread must be replaced by a fresh one.
export function afterSessionDeleted(rows, deletedId, activeId) {
  return {
    rows: (rows || []).filter((row) => row?.id !== deletedId),
    replaceActive: !!deletedId && deletedId === activeId,
  }
}

export function deleteSessionQuestion(title) {
  return `Delete the session "${sessionListTitle(title)}"? Its messages are removed and this cannot be undone.`
}
