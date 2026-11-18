// W2: each chat session keeps its own documents. A file uploaded while a
// session is open carries that session's id, so the server records it on
// the session (listed by GET .../conversations/{cid}/documents, pinned to
// that session's questions, kept out of the project's search). The sidebar
// shows the open session's files under "This session", each tagged with how
// it got there, above the project's documents.

/** Add the open session's id to an upload form. No session, no field: the
 *  server then treats the upload as a project document, as before. */
export function withSessionId(form, sessionId) {
  const sid = String(sessionId ?? '').trim()
  if (sid) form.append('session_id', sid)
  return form
}

/** The by-id route of one document (``tail``: ``/preview``,
 *  ``/preview/raw``, ``/download``), naming the open chat. A file attached to
 *  a chat is reachable by id only from that chat (the server answers 404
 *  anywhere else); a project document ignores the chat id. */
export function documentPath(projectId, docId, tail, sessionId) {
  const sid = String(sessionId ?? '').trim()
  const path = `/v1/projects/${projectId}/documents/${docId}${tail}`
  return sid ? `${path}?session_id=${encodeURIComponent(sid)}` : path
}

/** True when a document record may sit in the project's document list.
 *  A file attached to a chat session belongs to THAT session: it is shown
 *  in the session's own group and nowhere in the project's list, in any
 *  chat. The server records the session in the row's metadata and its
 *  project listings already leave such rows out; this is the same rule for
 *  a row the client adds by itself (an upload's response). */
export function belongsToProjectList(doc) {
  const meta = doc && typeof doc === 'object' ? doc.metadata : null
  const sid = meta && typeof meta === 'object' ? meta.session_id : null
  return !String(sid ?? '').trim()
}

/** The sidebar tag for a session document's origin. The server writes
 *  "uploaded" or "produced"; anything else is shown as uploaded, the
 *  server's own default for a session file without an origin. */
export function sessionDocumentTag(origin) {
  return origin === 'produced' ? 'Produced' : 'Uploaded'
}

/** The rows the "This session" group shows, only when the payload is for
 *  the session still open: a slow answer for a session the user has since
 *  left must not fill the new one. */
export function sessionDocumentsFor(fetchedId, activeId, payload) {
  if (!fetchedId || fetchedId !== activeId) return null
  const docs = payload && Array.isArray(payload.documents) ? payload.documents : []
  return docs
    .filter((d) => d && typeof d.id === 'string' && d.id)
    .map((d) => ({
      id: d.id,
      name: String(d.original_name ?? ''),
      tag: sessionDocumentTag(d.origin),
    }))
}

// G-1: what the "This session" group shows is a state, not just a list. A
// fetch that failed is not an empty session: saying "No files in this
// session yet" after a failed read is false, so a failure is kept as its own
// state, with any rows already known for that session, and shown as an
// error the user can retry. "No files" is said only after a read succeeded.

/** The state after a read of `fetchedId`'s documents succeeded. Null when
 *  the user has since opened another session (the reply is dropped). */
export function sessionDocumentsLoaded(fetchedId, activeId, payload) {
  const rows = sessionDocumentsFor(fetchedId, activeId, payload)
  return rows ? { sid: fetchedId, status: 'ready', rows } : null
}

/** The state after a read of `fetchedId`'s documents failed. Rows already
 *  shown for that session are kept; null when the user has moved on. */
export function sessionDocumentsFailed(fetchedId, activeId, previous) {
  if (!fetchedId || fetchedId !== activeId) return null
  const rows = previous && previous.sid === fetchedId && Array.isArray(previous.rows) ? previous.rows : []
  return { sid: fetchedId, status: 'error', rows }
}

/** What the group shows for the open session: 'loading' until its first
 *  read returns, 'error' after a failed read, else 'list' or 'empty'. */
export function sessionDocumentsView(state, activeId) {
  if (!activeId || !state || state.sid !== activeId) return { kind: 'loading', rows: [] }
  if (state.status === 'error') return { kind: 'error', rows: state.rows }
  return state.rows.length > 0 ? { kind: 'list', rows: state.rows } : { kind: 'empty', rows: [] }
}

/** Read the open session's documents once and return the next state (or
 *  null to keep the current one). `getJson` is the API read; `activeIdNow`
 *  is read after the reply so a reply for a session left behind is dropped. */
export async function readSessionDocuments(getJson, path, fetchedId, activeIdNow, previous) {
  try {
    const payload = await getJson(path)
    return sessionDocumentsLoaded(fetchedId, activeIdNow(), payload)
  } catch {
    return sessionDocumentsFailed(fetchedId, activeIdNow(), previous)
  }
}
