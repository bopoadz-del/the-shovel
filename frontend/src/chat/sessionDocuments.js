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
