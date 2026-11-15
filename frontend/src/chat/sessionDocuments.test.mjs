// node --test frontend/src/chat/sessionDocuments.test.mjs
// W2 slice 2 (exam 11.3): uploads carry the open session's id, and the
// sidebar's "This session" group lists only the open session's files, each
// tagged Uploaded or Produced. Every case has an input that must pass and
// one that must fail.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'

import { sessionDocumentTag, sessionDocumentsFor, withSessionId } from './sessionDocuments.js'

class FakeForm {
  constructor() { this.fields = [] }
  append(k, v) { this.fields.push([k, v]) }
}

const read = (rel) => fs.readFileSync(new URL(rel, import.meta.url), 'utf8')

test('an upload in a session carries its id; without one it carries none', () => {
  const inSession = withSessionId(new FakeForm(), 'ws-p1-abc')
  assert.deepEqual(inSession.fields, [['session_id', 'ws-p1-abc']])
  for (const none of [null, undefined, '', '   ']) {
    assert.deepEqual(withSessionId(new FakeForm(), none).fields, [])
  }
})

test('the tag follows the origin the server recorded', () => {
  assert.equal(sessionDocumentTag('produced'), 'Produced')
  assert.equal(sessionDocumentTag('uploaded'), 'Uploaded')
  assert.equal(sessionDocumentTag(undefined), 'Uploaded')
})

test('the rows fill only the session they were fetched for', () => {
  const payload = {
    documents: [
      { id: 'd1', original_name: 'a.md', origin: 'uploaded' },
      { id: 'd2', original_name: 'b.xlsx', origin: 'produced' },
    ],
  }
  assert.deepEqual(sessionDocumentsFor('ws-p-1', 'ws-p-1', payload), [
    { id: 'd1', name: 'a.md', tag: 'Uploaded' },
    { id: 'd2', name: 'b.xlsx', tag: 'Produced' },
  ])
  assert.equal(sessionDocumentsFor('ws-p-1', 'ws-p-2', payload), null)
  assert.equal(sessionDocumentsFor(null, 'ws-p-2', payload), null)
})

test('a row without an id is dropped; a malformed payload is no rows', () => {
  const rows = sessionDocumentsFor('s', 's', { documents: [{ original_name: 'x' }, { id: 'k' }] })
  assert.deepEqual(rows, [{ id: 'k', name: '', tag: 'Uploaded' }])
  assert.deepEqual(sessionDocumentsFor('s', 's', { documents: 'nope' }), [])
  assert.deepEqual(sessionDocumentsFor('s', 's', null), [])
})

test('both upload paths send the open session id', () => {
  const workspace = read('../pages/ProjectWorkspace.tsx')
  const composer = read('./ChatComposer.tsx')
  assert.ok(workspace.includes('withSessionId(form, sessionId)'), 'documents panel upload')
  assert.ok(composer.includes('withSessionId(fd, sessionId)'), 'composer upload')
  assert.ok(workspace.includes('sessionId={conversationId}'), 'open session wired to both')
})

test('"This session" sits above the project documents in the sidebar', () => {
  const panel = read('../layout/LeftPanel.tsx')
  const sessionAt = panel.indexOf('>This session</header>')
  const docsAt = panel.indexOf('>Documents</header>')
  assert.ok(sessionAt > 0 && docsAt > 0, 'both groups present')
  assert.ok(sessionAt < docsAt, 'This session above Documents')
})
