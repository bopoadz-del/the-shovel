// node --test frontend/src/chat/sessionDocuments.test.mjs
// W2 slice 2 (exam 11.3): uploads carry the open session's id, and the
// sidebar's "This session" group lists only the open session's files, each
// tagged Uploaded or Produced. Every case has an input that must pass and
// one that must fail.
import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  belongsToProjectList,
  readSessionDocuments,
  sessionDocumentTag,
  sessionDocumentsFailed,
  sessionDocumentsFor,
  sessionDocumentsLoaded,
  sessionDocumentsView,
  withSessionId,
} from './sessionDocuments.js'

class FakeForm {
  constructor() { this.fields = [] }
  append(k, v) { this.fields.push([k, v]) }
}

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

// G-1: a failed read is an error state, never an empty session. The two
// source-grep tests that stood here are replaced by rendered behaviour in
// SessionDocuments.visible.test.mjs (upload paths and the sidebar order).

const ROWS = { documents: [{ id: 'd1', original_name: 'a.md', origin: 'uploaded' }] }
const EMPTY = { documents: [] }

test('a failed read shows an error, an empty read shows "no files"', async () => {
  const fail = async () => { throw new Error('503') }
  const ok = (payload) => async () => payload
  const failed = await readSessionDocuments(fail, '/p', 's1', () => 's1', null)
  assert.deepEqual(sessionDocumentsView(failed, 's1'), { kind: 'error', rows: [] })
  const empty = await readSessionDocuments(ok(EMPTY), '/p', 's1', () => 's1', null)
  assert.deepEqual(sessionDocumentsView(empty, 's1'), { kind: 'empty', rows: [] })
  const listed = await readSessionDocuments(ok(ROWS), '/p', 's1', () => 's1', null)
  assert.equal(sessionDocumentsView(listed, 's1').kind, 'list')
})

test('a failed re-read keeps the rows already shown for that session only', () => {
  for (const [previous, kept] of [
    [sessionDocumentsLoaded('s1', 's1', ROWS), 1],
    [sessionDocumentsLoaded('s0', 's0', ROWS), 0],
  ]) {
    const state = sessionDocumentsFailed('s1', 's1', previous)
    const view = sessionDocumentsView(state, 's1')
    assert.equal(view.kind, 'error')
    assert.equal(view.rows.length, kept)
  }
})

test('a reply for a session the user has left is dropped, success or failure', async () => {
  const fail = async () => { throw new Error('timeout') }
  for (const getJson of [fail, async () => ROWS]) {
    assert.equal(await readSessionDocuments(getJson, '/p', 's1', () => 's2', null), null)
  }
})

test('before the first reply the group is loading, not empty', () => {
  for (const [state, active] of [[null, 's1'], [sessionDocumentsLoaded('s0', 's0', EMPTY), 's1']]) {
    assert.equal(sessionDocumentsView(state, active).kind, 'loading')
  }
})

// TESTER 10:40Z: a file attached in one chat showed in the project's
// document list of every later chat. The server never listed it; the client
// put the upload's response row into the project's list, which outlives the
// chat. A row naming a session is the session's, not the project's.
test('a session file does not belong in the project list; a project file does', () => {
  for (const sessionFile of [
    { id: 'd1', original_name: 'a.txt', metadata: { session_id: 'ws-p1-1000', origin: 'uploaded' } },
    { id: 'd2', original_name: 'b.txt', metadata: { session_id: 'ws-p1-mwiq9k', uploader_id: 'u1' } },
  ]) {
    assert.equal(belongsToProjectList(sessionFile), false)
  }
  for (const projectFile of [
    { id: 'd3', original_name: 'c.txt', metadata: { provenance: 'user_upload', uploader_id: 'u1' } },
    { id: 'd4', original_name: 'd.txt' },
    { id: 'd5', original_name: 'e.txt', metadata: null },
    { id: 'd6', original_name: 'f.txt', metadata: { session_id: '  ' } },
  ]) {
    assert.equal(belongsToProjectList(projectFile), true)
  }
})

// The hole the server-side enumeration test cannot see: rows the CLIENT
// writes into the project's list. Every write that adds rows goes through
// belongsToProjectList: the project load and the one add handler.
test('every row the workspace adds to the project list passes the session rule', async () => {
  const fs = await import('node:fs')
  const src = fs.readFileSync(new URL('../pages/ProjectWorkspace.tsx', import.meta.url), 'utf8')
  const adds = src.split('\n').filter((line) => /setDocuments\(/.test(line)
    && (/project\.documents/.test(line) || /\[doc, \.\.\.prev\]/.test(line)))
  assert.equal(adds.length, 2, adds.join('\n'))
  assert.ok(adds.some((l) => /\.filter\(belongsToProjectList\)/.test(l)), adds.join('\n'))
  const handler = src.slice(src.indexOf('const handleDocumentAdded = useCallback'))
  const body = handler.slice(0, handler.indexOf('}, [])'))
  assert.ok(/if \(!belongsToProjectList\(doc\)\) return/.test(body), body)
  assert.ok(body.indexOf('belongsToProjectList') < body.indexOf('setDocuments('), body)
})
