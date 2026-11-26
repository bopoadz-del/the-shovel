// node --test frontend/src/chat/sessionDocuments.test.mjs
// W2 slice 2 (exam 11.3): uploads carry the open session's id, and the
// sidebar's "This session" group lists only the open session's files, each
// tagged Uploaded or Produced. Every case has an input that must pass and
// one that must fail.
import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  belongsToProjectList,
  documentPath,
  readSessionDocuments,
  sessionDocumentTag,
  sessionDocumentsFailed,
  sessionDocumentsFor,
  sessionDocumentsLoaded,
  sessionDocumentsView,
  uploadNotice,
  uploadSessionFor,
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

// #451: a content-by-id request names the open chat; a file attached to a
// chat is reachable by id only from it.
for (const [sid, tail, expected] of [
  ['ws-p1-1000', '/preview', '/v1/projects/p1/documents/d9/preview?session_id=ws-p1-1000'],
  ['ws p1/2000', '/download', '/v1/projects/p1/documents/d9/download?session_id=ws%20p1%2F2000'],
]) {
  test(`documentPath names the open chat (${tail})`, () => {
    assert.equal(documentPath('p1', 'd9', tail, sid), expected)
  })
}

for (const sid of [null, '   ']) {
  test(`documentPath with no open chat carries no session (${JSON.stringify(sid)})`, () => {
    assert.equal(documentPath('p1', 'd9', '/preview/raw', sid), '/v1/projects/p1/documents/d9/preview/raw')
  })
}

// The workspace always has a chat open, so "is there a session" cannot pick
// project vs session. The control the user used picks it: "+ Add to project"
// sends no session, "+ Upload file" sends the open one.
for (const sid of ['ws-p1-1000', 'ws-p7-2000']) {
  test(`a project upload carries no session; a session upload carries ${sid}`, () => {
    assert.deepEqual(withSessionId(new FakeForm(), uploadSessionFor('project', sid)).fields, [])
    assert.deepEqual(withSessionId(new FakeForm(), uploadSessionFor('session', sid)).fields,
      [['session_id', sid]])
  })
}

test('only the project target drops the session; no session stays none', () => {
  for (const target of ['session', 'Project', '', undefined]) {
    assert.equal(uploadSessionFor(target, 'ws-p1-1000'), 'ws-p1-1000', String(target))
  }
  assert.equal(uploadSessionFor('session', null), null)
  assert.equal(uploadSessionFor('project', null), null)
})

test('the sidebar has one picker per target, each wired to its own', async () => {
  const fs = await import('node:fs')
  const src = fs.readFileSync(new URL('../pages/ProjectWorkspace.tsx', import.meta.url), 'utf8')
  const wiring = (id) => {
    const at = src.indexOf(`id="${id}"`)
    assert.ok(at > 0, id)
    return src.slice(at, src.indexOf('/>', at))
  }
  assert.match(wiring('doc-file-input'), /handleFileChange\(e, 'session'\)/)
  assert.match(wiring('doc-project-file-input'), /handleFileChange\(e, 'project'\)/)
  assert.match(src, /withSessionId\(form, uploadSessionFor\(target, sessionId\)\)/)
})

// The project control does not decide who may index: the server does, and
// says so on the stored row. The server's card state picks whether the user
// is told: a file stored outside search, or one that failed, repeats the
// server's note; a file still waiting to be read (indexing scheduled) or
// already searchable says nothing.
const STORED = { state: 'stored', line: 'Stored, not searchable' }
const FAILED = { state: 'failed', line: 'Failed' }
for (const [detail, state] of [
  ["Only an admin adds documents to a project's knowledge base. The file was not indexed.", STORED],
  ['No text could be read from this file.', FAILED],
]) {
  test(`a ${state.state} upload repeats the server note (${detail.slice(0, 20)})`, () => {
    const doc = {
      id: 'd1',
      panel_status: { label: 'Not indexed', detail: `  ${detail} `, status: 'not_indexed' },
      document_state: state,
    }
    assert.equal(uploadNotice(doc), detail)
  })
}

// The shape the server returns for an admin's project upload with indexing
// scheduled in the background: a "Not indexed" panel note with a detail,
// but the card state says the file is waiting to be read. Not an error.
for (const [status, detail] of [
  ['UNVERIFIED', 'This file is stored and has not been added to the project knowledge base.'],
  ['queued', 'Indexing is queued.'],
]) {
  test(`a waiting-to-be-read upload has no notice (${status})`, () => {
    const doc = {
      id: 'd9',
      panel_status: { label: 'Not indexed', detail, status },
      document_state: { state: 'reading', line: 'Waiting to be read' },
    }
    assert.equal(uploadNotice(doc), null)
    // The same note under a stored card state is shown: the state decides.
    assert.equal(uploadNotice({ ...doc, document_state: STORED }), detail)
  })
}

test('an upload the server holds for search, or a malformed reply, has no notice', () => {
  const note = { label: 'Not indexed', detail: 'A real note.', status: 'not_indexed' }
  for (const doc of [
    { id: 'd1' },
    { id: 'd1b', document_state: { state: 'searchable', line: 'Searchable' } },
    { id: 'd1c', panel_status: note, document_state: { state: 'searchable', line: 'Searchable' } },
    { id: 'd1d', panel_status: note },
    { id: 'd1e', panel_status: note, document_state: 'stored' },
    { id: 'd1f', panel_status: note, document_state: { state: 'Stored' } },
    { id: 'd2', panel_status: null, document_state: STORED },
    { id: 'd3', panel_status: { label: 'Not indexed', detail: '   ' }, document_state: STORED },
    { id: 'd4', panel_status: { detail: 7 }, document_state: FAILED },
    { id: 'd5', panel_status: 'Not indexed', document_state: STORED },
    null,
    undefined,
    'stored',
  ]) {
    assert.equal(uploadNotice(doc), null, JSON.stringify(doc))
  }
})

test('the project upload shows the server note and its title promises no search', async () => {
  const fs = await import('node:fs')
  const src = fs.readFileSync(new URL('../pages/ProjectWorkspace.tsx', import.meta.url), 'utf8')
  const body = src.slice(src.indexOf('async function handleFileChange('), src.indexOf('function beginReplace('))
  assert.match(body, /const notice = uploadNotice\(resp\.document\)\s*if \(notice\) \{\s*setUploadError\(notice\)/)
  const labelAt = src.indexOf('htmlFor="doc-project-file-input"')
  assert.ok(labelAt > 0)
  const label = src.slice(labelAt, src.indexOf('+ Add to project', labelAt))
  assert.match(label, /title=/)
  assert.doesNotMatch(label, /search/i)
})

// Each picker opens only from its own label, is off while an upload runs,
// and is cleared after every pick so the same file can be picked again.
for (const [id, text] of [['doc-file-input', '+ Upload file'], ['doc-project-file-input', '+ Add to project']]) {
  test(`the ${text} label opens ${id}, which is off while uploading`, async () => {
    const fs = await import('node:fs')
    const src = fs.readFileSync(new URL('../pages/ProjectWorkspace.tsx', import.meta.url), 'utf8')
    const labelAt = src.indexOf(`htmlFor="${id}"`)
    assert.ok(labelAt > 0, id)
    const label = src.slice(labelAt, src.indexOf('</label>', labelAt))
    assert.ok(label.includes(text), label)
    assert.match(label, /aria-busy=\{uploading\}/)
    const inputAt = src.indexOf(`id="${id}"`)
    const input = src.slice(inputAt, src.indexOf('/>', inputAt))
    assert.match(input, /disabled=\{uploading\}/)
  })
}

test('every pick clears its input, so the same file can be picked twice', async () => {
  const fs = await import('node:fs')
  const src = fs.readFileSync(new URL('../pages/ProjectWorkspace.tsx', import.meta.url), 'utf8')
  const body = src.slice(src.indexOf('async function handleFileChange('), src.indexOf('function beginReplace('))
  const fin = body.slice(body.indexOf('} finally {'))
  assert.ok(body.indexOf('} finally {') > 0, body)
  assert.match(fin, /e\.target\.value = ''/)
})
