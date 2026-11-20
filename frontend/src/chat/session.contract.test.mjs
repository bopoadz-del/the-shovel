// node --test frontend/src/chat/session.contract.test.mjs
// Locks the workspace session mechanism: opening a project mints a new
// id, a stale history payload cannot fill that session, and a session
// can be renamed with a validated title.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'

import {
  afterSessionDeleted,
  deleteSessionQuestion,
  freshSessionEntropy,
  historyBelongsToSession,
  legacyConversationId,
  mintConversationId,
  orderSessionsNewestFirst,
  readRememberedSession,
  rememberSession,
  renameTitleError,
  restorableSessionId,
  sessionListTitle,
  sessionMemoryKey,
} from './session.js'

const workspace = fs.readFileSync(
  new URL('../pages/ProjectWorkspace.tsx', import.meta.url),
  'utf8',
)

test('opening a project does not fall back to the legacy stable thread', () => {
  assert.equal(
    workspace.includes('activeConversationId ?? (id ? `ws-${id}` : null)'),
    false,
  )
  assert.equal(
    workspace.includes('mintConversationId('),
    true,
  )
})

test('opening a project does not hydrate the legacy thread into the pane', () => {
  assert.equal(
    workspace.includes('/v1/agents/conversations/ws-${id}/messages'),
    false,
  )
})

test('a minted session id is never the legacy thread and never repeats an entropy', () => {
  const projectId = 'p' + '1'.repeat(7)
  assert.equal(legacyConversationId(projectId), `ws-${projectId}`)
  const first = mintConversationId(projectId, 'aaa111')
  const second = mintConversationId(projectId, 'bbb222')
  assert.notEqual(first, legacyConversationId(projectId))
  assert.notEqual(second, first)
  assert.equal(first, `ws-${projectId}-aaa111`)
  assert.equal(mintConversationId(projectId, ''), null)
})

test('two fresh entropies differ so a double click cannot reopen one session', () => {
  const a = freshSessionEntropy()
  const b = freshSessionEntropy()
  assert.notEqual(a, b)
  assert.match(a, /^[a-z0-9]+$/)
})

test('history for a different session is not applied to the one on screen', () => {
  const projectId = 'abcd1234'
  const current = mintConversationId(projectId, 'new1')
  const older = legacyConversationId(projectId)
  const prior = [{ role: 'user', content: 'earlier question' }]
  assert.equal(historyBelongsToSession(older, current, prior), null)
  assert.deepEqual(historyBelongsToSession(current, current, prior), prior)
})

test('sessions sort newest first', () => {
  const ordered = orderSessionsNewestFirst([
    { id: 'a', updated_at: '2020-01-01T00:00:00' },
    { id: 'c', updated_at: '2024-01-01T00:00:00' },
    { id: 'b', updated_at: '2022-01-01T00:00:00' },
  ])
  assert.deepEqual(ordered.map((row) => row.id), ['c', 'b', 'a'])
})

test('an untitled session displays a default title', () => {
  assert.equal(sessionListTitle(null), 'New chat')
  assert.equal(sessionListTitle('   '), 'New chat')
  assert.equal(sessionListTitle('  pipe specs  '), 'pipe specs')
})

test('a rename rejects empty, over-long, and HTML titles', () => {
  assert.equal(renameTitleError('Drainage review'), null)
  assert.equal(renameTitleError('  Drainage review  '), null)
  assert.match(renameTitleError('   '), /empty/)
  assert.match(renameTitleError('x'.repeat(81)), /long/)
  assert.match(renameTitleError('<b>no</b>'), /HTML/)
  assert.equal(renameTitleError('x'.repeat(80)), null)
})


function memoryStorage() {
  const data = new Map()
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => data.set(k, String(v)),
    removeItem: (k) => data.delete(k),
    data,
  }
}

for (const [pid, other] of [['p-quay-wall', 'p-pump-house'], ['b7c1e9', 'a0d4f2']]) {
  test(`the open session is remembered per project and reopened (${pid})`, () => {
    const storage = memoryStorage()
    const cid = mintConversationId(pid, freshSessionEntropy())
    rememberSession(pid, cid, storage)
    assert.equal(readRememberedSession(pid, storage), cid)
    // Another project does not inherit it.
    assert.equal(readRememberedSession(other, storage), null)
    assert.notEqual(sessionMemoryKey(pid), sessionMemoryKey(other))
  })

  test(`only an id minted for the project is restored (${pid})`, () => {
    assert.equal(restorableSessionId(pid, `ws-${pid}`), null)
    assert.equal(restorableSessionId(pid, `ws-${pid}-`), null)
    assert.equal(restorableSessionId(pid, `ws-${other}-abc123`), null)
    assert.equal(restorableSessionId(pid, `ws-${pid}-abc 123`), null)
    assert.equal(restorableSessionId(pid, `ws-${pid}-abc/../x`), null)
    assert.equal(restorableSessionId(pid, 42), null)
    assert.equal(restorableSessionId(pid, `ws-${pid}-k3x9q`), `ws-${pid}-k3x9q`)
    const storage = memoryStorage()
    storage.setItem(sessionMemoryKey(pid), `ws-${other}-k3x9q`)
    assert.equal(readRememberedSession(pid, storage), null)
    rememberSession(pid, null, storage)
    assert.equal(storage.data.has(sessionMemoryKey(pid)), false)
  })
}

test('a blocked storage never breaks the session', () => {
  const blocked = {
    getItem: () => { throw new Error('denied') },
    setItem: () => { throw new Error('denied') },
    removeItem: () => { throw new Error('denied') },
  }
  assert.equal(readRememberedSession('p1', blocked), null)
  assert.doesNotThrow(() => rememberSession('p1', 'ws-p1-a1', blocked))
  assert.equal(readRememberedSession('p1', null), null)
})

for (const [rows, del, active] of [
  [[{ id: 'ws-a-1' }, { id: 'ws-a-2' }, { id: 'ws-a-3' }], 'ws-a-2', 'ws-a-2'],
  [[{ id: 'ws-b-9' }, { id: 'ws-b-8' }], 'ws-b-8', 'ws-b-9'],
]) {
  test(`deleting ${del} drops it and replaces the open thread only when it was open`, () => {
    const out = afterSessionDeleted(rows, del, active)
    assert.deepEqual(out.rows.map((r) => r.id), rows.map((r) => r.id).filter((x) => x !== del))
    assert.equal(out.replaceActive, del === active)
  })
}

for (const title of ['Slab pour review', null]) {
  test(`delete asks first, naming the session (${title})`, () => {
    const q = deleteSessionQuestion(title)
    assert.match(q, /^Delete the session "/)
    assert.ok(q.includes(sessionListTitle(title)))
    assert.match(q, /cannot be undone/)
  })
}

const panel = fs.readFileSync(new URL('../layout/LeftPanel.tsx', import.meta.url), 'utf8')

test('sessions are listed above the documents in the sidebar', () => {
  const sessions = panel.indexOf('>Sessions</header>')
  const documents = panel.indexOf('>Documents</header>')
  assert.ok(sessions > 0 && documents > 0, 'both sections present')
  assert.ok(sessions < documents, 'Sessions comes before Documents')
  assert.ok(panel.includes('window.confirm(deleteSessionQuestion('), 'delete asks first')
})

test('the workspace restores the remembered session and reloads its history', () => {
  assert.ok(workspace.includes('readRememberedSession(id, browserStorage())'))
  assert.ok(workspace.includes('rememberSession(id, activeConversationId, browserStorage())'))
  assert.ok(workspace.includes('if (restoredConversationId) loadSessionHistory(restoredConversationId)'))
})
