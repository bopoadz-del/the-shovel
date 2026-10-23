// node --test frontend/src/chat/session.contract.test.mjs
// Locks the workspace session mechanism: opening a project mints a new
// id, a stale history payload cannot fill that session, and a session
// can be renamed with a validated title.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'

import {
  freshSessionEntropy,
  historyBelongsToSession,
  legacyConversationId,
  mintConversationId,
  orderSessionsNewestFirst,
  renameTitleError,
  sessionListTitle,
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
