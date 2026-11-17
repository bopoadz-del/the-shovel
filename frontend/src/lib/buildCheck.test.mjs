// node --test frontend/src/lib/buildCheck.test.mjs   (Node 20+, no dependencies)
import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  BUILD_HEADER,
  MISMATCH_MESSAGE,
  compareBuilds,
  createBuildWatcher,
  installFetchObserver,
} from './buildCheck.js'

// Each test makes its own two build ids; no real commit is named.
const OLDER = 'a'.repeat(40)
const NEWER = 'b'.repeat(40)

function fakeResponse(serverBuild) {
  const headers = new Headers()
  if (serverBuild !== undefined) headers.set(BUILD_HEADER, serverBuild)
  return new Response('{}', { status: 200, headers })
}

test('an older page on a newer server differs', () => {
  assert.equal(compareBuilds(OLDER, NEWER), 'differs')
})

test('a newer page on an older server differs too', () => {
  assert.equal(compareBuilds(NEWER, OLDER), 'differs')
})

test('the same build is the same, whatever the case or padding', () => {
  assert.equal(compareBuilds(OLDER, ` ${OLDER.toUpperCase()} `), 'same')
})

test('a build that is not known on either side is unknown, never differs', () => {
  for (const [c, s] of [['', NEWER], [OLDER, ''], [undefined, NEWER], [OLDER, null]]) {
    assert.equal(compareBuilds(c, s), 'unknown')
  }
})

test('the watcher tells the page once, through the response header', () => {
  const told = []
  const w = createBuildWatcher(OLDER, (s) => told.push(s))
  w.observe(fakeResponse(OLDER))
  assert.deepEqual(told, [])
  w.observe(fakeResponse(NEWER))
  w.observe(fakeResponse(NEWER))
  assert.deepEqual(told, [NEWER])
})

test('a response with no build header says nothing', () => {
  const told = []
  const w = createBuildWatcher(OLDER, (s) => told.push(s))
  w.observe(fakeResponse(undefined))
  assert.deepEqual(told, [])
})

test('the fetch observer sees every caller and returns the same response', async () => {
  const sent = fakeResponse(NEWER)
  const target = { fetch: async () => sent }
  const told = []
  const w = createBuildWatcher(OLDER, (s) => told.push(s))
  installFetchObserver(target, w)
  installFetchObserver(target, w) // twice is a no-op, not a double wrap
  const got = await target.fetch('/v1/anything', { method: 'POST' })
  assert.equal(got, sent)
  assert.deepEqual(told, [NEWER])
})

test('a page on the same build as the server is never told', async () => {
  const target = { fetch: async () => fakeResponse(OLDER) }
  const told = []
  installFetchObserver(target, createBuildWatcher(OLDER, (s) => told.push(s)))
  await target.fetch('/v1/anything')
  assert.deepEqual(told, [])
})

test('a failed fetch is rethrown unchanged', async () => {
  const boom = new TypeError('Failed to fetch')
  const target = { fetch: async () => { throw boom } }
  installFetchObserver(target, createBuildWatcher(OLDER, () => {}))
  await assert.rejects(target.fetch('/v1/x'), (e) => e === boom)
})

test('the message tells the user what to do', () => {
  assert.match(MISMATCH_MESSAGE, /reload/i)
})
