// node --test frontend/src/lib/buildCheck.test.mjs   (Node 20+, no dependencies)
import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  BUILD_HEADER,
  MISMATCH_MESSAGE,
  compareBuilds,
  createBuildWatcher,
  installFetchObserver,
  installVisibilityCheck,
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

// --- the tab-visible path ---------------------------------------------------
// A fake tab: an EventTarget whose visibilityState the test sets, and a fake
// server that counts probes and answers with whatever build it is running.

function fakeTab(state) {
  const doc = new EventTarget()
  doc.visibilityState = state
  doc.show = () => {
    doc.visibilityState = 'visible'
    doc.dispatchEvent(new Event('visibilitychange'))
  }
  doc.hide = () => {
    doc.visibilityState = 'hidden'
    doc.dispatchEvent(new Event('visibilitychange'))
  }
  return doc
}

function fakeServer(serverBuild) {
  const calls = []
  const target = {
    fetch: async (url, init) => {
      calls.push({ url, init })
      return fakeResponse(serverBuild)
    },
  }
  return { target, calls }
}

const settle = () => new Promise((resolve) => setImmediate(resolve))

test('a tab brought back to the front after a deploy is told, through the observer', async () => {
  const told = []
  const w = createBuildWatcher(OLDER, (s) => told.push(s))
  const { target, calls } = fakeServer(NEWER)
  installFetchObserver(target, w)
  const doc = fakeTab('hidden')
  const probeUrl = 'https://server.invalid/livez'
  installVisibilityCheck(doc, target, probeUrl, () => told.length > 0)
  doc.show()
  await settle()
  assert.equal(calls.length, 1)
  assert.equal(calls[0].url, probeUrl)
  assert.equal(calls[0].init.cache, 'no-store')
  assert.deepEqual(told, [NEWER])
})

test('a tab on the same build is probed and stays silent', async () => {
  const told = []
  const w = createBuildWatcher(NEWER, (s) => told.push(s))
  const { target, calls } = fakeServer(NEWER)
  installFetchObserver(target, w)
  const doc = fakeTab('hidden')
  installVisibilityCheck(doc, target, '/livez', () => told.length > 0)
  doc.show()
  await settle()
  assert.equal(calls.length, 1)
  assert.deepEqual(told, [])
})

test('a tab going to the back sends nothing', async () => {
  const { target, calls } = fakeServer(NEWER)
  const doc = fakeTab('visible')
  installVisibilityCheck(doc, target, '/livez', () => false)
  doc.hide()
  await settle()
  assert.equal(calls.length, 0)
})

test('once quiet (already told, or no build id) a return to the front sends nothing', async () => {
  const told = []
  const w = createBuildWatcher(OLDER, (s) => told.push(s))
  const { target, calls } = fakeServer(NEWER)
  installFetchObserver(target, w)
  const doc = fakeTab('hidden')
  installVisibilityCheck(doc, target, '/livez', () => told.length > 0)
  doc.show()
  await settle()
  doc.hide()
  doc.show()
  doc.hide()
  doc.show()
  await settle()
  assert.equal(calls.length, 1)
  assert.deepEqual(told, [NEWER])

  const silent = fakeServer(NEWER)
  const doc2 = fakeTab('hidden')
  installVisibilityCheck(doc2, silent.target, '/livez', () => true)
  doc2.show()
  await settle()
  assert.equal(silent.calls.length, 0)
})

test('every return to the front probes while nothing has been told', async () => {
  const { target, calls } = fakeServer(OLDER)
  const doc = fakeTab('hidden')
  installVisibilityCheck(doc, target, '/livez', () => false)
  doc.show()
  doc.hide()
  doc.show()
  await settle()
  assert.equal(calls.length, 2)
})

test('a probe that fails, by rejection or by throwing, never surfaces', async () => {
  const unhandled = []
  const onUnhandled = (err) => unhandled.push(err)
  process.on('unhandledRejection', onUnhandled)
  try {
    const rejecting = { fetch: async () => { throw new TypeError('offline') } }
    const doc = fakeTab('hidden')
    installVisibilityCheck(doc, rejecting, '/livez', () => false)
    doc.show()
    const throwing = { fetch: () => { throw new TypeError('blocked') } }
    const doc2 = fakeTab('hidden')
    installVisibilityCheck(doc2, throwing, '/livez', () => false)
    assert.doesNotThrow(() => doc2.show())
    await settle()
    await settle()
  } finally {
    process.off('unhandledRejection', onUnhandled)
  }
  assert.deepEqual(unhandled, [])
})

test('the returned function removes the listener', async () => {
  const { target, calls } = fakeServer(NEWER)
  const doc = fakeTab('hidden')
  const remove = installVisibilityCheck(doc, target, '/livez', () => false)
  remove()
  doc.show()
  await settle()
  assert.equal(calls.length, 0)
})
