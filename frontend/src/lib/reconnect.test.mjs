// node --test frontend/src/lib/reconnect.test.mjs   (Node 20+, no dependencies)
import assert from 'node:assert/strict'
import { test } from 'node:test'

import { ReconnectGaveUp, sendWithReconnect } from './reconnect.js'

const res = (status) => ({ status })
const clock = () => {
  let t = 0
  return { now: () => t, sleep: async (ms) => { t += ms } }
}
const script = (...steps) => {
  let i = 0
  const calls = { n: 0 }
  const fn = async () => {
    calls.n += 1
    const s = steps[Math.min(i++, steps.length - 1)]
    if (s instanceof Error) throw s
    return s
  }
  fn.calls = calls
  return fn
}

test('a healthy server is asked exactly once', async () => {
  const send = script(res(200))
  const out = await sendWithReconnect(send, async () => true, clock())
  assert.equal(out.status, 200)
  assert.equal(send.calls.n, 1)
})

test('502 during a deploy is retried until the server is back', async () => {
  const send = script(res(502), res(503), res(200))
  const seen = []
  const up = script(false, true)
  const out = await sendWithReconnect(send, up, { ...clock(), onReconnecting: (n) => seen.push(n) })
  assert.equal(out.status, 200)
  assert.equal(send.calls.n, 3)
  assert.deepEqual(seen, [1, 2])
})

test('a real answer is never retried: 500, 401 and 413 come straight back', async () => {
  for (const status of [500, 401, 413, 404]) {
    const send = script(res(status))
    const out = await sendWithReconnect(send, async () => true, clock())
    assert.equal(out.status, status)
    assert.equal(send.calls.n, 1)
  }
})

test('a network error while the server is UP is not retried — the turn may be running', async () => {
  const send = script(new TypeError('Failed to fetch'), res(200))
  await assert.rejects(sendWithReconnect(send, async () => true, clock()), /Failed to fetch/)
  assert.equal(send.calls.n, 1)
})

test('a network error while the server is DOWN is retried once it is back', async () => {
  const send = script(new TypeError('Failed to fetch'), res(200))
  const up = script(false, false, true)
  const out = await sendWithReconnect(send, up, clock())
  assert.equal(out.status, 200)
  assert.equal(send.calls.n, 2)
})

test('it gives up after the budget with a message that says the message was not sent', async () => {
  const send = script(res(502))
  await assert.rejects(
    sendWithReconnect(send, async () => false, { ...clock(), maxMs: 30_000 }),
    (err) => err instanceof ReconnectGaveUp && /was not sent/.test(err.message),
  )
})

test('an aborted turn stops immediately and is not retried', async () => {
  const abort = Object.assign(new Error('Aborted'), { name: 'AbortError' })
  const send = script(abort, res(200))
  await assert.rejects(sendWithReconnect(send, async () => false, clock()), { name: 'AbortError' })
  assert.equal(send.calls.n, 1)
})
