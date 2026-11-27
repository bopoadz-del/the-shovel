// node --test src/lib/apiDeadline.test.mjs
// apiGet's read deadline: a read the server never answers ends at the
// deadline with an ApiError of status 0; a read answered in time, or a read
// without a deadline, is untouched.
import assert from 'node:assert/strict'
import path from 'node:path'
import { after, test } from 'node:test'
import { fileURLToPath } from 'node:url'

import { createServer } from 'vite'

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const server = await createServer({
  root: frontendRoot,
  configFile: path.join(frontendRoot, 'vite.config.ts'),
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
})
after(() => server.close())

const emptyStorage = { getItem: () => null, setItem() {}, removeItem() {} }
globalThis.localStorage = emptyStorage
globalThis.sessionStorage = emptyStorage
globalThis.window = { location: { pathname: '/', assign() {} }, localStorage: emptyStorage, sessionStorage: emptyStorage }
const { apiGet, ApiError } = await server.ssrLoadModule('/src/lib/api.ts')

// A server that answers after `delayMs`, or never when delayMs is null,
// and honours the abort signal the way the browser's fetch does.
function serverAnswering(delayMs, body) {
  const seen = []
  globalThis.fetch = (url, init = {}) => {
    seen.push({ url: String(url), signal: init.signal })
    return new Promise((resolve, reject) => {
      const t = delayMs === null ? null : setTimeout(
        () => resolve(new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } })),
        delayMs,
      )
      init.signal?.addEventListener('abort', () => {
        if (t) clearTimeout(t)
        reject(new DOMException('The operation was aborted.', 'AbortError'))
      })
    })
  }
  return seen
}

for (const deadline of [30, 90]) {
  test(`a read never answered ends at its ${deadline} ms deadline with status 0`, async () => {
    serverAnswering(null, {})
    const started = Date.now()
    await assert.rejects(apiGet('/v1/projects/x', { timeoutMs: deadline }), (err) => {
      assert.ok(err instanceof ApiError, String(err))
      assert.equal(err.status, 0)
      assert.match(err.message, /did not answer/)
      return true
    })
    const took = Date.now() - started
    assert.ok(took >= deadline - 5 && took < deadline + 500, `ended after ${took} ms`)
  })
}

for (const [delay, deadline, body] of [[5, 200, { id: 'a' }], [40, 300, { id: 'b', n: 2 }]]) {
  test(`a read answered in ${delay} ms inside a ${deadline} ms deadline returns its body`, async () => {
    serverAnswering(delay, body)
    assert.deepEqual(await apiGet('/v1/projects/y', { timeoutMs: deadline }), body)
  })
}

test('a read without a deadline carries no abort signal and waits for its answer', async () => {
  const seen = serverAnswering(60, { id: 'c' })
  assert.deepEqual(await apiGet('/v1/projects/z'), { id: 'c' })
  assert.equal(seen.at(-1).signal, undefined)
})
