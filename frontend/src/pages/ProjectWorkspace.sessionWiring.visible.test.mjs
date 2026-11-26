// node --test src/pages/ProjectWorkspace.sessionWiring.visible.test.mjs
// G-1 / W2, through the real workspace: mounts the routed ProjectWorkspace
// page in jsdom with a stubbed server and drives it the way a person does.
// The component-level tests (src/chat/SessionDocuments.visible.test.mjs)
// hand the session id in by hand; this file proves the workspace itself
//  - passes the session the user OPENED from the list into both upload
//    paths (the documents panel and the chat composer), and
//  - wires the "This session" retry: a failed read shows the error and a
//    retry control, and using it reads that session's files again.
import assert from 'node:assert/strict'
import path from 'node:path'
import { after, test } from 'node:test'
import { fileURLToPath } from 'node:url'

import React from 'react'
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

const { JSDOM } = await import('jsdom')
const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/projects/p7' })
globalThis.window = dom.window
globalThis.document = dom.window.document
globalThis.IS_REACT_ACT_ENVIRONMENT = true
if (globalThis.navigator === undefined) globalThis.navigator = dom.window.navigator
for (const k of ['HTMLElement', 'Element', 'Node', 'Text', 'Event', 'MouseEvent',
  'DocumentFragment', 'getComputedStyle', 'requestAnimationFrame', 'cancelAnimationFrame',
  'localStorage', 'sessionStorage']) {
  if (dom.window[k] !== undefined) globalThis[k] = dom.window[k]
}
if (!dom.window.HTMLElement.prototype.scrollIntoView) dom.window.HTMLElement.prototype.scrollIntoView = () => {}
const { createRoot } = await import('react-dom/client')
const { act } = await import('react')
const { MemoryRouter, Routes, Route } = await import('react-router-dom')

const { default: ProjectWorkspace } = await server.ssrLoadModule('/src/pages/ProjectWorkspace.tsx')
const { AuthProvider } = await server.ssrLoadModule('/src/auth/AuthContext.tsx')
const { ThemeProvider } = await server.ssrLoadModule('/src/theme/ThemeContext.tsx')

// Invented fixture: one project with two stored sessions. The test opens
// the OLDER one, so a workspace that kept its freshly minted id (or the
// first listed one) would post the wrong id.
const PROJECT = 'p7'
const OPENED = 'ws-p7-1700000000000'
const OTHER = 'ws-p7-1700000999000'
const SESSIONS = [
  { id: OTHER, title: 'Façade mock-up review', updated_at: '2026-01-02T00:00:00Z' },
  { id: OPENED, title: 'Pour sequence level 3', updated_at: '2026-01-01T00:00:00Z' },
]

const server_ = {
  sent: [],
  failSessionReads: new Set(), // session ids whose documents read answers 500
  projectFailures: [], // statuses the next project reads answer, in order
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { 'content-type': 'application/json' } })
}

globalThis.fetch = async (url, init = {}) => {
  const u = new URL(String(url), 'http://localhost')
  const method = (init.method ?? 'GET').toUpperCase()
  server_.sent.push({ path: u.pathname, method, body: init.body, signal: init.signal })
  const p = u.pathname
  if (method === 'POST' && p === `/v1/projects/${PROJECT}/documents`) {
    return json({ document: { id: `doc-${server_.sent.length}`, original_name: 'spec.md', status: 'ready' } })
  }
  const sessionDocs = p.match(new RegExp(`^/v1/projects/${PROJECT}/conversations/([^/]+)/documents$`))
  if (sessionDocs) {
    const sid = decodeURIComponent(sessionDocs[1])
    if (server_.failSessionReads.has(sid)) return json({ detail: 'database unavailable' }, 500)
    return json({ documents: [{ id: `own-${sid}`, original_name: 'pour-plan.md', origin: 'uploaded' }] })
  }
  if (p === `/v1/projects/${PROJECT}/conversations`) return json({ conversations: SESSIONS })
  if (p === `/v1/projects/${PROJECT}`) {
    const failWith = server_.projectFailures.shift()
    if (failWith) return json({ detail: `project store unavailable (${failWith})` }, failWith)
    return json({ id: PROJECT, name: 'Invented tower', documents: [], held_document_count: 0, searchable_document_count: 0 })
  }
  if (p.startsWith('/v1/agents/conversations/')) return json({ messages: [] })
  if (p === '/v1/agents') return json({ agents: [] })
  if (p === '/v1/users/me') return json({ id: 'u1', email: 'tester@example.invalid', name: 'Tester' })
  return json({}, 404)
}

async function settle() {
  for (let i = 0; i < 8; i++) await act(async () => { await new Promise((r) => setTimeout(r, 0)) })
}

// A fresh tab has empty storage of its own; a reload in the same tab keeps it.
async function mountWorkspace({ sameTab = false } = {}) {
  if (!sameTab) {
    dom.window.localStorage.clear()
    dom.window.sessionStorage.clear()
  }
  const container = dom.window.document.createElement('div')
  dom.window.document.body.appendChild(container)
  const root = createRoot(container)
  await act(async () => {
    root.render(React.createElement(MemoryRouter, { initialEntries: [`/projects/${PROJECT}`] },
      React.createElement(ThemeProvider, null, React.createElement(AuthProvider, null,
        React.createElement(Routes, null,
          React.createElement(Route, { path: '/projects/:id', element: React.createElement(ProjectWorkspace) }))))))
  })
  await settle()
  const view = {
    container,
    text: () => container.textContent.replace(/\s+/g, ' ').trim(),
    async click(el) {
      await act(async () => { el.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true })) })
      await settle()
    },
    async chooseFile(input, file) {
      Object.defineProperty(input, 'files', { configurable: true, value: [file] })
      await act(async () => { input.dispatchEvent(new dom.window.Event('change', { bubbles: true })) })
      await settle()
    },
    buttonNamed(text) {
      return [...container.querySelectorAll('button')].find((b) => b.textContent.replace(/\s+/g, ' ').trim() === text)
    },
    async openSession(title) {
      const btn = [...container.querySelectorAll('button')].find((b) => b.textContent.includes(title))
      assert.ok(btn, `session "${title}" listed: ${view.text().slice(0, 300)}`)
      await view.click(btn)
    },
    unmount() { act(() => root.unmount()); container.remove() },
  }
  return view
}

const sessionReads = (sid) => server_.sent.filter((r) => r.method === 'GET'
  && r.path === `/v1/projects/${PROJECT}/conversations/${encodeURIComponent(sid)}/documents`).length

function lastUploadForm() {
  const posts = server_.sent.filter((r) => r.method === 'POST' && r.path === `/v1/projects/${PROJECT}/documents`)
  assert.ok(posts.length > 0, `an upload was sent: ${server_.sent.map((r) => `${r.method} ${r.path}`).join(', ')}`)
  return posts.at(-1).body
}

test('an upload from either path carries the session the user opened', async () => {
  server_.sent = []
  server_.failSessionReads = new Set()
  const ws = await mountWorkspace()
  await ws.openSession('Pour sequence level 3')

  const panelInput = ws.container.querySelector('input[aria-label="Choose file to upload"]')
  assert.ok(panelInput, 'documents panel upload input rendered')
  await ws.chooseFile(panelInput, new File(['# a'], 'spec.md', { type: 'text/markdown' }))
  assert.equal(lastUploadForm().get('session_id'), OPENED)

  const composerInput = ws.container.querySelector('input[type="file"][multiple]')
  assert.ok(composerInput, 'composer upload input rendered')
  await ws.chooseFile(composerInput, new File(['# b'], 'spec.md', { type: 'text/markdown' }))
  assert.equal(lastUploadForm().get('session_id'), OPENED)

  // Switching session moves the id with it: the next upload names the other one.
  await ws.openSession('Façade mock-up review')
  await ws.chooseFile(ws.container.querySelector('input[aria-label="Choose file to upload"]'),
    new File(['# c'], 'spec.md', { type: 'text/markdown' }))
  assert.equal(lastUploadForm().get('session_id'), OTHER)
  ws.unmount()
})

test('a failed read of the opened session shows an error and a retry that reads it again', async () => {
  server_.sent = []
  server_.failSessionReads = new Set([OPENED])
  const ws = await mountWorkspace()
  await ws.openSession('Pour sequence level 3')

  assert.ok(sessionReads(OPENED) >= 1, 'the opened session was read')
  assert.match(ws.text(), /Could not load this session's files/)
  assert.ok(!ws.text().includes('No files in this session yet'))
  const retry = ws.buttonNamed('Try again')
  assert.ok(retry, 'a retry control is shown')

  const before = sessionReads(OPENED)
  server_.failSessionReads.delete(OPENED)
  await ws.click(retry)
  assert.equal(sessionReads(OPENED), before + 1, 'retry reads the opened session again')
  assert.ok(!/Could not load this session's files/.test(ws.text()), 'error cleared after a good read')
  assert.ok(ws.text().includes('pour-plan.md'), 'the session file is listed')
  ws.unmount()
})

for (const [chosen, title] of [[OPENED, 'Pour sequence level 3'], [OTHER, 'Façade mock-up review']]) {
  test(`a reload in the same tab reopens the thread it left open (${chosen})`, async () => {
    server_.sent = []
    server_.failSessionReads = new Set()
    const first = await mountWorkspace()
    await first.openSession(title)
    first.unmount()
    const before = sessionReads(chosen)
    const again = await mountWorkspace({ sameTab: true })
    assert.ok(sessionReads(chosen) > before, `reload read ${chosen}: ${server_.sent.map((r) => r.path).join(', ')}`)
    again.unmount()
  })

  test(`a fresh tab starts its own thread, not the one another tab left open (${chosen})`, async () => {
    server_.sent = []
    server_.failSessionReads = new Set()
    const first = await mountWorkspace()
    await first.openSession(title)
    first.unmount()
    // Another tab: its own empty session storage. What the browser holds
    // across tabs (local storage) still names the old thread, as an older
    // build left it; the fresh tab must not read it.
    const shared = Object.fromEntries(Object.keys(dom.window.sessionStorage).map((k) => [k, dom.window.sessionStorage.getItem(k)]))
    assert.ok(Object.values(shared).includes(chosen), `the first tab remembered ${chosen}: ${JSON.stringify(shared)}`)
    dom.window.sessionStorage.clear()
    for (const [k, v] of Object.entries(shared)) dom.window.localStorage.setItem(k, v)
    const before = sessionReads(chosen)
    const fresh = await mountWorkspace({ sameTab: true })
    assert.equal(sessionReads(chosen), before, `fresh tab did not read ${chosen}: ${server_.sent.map((r) => r.path).join(', ')}`)
    const minted = server_.sent.filter((r) => r.path.includes('/conversations/ws-') && !r.path.includes(encodeURIComponent(OPENED)) && !r.path.includes(encodeURIComponent(OTHER)))
    assert.ok(minted.length > 0, `fresh tab read a thread of its own: ${server_.sent.map((r) => r.path).join(', ')}`)
    fresh.unmount()
  })
}

for (const status of [500, 503]) {
  test(`a failed project read (${status}) offers a retry that opens the workspace`, async () => {
    server_.sent = []
    server_.failSessionReads = new Set()
    server_.projectFailures = [status]
    const view = await mountWorkspace()
    assert.match(view.text(), /Could not load project/)
    assert.match(view.text(), new RegExp(`\(${status}\)`))
    assert.equal(view.container.querySelector('textarea'), null, 'no composer before the project is read')
    const retry = view.buttonNamed('Try again')
    assert.ok(retry, `a retry control: ${view.text().slice(0, 300)}`)
    await view.click(retry)
    assert.doesNotMatch(view.text(), /Could not load project/)
    assert.match(view.text(), /Invented tower/)
    assert.ok(view.container.querySelector('textarea'), 'the composer is shown after the retry')
    const reads = server_.sent.filter((r) => r.method === 'GET' && r.path === `/v1/projects/${PROJECT}`)
    assert.equal(reads.length, 2, 'the retry read the project again')
    view.unmount()
  })
}

test('the project read carries a deadline; the session list read is not changed', async () => {
  server_.sent = []
  server_.failSessionReads = new Set()
  server_.projectFailures = []
  const view = await mountWorkspace()
  const read = server_.sent.find((r) => r.method === 'GET' && r.path === `/v1/projects/${PROJECT}`)
  assert.ok(read?.signal, 'the project read can be aborted at its deadline')
  const list = server_.sent.find((r) => r.method === 'GET' && r.path === `/v1/projects/${PROJECT}/conversations`)
  assert.equal(list?.signal, undefined)
  view.unmount()
})
