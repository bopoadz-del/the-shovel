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
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { 'content-type': 'application/json' } })
}

globalThis.fetch = async (url, init = {}) => {
  const u = new URL(String(url), 'http://localhost')
  const method = (init.method ?? 'GET').toUpperCase()
  server_.sent.push({ path: u.pathname, method, body: init.body })
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

async function mountWorkspace() {
  dom.window.localStorage.clear()
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
