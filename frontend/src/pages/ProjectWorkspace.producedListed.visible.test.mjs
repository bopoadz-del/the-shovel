// node --test src/pages/ProjectWorkspace.producedListed.visible.test.mjs
// W2-b, through the real workspace: a file the product produces in a
// session is listed under "This session" AT ONCE, whichever export control
// produced it. The server keeps an exported file as the session's Produced
// document (#360); this proves the page re-reads the session after every
// successful export, not only after one of them.
// #360's own tests called the export endpoint directly with a conversation
// written straight into memory: they never crossed the page, so the Word /
// Excel buttons under an answer were never checked to re-read the list.
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
const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/projects/p9' })
globalThis.window = dom.window
globalThis.document = dom.window.document
globalThis.IS_REACT_ACT_ENVIRONMENT = true
if (globalThis.navigator === undefined) globalThis.navigator = dom.window.navigator
for (const k of ['HTMLElement', 'Element', 'Node', 'Text', 'Event', 'MouseEvent',
  'DocumentFragment', 'getComputedStyle', 'requestAnimationFrame', 'cancelAnimationFrame',
  'localStorage', 'sessionStorage']) {
  if (dom.window[k] !== undefined) globalThis[k] = dom.window[k]
}
dom.window.HTMLElement.prototype.scrollIntoView ??= () => {}
// The download itself is the browser's; only the object URL is needed here.
URL.createObjectURL = () => 'blob:produced'
URL.revokeObjectURL = () => {}
dom.window.HTMLAnchorElement.prototype.click = function click() {}
const { createRoot } = await import('react-dom/client')
const { act } = await import('react')
const { MemoryRouter, Routes, Route } = await import('react-router-dom')

const { default: ProjectWorkspace } = await server.ssrLoadModule('/src/pages/ProjectWorkspace.tsx')
const { AuthProvider } = await server.ssrLoadModule('/src/auth/AuthContext.tsx')
const { ThemeProvider } = await server.ssrLoadModule('/src/theme/ThemeContext.tsx')

// Invented fixture: one project, one stored session with one answer.
const PROJECT = 'p9'
const SESSION = 'ws-p9-1700000123000'
const TITLE = 'Formwork striking times'

const stub = { sent: [], kept: [] }

function json(obj, status = 200, headers = {}) {
  return new Response(JSON.stringify(obj), { status, headers: { 'content-type': 'application/json', ...headers } })
}

globalThis.fetch = async (url, init = {}) => {
  const u = new URL(String(url), 'http://localhost')
  const method = (init.method ?? 'GET').toUpperCase()
  const p = u.pathname
  stub.sent.push({ method, path: p })
  if (method === 'POST' && p === `/v1/projects/${PROJECT}/conversations/${SESSION}/export`) {
    // The server keeps what it serves as the session's Produced document.
    const fmt = u.searchParams.get('format') ?? 'docx'
    const name = `Invented_Tower-${stub.kept.length + 1}.${fmt}`
    stub.kept.push({ id: `prod-${stub.kept.length + 1}`, original_name: name, origin: 'produced' })
    return new Response(new Blob(['PK']), {
      status: 200, headers: { 'content-disposition': `attachment; filename="${name}"` },
    })
  }
  if (p === `/v1/projects/${PROJECT}/conversations/${encodeURIComponent(SESSION)}/documents`) {
    return json({ documents: [...stub.kept] })
  }
  if (p.match(new RegExp(`^/v1/projects/${PROJECT}/conversations/[^/]+/documents$`))) return json({ documents: [] })
  if (p === `/v1/projects/${PROJECT}/conversations`) {
    return json({ conversations: [{ id: SESSION, title: TITLE, updated_at: '2026-01-01T00:00:00Z' }] })
  }
  if (p === `/v1/projects/${PROJECT}`) {
    return json({ id: PROJECT, name: 'Invented tower', documents: [], held_document_count: 0, searchable_document_count: 0 })
  }
  if (p === `/v1/agents/conversations/${SESSION}/messages`) {
    return json({ messages: [
      { role: 'user', content: 'When can the soffit forms be struck?' },
      { role: 'assistant', content: 'Strike soffit forms after seven days.' },
    ] })
  }
  if (p.startsWith('/v1/agents/conversations/')) return json({ messages: [] })
  if (p === '/v1/agents') return json({ agents: [] })
  return json({}, 404)
}

async function settle() {
  for (let i = 0; i < 10; i++) await act(async () => { await new Promise((r) => setTimeout(r, 0)) })
}

async function openWorkspaceOnTheSession() {
  dom.window.localStorage.clear()
  stub.sent = []
  stub.kept = []
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
  const click = async (el) => {
    await act(async () => { el.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true })) })
    await settle()
  }
  const open = [...container.querySelectorAll('button')].find((b) => b.textContent.includes(TITLE))
  assert.ok(open, 'the stored session is listed')
  await click(open)
  const sessionGroup = () => container.querySelector('section[aria-label="This session\'s documents"]')?.textContent ?? ''
  assert.match(sessionGroup(), /No files in this session yet/, 'nothing produced yet')
  return {
    container, click, sessionGroup,
    unmount() { act(() => root.unmount()); container.remove() },
  }
}

const exportControls = [
  ['the Word button under an answer', (c) => c.querySelector('button[aria-label="Download as Word document"]')],
  ['the sidebar conversation export', (c) => c.querySelector('button[title="Export this conversation as a .docx file"]')],
]

for (const [name, find] of exportControls) {
  test(`a file produced from ${name} is listed under This session at once`, async () => {
    const ws = await openWorkspaceOnTheSession()
    const control = find(ws.container)
    assert.ok(control, `${name} rendered`)
    await ws.click(control)
    assert.equal(stub.kept.length, 1, 'the export ran')
    assert.ok(ws.sessionGroup().includes(stub.kept[0].original_name),
      `listed at once: ${ws.sessionGroup()}`)
    assert.match(ws.sessionGroup(), /Produced/)
    ws.unmount()
  })
}
