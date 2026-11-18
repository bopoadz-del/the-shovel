// node --test src/pages/ProjectWorkspace.panelRequestOrder.visible.test.mjs
// W3 follow-up (#379): the side panel shows the NEWER of two requests, a
// document opened from the Sources list or a marker opened from the answer.
// The workspace numbered both with the clock, so two clicks in one
// millisecond tied and a clock set back ran the order backwards: the marker
// clicked last did not open its entry. Through the real page: list ->
// workspace -> panel, with the clock frozen and with the clock running back.
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
dom.window.HTMLElement.prototype.scrollIntoView ??= () => {}
const { createRoot } = await import('react-dom/client')
const { act } = await import('react')
const { MemoryRouter, Routes, Route } = await import('react-router-dom')

const { default: ProjectWorkspace } = await server.ssrLoadModule('/src/pages/ProjectWorkspace.tsx')
const { AuthProvider } = await server.ssrLoadModule('/src/auth/AuthContext.tsx')
const { ThemeProvider } = await server.ssrLoadModule('/src/theme/ThemeContext.tsx')

// Invented fixture: one project, one stored session, an answer citing two documents.
const PROJECT = 'p7'
const SESSION = 'ws-p7-1700000456000'
const TITLE = 'Lintel bearing lengths'
const DOCS = [
  { doc_id: 'inv-a', doc_name: 'Invented masonry spec.pdf', page_or_section: 'p. 4', score: 0.91, confidence: 'High' },
  { doc_id: 'inv-b', doc_name: 'Invented lintel schedule.pdf', page_or_section: 'sheet 2', score: 0.82, confidence: 'Medium' },
]
const END = {
  type: 'end',
  content: 'Bearing is 150 mm each end [1]; precast units follow the schedule [2].',
  sources: DOCS,
  citations: [
    { marker: 1, document: DOCS[0].doc_name, section: 'p. 4', passage: 'Minimum bearing 150 mm.' },
    { marker: 2, document: DOCS[1].doc_name, section: 'sheet 2', passage: 'Precast lintel types L1 to L4.' },
  ],
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { 'content-type': 'application/json' } })
}

globalThis.fetch = async (url, init = {}) => {
  const u = new URL(String(url), 'http://localhost')
  const method = (init.method ?? 'GET').toUpperCase()
  const p = u.pathname
  if (method === 'POST' && p === '/v1/chat/stream') {
    return new Response(`data: ${JSON.stringify(END)}\n\n`, {
      status: 200, headers: { 'content-type': 'text/event-stream' },
    })
  }
  if (p === '/livez') return json({ status: 'alive' })
  if (p.match(new RegExp(`^/v1/projects/${PROJECT}/conversations/[^/]+/documents$`))) return json({ documents: [] })
  if (p === `/v1/projects/${PROJECT}/conversations`) {
    return json({ conversations: [{ id: SESSION, title: TITLE, updated_at: '2026-01-01T00:00:00Z' }] })
  }
  if (p === `/v1/projects/${PROJECT}`) {
    return json({ id: PROJECT, name: 'Invented school', documents: [], held_document_count: 0, searchable_document_count: 0 })
  }
  if (p.startsWith('/v1/agents/conversations/')) return json({ messages: [] })
  if (p === '/v1/agents') return json({ agents: [] })
  return json({}, 404)
}

async function settle() {
  for (let i = 0; i < 10; i++) await act(async () => { await new Promise((r) => setTimeout(r, 0)) })
}

async function workspaceWithACitedAnswer() {
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
  const click = async (el) => {
    assert.ok(el, 'the control to click rendered')
    await act(async () => { el.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true })) })
    await settle()
  }
  await click([...container.querySelectorAll('button')].find((b) => b.textContent.includes(TITLE)))
  const box = container.querySelector('textarea[aria-label="Chat message"]')
  assert.ok(box, 'the composer rendered')
  const setValue = Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype, 'value').set
  await act(async () => {
    setValue.call(box, 'What bearing do the lintels need?')
    box.dispatchEvent(new dom.window.Event('input', { bubbles: true }))
  })
  await click(container.querySelector('button[aria-label="Send message"]'))
  return {
    click,
    tab: () => container.querySelector('.right-panel [role="tab"][aria-selected="true"]')?.textContent,
    listEntry: (name) => container.querySelector(`.right-panel button[aria-label="Preview ${name}"]`),
    marker: (n) => container.querySelector(`button[aria-label="Source ${n}"]`),
    activeEntry: () => container.querySelector('.sources-list__item--active')?.getAttribute('data-citation-marker'),
    unmount() { act(() => root.unmount()); container.remove() },
  }
}

// Two clocks that give no order: frozen, and set back between clicks.
const clocks = [
  ['a frozen clock', () => { const t = 1_800_000_000_000; return () => t }],
  ['a clock set back', () => { let t = 1_800_000_000_000; return () => (t -= 1000) }],
]

for (const [name, makeClock] of clocks) {
  test(`with ${name}, the request clicked last decides the panel: list -> marker -> list -> marker`, async () => {
    const ws = await workspaceWithACitedAnswer()
    assert.equal(ws.tab(), 'Sources', 'the cited answer opens on Sources')
    const realNow = Date.now
    Date.now = makeClock()
    try {
      await ws.click(ws.listEntry(DOCS[0].doc_name))
      assert.equal(ws.tab(), 'Sheet', 'a document opened from the list shows its preview')

      await ws.click(ws.marker(2))
      assert.equal(ws.tab(), 'Sources', 'the marker clicked after the list opens Sources')
      assert.equal(ws.activeEntry(), '2', 'on the entry of the marker clicked')

      await ws.click(ws.listEntry(DOCS[1].doc_name))
      assert.equal(ws.tab(), 'Sheet', 'the list clicked after the marker opens the preview')

      await ws.click(ws.marker(1))
      assert.equal(ws.tab(), 'Sources', 'the second marker opens Sources again')
      assert.equal(ws.activeEntry(), '1')
    } finally {
      Date.now = realNow
      ws.unmount()
    }
  })
}
