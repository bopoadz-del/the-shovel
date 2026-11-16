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

const stub = { sent: [], kept: [], failExports: false }
// The offer a finished answer carries (invented): generated server-side on click.
const OFFER = {
  label: 'Invented cost workbook', format: 'xlsx', method: 'POST',
  endpoint: `/v1/projects/${PROJECT}/exports/invented-workbook`, payload: { conversation_id: SESSION },
}

function keepAndServe(name) {
  // The server keeps what it serves as the session's Produced document.
  stub.kept.push({ id: `prod-${stub.kept.length + 1}`, original_name: name, origin: 'produced' })
  return new Response(new Blob(['PK']), {
    status: 200, headers: { 'content-disposition': `attachment; filename="${name}"` },
  })
}

function json(obj, status = 200, headers = {}) {
  return new Response(JSON.stringify(obj), { status, headers: { 'content-type': 'application/json', ...headers } })
}

globalThis.fetch = async (url, init = {}) => {
  const u = new URL(String(url), 'http://localhost')
  const method = (init.method ?? 'GET').toUpperCase()
  const p = u.pathname
  stub.sent.push({ method, path: p })
  const isExport = method === 'POST' && (
    p === `/v1/projects/${PROJECT}/conversations/${SESSION}/export` || p === OFFER.endpoint)
  if (isExport && stub.failExports) return json({ detail: 'invented failure' }, 500)
  if (method === 'POST' && p === `/v1/projects/${PROJECT}/conversations/${SESSION}/export`) {
    const fmt = u.searchParams.get('format') ?? 'docx'
    return keepAndServe(`Invented_Tower-${stub.kept.length + 1}.${fmt}`)
  }
  if (method === 'POST' && p === OFFER.endpoint) {
    return keepAndServe(`Invented_workbook-${stub.kept.length + 1}.${OFFER.format}`)
  }
  if (method === 'POST' && p === '/v1/chat/stream') {
    const end = { type: 'end', content: 'The workbook can be generated from the priced bill.', exports: [OFFER] }
    return new Response(`data: ${JSON.stringify(end)}

`, {
      status: 200, headers: { 'content-type': 'text/event-stream' },
    })
  }
  if (p === '/livez') return json({ status: 'alive' })
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
  stub.failExports = false
  stub.alerts = []
  globalThis.alert = dom.window.alert = (msg) => { stub.alerts.push(String(msg)) }
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

async function askForTheOffer(ws) {
  // A live answer carries the offer; it is never on a stored message.
  const box = ws.container.querySelector('textarea[aria-label="Chat message"]')
  assert.ok(box, 'the composer rendered')
  const setValue = Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype, 'value').set
  await act(async () => {
    setValue.call(box, 'Can I have the priced bill as a workbook?')
    box.dispatchEvent(new dom.window.Event('input', { bubbles: true }))
  })
  await ws.click(ws.container.querySelector('button[aria-label="Send message"]'))
  return ws.container.querySelector(`button[aria-label="Download ${OFFER.label}"]`)
}

const exportControls = [
  ['the Word button under an answer', async (ws) => ws.container.querySelector('button[aria-label="Download as Word document"]')],
  ['the sidebar conversation export', async (ws) => ws.container.querySelector('button[title="Export this conversation as a .docx file"]')],
  ['the download offer an answer carries', askForTheOffer],
]

const sessionDocReads = () => stub.sent.filter((r) => r.method === 'GET'
  && r.path === `/v1/projects/${PROJECT}/conversations/${encodeURIComponent(SESSION)}/documents`).length

for (const [name, find] of exportControls) {
  test(`a FAILED export from ${name} is reported and does not re-read the session`, async () => {
    const ws = await openWorkspaceOnTheSession()
    const control = await find(ws)
    assert.ok(control, `${name} rendered`)
    stub.failExports = true
    const readsBefore = sessionDocReads()
    await ws.click(control)
    assert.equal(stub.kept.length, 0, 'nothing was produced')
    assert.equal(stub.alerts.length, 1, `the failure is reported: ${stub.alerts}`)
    assert.match(stub.alerts[0], /failed \(500\)/)
    assert.equal(sessionDocReads(), readsBefore, 'a failed export re-read the session list')
    assert.match(ws.sessionGroup(), /No files in this session yet/)
    ws.unmount()
  })

  test(`a file produced from ${name} is listed under This session at once`, async () => {
    const ws = await openWorkspaceOnTheSession()
    const control = await find(ws)
    assert.ok(control, `${name} rendered`)
    await ws.click(control)
    assert.equal(stub.kept.length, 1, 'the export ran')
    assert.ok(ws.sessionGroup().includes(stub.kept[0].original_name),
      `listed at once: ${ws.sessionGroup()}`)
    assert.match(ws.sessionGroup(), /Produced/)
    ws.unmount()
  })
}
