// node --test src/chat/SessionDocuments.visible.test.mjs
// W2 / G-1, rendered: mounts the real components in jsdom and reads what a
// person sees, or what the browser would send. Replaces two tests that only
// grepped the source (they passed on dead code):
//  - both upload paths send the open session's id (and none without one);
//  - "This session" sits above the project's Documents in the sidebar.
// Adds G-1: a failed read of the session's files shows an error with a
// retry, never "No files in this session yet".
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
const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' })
globalThis.window = dom.window
globalThis.document = dom.window.document
globalThis.IS_REACT_ACT_ENVIRONMENT = true
if (globalThis.navigator === undefined) globalThis.navigator = dom.window.navigator
for (const k of ['HTMLElement', 'Element', 'Node', 'Text', 'Event', 'MouseEvent',
  'DocumentFragment', 'getComputedStyle', 'requestAnimationFrame', 'cancelAnimationFrame',
  'localStorage', 'sessionStorage']) {
  if (dom.window[k] !== undefined) globalThis[k] = dom.window[k]
}
const { createRoot } = await import('react-dom/client')
const { act } = await import('react')

const { default: SessionDocumentsGroup } = await server.ssrLoadModule('/src/layout/SessionDocumentsGroup.tsx')
const { default: LeftPanel } = await server.ssrLoadModule('/src/layout/LeftPanel.tsx')
const { AuthProvider } = await server.ssrLoadModule('/src/auth/AuthContext.tsx')
const { MemoryRouter } = await import('react-router-dom')
const { DocumentsPanel } = await server.ssrLoadModule('/src/pages/ProjectWorkspace.tsx')
const { default: ChatComposer } = await server.ssrLoadModule('/src/chat/ChatComposer.tsx')

// Every request the components make is recorded; uploads answer with a
// stored document, any other read with an empty list.
let sent = []
globalThis.fetch = async (url, init = {}) => {
  sent.push({ url: String(url), method: init.method ?? 'GET', body: init.body })
  const json = (obj, status = 200) => new Response(JSON.stringify(obj), {
    status, headers: { 'content-type': 'application/json' },
  })
  if ((init.method ?? 'GET') === 'POST') {
    return json({ document: { id: 'doc-new', original_name: 'spec.md' } })
  }
  if (String(url).includes('/v1/upload-limits')) return json({}, 404)
  return json({ projects: [] })
}

async function mount(element) {
  const container = dom.window.document.createElement('div')
  dom.window.document.body.appendChild(container)
  const root = createRoot(container)
  await act(async () => { root.render(element) })
  return {
    container,
    text: () => container.textContent.replace(/\s+/g, ' ').trim(),
    async click(el) {
      await act(async () => { el.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true })) })
    },
    async chooseFile(input, file) {
      Object.defineProperty(input, 'files', { configurable: true, value: [file] })
      await act(async () => { input.dispatchEvent(new dom.window.Event('change', { bubbles: true })) })
      // the upload awaits the bytes and the reply; let them settle
      for (let i = 0; i < 5; i++) await act(async () => { await new Promise((r) => setTimeout(r, 0)) })
    },
    unmount() { act(() => root.unmount()); container.remove() },
  }
}

const NO_FILES = 'No files in this session yet.'
const ROW = { id: 'd1', name: 'site-diary.md', tag: 'Uploaded' }

test('a failed read says so and offers a retry; it never says "no files"', async () => {
  let retries = 0
  const failed = await mount(React.createElement(SessionDocumentsGroup, {
    view: { kind: 'error', rows: [] }, onRetry: () => { retries += 1 },
  }))
  assert.match(failed.text(), /Could not load this session's files/)
  assert.ok(!failed.text().includes(NO_FILES))
  await failed.click(failed.container.querySelector('button'))
  assert.equal(retries, 1)
  failed.unmount()

  const empty = await mount(React.createElement(SessionDocumentsGroup, { view: { kind: 'empty', rows: [] } }))
  assert.ok(empty.text().includes(NO_FILES))
  assert.ok(!/Could not load/.test(empty.text()))
  empty.unmount()
})

test('a failed re-read keeps the files already listed; loading is not "no files"', async () => {
  const stale = await mount(React.createElement(SessionDocumentsGroup, { view: { kind: 'error', rows: [ROW] } }))
  assert.ok(stale.text().includes('site-diary.md'))
  assert.match(stale.text(), /Could not load/)
  stale.unmount()

  const loading = await mount(React.createElement(SessionDocumentsGroup, { view: { kind: 'loading', rows: [] } }))
  assert.ok(!loading.text().includes(NO_FILES))
  loading.unmount()
})

function sidebar(view) {
  return React.createElement(MemoryRouter, null,
    React.createElement(AuthProvider, null,
      React.createElement(LeftPanel, {
        activeProjectId: 'p1',
        activeProjectName: 'Project',
        activeConversationId: 'ws-p1-1',
        conversations: [],
        sessionDocuments: view,
        documents: React.createElement('p', null, 'project-doc-list'),
      })))
}

test('"This session" is rendered above the project Documents, in every state', async () => {
  for (const view of [{ kind: 'list', rows: [ROW] }, { kind: 'error', rows: [] }]) {
    const panel = await mount(sidebar(view))
    const heads = [...panel.container.querySelectorAll('header')].map((h) => h.textContent.trim())
    const sessionAt = heads.indexOf('This session')
    const docsAt = heads.indexOf('Documents')
    assert.ok(sessionAt >= 0 && docsAt >= 0, `both groups rendered: ${heads.join(' | ')}`)
    assert.ok(sessionAt < docsAt, 'This session above Documents')
    assert.equal(panel.text().includes(NO_FILES), false)
    panel.unmount()
  }
})

function uploadedForm() {
  const post = sent.find((r) => r.method === 'POST' && r.url.includes('/documents'))
  assert.ok(post, `an upload was sent: ${sent.map((r) => r.url).join(', ')}`)
  return post.body
}

const documentsPanel = (sessionId) => React.createElement(DocumentsPanel, {
  projectId: 'p1', sessionId, documents: [], pinnedIds: [],
  onDocumentAdded() {}, onDocumentRemoved() {}, onDocumentReplaced() {}, onAsk() {},
})
const composer = (sessionId) => React.createElement(ChatComposer, {
  onSend: () => true, disabled: false, projectId: 'p1', sessionId,
})

for (const [name, render, pick] of [
  ['documents panel', documentsPanel, (c) => c.querySelector('input[aria-label="Choose file to upload"]')],
  ['chat composer', composer, (c) => c.querySelector('input[type="file"][multiple]')],
]) {
  test(`the ${name} upload carries the open session id, and none without a session`, async () => {
    for (const [sessionId, expected] of [['ws-p1-1700000000000', 'ws-p1-1700000000000'], [null, null]]) {
      sent = []
      const view = await mount(render(sessionId))
      const input = pick(view.container)
      assert.ok(input, 'file input rendered')
      await view.chooseFile(input, new File(['# spec'], 'spec.md', { type: 'text/markdown' }))
      assert.equal(uploadedForm().get('session_id'), expected)
      view.unmount()
    }
  })
}
