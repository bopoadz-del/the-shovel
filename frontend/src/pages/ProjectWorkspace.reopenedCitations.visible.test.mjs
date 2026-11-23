// node --test src/pages/ProjectWorkspace.reopenedCitations.visible.test.mjs
// W3, through the real workspace: a session reopened from the list shows its
// answers as they were shown live -- the numbered markers link to entries in
// the Sources panel, the panel lists each entry with its document and
// section, and the coverage line comes back. Before this, the workspace read
// only the role and text of a stored answer and the panel said "No citations
// for the current answer". Mounts the routed ProjectWorkspace page in jsdom
// with a stubbed server; every name below is invented.
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


const PROJECT = 'p7'
const OPENED = 'ws-p7-1700000000000'
const OTHER = 'ws-p7-1700000999000'
const SESSIONS = [
  { id: OTHER, title: 'Kerb line survey', updated_at: '2026-01-02T00:00:00Z' },
  { id: OPENED, title: 'Plinth depth check', updated_at: '2026-01-01T00:00:00Z' },
]
const STORED = {
  [OPENED]: [
    { role: 'user', content: 'How deep is the plinth?' },
    {
      role: 'assistant',
      content: 'The plinth is 450 mm deep [1]; the kerb upstand is 125 mm [2].',
      citations: [
        { marker: 1, document: 'Plinth_Notes_311.pdf', section: 'p. 7', tier: 'Your project document', kind: 'document' },
        { marker: 2, document: 'Kerb_Schedule_207.xlsx', section: 'Sheet 2', tier: null, kind: 'document' },
        { marker: null, document: 'Uncited_Memo_905.docx', section: '', tier: null, kind: 'document' },
      ],
      coverage: { indexed: 4, total: 6 },
    },
  ],
  [OTHER]: [
    { role: 'user', content: 'Any kerb notes?' },
    { role: 'assistant', content: 'No kerb notes were found.', citations: null },
  ],
}

globalThis.fetch = async (url, init = {}) => {
  const u = new URL(String(url), 'http://localhost')
  const p = u.pathname
  const hist = p.match(/^\/v1\/agents\/conversations\/([^/]+)\/messages$/)
  if (hist) return json({ messages: STORED[decodeURIComponent(hist[1])] ?? [] })
  if (/^\/v1\/projects\/[^/]+\/conversations\/[^/]+\/documents$/.test(p)) return json({ documents: [] })
  if (p === `/v1/projects/${PROJECT}/conversations`) return json({ conversations: SESSIONS })
  if (p === `/v1/projects/${PROJECT}`) {
    return json({ id: PROJECT, name: 'Invented depot', documents: [], held_document_count: 0, searchable_document_count: 0 })
  }
  if (p === '/v1/agents') return json({ agents: [] })
  if (p === '/v1/users/me') return json({ id: 'u1', email: 'tester@example.invalid', name: 'Tester' })
  void init
  return json({}, 404)
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { 'content-type': 'application/json' } })
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
  return {
    container,
    text: () => container.textContent.replace(/\s+/g, ' ').trim(),
    async openSession(title) {
      const btn = [...container.querySelectorAll('button')].find((b) => b.textContent.includes(title))
      assert.ok(btn, `session "${title}" listed`)
      await act(async () => { btn.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true })) })
      await settle()
    },
    unmount() { act(() => root.unmount()); container.remove() },
  }
}

test('a reopened answer links its markers to its Sources entries', async () => {
  const ws = await mountWorkspace()
  await ws.openSession('Plinth depth check')
  assert.match(ws.text(), /The plinth is 450 mm deep/)

  const markers = [...ws.container.querySelectorAll('[data-citation-marker]')]
  const inText = markers.filter((el) => el.tagName !== 'LI').map((el) => Number(el.getAttribute('data-citation-marker')))
  assert.deepEqual(inText.sort(), [1, 2], 'both markers in the reopened answer are links')

  const entries = [...ws.container.querySelectorAll('li[data-citation-marker]')]
  assert.deepEqual(entries.map((li) => Number(li.getAttribute('data-citation-marker'))), [1, 2])
  assert.match(entries[0].textContent, /Plinth_Notes_311\.pdf/)
  assert.match(entries[0].textContent, /p\. 7/)
  assert.match(entries[1].textContent, /Kerb_Schedule_207\.xlsx/)
  assert.ok(!ws.text().includes('No citations for the current answer'), 'the panel is not empty')
  assert.match(ws.text(), /4 of 6 project documents indexed/)

  // Switching to a session whose answer cited nothing clears the panel.
  await ws.openSession('Kerb line survey')
  assert.equal(ws.container.querySelectorAll('li[data-citation-marker]').length, 0)
  assert.match(ws.text(), /No citations for the current answer/)
  ws.unmount()
})
