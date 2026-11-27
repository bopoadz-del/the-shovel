// node --test src/pages/MemoryNotes.visible.test.mjs
// The owner's screen for the model's own notes, mounted in a real DOM (jsdom +
// react-dom/client) against a fake server that holds its notes and decisions in
// memory. Every case runs on two different projects whose notes have different
// shapes (a JSON object, a JSON array), so a constant render cannot pass.
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
const { default: MemoryNotes } = await server.ssrLoadModule('/src/pages/MemoryNotes.tsx')

const CASES = [
  {
    project: 'proj-object',
    notes: [{ name: 'paths', data: { ask_first: 'which bill', tool_for: { rates: 'rows_aggregate' } },
      turn: 'conv-a#2', written_at: '2026-11-20T08:15:00+00:00' }],
    shows: 'ask_first',
  },
  {
    project: 'proj-array',
    notes: [
      { name: 'how-to', data: [{ kind: 'schedule', start_with: 'list_documents' }],
        turn: 'conv-b#5', written_at: '2026-11-25T17:40:00+00:00' },
      { name: 'other', data: { keep: true }, turn: 'conv-b#6', written_at: '2026-11-26T09:00:00+00:00' },
    ],
    shows: 'start_with',
  },
]

let store = {}
let posts = []
let refuse = false

function reset() {
  store = {}
  for (const c of CASES) {
    store[c.project] = {
      notes: c.notes.map((n) => ({ name: n.name, data: n.data, bytes: 10, written_at: n.written_at,
        written_by_turn: n.turn, review: '', reviewed_at: '', reviewed_by: '' })),
      reviews: [],
    }
  }
  store['proj-empty'] = { notes: [], reviews: [] }
  posts = []
  refuse = false
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { 'content-type': 'application/json' } })
}

globalThis.fetch = async (url, init = {}) => {
  const u = new URL(String(url), 'http://localhost')
  const method = (init.method ?? 'GET').toUpperCase()
  const p = decodeURIComponent(u.pathname)
  if (p === '/v1/projects') return json({ projects: Object.keys(store).map((id) => ({ id, name: `Name of ${id}` })) })
  if (refuse) return json({ detail: 'Admin access required' }, 403)
  let m = p.match(/^\/v1\/admin\/projects\/([^/]+)\/memory\/reviews$/)
  if (m) return json({ reviews: store[m[1]].reviews })
  m = p.match(/^\/v1\/admin\/projects\/([^/]+)\/memory\/([^/]+)\/review$/)
  if (m && method === 'POST') {
    const body = JSON.parse(init.body)
    posts.push({ project: m[1], name: m[2], body })
    const note = store[m[1]].notes.find((n) => n.name === m[2])
    note.review = body.decision === 'keep' ? 'kept' : 'discarded'
    store[m[1]].reviews.push({ name: m[2], decision: body.decision, data: note.data,
      note_written_at: note.written_at, decided_at: '2026-11-27T01:00:00+00:00', decided_by: 'owner' })
    return json({ name: m[2], decision: body.decision })
  }
  m = p.match(/^\/v1\/admin\/projects\/([^/]+)\/memory$/)
  if (m) return json({ notes: store[m[1]].notes })
  return json({}, 404)
}

async function settle() {
  for (let i = 0; i < 10; i++) await act(async () => { await new Promise((r) => setTimeout(r, 0)) })
}

async function mount(projectId) {
  const container = dom.window.document.createElement('div')
  dom.window.document.body.appendChild(container)
  const root = createRoot(container)
  await act(async () => { root.render(React.createElement(MemoryNotes, { initialProjectId: projectId })) })
  await settle()
  return {
    container,
    note: (name) => container.querySelector(`[data-note="${name}"]`),
    button: (name, label) => [...container.querySelector(`[data-note="${name}"]`).querySelectorAll('button')]
      .find((b) => b.textContent === label),
    async click(el) {
      await act(async () => { el.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true })) })
      await settle()
    },
    unmount() { act(() => root.unmount()); container.remove() },
  }
}

for (const c of CASES) {
  test(`every note of ${c.project} shows its name, date, turn and whole content`, async () => {
    reset()
    const v = await mount(c.project)
    try {
      for (const n of c.notes) {
        const text = v.note(n.name).textContent
        assert.ok(text.includes(n.name), text)
        assert.ok(text.includes(n.turn), text)
        assert.ok(text.includes(n.written_at.slice(0, 10)), text)
        assert.ok(text.includes('Not reviewed'), text)
        assert.deepEqual(JSON.parse(v.note(n.name).querySelector('pre').textContent), n.data)
      }
      assert.ok(v.container.textContent.includes(c.shows))
      assert.equal(v.container.querySelectorAll('[data-note]').length, c.notes.length)
      assert.equal(v.container.textContent.includes('written no notes'), false)
    } finally { v.unmount() }
  })

  test(`discarding a note of ${c.project} posts the decision and shows it marked, content unchanged`, async () => {
    reset()
    const target = c.notes[0]
    const v = await mount(c.project)
    try {
      await v.click(v.button(target.name, 'Discard'))
      assert.deepEqual(posts, [{ project: c.project, name: target.name, body: { decision: 'discard' } }])
      const el = v.note(target.name)
      assert.equal(el.getAttribute('data-review'), 'discarded')
      assert.ok(el.textContent.includes('the model no longer sees it'))
      assert.deepEqual(JSON.parse(el.querySelector('pre').textContent), target.data)
      assert.equal(v.button(target.name, 'Discard').disabled, true)
      assert.equal(v.button(target.name, 'Keep').disabled, false)
      // Other notes of the project are not marked.
      for (const n of c.notes.slice(1)) assert.equal(v.note(n.name).getAttribute('data-review'), '')
      const record = v.container.querySelector('.memory-decisions')
      assert.ok(record.textContent.includes('Every decision (1)'))
      assert.ok(record.textContent.includes(`Discarded ${target.name}`))
    } finally { v.unmount() }
  })

  test(`keeping a note of ${c.project} posts keep and the record grows with each decision`, async () => {
    reset()
    const target = c.notes[0]
    const v = await mount(c.project)
    try {
      await v.click(v.button(target.name, 'Keep'))
      await v.click(v.button(target.name, 'Discard'))
      assert.deepEqual(posts.map((x) => x.body.decision), ['keep', 'discard'])
      assert.ok(v.container.querySelector('.memory-decisions').textContent.includes('Every decision (2)'))
    } finally { v.unmount() }
  })
}

for (const projectId of ['proj-empty', '']) {
  test(`no note listed and no decision possible for ${projectId || 'no project chosen'}`, async () => {
    reset()
    const v = await mount(projectId)
    try {
      assert.equal(v.container.querySelectorAll('[data-note]').length, 0)
      assert.equal(v.container.querySelectorAll('.memory-note__actions button').length, 0)
      assert.equal(v.container.textContent.includes('The model has written no notes in this project.'),
        projectId !== '')
      assert.deepEqual(posts, [])
    } finally { v.unmount() }
  })
}

for (const c of CASES) {
  test(`a refused read of ${c.project} says why and lists nothing`, async () => {
    reset()
    refuse = true
    const v = await mount(c.project)
    try {
      assert.equal(v.container.querySelectorAll('[data-note]').length, 0)
      assert.ok(v.container.querySelector('[role="alert"]').textContent.includes('Admin access required'))
    } finally { v.unmount() }
  })
}
