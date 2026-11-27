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
      ref: 'ws-proj-object-1763626500000#2', label: { conversation: 'Bill rates check', message: 2 },
      words: 'message 2 of “Bill rates check”', written_at: '2026-11-20T08:15:00+00:00' }],
    shows: 'ask_first',
  },
  {
    project: 'proj-array',
    notes: [
      { name: 'how-to', data: [{ kind: 'schedule', start_with: 'list_documents' }],
        ref: 'ws-proj-array-1764092400000#5', label: { conversation: null, message: 5 },
        words: 'message 5 of a conversation', written_at: '2026-11-25T17:40:00+00:00' },
      { name: 'other', data: { keep: true }, ref: 'ws-proj-array-1764147600000#6',
        label: { conversation: 'Pour dates', message: 6 }, words: 'message 6 of “Pour dates”',
        written_at: '2026-11-26T09:00:00+00:00' },
    ],
    shows: 'start_with',
  },
]

let store = {}
let posts = []
let refuse = false
let gates = {}
let full = null
const DECIDER = 'u-3f9a12bc'

function reset() {
  store = {}
  for (const c of CASES) {
    store[c.project] = {
      notes: c.notes.map((n) => ({ name: n.name, data: n.data, bytes: 10, written_at: n.written_at,
        written_by_turn: n.ref, turn: n.label, review: '', reviewed_at: '', reviewed_by: '',
        version: `v1-${c.project}-${n.name}` })),
      reviews: [],
    }
  }
  store['proj-empty'] = { notes: [], reviews: [] }
  posts = []
  refuse = false
  gates = {}
  full = null
}

// The model writes the note again on the server: new content, new version.
function rewrite(project, name, data) {
  const note = store[project].notes.find((n) => n.name === name)
  Object.assign(note, { data, version: `v2-${project}-${name}`, review: '' })
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
    if (body.version !== note.version) return json({ detail: 'written again after it was opened' }, 409)
    if (full && body.decision === 'keep' && note.review === 'discarded') return json({ detail: full }, 422)
    note.review = body.decision === 'keep' ? 'kept' : 'discarded'
    store[m[1]].reviews.push({ name: m[2], decision: body.decision, data: note.data,
      note_written_at: note.written_at, decided_at: '2026-11-27T01:00:00+00:00', decided_by: DECIDER })
    return json({ name: m[2], decision: body.decision })
  }
  m = p.match(/^\/v1\/admin\/projects\/([^/]+)\/memory$/)
  if (m) {
    const notes = structuredClone(store[m[1]].notes)
    if (gates[m[1]]) await gates[m[1]].held
    return json({ notes })
  }
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
    async choose(projectId) {
      const sel = container.querySelector('#memory-project')
      await act(async () => {
        sel.value = projectId
        sel.dispatchEvent(new dom.window.Event('change', { bubbles: true }))
      })
    },
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
        assert.ok(text.includes(n.words), text)
        // The stored turn reference is an internal id: never on screen.
        assert.equal(text.includes(n.ref.split('#')[0]), false, text)
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
      assert.deepEqual(posts, [{ project: c.project, name: target.name,
        body: { decision: 'discard', version: `v1-${c.project}-${target.name}` } }])
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
      // The decider's internal id is never printed on the screen.
      assert.equal(v.container.textContent.includes(DECIDER), false)
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

for (const c of CASES) {
  test(`a click on a version of ${c.project} the model has since rewritten records nothing and shows the new one`,
    async () => {
      reset()
      const target = c.notes[0]
      const v = await mount(c.project)
      try {
        const newer = { rewritten_for: c.project }
        rewrite(c.project, target.name, newer)
        await v.click(v.button(target.name, 'Keep'))
        assert.deepEqual(posts.map((x) => x.body.version), [`v1-${c.project}-${target.name}`])
        assert.deepEqual(store[c.project].reviews, [])
        assert.ok(v.container.querySelector('[role="alert"]').textContent.includes(`wrote ${target.name} again`))
        assert.deepEqual(JSON.parse(v.note(target.name).querySelector('pre').textContent), newer)
        assert.equal(v.note(target.name).getAttribute('data-review'), '')
        // The decision on the version now shown lands.
        await v.click(v.button(target.name, 'Keep'))
        assert.equal(posts[1].body.version, `v2-${c.project}-${target.name}`)
        assert.deepEqual(store[c.project].reviews.map((r) => r.data), [newer])
      } finally { v.unmount() }
    })
}

for (const [slow, fast] of [['proj-object', 'proj-array'], ['proj-array', 'proj-object']]) {
  test(`a slow answer for ${slow} after switching to ${fast} is dropped; Discard acts on what is shown`, async () => {
    reset()
    // Both lists are held until the test lets them go: the new project's
    // first, the old project's last.
    const hold = () => { let go; const held = new Promise((r) => { go = r }); return { held, go } }
    gates[slow] = hold()
    gates[fast] = hold()
    const v = await mount(slow)
    try {
      await v.choose(fast)
      await settle()
      // Nothing is listed under the new project before its own list arrives.
      assert.equal(v.container.querySelectorAll('[data-note]').length, 0)
      gates[fast].go()
      await settle()
      gates[slow].go()
      await settle()
      assert.equal(v.container.querySelector('#memory-project').value, fast)
      const want = store[fast].notes
      assert.deepEqual([...v.container.querySelectorAll('[data-note]')].map((el) => el.getAttribute('data-note')),
        want.map((n) => n.name))
      const shown = want[0]
      assert.deepEqual(JSON.parse(v.note(shown.name).querySelector('pre').textContent), shown.data)
      await v.click(v.button(shown.name, 'Discard'))
      assert.deepEqual(posts, [{ project: fast, name: shown.name,
        body: { decision: 'discard', version: shown.version } }])
      assert.ok(store[slow].notes.every((n) => n.review === ''))
    } finally { v.unmount() }
  })
}

for (const [from, to] of [['proj-object', 'proj-array'], ['proj-array', 'proj-object']]) {
  test(`switching from ${from} to ${to} lists none of ${from}'s notes while ${to}'s list loads`, async () => {
    reset()
    const v = await mount(from)
    try {
      assert.equal(v.container.querySelectorAll('[data-note]').length, store[from].notes.length)
      let go
      gates[to] = { held: new Promise((r) => { go = r }) }
      await v.choose(to)
      await settle()
      assert.equal(v.container.querySelectorAll('[data-note]').length, 0)
      assert.equal(v.container.querySelectorAll('.memory-note__actions button').length, 0)
      go()
      await settle()
      assert.deepEqual([...v.container.querySelectorAll('[data-note]')].map((el) => el.getAttribute('data-note')),
        store[to].notes.map((n) => n.name))
    } finally { v.unmount() }
  })
}

const { turnWords } = await server.ssrLoadModule('/src/pages/memoryTurn.ts')

test('a turn reads as message and conversation title, or as little as is known', () => {
  assert.equal(turnWords({ conversation: 'Bill rates', message: 3 }), 'message 3 of “Bill rates”')
  assert.equal(turnWords({ conversation: null, message: 7 }), 'message 7 of a conversation')
  assert.equal(turnWords({ conversation: 'Pour dates', message: null }), 'in “Pour dates”')
  assert.equal(turnWords({ conversation: null, message: null }), '')
  assert.equal(turnWords(undefined), '')
})

for (const c of CASES) {
  test(`a Keep on a discarded note of ${c.project} refused at the limit says why and leaves it discarded`,
    async () => {
      reset()
      const target = c.notes[0]
      const v = await mount(c.project)
      try {
        await v.click(v.button(target.name, 'Discard'))
        full = `keeping '${target.name}' would put ${c.notes.length + 1} notes in the model's sight`
        await v.click(v.button(target.name, 'Keep'))
        assert.deepEqual(posts.map((x) => x.body.decision), ['discard', 'keep'])
        assert.ok(v.container.querySelector('[role="alert"]').textContent.includes(full))
        assert.ok(!v.container.querySelector('[role="alert"]').textContent.includes('again'))
        assert.equal(v.note(target.name).getAttribute('data-review'), 'discarded')
        assert.deepEqual(store[c.project].reviews.map((r) => r.decision), ['discard'])
      } finally { v.unmount() }
    })
}
