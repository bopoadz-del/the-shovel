// node --test src/chat/SourcesList.visible.test.mjs
// Renders the Sources panel the way the chat page does and reads the text
// a person sees. Machine ids may sit on data-* attributes; they must not
// appear as words or as key=value.
import assert from 'node:assert/strict'
import path from 'node:path'
import { after, test } from 'node:test'
import { fileURLToPath } from 'node:url'

import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const SNAKE = /\b[a-z]+_[a-z_]+\b/
const KEY_VALUE = /\w+=\w+/

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')

const server = await createServer({
  root: frontendRoot,
  configFile: path.join(frontendRoot, 'vite.config.ts'),
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
})
after(() => server.close())

const { default: SourcesList } = await server.ssrLoadModule('/src/chat/SourcesList.tsx')
const { default: ChatBubble } = await server.ssrLoadModule('/src/chat/ChatBubble.tsx')
const { SOURCE_CLASS_LABELS } = await server.ssrLoadModule('/src/chat/sourceClassLabels.ts')
const { activityForTool } = await server.ssrLoadModule('/src/chat/userFacingText.ts')

// jsdom for the W3 click and tab tests below; every module is loaded here,
// before the first test runs (a later top-level await lets the runner
// finish and close the vite server first).
const { JSDOM } = await import('jsdom')
const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' })
globalThis.window = dom.window
globalThis.document = dom.window.document
globalThis.IS_REACT_ACT_ENVIRONMENT = true
for (const k of ['HTMLElement', 'Element', 'Node', 'Text', 'Event', 'MouseEvent', 'DocumentFragment',
  'getComputedStyle', 'requestAnimationFrame', 'cancelAnimationFrame', 'localStorage']) {
  if (dom.window[k] !== undefined) globalThis[k] = dom.window[k]
}
globalThis.fetch = async () => new Response('{}', { status: 404, headers: { 'content-type': 'application/json' } })
const { createRoot } = await import('react-dom/client')
const { act } = await import('react')
const { default: RightPanel } = await server.ssrLoadModule('/src/layout/RightPanel.tsx')
const { readCoverage } = await server.ssrLoadModule('/src/chat/citationMarkers.ts')

function visibleText(html) {
  return html
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&#39;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/\s+/g, ' ')
    .trim()
}

function source(sourceClass, label) {
  const row = {
    doc_id: 'doc-1',
    doc_name: 'Contract Data.pdf',
    page_or_section: 'p. 12',
    score: 0.91,
    confidence: 'High',
    layer_label: 'Project document',
    source_class: sourceClass,
  }
  if (label !== undefined) row.source_class_label = label
  return row
}

function render(sources) {
  return renderToStaticMarkup(React.createElement(SourcesList, { sources }))
}

const known = Object.entries(SOURCE_CLASS_LABELS)
assert.ok(known.length > 0, 'the source-class map is empty')

for (const [sourceClass, label] of known) {
  test(`Sources panel shows "${label}" for ${sourceClass} and no internal code`, () => {
    const html = render([source(sourceClass, label)])
    const text = visibleText(html)
    assert.match(text, new RegExp(label, 'i'))
    assert.doesNotMatch(text, SNAKE)
    assert.doesNotMatch(text, KEY_VALUE)
    assert.match(html, new RegExp(`data-source-class="${sourceClass}"`))
  })

  test(`Sources panel maps ${sourceClass} itself when no label is supplied`, () => {
    const html = render([source(sourceClass)])
    const text = visibleText(html)
    assert.match(text, new RegExp(label, 'i'))
    assert.doesNotMatch(text, SNAKE)
    assert.doesNotMatch(text, KEY_VALUE)
  })
}

test('an unseen source class is still plain words, with the id only in data-*', () => {
  const sourceClass = 'alpha_beta'
  assert.equal(SOURCE_CLASS_LABELS[sourceClass], undefined)
  const html = render([source(sourceClass)])
  const text = visibleText(html)
  assert.match(text, /alpha beta/)
  assert.doesNotMatch(text, SNAKE)
  assert.doesNotMatch(text, KEY_VALUE)
  assert.match(html, new RegExp(`data-source-class="${sourceClass}"`))
})

test('a label that is itself the machine id is not shown', () => {
  const html = render([source('alpha_beta', 'alpha_beta')])
  const text = visibleText(html)
  assert.match(text, /alpha beta/)
  assert.doesNotMatch(text, SNAKE)
  assert.doesNotMatch(text, KEY_VALUE)
})

test('every known class in one panel stays free of internal codes', () => {
  const html = render(known.map(([sourceClass, label]) => source(sourceClass, label)))
  const text = visibleText(html)
  for (const [, label] of known) assert.match(text, new RegExp(label, 'i'))
  assert.doesNotMatch(text, SNAKE)
  assert.doesNotMatch(text, KEY_VALUE)
})

test('a document title stays the file name', () => {
  const row = source('project_corpus', 'this contract')
  row.doc_name = 'fenwick_waterproofing_spec.docx'
  const text = visibleText(render([row]))
  assert.match(text, /fenwick_waterproofing_spec\.docx/)
  assert.match(text, /this contract/)
  assert.doesNotMatch(text, /class=/)
  assert.doesNotMatch(text, /project_corpus/)
})

test('a tool status shows words, not the tool id or a key=value list', () => {
  const html = renderToStaticMarkup(React.createElement(ChatBubble, {
    message: {
      id: 'm1',
      role: 'assistant',
      content: '',
      streaming: true,
      toolStatus: 'Running construction_calc (contract_amount=50000000, rate_percent=0.1)…',
    },
  }))
  const text = visibleText(html)
  assert.match(text, /construction calc/)
  assert.doesNotMatch(text, SNAKE)
  assert.doesNotMatch(text, KEY_VALUE)
})

test('a tool with no dedicated sentence is still plain words', () => {
  const line = activityForTool('generate_wbs')
  assert.match(line, /generate wbs/)
  assert.doesNotMatch(line, SNAKE)
  assert.doesNotMatch(line, KEY_VALUE)
})

// ── W3: citations live in the side panel only ──────────────────────────────
// The answer text carries numbered markers; each marker opens its entry in
// the Sources panel, where document, section, passage and tier are shown.
// Coverage (N of M documents indexed) is a panel line, never answer text.
// Every case builds its own invented answer and evidence.

const W3_CASES = [
  {
    content: 'The slab is 250 mm thick [1] and the cover is 40 mm [2].',
    citations: [
      { marker: 1, document: 'Alpha Structural Notes.pdf', section: 'p. 7', passage: 'slab thickness 250 mm', tier: 'Project document', kind: 'document' },
      { marker: 2, document: 'Beta Concrete Spec.docx', section: 'Section 3.2', passage: 'nominal cover 40 mm', tier: 'Reference knowledge', kind: 'document' },
    ],
    coverage: { indexed: 4, total: 6 },
  },
  {
    content: 'Retention is 5 % [1].\n\nThe excavation is 96 m³ [2].',
    citations: [
      { marker: 1, document: 'Gamma Conditions of Contract.pdf', section: 'Clause 14.3', passage: 'retention of five percent', tier: 'Benchmark', kind: 'document' },
      { marker: 2, document: 'Excavation volume', section: '', passage: 'length 8 m, width 4 m, depth 3 m', tier: null, kind: 'calculation' },
    ],
    coverage: { indexed: 11, total: 11 },
  },
]

function bubble(c) {
  return renderToStaticMarkup(React.createElement(ChatBubble, {
    message: { id: 'a1', role: 'assistant', content: c.content, citations: c.citations, coverage: c.coverage },
  }))
}

function panel(c) {
  return renderToStaticMarkup(React.createElement(SourcesList, { citations: c.citations, coverage: c.coverage }))
}

for (const [i, c] of W3_CASES.entries()) {
  test(`W3 case ${i}: each marker in the answer is a control that opens its Sources entry`, () => {
    const html = bubble(c)
    for (const cite of c.citations) {
      assert.match(html, new RegExp(`<button[^>]*data-citation-marker="${cite.marker}"`),
        `marker ${cite.marker} is not a control`)
    }
  })

  test(`W3 case ${i}: the answer text names no document and carries no source line or coverage`, () => {
    const text = visibleText(bubble(c))
    for (const cite of c.citations) assert.ok(!text.includes(cite.document), `answer shows ${cite.document}`)
    assert.doesNotMatch(text, /\bsources?\s*:/i)
    assert.doesNotMatch(text, /documents indexed/i)
    assert.doesNotMatch(text, /\[\d+\]/, 'a bare [n] is left in the text')
  })

  test(`W3 case ${i}: the panel shows every entry by its marker with document, section, passage and tier`, () => {
    const html = panel(c)
    const text = visibleText(html)
    for (const cite of c.citations) {
      assert.match(html, new RegExp(`data-citation-marker="${cite.marker}"`))
      assert.ok(text.includes(cite.document), `panel lacks ${cite.document}`)
      if (cite.section) assert.ok(text.includes(cite.section), `panel lacks ${cite.section}`)
      assert.ok(text.includes(cite.passage), `panel lacks the passage ${cite.passage}`)
      if (cite.tier) assert.ok(text.includes(cite.tier), `panel lacks the tier ${cite.tier}`)
      if (cite.kind === 'calculation') assert.match(text, /calculation/i)
    }
    const order = c.citations.map((cite) => html.indexOf(`data-citation-marker="${cite.marker}"`))
    assert.deepEqual(order, [...order].sort((a, b) => a - b), 'entries are not in marker order')
    assert.doesNotMatch(text, SNAKE)
    assert.doesNotMatch(text, KEY_VALUE)
  })

  test(`W3 case ${i}: coverage is a panel line`, () => {
    const text = visibleText(panel(c))
    assert.match(text, new RegExp(String.raw`\b${c.coverage.indexed}\s+of\s+${c.coverage.total}\b`))
  })

  test(`W3 case ${i}: a bracketed number with no entry stays plain text, never a control`, () => {
    const stray = c.citations.length + 5
    const html = bubble({ ...c, content: `${c.content} See figure [${stray}].` })
    assert.doesNotMatch(html, new RegExp(`data-citation-marker="${stray}"`))
    assert.ok(visibleText(html).includes(`[${stray}]`))
  })
}

// ── W3, driven: a click on a marker, and the tab the side panel shows ───────
// Mounted in jsdom and clicked, not only rendered: marker 1 then marker 2
// reach the workspace's handler with their own numbers; the side panel
// shows Sources or the preview by whichever request is NEWER; and the
// coverage reader refuses malformed input.


async function mountLive(element) {
  const container = dom.window.document.createElement('div')
  dom.window.document.body.appendChild(container)
  const root = createRoot(container)
  await act(async () => { root.render(element) })
  return {
    container,
    async rerender(next) { await act(async () => { root.render(next) }) },
    async click(el) {
      await act(async () => { el.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true })) })
    },
    activeTab: () => container.querySelector('[role="tab"][aria-selected="true"]')?.textContent.trim(),
    tab: (label) => [...container.querySelectorAll('[role="tab"]')].find((t) => t.textContent.trim() === label),
    unmount() { act(() => root.unmount()); container.remove() },
  }
}

for (const [i, c] of W3_CASES.entries()) {
  test(`W3 case ${i}: clicking marker 1 then marker 2 hands each number to the handler`, async () => {
    const got = []
    const view = await mountLive(React.createElement(ChatBubble, {
      message: { id: `m${i}`, role: 'assistant', content: c.content, citations: c.citations, coverage: c.coverage },
      onCitation: (n) => got.push(n),
    }))
    const markerButton = (n) => view.container.querySelector(`button[data-citation-marker="${n}"]`)
    for (const cite of c.citations) {
      assert.ok(markerButton(cite.marker), `marker ${cite.marker} rendered as a control`)
      await view.click(markerButton(cite.marker))
    }
    assert.deepEqual(got, c.citations.map((cite) => cite.marker))
    view.unmount()
  })
}

const DOC = { id: 'doc-a', name: 'Alpha Structural Notes.xlsx', type: 'xlsx', size: 10 }
const sidePanel = (sourcesRequest, previewRequest) => React.createElement(RightPanel, {
  sources: React.createElement('p', null, 'sources-body'),
  graph: React.createElement('p', null, 'graph-body'),
  projectId: 'p1', documents: [DOC], citedDocuments: [DOC],
  sourcesRequest, previewRequest,
})

test('W3: the newer of a marker request and a preview request picks the side panel tab', async () => {
  // preview first (lower nonce), marker second (higher): Sources
  let view = await mountLive(sidePanel({ marker: 1, nonce: 20 }, { docId: DOC.id, nonce: 10 }))
  assert.equal(view.activeTab(), 'Sources')
  // a newer preview then wins over that marker
  await view.rerender(sidePanel({ marker: 1, nonce: 20 }, { docId: DOC.id, nonce: 30 }))
  assert.equal(view.activeTab(), 'Sheet')
  view.unmount()

  // marker first (lower nonce), preview second (higher): the preview's tab
  view = await mountLive(sidePanel({ marker: 2, nonce: 10 }, { docId: DOC.id, nonce: 20 }))
  assert.equal(view.activeTab(), 'Sheet')
  view.unmount()
})

test('W3: a marker opens Sources from another tab, and again after the user moved away', async () => {
  const view = await mountLive(sidePanel(null, null))
  await view.click(view.tab('Doc'))
  assert.equal(view.activeTab(), 'Doc')
  await view.rerender(sidePanel({ marker: 1, nonce: 1 }, null))
  assert.equal(view.activeTab(), 'Sources', 'a marker click switches to Sources')
  await view.click(view.tab('Doc'))
  assert.equal(view.activeTab(), 'Doc', 'the user can leave Sources')
  await view.rerender(sidePanel({ marker: 1, nonce: 2 }, null))
  assert.equal(view.activeTab(), 'Sources', 'the same marker clicked again reopens Sources')
  view.unmount()
})

test('W3: the coverage reader keeps a well-formed value and refuses malformed ones', () => {
  assert.deepEqual(readCoverage({ indexed: 3, total: 9 }), { indexed: 3, total: 9 })
  for (const bad of [
    { indexed: '3', total: 9 },
    { indexed: 3, total: 0 },
    { indexed: Number.NaN, total: 4 },
    { total: 4 },
    'indexed 3 of 9',
    null,
  ]) assert.equal(readCoverage(bad), undefined, `accepted ${JSON.stringify(bad)}`)
})
