// node --test src/chat/WhyThisAnswer.visible.test.mjs
// Renders the "Why this answer" panel the way the RightPanel does and reads the
// text a person sees. The envelope badge must read as plain words (Certified
// calculation / Reference data / Estimated) -- the machine provenance value
// (e.g. reference_table, user_input) must never leak as a word or key=value.
// S-12b: each calculator input carries a plain WHERE word (your words / project
// data / document / unresolved); an absent source shows nothing; an unknown
// source renders raw so a regression is visible.
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

const { default: WhyThisAnswer, resolveDocumentSource, CalcInputs } =
  await server.ssrLoadModule('/src/chat/WhyThisAnswer.tsx')

// Walk a React element tree (plain {type, props} objects — no DOM needed) and
// collect host nodes carrying a given class. CalcInputs is a pure view, so we
// can call it as a function, render a state, and invoke a button's onClick the
// way a real click would — without a browser harness this repo does not have.
function elements(node, out = []) {
  if (node == null || typeof node !== 'object') return out
  if (Array.isArray(node)) {
    for (const n of node) elements(n, out)
    return out
  }
  out.push(node)
  if (node.props && node.props.children != null) elements(node.props.children, out)
  return out
}
function byClass(tree, cls) {
  return elements(tree).filter(
    (n) => n.props && typeof n.props.className === 'string'
      && n.props.className.split(' ').includes(cls),
  )
}

function visibleText(html) {
  return html
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&#39;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/\s+/g, ' ')
    .trim()
}

function render(entries, opts = {}) {
  return renderToStaticMarkup(React.createElement(WhyThisAnswer, { entries, ...opts }))
}

// Two calculator badges -> two distinct visible labels; neither leaks the
// machine provenance value.
const CALC_CASES = [
  { provenance: 'kernel', certified: true, label: 'Certified calculation' },
  { provenance: 'reference_table', certified: false, label: 'Reference data' },
  { provenance: 'improvised', certified: false, label: 'Estimated' },
]

for (const c of CALC_CASES) {
  test(`a ${c.provenance} calc figure shows "${c.label}" and never the machine value`, () => {
    const html = render([{
      figure: '8.75 T/m2',
      source: 'calculator',
      formula: 'dewatering_uplift_check',
      inputs: [
        { name: 'water_depth', value: 23, unit: 'm', source: 'user_text' },
        { name: 'raft_thickness', value: 2, unit: 'm', source: 'project_data' },
      ],
      provenance: c.provenance,
      standard: c.provenance === 'kernel' ? 'BS EN 1997' : 'none declared',
      certified: c.certified,
    }])
    const text = visibleText(html)
    assert.match(text, /8\.75 T\/m2/)
    assert.match(text, new RegExp(c.label, 'i'))
    assert.match(text, /Dewatering uplift check/i)
    if (c.provenance === 'kernel') assert.match(text, /per BS EN 1997/)
    // No internal codes: not the provenance value, not the raw source, not key=value.
    assert.doesNotMatch(text, SNAKE)
    assert.doesNotMatch(text, KEY_VALUE)
    assert.doesNotMatch(text, /none declared/)
  })
}

// S-12b: each input names WHERE its value came from, in plain words.
test('each input shows its source in plain words; unresolved is plain; absent shows nothing', () => {
  const html = render([{
    figure: 'SAR 53,564.80',
    source: 'calculator',
    formula: 'delay_damages_daily',
    inputs: [
      { name: 'rate_percent', value: 0.137, unit: '%', source: 'project_data' },
      { name: 'contract_amount', value: 39098392.98, unit: 'SAR', source: 'document:cd_chunk_7' },
      { name: 'days', value: 60, source: 'user_text' },
      { name: 'factor', value: 1.0, source: 'unresolved' },
      { name: 'rounding', value: 2 }, // no source -> no label
    ],
    provenance: 'kernel',
    standard: 'ACA',
    certified: true,
  }])
  const text = visibleText(html)
  assert.match(text, /rate percent/)
  assert.match(text, /project data/)
  assert.match(text, /document/) // the chunk id is never shown
  assert.doesNotMatch(text, /cd_chunk_7/)
  assert.match(text, /your words/)
  assert.match(text, /unresolved/)
  // the no-source input (rounding, the last one) carries no source word:
  // nothing after it gets a " · " label. (visibleText turns </span> into a
  // space, so the key reads "rounding : 2" — assert on the name, not spacing.)
  assert.match(text, /rounding/)
  assert.doesNotMatch(text.slice(text.indexOf('rounding')), /·/)
  // Collapsed by default behind an aria-expanded toggle; the inputs are in the
  // DOM (hidden) so the panel can reveal them on click.
  assert.match(html, /aria-expanded="false"/)
  assert.match(html, /class="why-row__inputs" hidden/)
  assert.match(text, /Inputs \(5\)/)
})

// An unknown/unexpected source renders raw so a backend regression is visible.
test('an unknown source value renders raw, not hidden', () => {
  const html = render([{
    figure: '10',
    source: 'calculator',
    formula: 'excavation_volume',
    inputs: [{ name: 'depth', value: 10, source: 'mystery' }],
    provenance: 'kernel',
    certified: true,
  }])
  assert.match(visibleText(html), /mystery/)
})

test('defaulted inputs are listed as the sentences the calculator produced', () => {
  const html = render([{
    figure: '12.0 hours',
    source: 'calculator',
    formula: 'formwork_striking_time',
    inputs: [{ name: 'concrete_strength_7h', value: 2, source: 'user_text' }],
    provenance: 'kernel',
    standard: 'BS 8110',
    certified: true,
    defaults: ['The minimum striking time used, 12 h, is the platform default, not a project figure.'],
  }])
  const text = visibleText(html)
  assert.match(text, /minimum striking time used/)
  assert.doesNotMatch(text, SNAKE)
})

test('a user figure reads "From your message", a document figure names the file', () => {
  const html = render([
    { figure: '5,000', source: 'user_input' },
    { figure: '1,754,504.25', source: 'project_document', doc_name: 'Contract Data.pdf', page_or_section: 'p. 3' },
  ])
  const text = visibleText(html)
  assert.match(text, /From your message/)
  assert.match(text, /Contract Data\.pdf/)
  assert.doesNotMatch(text, /user_input/)
  assert.doesNotMatch(text, KEY_VALUE)
})

test('streaming with nothing yet shows the waiting line; empty + done renders nothing', () => {
  assert.match(visibleText(render([], { streaming: true })), /appears once the answer completes/i)
  assert.equal(render([], { streaming: false }), '')
})

// S-12b: a document-sourced input resolves to its cited source and renders the
// word "document" as a link that opens that source; the chunk id is never shown.
const CITED = [{ doc_id: 'spec_a', doc_name: 'Concrete Spec.pdf', page_or_section: 'p. 3', score: 0.9, confidence: 'High' }]

test('resolveDocumentSource matches document:<doc_id>[#chunk] to its cited source', () => {
  // the stamped source is document:<doc_id> or document:<doc_id>#<n>; both resolve
  assert.equal(resolveDocumentSource('document:spec_a#3', CITED), CITED[0])
  assert.equal(resolveDocumentSource('document:spec_a', CITED), CITED[0])
  // no matching cited source, a non-document source, and missing inputs -> null
  assert.equal(resolveDocumentSource('document:other', CITED), null)
  assert.equal(resolveDocumentSource('project_data', CITED), null)
  assert.equal(resolveDocumentSource('document:spec_a', undefined), null)
})

const DOC_INPUTS = [
  { name: 'contract_amount', value: 39098392.98, unit: 'SAR', source: 'document:spec_a#3' },
  { name: 'volume_m3', value: 900, unit: 'm3', source: 'user_text' },
]

test('the toggle expands and collapses the inputs — aria-expanded + ul visibility follow open', () => {
  // collapsed (default): toggle aria-expanded=false, inputs ul hidden
  let tree = CalcInputs({ inputs: DOC_INPUTS, open: false, onToggle: () => {} })
  assert.equal(byClass(tree, 'why-row__toggle')[0].props['aria-expanded'], false)
  assert.equal(byClass(tree, 'why-row__inputs')[0].props.hidden, true)
  // expanded (after a click): aria-expanded=true and the ul is visible
  tree = CalcInputs({ inputs: DOC_INPUTS, open: true, onToggle: () => {} })
  assert.equal(byClass(tree, 'why-row__toggle')[0].props['aria-expanded'], true)
  assert.ok(!byClass(tree, 'why-row__inputs')[0].props.hidden) // visible
  // clicking the toggle asks the parent to flip state, then it collapses again
  let toggles = 0
  tree = CalcInputs({ inputs: DOC_INPUTS, open: false, onToggle: () => { toggles += 1 } })
  byClass(tree, 'why-row__toggle')[0].props.onClick()
  assert.equal(toggles, 1)
})

test('a matched document input renders the word as a link that calls onOpenSource with its cited source', () => {
  const opened = []
  const tree = CalcInputs({
    inputs: DOC_INPUTS, open: true, onToggle: () => {},
    sources: CITED, onOpenSource: (s) => opened.push(s),
  })
  const link = byClass(tree, 'why-row__srclink')[0]
  assert.ok(link, 'the document input should render as a link')
  assert.equal(link.props.children, 'document') // the word, chunk id never shown
  link.props.onClick() // the click
  assert.deepEqual(opened, [CITED[0]]) // opened the matched cited source
})

test('an unmatched document input renders the plain word, no link', () => {
  const tree = CalcInputs({
    inputs: [{ name: 'x', value: 1, source: 'document:not_cited#0' }],
    open: true, onToggle: () => {}, sources: CITED, onOpenSource: () => {},
  })
  assert.equal(byClass(tree, 'why-row__srclink').length, 0)
  assert.equal(byClass(tree, 'why-row__src')[0].props.children.join(''), ' · document')
})
