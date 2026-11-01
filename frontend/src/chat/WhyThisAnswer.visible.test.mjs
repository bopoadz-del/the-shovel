// node --test src/chat/WhyThisAnswer.visible.test.mjs
// Renders the "Why this answer" panel the way the RightPanel does and reads the
// text a person sees. The envelope badge must read as plain words (Certified
// calculation / Reference data / Estimated) -- the machine provenance value
// (e.g. reference_table, user_input) must never leak as a word or key=value.
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

const { default: WhyThisAnswer } = await server.ssrLoadModule('/src/chat/WhyThisAnswer.tsx')

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

function render(entries, streaming = false) {
  return renderToStaticMarkup(React.createElement(WhyThisAnswer, { entries, streaming }))
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
      inputs: { water_depth: 23, raft_thickness: 2 },
      provenance: c.provenance,
      standard: c.provenance === 'kernel' ? 'BS EN 1997' : 'none declared',
      certified: c.certified,
    }])
    const text = visibleText(html)
    assert.match(text, /8\.75 T\/m2/)
    assert.match(text, new RegExp(c.label, 'i'))
    // The formula + inputs are humanised, the standard shows when declared.
    assert.match(text, /Dewatering uplift check/i)
    if (c.provenance === 'kernel') assert.match(text, /per BS EN 1997/)
    // No internal codes: not the provenance value, not the source, not key=value.
    assert.doesNotMatch(text, SNAKE)
    assert.doesNotMatch(text, KEY_VALUE)
    assert.doesNotMatch(text, /none declared/)
  })
}

test('defaulted inputs are listed as the sentences the calculator produced', () => {
  const html = render([{
    figure: '12.0 hours',
    source: 'calculator',
    formula: 'formwork_striking_time',
    inputs: { concrete_strength_7h: 2 },
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
  assert.match(visibleText(render([], true)), /appears once the answer completes/i)
  assert.equal(render([], false), '')
})
