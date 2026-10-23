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
