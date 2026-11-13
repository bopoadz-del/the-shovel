// node --test src/pages/KnowledgeLayers.visible.test.mjs
// S-U1: system knowledge layers render as one read-only row (name + chunk
// count, no Delete), separate from the user's own project cards, and the
// Projects page no longer carries a pilot tip. Each case builds its own
// projects and runs on two inputs.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { after, test } from 'node:test'
import { fileURLToPath } from 'node:url'

import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
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

const { default: KnowledgeLayers } = await server.ssrLoadModule('/src/pages/KnowledgeLayers.tsx')
const { partitionProjects } = await server.ssrLoadModule('/src/pages/partitionProjects.ts')
const { default: ProjectCard } = await server.ssrLoadModule('/src/pages/ProjectCard.tsx')
// Vite SSR externalises node_modules, so this is the same router copy the
// components import.
const { MemoryRouter } = await import('react-router-dom')

function render(node) {
  return renderToStaticMarkup(React.createElement(MemoryRouter, null, node))
}

function visibleText(html) {
  return html.replace(/<[^>]+>/g, ' ').replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim()
}

function project(id, name, extra = {}) {
  return { id, name, status: 'active', created_at: '2026-01-01T00:00:00Z', ...extra }
}

for (const [layerName, count, expected] of [
  ['Reference library A', 35192, '35,192 chunks'],
  ['Training notes B', 1, '1 chunk'],
]) {
  test(`a knowledge layer shows its name and chunk count, no delete (${layerName})`, () => {
    const layer = project(`id-${count}`, layerName, { is_system: true, chunk_count: count })
    const text = visibleText(render(React.createElement(KnowledgeLayers, { layers: [layer] })))
    assert.match(text, /Knowledge layers/)
    assert.ok(text.includes(layerName))
    assert.ok(text.includes(expected), text)
    assert.doesNotMatch(text, /delete/i)
  })
}

for (const [systemFlags, userIds] of [
  [[true, false, false], ['u1', 'u2']],
  [[false, true], ['u0']],
]) {
  test(`partition puts flagged layers apart from user projects (${systemFlags})`, () => {
    const all = systemFlags.map((flag, i) => project(flag ? `s${i}` : `u${i}`, `P${i}`, { is_system: flag }))
    const { userProjects, layers } = partitionProjects(all)
    assert.deepEqual(userProjects.map((p) => p.id), userIds)
    assert.ok(layers.every((p) => p.is_system))
    assert.equal(layers.length + userProjects.length, all.length)
  })
}

for (const layers of [[], undefined]) {
  test(`no layers renders nothing (${JSON.stringify(layers)})`, () => {
    assert.equal(render(React.createElement(KnowledgeLayers, { layers: layers ?? [] })), '')
  })
}

test('a user project card still renders as a card (contrast with the layer row)', () => {
  const html = render(React.createElement(ProjectCard, { project: project('u9', 'My tower') }))
  assert.ok(visibleText(html).includes('My tower'))
  assert.doesNotMatch(html, /knowledge-layers/)
})

for (const phrase of ['Pilot tip', 'still being indexed']) {
  test(`the Projects page carries no stale pilot tip (${phrase})`, () => {
    const src = readFileSync(path.join(frontendRoot, 'src/pages/Projects.tsx'), 'utf8')
    assert.ok(!src.includes(phrase))
  })
}
