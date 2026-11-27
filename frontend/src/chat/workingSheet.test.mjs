// node --test src/chat/workingSheet.test.mjs
// The display view of a working sheet. Two sheet shapes, every value invented:
// a three-step chain that finished, and a chain that stopped on a missing
// input. One view for both: nothing here names a formula or a project.
import assert from 'node:assert/strict'
import { test } from 'node:test'

import { endingWords, entryWords, figureNumber, givenSource, readSheet, showNumber, workingView } from './workingSheet.js'

const CHAIN = {
  givens: [
    { id: 'g1', what: 'length', value: 12, unit: 'm', source: 'user' },
    { id: 'g2', what: 'width', value: 0.6, unit: 'm', source: 'user' },
    { id: 'g3', what: 'count', value: 7, unit: 'nr', source: 'bill' },
  ],
  plan: [
    { id: 's1', entry: 'formula:arithmetic_step', inputs: { a: 'g1', b: 'g2', op: 'multiply' } },
    { id: 's2', entry: 'formula:arithmetic_step', inputs: { a: 's1', b: 'g3', op: 'multiply' } },
    { id: 's3', entry: 'formula:arithmetic_step', inputs: { a: 's2', b: 1.05, op: 'multiply' } },
  ],
  steps: [
    { id: 's1', status: 'done', result: { name: 'result', value: 7.2, unit: 'm2' } },
    { id: 's2', status: 'done', result: { name: 'result', value: 50.4, unit: 'm2' } },
    { id: 's3', status: 'done', result: { name: 'result', value: 52.92000000000001, unit: 'm2' } },
  ],
  missing: [],
  done: true,
}

const STOPPED = {
  givens: [{ id: 'g1', what: 'depth', value: 2.5, unit: 'm', source: 'user' }],
  plan: [
    { id: 's1', entry: 'formula:trench_volume', inputs: { depth: 'g1' } },
    { id: 's2', entry: 'formula:arithmetic_step', inputs: { a: 's1', b: 3, op: 'multiply' } },
  ],
  steps: [{ id: 's1', status: 'waiting_input', needs: ['Trench length', 'Trench width'] }],
  missing: [{ for: 's1', input: 'length', label: 'Trench length', asked: true }],
  done: false,
}

const PROV_CHAIN = [
  { figure: '12 m', source: 'user_input' },
  { figure: '7 nr', source: 'project_document', doc_name: 'Bill 2', page_or_section: 'p. 14' },
  { figure: '52.92 m2', source: 'calculator' },
]

test('three-step chain: every step in plan order with its tool, inputs and result', () => {
  const v = workingView(CHAIN, PROV_CHAIN)
  assert.deepEqual(v.steps.map((s) => s.id), ['s1', 's2', 's3'])
  assert.deepEqual(v.steps.map((s) => s.status), ['done', 'done', 'done'])
  assert.equal(v.steps[0].tool, 'arithmetic step')
  assert.deepEqual(v.steps.map((s) => s.result.value), [7.2, 50.4, 52.92000000000001])
  // s2's first input is s1's result, resolved from the sheet, not re-typed
  assert.deepEqual(v.steps[1].inputs[0], { name: 'a', ref: 's1', value: 7.2, unit: 'm2' })
  // a literal input is passed as written, with no reference
  assert.deepEqual(v.steps[2].inputs[1], { name: 'b', ref: '', value: 1.05, unit: '' })
  assert.equal(v.done, true)
  assert.deepEqual(v.missing, [])
})

test('stopped chain: the waiting step names its needs, no result is shown, missing listed', () => {
  const v = workingView(STOPPED, [])
  assert.deepEqual(v.steps.map((s) => s.status), ['waiting_input', 'not_run'])
  assert.equal(v.steps[0].result, null)
  assert.deepEqual(v.steps[0].needs, ['Trench length', 'Trench width'])
  assert.equal(v.steps[1].result, null)
  // s2 reads s1, which has no result: shown as not available, never a number
  assert.deepEqual(v.steps[1].inputs[0], { name: 'a', ref: 's1', missing: true })
  assert.deepEqual(v.missing, [{ for: 's1', label: 'Trench length' }])
  assert.equal(v.done, false)
})

test('a failed step shows its reason and never a figure, even if a result is present', () => {
  const sheet = { ...STOPPED, steps: [{ id: 's1', status: 'failed', error: 'depth: out of range', result: { value: 9, unit: 'm3' } }] }
  const v = workingView(sheet, [])
  assert.equal(v.steps[0].result, null)
  assert.equal(v.steps[0].error, 'depth: out of range')
  const other = workingView({ ...sheet, steps: [{ id: 's1', status: 'failed', error: 'no output' }] }, [])
  assert.equal(other.steps[0].error, 'no output')
})

test("a given's source is code's provenance match, never the model's own label", () => {
  const v = workingView(CHAIN, PROV_CHAIN)
  assert.deepEqual(v.givens.map((g) => g.source), ['your words', '', 'project document (Bill 2, p. 14)'])
  assert.equal(v.unsourced, 1)
  // the model labelled g3 'bill' and g1 'user': with no provenance, no source at all
  const bare = workingView(CHAIN, [])
  assert.deepEqual(bare.givens.map((g) => g.source), ['', '', ''])
  assert.equal(bare.unsourced, 3)
})

test('an unresolved or empty provenance source is not a source; a different number does not match', () => {
  assert.equal(givenSource({ value: 12 }, [{ figure: '12 m', source: 'unresolved' }]), '')
  assert.equal(givenSource({ value: 12 }, [{ figure: '12 m', source: '' }]), '')
  assert.equal(givenSource({ value: 12 }, [{ figure: '12.5 m', source: 'user_input' }]), '')
  assert.equal(givenSource({ value: 1250.5 }, [{ figure: '1,250.5 m3', source: 'calculator' }]), 'calculator')
})

test('not a sheet, or an empty sheet: nothing to show', () => {
  for (const raw of [null, undefined, 'x', 3, [], {}, { givens: [], plan: [], missing: [] }]) {
    assert.equal(readSheet(raw), null)
    assert.equal(workingView(raw, []), null)
  }
  assert.notEqual(readSheet({ missing: [{ input: 'x' }] }), null)
})

test('small helpers on two inputs each', () => {
  assert.equal(figureNumber('1,250.5 m3'), 1250.5)
  assert.equal(figureNumber('-3 kN'), -3)
  assert.equal(figureNumber('none'), null)
  assert.equal(entryWords('formula:pile_cap_volume'), 'pile cap volume')
  assert.equal(entryWords('arithmetic_step'), 'arithmetic step')
  assert.equal(showNumber(52.92000000000001), '52.92')
  assert.equal(showNumber(0.1 + 0.2), '0.3')
  assert.equal(endingWords('answer'), 'answered')
  assert.equal(endingWords('ceiling'), 'stopped at the round limit')
  assert.equal(endingWords('other'), '')
})
