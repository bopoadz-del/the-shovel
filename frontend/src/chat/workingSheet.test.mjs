// node --test src/chat/workingSheet.test.mjs
// The display view of a working sheet. The first two tests read REAL end
// events of the catalogue path (tests/fixtures/show_working, re-run and
// compared by tests/test_driver_catalogue_path.py::
// test_the_panel_fixtures_are_real_end_events): a three-step chain that
// finished, and a chain that stopped on a missing input; a third has plan
// literals written as {value, unit} objects. The rest build
// their own sheets, two inputs each. Nothing here names a formula or a
// project.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'

import {
  checkedWord,
  endingWords,
  entryWords,
  givenSource,
  literalQuantity,
  quantityText,
  readSheet,
  showNumber,
  unitWords,
  workingView,
} from './workingSheet.js'

const FIXTURES = new URL('../../../tests/fixtures/show_working/', import.meta.url)
const recorded = (name) => JSON.parse(readFileSync(new URL(name, FIXTURES), 'utf8'))

// What the screen shows of a view: every label and source word, no keys.
function shownWords(v) {
  return [
    ...v.givens.flatMap((g) => [g.what, g.source]),
    ...v.steps.flatMap((s) => [s.label, s.tool, ...s.needs, ...s.inputs.flatMap((i) => [i.name, i.from])]),
    ...v.missing.flatMap((m) => [m.label, m.for]),
  ].join(' | ')
}

test('real chain end event: steps in plan order, results resolved, every typed given from your words', () => {
  const turn = recorded('three_step_chain.json')
  const v = workingView(turn.sheet)
  assert.deepEqual(v.steps.map((s) => s.label), ['step 1', 'step 2', 'step 3'])
  assert.deepEqual(v.steps.map((s) => s.status), ['done', 'done', 'done'])
  assert.deepEqual(v.steps.map((s) => s.result.value), [7.2, 50.4, 100.8])
  // step 2 reads step 1's result, resolved from the sheet and named by place
  assert.deepEqual(v.steps[1].inputs[1], { name: 'a', from: 'step 1', value: 7.2, unit: 'm*m' })
  assert.deepEqual(v.steps[0].inputs[1], { name: 'a', from: 'wall length', value: 12, unit: 'm' })
  // the user typed every figure: code marked each "user", none is unchecked
  assert.deepEqual(v.givens.map((g) => g.source), ['your words', 'your words', 'your words', 'your words'])
  assert.equal(v.unchecked, 0)
  assert.equal(endingWords(turn.ended), 'answered')
})

test('real stopped end event: the waiting step names its needs, the model-written given is not checked', () => {
  const turn = recorded('stopped_on_missing_input.json')
  const v = workingView(turn.sheet)
  assert.deepEqual(v.steps.map((s) => s.status), ['waiting_input', 'not_run'])
  assert.deepEqual(v.steps.map((s) => s.result), [null, null])
  assert.deepEqual(v.steps[0].needs, ['second value'])
  // step 2 reads step 1, which has no result: not available, never a number
  assert.deepEqual(v.steps[1].inputs[1], { name: 'a', from: 'step 1', missing: true })
  assert.deepEqual(v.missing, [{ for: 'step 1', label: 'pit plan area' }])
  // the model labelled the bulking factor "general knowledge"; code did not check it
  assert.deepEqual(v.givens.map((g) => [g.what, g.source]), [['pit depth', 'your words'], ['bulking factor', '']])
  assert.equal(v.unchecked, 1)
  assert.equal(endingWords(turn.ended), 'asked you for an input')
})

test('no internal id is put on screen, on both real end events', () => {
  for (const name of ['three_step_chain.json', 'stopped_on_missing_input.json']) {
    const words = shownWords(workingView(recorded(name).sheet))
    assert.doesNotMatch(words, /\b[gs]\d+\b/, `${name}: ${words}`)
    assert.doesNotMatch(words, /_/, `${name}: ${words}`)
  }
})

test("each check code records is shown as what code recorded, two givens each", () => {
  const cases = [
    ['user', 'your words'],
    ['calculator', 'a calculation'],
    ['passage', 'a retrieved passage'],
    [true, 'checked'],
    ['a_kind_added_later', 'checked'],
  ]
  for (const [checked, word] of cases) {
    for (const value of [450000, 1.75]) {
      assert.equal(givenSource({ id: 'g1', value, checked }), word, String(checked))
    }
  }
})

test("a given code did not check is shown as not checked, whatever label the model wrote", () => {
  for (const given of [
    { id: 'g1', value: 12, source: 'user' },
    { id: 'g2', value: 7, source: 'bill p. 14' },
    { id: 'g3', value: 7, source: 'user', checked: '' },
    { id: 'g4', value: 7, source: 'user', checked: false },
    { id: 'g5', value: 7, source: 'user', checked: '  ' },
    { id: 'g6', value: 7, source: 'user', checked: null },
  ]) {
    assert.equal(givenSource(given), '', JSON.stringify(given))
  }
  assert.equal(givenSource(null), '')
  assert.equal(givenSource([{ checked: 'user' }]), '')
  for (const [marks, unchecked] of [[['user', undefined, 'passage'], 1], [[undefined, undefined, 'calculator'], 2]]) {
    const sheet = { givens: marks.map((checked, i) => ({ id: `g${i + 1}`, what: 'x', value: i, unit: 'm', checked })) }
    assert.equal(workingView(sheet).unchecked, unchecked)
  }
})

test('checkedWord on its own, two inputs per word', () => {
  assert.equal(checkedWord('user'), 'your words')
  assert.equal(checkedWord('calculator'), 'a calculation')
  assert.equal(checkedWord('passage'), 'a retrieved passage')
  assert.equal(checkedWord(true), 'checked')
  assert.equal(checkedWord('later'), 'checked')
  assert.equal(checkedWord(undefined), '')
  assert.equal(checkedWord(0), '')
})

test("a failed step shows no figure and no backend message, even with a result present", () => {
  for (const error of ["'formula:no_such_formula' is not a registered formula", 's1 is not a plan step']) {
    const sheet = {
      givens: [{ id: 'g1', what: 'x', value: 5, unit: 'm' }],
      plan: [{ id: 's1', entry: 'formula:arithmetic_step', inputs: { a: 'g1' } },
        { id: 's2', entry: 'formula:arithmetic_step', inputs: { a: 's1' } }],
      steps: [{ id: 's1', status: 'failed', error, result: { value: 9, unit: 'm3' } }],
    }
    const v = workingView(sheet)
    assert.equal(v.steps[0].status, 'failed')
    assert.equal(v.steps[0].result, null)
    assert.equal(JSON.stringify(v).includes(error), false)
    // step 2 reads the failed step: not available, not its stray result
    assert.deepEqual(v.steps[1].inputs[0], { name: 'a', from: 'step 1', missing: true })
  }
})

test('needs are shown only for a step waiting for an input', () => {
  for (const [status, needs] of [['failed', []], ['done', []], ['waiting_input', ['unit rate']]]) {
    const sheet = {
      plan: [{ id: 's1', entry: 'formula:arithmetic_step', inputs: {} }],
      steps: [{ id: 's1', status, needs: ['unit_rate'], result: { value: 1, unit: '' } }],
    }
    assert.deepEqual(workingView(sheet).steps[0].needs, needs, status)
  }
})

test('a status run_step does not write is shown as unknown, never raw', () => {
  for (const status of ['half_done', '']) {
    const sheet = { plan: [{ id: 's1', entry: 'x', inputs: {} }], steps: [{ id: 's1', status }] }
    assert.equal(workingView(sheet).steps[0].status, 'unknown')
  }
  const notRun = workingView({ plan: [{ id: 's1', entry: 'x', inputs: {} }], steps: [] })
  assert.equal(notRun.steps[0].status, 'not_run')
})

test('a missing input for a step off the plan names no step', () => {
  for (const ref of ['s9', '']) {
    const v = workingView({ plan: [{ id: 's1', entry: 'x' }], missing: [{ for: ref, input: 'unit_rate' }] })
    assert.deepEqual(v.missing, [{ for: '', label: 'unit rate' }])
  }
})

test('a plan input naming a given the sheet does not hold is shown as not available, on two sheets', () => {
  for (const [held, absent] of [['g1', 'g4'], ['g2', 'g9']]) {
    const raw = {
      givens: [{ id: held, what: 'depth', value: 2.5, unit: 'm', source: 'user' }],
      plan: [{ id: 's1', entry: 'formula:arithmetic_step', inputs: { a: held, b: absent, op: 'add' } }],
      steps: [],
      missing: [],
    }
    const [a, b] = workingView(raw).steps[0].inputs
    assert.deepEqual(a, { name: 'a', from: 'depth', value: 2.5, unit: 'm' })
    assert.deepEqual(b, { name: 'b', from: '', missing: true })
  }
})

test('not a sheet, or an empty sheet: nothing to show', () => {
  for (const raw of [null, undefined, 'x', 3, [], {}, { givens: [], plan: [], missing: [] }]) {
    assert.equal(readSheet(raw), null)
    assert.equal(workingView(raw), null)
  }
  assert.notEqual(readSheet({ missing: [{ input: 'x' }] }), null)
})

test('small helpers on two inputs each', () => {
  assert.equal(entryWords('formula:pile_cap_volume'), 'pile cap volume')
  assert.equal(entryWords('arithmetic_step'), 'arithmetic step')
  assert.equal(showNumber(52.92000000000001), '52.92')
  assert.equal(showNumber(0.1 + 0.2), '0.3')
  assert.equal(endingWords('answer'), 'answered')
  assert.equal(endingWords('asked_input'), 'asked you for an input')
  assert.equal(endingWords('ceiling'), 'stopped at the round limit')
  assert.equal(endingWords('other'), '')
})

// An input line as the panel prints it (ShowWorking.tsx inputLine).
const inputText = (i) => (i.missing ? 'not available yet' : quantityText(i.value, i.unit))

test('real end event with {value, unit} literals in the plan: each shown as its number and unit', () => {
  const turn = recorded('plan_literals.json')
  const v = workingView(turn.sheet)
  const b = v.steps.map((s) => s.inputs.find((i) => i.name === 'b'))
  assert.deepEqual(b.map(inputText), ['1.05', '2.5 m3'])
  assert.deepEqual(v.steps.map((s) => quantityText(s.result.value, s.result.unit)), ['42 m3', '44.5 m3'])
  const shown = v.steps.flatMap((s) => s.inputs.map(inputText)).join(' | ')
  assert.ok(!shown.includes('[object'), shown)
})

// Every shape the backend accepts for a plan literal (sheet.written_quantities,
// _formula_values), two values each: what the input line shows.
const SHAPES = [
  ['a number', 3, '3'],
  ['a number', 0.1 + 0.2, '0.3'],
  ['text with a unit', '2.5 m3', '2.5 m3'],
  ['text with no number', 'multiply', 'multiply'],
  ['a {value, unit} object', { value: 1.05, unit: '-' }, '1.05'],
  ['a {value, unit} object', { value: 12, unit: 'm' }, '12 m'],
  ['a {value} object with no unit', { value: 7 }, '7'],
  ['a list', [1, 2], '1, 2'],
  ['a list of mixed shapes', [{ value: 2, unit: 'm' }, 'each'], '2 m, each'],
  ['an object of named values', { unit_rate: 5, currency: 'AED' }, 'unit rate 5, currency AED'],
  ['an object of named quantities', { length: { value: 3, unit: 'm' } }, 'length 3 m'],
  ['a {value} object holding an object', { value: { value: 4, unit: 'kg' } }, '4 kg'],
  ['a flag', true, 'yes'],
  ['a flag', false, 'no'],
]

for (const [shape, literal, want] of SHAPES) {
  test(`a plan literal written as ${shape} (${JSON.stringify(literal)}) is shown as "${want}"`, () => {
    const sheet = {
      givens: [{ id: 'g1', what: 'x', value: 1, unit: 'm' }],
      plan: [{ id: 's1', entry: 'formula:arithmetic_step', inputs: { a: 'g1', b: literal } }],
    }
    const b = workingView(sheet).steps[0].inputs[1]
    assert.equal(b.from, '')
    assert.equal(inputText(b), want)
  })
}

test('literalQuantity keeps a {value, unit} number and its unit apart, on two objects', () => {
  assert.deepEqual(literalQuantity({ value: 1.05, unit: '-' }), { value: 1.05, unit: '-' })
  assert.deepEqual(literalQuantity({ value: 40, unit: 'm3' }), { value: 40, unit: 'm3' })
  assert.deepEqual(literalQuantity(9), { value: 9, unit: '' })
  assert.deepEqual(literalQuantity([{ value: 2, unit: 'm' }]), { value: '2 m', unit: '' })
})

test('an empty or absent literal shows nothing, never a raw value', () => {
  for (const literal of [null, undefined, [], {}]) {
    assert.equal(inputText(literalQuantity(literal)), '')
  }
})

test('unitWords and quantityText: "-" and blank are no unit, two inputs each', () => {
  assert.equal(unitWords('-'), '')
  assert.equal(unitWords(' '), '')
  assert.equal(unitWords('m2'), 'm2')
  assert.equal(unitWords(' kg '), 'kg')
  assert.equal(quantityText(42, 'm3'), '42 m3')
  assert.equal(quantityText(1.05, '-'), '1.05')
  assert.equal(quantityText('', 'm'), 'm')
})
