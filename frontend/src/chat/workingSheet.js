// The working sheet of a catalogue-path turn, read for display.
//
// The backend writes the sheet (app/agents/driver/sheet.py) and sends the
// final one on the SSE 'end' event. This module only READS it: one view for
// any sheet, whatever the formulas, the project or the number of steps. It
// never changes the answer text and it never works a source out again.
//
// Who wrote what, and so what the view may claim:
//   * steps[].result / status / needs -- code (run_step), shown as is;
//     a failed step is shown as not calculated: its `error` is a message to
//     the model, so it is not put on screen;
//   * givens -- the model wrote the value; a given's own `source` field is
//     the model's label, so it is NEVER shown as a source. What is shown is
//     the given's `checked` mark, which only code writes (sheet.mark_given
//     with citation_provenance.given_check): who gave that quantity in the
//     turn. A given with no mark is shown as not checked;
//   * plan inputs -- references the model wrote; the value shown is resolved
//     here from the sheet, the same way resolve_inputs does. A literal the
//     model wrote in the plan is shown as text in each shape the backend
//     accepts (literalQuantity), never as a raw object. Internal ids
//     (g1, s1) are not shown: a given is named by what it is, a step by its
//     place in the plan.

const GIVEN_REF = /^g\d+$/
const STEP_REF = /^s\d+$/

/** The step statuses run_step writes; anything else is shown as unknown. */
export const STEP_STATUSES = ['done', 'failed', 'waiting_input', 'not_run']

function isObject(v) {
  return v !== null && typeof v === 'object' && !Array.isArray(v)
}

function list(v) {
  return Array.isArray(v) ? v.filter(isObject) : []
}

function words(v) {
  return String(v ?? '').replace(/_/g, ' ').trim()
}

/** The sheet from an 'end' event (or a stored message), or null when the
 *  value is not a sheet. A sheet with no plan, no givens and nothing missing
 *  has nothing to show and is null too. */
export function readSheet(raw) {
  if (!isObject(raw)) return null
  const sheet = {
    givens: list(raw.givens),
    plan: list(raw.plan),
    steps: list(raw.steps),
    missing: list(raw.missing),
    done: raw.done === true,
  }
  if (!sheet.givens.length && !sheet.plan.length && !sheet.missing.length) return null
  return sheet
}

/** Code's mark on a given (`checked`, written by sheet.mark_given) as words:
 *  who gave the quantity in the turn. '' when code did not mark it. A mark
 *  code wrote without naming who (true, or a kind this view does not know)
 *  is shown as checked, never as a named source. */
export function checkedWord(checked) {
  if (checked === 'user') return 'your words'
  if (checked === 'calculator') return 'a calculation'
  if (checked === 'passage') return 'a retrieved passage'
  if (checked === true || (typeof checked === 'string' && checked.trim())) return 'checked'
  return ''
}

/** The source shown for a given: its `checked` mark and nothing else. */
export function givenSource(given) {
  return isObject(given) ? checkedWord(given.checked) : ''
}

/** A catalogue entry as words: "formula:pile_volume" -> "pile volume". */
export function entryWords(entry) {
  const s = String(entry ?? '')
  const name = s.includes(':') ? s.slice(s.indexOf(':') + 1) : s
  return words(name)
}

function stepOf(sheet, id) {
  return sheet.steps.find((s) => s.id === id) ?? null
}

/** A step's place in the plan as shown ("step 2"), or '' off the plan. */
function stepLabel(sheet, id) {
  const at = sheet.plan.findIndex((p) => p.id === id)
  return at < 0 ? '' : `step ${at + 1}`
}

/** A given as named on screen: what the model said it is. */
function givenLabel(g) {
  return words(g.what) || 'a given'
}

/** One plan input as shown: where its value comes from and the value. */
function inputView(sheet, name, ref) {
  const label = words(name)
  if (typeof ref === 'string' && GIVEN_REF.test(ref)) {
    const g = sheet.givens.find((x) => x.id === ref)
    return g
      ? { name: label, from: givenLabel(g), value: g.value, unit: String(g.unit ?? '') }
      : { name: label, from: '', missing: true }
  }
  if (typeof ref === 'string' && STEP_REF.test(ref)) {
    const s = stepOf(sheet, ref)
    const from = stepLabel(sheet, ref)
    if (s && s.status === 'done' && isObject(s.result)) {
      return { name: label, from, value: s.result.value, unit: String(s.result.unit ?? '') }
    }
    return { name: label, from, missing: true }
  }
  return { name: label, from: '', ...literalQuantity(ref) }
}

/** A unit as shown: "-" (no unit) and blank are not shown. */
export function unitWords(unit) {
  const u = String(unit ?? '').trim()
  return u === '-' ? '' : u
}

/** A value and its unit as one piece of text ("42 m3", "1.05"). */
export function quantityText(value, unit) {
  return [literalText(value), unitWords(unit)].filter(Boolean).join(' ')
}

/** A value the model wrote into the plan, as text, in every shape the
 *  backend accepts for one (sheet.written_quantities / _formula_values): a
 *  number, text ("2.5 m3", an operator), a {value, unit} object, a list of
 *  any of these, or another object of named values. Never a raw object. */
function literalText(v) {
  if (v === null || v === undefined) return ''
  if (typeof v === 'number') return showNumber(v)
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (Array.isArray(v)) return v.map(literalText).filter(Boolean).join(', ')
  if (isObject(v)) {
    if ('value' in v) return quantityText(v.value, v.unit)
    return Object.entries(v)
      .map(([k, x]) => [words(k), literalText(x)].filter(Boolean).join(' '))
      .filter(Boolean)
      .join(', ')
  }
  return String(v)
}

/** A plan literal as {value, unit}: a {value, unit} object keeps its number
 *  and unit apart; every other shape is its text with no unit. */
export function literalQuantity(ref) {
  if (isObject(ref) && 'value' in ref && !isObject(ref.value) && !Array.isArray(ref.value)) {
    return { value: ref.value, unit: String(ref.unit ?? '') }
  }
  return { value: typeof ref === 'number' ? ref : literalText(ref), unit: '' }
}

/** The whole view: givens with code's check, plan steps in plan order with
 *  their inputs, status and result, and the inputs still missing. */
export function workingView(rawSheet) {
  const sheet = readSheet(rawSheet)
  if (!sheet) return null
  const givens = sheet.givens.map((g, i) => ({
    key: `${i}:${String(g.id ?? '')}`,
    what: givenLabel(g),
    value: g.value,
    unit: String(g.unit ?? ''),
    source: givenSource(g),
  }))
  const steps = sheet.plan.map((p, i) => {
    const rec = stepOf(sheet, p.id)
    const raw = rec ? String(rec.status ?? '') : 'not_run'
    const status = STEP_STATUSES.includes(raw) ? raw : 'unknown'
    const inputs = Object.entries(isObject(p.inputs) ? p.inputs : {}).map(([n, r]) => inputView(sheet, n, r))
    return {
      key: `${i}:${String(p.id ?? '')}`,
      label: `step ${i + 1}`,
      tool: entryWords(p.entry),
      status,
      inputs,
      result: status === 'done' && rec && isObject(rec.result) ? rec.result : null,
      needs: status === 'waiting_input' && rec && Array.isArray(rec.needs) ? rec.needs.map(words) : [],
    }
  })
  const missing = sheet.missing.map((m) => ({
    for: stepLabel(sheet, m.for),
    label: words(m.label || m.input),
  }))
  const unchecked = givens.filter((g) => !g.source).length
  return { givens, steps, missing, done: sheet.done, unchecked }
}

/** The closing line's words for how the turn ended (trace/1 `ended`). */
export function endingWords(ended) {
  if (ended === 'answer') return 'answered'
  if (ended === 'asked_input') return 'asked you for an input'
  if (ended === 'ceiling') return 'stopped at the round limit'
  return ''
}

/** A number as shown: as sent, without float noise. */
export function showNumber(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return String(v ?? '')
  return String(Number(v.toPrecision(12)))
}
