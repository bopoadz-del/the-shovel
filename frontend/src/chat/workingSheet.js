// The working sheet of a catalogue-path turn, read for display.
//
// The backend writes the sheet (app/agents/driver/sheet.py) and sends the
// final one on the SSE 'end' event with the turn's figure provenance. This
// module only READS it: one view for any sheet, whatever the formulas, the
// project or the number of steps. It never changes the answer text.
//
// Who wrote what, and so what the view may claim:
//   * steps[].result / status / error / needs -- code (run_step), shown as is;
//   * givens -- the model; a given's own `source` field is the model's label,
//     so it is NEVER shown as a checked source. The source shown is code's
//     match of the given's value against the turn's provenance entries; with
//     no match the given shows that no source was found;
//   * plan inputs -- references the model wrote (g1 / s1); the value shown is
//     resolved here from the sheet, the same way resolve_inputs does.

const GIVEN_REF = /^g\d+$/
const STEP_REF = /^s\d+$/

function isObject(v) {
  return v !== null && typeof v === 'object' && !Array.isArray(v)
}

function list(v) {
  return Array.isArray(v) ? v.filter(isObject) : []
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

/** The leading number of a provenance figure ("1,250.5 m3" -> 1250.5). */
export function figureNumber(figure) {
  const m = String(figure ?? '').replace(/,/g, '').match(/-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?/)
  return m ? Number(m[0]) : null
}

function sameNumber(a, b) {
  if (typeof a !== 'number' || typeof b !== 'number' || !Number.isFinite(a) || !Number.isFinite(b)) return false
  return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b))
}

/** A plain word for a provenance source kind. Unknown kinds are shown raw so
 *  a backend change is visible; an empty or unresolved source is no source. */
export function sourceWord(source) {
  const s = String(source ?? '')
  if (!s || s === 'unresolved') return ''
  if (s === 'user_input' || s === 'user_text') return 'your words'
  if (s === 'project_document' || s === 'project_data' || s.startsWith('document:')) return 'project document'
  if (s === 'general_knowledge') return 'general knowledge'
  if (s === 'calculator') return 'calculator'
  return s
}

/** Code's source for a given: the first provenance entry whose figure is the
 *  same number and whose source is a real one. '' when none matches. */
export function givenSource(given, provenance) {
  for (const entry of list(provenance)) {
    const word = sourceWord(entry.source)
    if (word && sameNumber(figureNumber(entry.figure), given.value)) {
      const where = [entry.doc_name, entry.page_or_section].filter(Boolean).join(', ')
      return where ? `${word} (${where})` : word
    }
  }
  return ''
}

/** A catalogue entry as words: "formula:pile_volume" -> "pile volume". */
export function entryWords(entry) {
  const s = String(entry ?? '')
  const name = s.includes(':') ? s.slice(s.indexOf(':') + 1) : s
  return name.replace(/_/g, ' ').trim()
}

function stepOf(sheet, id) {
  return sheet.steps.find((s) => s.id === id) ?? null
}

/** One plan input as shown: where its value comes from and the value. */
function inputView(sheet, name, ref) {
  const label = String(name).replace(/_/g, ' ')
  if (typeof ref === 'string' && GIVEN_REF.test(ref)) {
    const g = sheet.givens.find((x) => x.id === ref)
    return g ? { name: label, ref, value: g.value, unit: g.unit ?? '' } : { name: label, ref, missing: true }
  }
  if (typeof ref === 'string' && STEP_REF.test(ref)) {
    const s = stepOf(sheet, ref)
    if (s && s.status === 'done' && isObject(s.result)) {
      return { name: label, ref, value: s.result.value, unit: s.result.unit ?? '' }
    }
    return { name: label, ref, missing: true }
  }
  return { name: label, ref: '', value: ref, unit: '' }
}

/** The whole view: givens with code's source, plan steps in plan order with
 *  their inputs, status and result, and the inputs still missing. */
export function workingView(rawSheet, provenance) {
  const sheet = readSheet(rawSheet)
  if (!sheet) return null
  const givens = sheet.givens.map((g) => ({
    id: String(g.id ?? ''),
    what: String(g.what ?? ''),
    value: g.value,
    unit: String(g.unit ?? ''),
    source: givenSource(g, provenance),
  }))
  const steps = sheet.plan.map((p) => {
    const rec = stepOf(sheet, p.id)
    const status = rec ? String(rec.status ?? '') : 'not_run'
    const inputs = Object.entries(isObject(p.inputs) ? p.inputs : {}).map(([n, r]) => inputView(sheet, n, r))
    return {
      id: String(p.id ?? ''),
      tool: entryWords(p.entry),
      status,
      inputs,
      result: status === 'done' && rec && isObject(rec.result) ? rec.result : null,
      error: status === 'failed' && rec ? String(rec.error ?? '') : '',
      needs: status === 'waiting_input' && rec && Array.isArray(rec.needs) ? rec.needs.map(String) : [],
    }
  })
  const missing = sheet.missing.map((m) => ({
    for: String(m.for ?? ''),
    label: String(m.label || m.input || ''),
  }))
  const unsourced = givens.filter((g) => !g.source).length
  return { givens, steps, missing, done: sheet.done, unsourced }
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
