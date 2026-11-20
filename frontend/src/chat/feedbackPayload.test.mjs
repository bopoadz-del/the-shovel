// node --test src/chat/feedbackPayload.test.mjs
import assert from 'node:assert/strict'
import { test } from 'node:test'

import { buildFeedbackBody, echoMatches } from './feedbackPayload.js'

const base = { projectId: 'p-1', conversationId: 'ws-p-1-1', assistantIndex: 2 }

test('each kind sends only the fields it accepts', () => {
  const fig = buildFeedbackBody({ ...base, kind: 'wrong_figure', value: ' 300 ', unit: 'mm', note: 'x', content: 'Slab  is\n250 mm' })
  assert.deepEqual(fig.body, {
    project_id: 'p-1', conversation_id: 'ws-p-1-1', message_index: 2,
    kind: 'wrong_figure', answer_head: 'Slab is 250 mm', correct_value: 300, unit: 'mm',
  })
  const miss = buildFeedbackBody({ ...base, kind: 'missing', value: '3', unit: 'm', note: ' the DNP ', content: 'Rate' })
  assert.deepEqual(miss.body, {
    project_id: 'p-1', conversation_id: 'ws-p-1-1', message_index: 2,
    kind: 'missing', answer_head: 'Rate', note: 'the DNP',
  })
})

test('bad inputs are refused before sending', () => {
  assert.match(buildFeedbackBody({ ...base, kind: 'wrong_figure', value: 'abc', content: 'a' }).error, /number/)
  assert.match(buildFeedbackBody({ ...base, kind: 'wrong_document', note: 'n'.repeat(201), content: 'a' }).error, /200/)
})

test('recorded only when the echo matches the answer shown', () => {
  const shown = 'The slab is 250 mm thick, grade C40.'
  assert.equal(echoMatches({ status: 'recorded', message_id: 'm1', answer_head: 'The slab is 250 mm' }, shown), true)
  assert.equal(echoMatches({ status: 'recorded', message_id: 'm1', answer_head: 'The rate is SAR' }, shown), false)
  assert.equal(echoMatches({ status: 'recorded', message_id: '', answer_head: 'The slab' }, shown), false)
})
