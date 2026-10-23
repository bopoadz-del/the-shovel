// node --test frontend/src/lib/streamOutcome.test.mjs   (Node 20+, no dependencies)
import assert from 'node:assert/strict'
import { test } from 'node:test'

import { INTERRUPTED_MESSAGE, isInterruptedStream } from './streamOutcome.js'

test('a stream that closed with no end event and no text is interrupted', () => {
  assert.equal(isInterruptedStream(false, ''), true)
  assert.equal(isInterruptedStream(false, '   \n'), true)
})

test('a finished stream, or one that delivered text, is not', () => {
  assert.equal(isInterruptedStream(true, ''), false)
  assert.equal(isInterruptedStream(false, 'Part of an answer'), false)
})

test('the message tells the user what to do', () => {
  assert.match(INTERRUPTED_MESSAGE, /send your question again/)
})
