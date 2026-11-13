// node --test frontend/src/chat/uploadName.test.mjs
// S-U2: a share-sheet document (non-ASCII name, emoji, no extension) is
// sent with an ASCII stand-in name and routed by its bytes.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { test } from 'node:test'

import { asciiUploadName, bytesAreImage, isImageUpload, unreadableFileMessage } from './uploadName.js'

const JPEG = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 0x10])
const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
const WEBP = new Uint8Array([0x52, 0x49, 0x46, 0x46, 1, 0, 0, 0, 0x57, 0x45, 0x42, 0x50])
const PDF = new TextEncoder().encode('%PDF-1.4')
const TEXT = new TextEncoder().encode('item,qty')

for (const [label, bytes, expected] of [
  ['jpeg', JPEG, true],
  ['png', PNG, true],
  ['webp', WEBP, true],
  ['pdf', PDF, false],
  ['text', TEXT, false],
  ['empty', new Uint8Array(), false],
]) {
  test(`bytesAreImage reads the bytes (${label})`, () => {
    assert.equal(bytesAreImage(bytes), expected)
  })
}

for (const [name, mime, bytes, expected] of [
  ['Document from شادي Chadi 🇸🇦', '', JPEG, true],
  ['Document from شادي Chadi 🇸🇦', '', PDF, false],
  ['site.JPG', '', TEXT, true],
  ['shared', 'image/heic', TEXT, true],
  ['minutes.pdf', 'application/pdf', PDF, false],
]) {
  test(`isImageUpload routes by name, type or bytes (${name}, ${mime || 'no type'})`, () => {
    assert.equal(isImageUpload(name, mime, bytes), expected)
  })
}

for (const [name, expected] of [
  ['Document from شادي Chadi 🇸🇦', 'Document_from_Chadi'],
  ['تقرير.pdf', 'upload.pdf'],
  ['Site Report v1.2.docx', 'Site_Report_v1.2.docx'],
  ['Café menu.txt', 'Cafe_menu.txt'],
  ['', 'upload'],
]) {
  test(`asciiUploadName is ASCII-safe and keeps the extension (${name || 'empty'})`, () => {
    const out = asciiUploadName(name)
    assert.equal(out, expected)
    assert.match(out, /^[\x20-\x7e]+$/)
  })
}

for (const name of ['Document from شادي', 'plan.pdf']) {
  test(`unreadableFileMessage names the file and the cause (${name})`, () => {
    const msg = unreadableFileMessage(name)
    assert.ok(msg.includes(name))
    assert.match(msg, /could not read/)
  })
}

test('the composer sends the stand-in name and the real name as display_name', () => {
  const src = fs.readFileSync(new URL('./ChatComposer.tsx', import.meta.url), 'utf8')
  assert.ok(src.includes("fd.append('display_name', file.name)"))
  assert.ok(src.includes('asciiUploadName(file.name)'))
  assert.ok(src.includes('isImageUpload(file.name, file.type, head)'))
})
