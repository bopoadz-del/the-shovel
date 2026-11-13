// How the composer names and routes an upload (S-U2).
//
// A phone's share sheet hands the browser a file named after its sender, in
// any script, with emoji and often no extension. The multipart filename is
// sent as an ASCII-safe stand-in and the user's own name travels in the
// `display_name` form field; the server types the file from its bytes.

const IMAGE_EXT_RE = /\.(jpe?g|png|webp|tiff?|bmp|gif)$/i

// Leading bytes of the image formats the photo route reads.
const IMAGE_SIGNATURES = [
  [0xff, 0xd8, 0xff], // JPEG
  [0x89, 0x50, 0x4e, 0x47], // PNG
  [0x47, 0x49, 0x46, 0x38], // GIF8
  [0x49, 0x49, 0x2a, 0x00], // TIFF little-endian
  [0x4d, 0x4d, 0x00, 0x2a], // TIFF big-endian
]

function startsWith(bytes, sig, at = 0) {
  if (bytes.length < at + sig.length) return false
  return sig.every((b, i) => bytes[at + i] === b)
}

/** True when the leading bytes are an image the photo route reads. */
export function bytesAreImage(bytes) {
  if (IMAGE_SIGNATURES.some((sig) => startsWith(bytes, sig))) return true
  // RIFF....WEBP
  return startsWith(bytes, [0x52, 0x49, 0x46, 0x46]) && startsWith(bytes, [0x57, 0x45, 0x42, 0x50], 8)
}

/** The upload is a photo: by its name, its declared type, or its bytes. */
export function isImageUpload(name, mimeType, bytes) {
  return IMAGE_EXT_RE.test(name || '') || /^image\//i.test(mimeType || '') || bytesAreImage(bytes)
}

/** An ASCII-only stand-in for `name`, keeping an ASCII extension. */
export function asciiUploadName(name) {
  const raw = String(name || '')
  const dot = raw.lastIndexOf('.')
  const ext = dot > 0 && /^\.[A-Za-z0-9]{1,8}$/.test(raw.slice(dot)) ? raw.slice(dot) : ''
  const stem = (ext ? raw.slice(0, dot) : raw)
    .normalize('NFKD')
    .replace(/[^A-Za-z0-9._-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^[_.]+|[_.]+$/g, '')
  return (stem || 'upload') + ext
}

/** What to tell the user when the browser could not read the file at all. */
export function unreadableFileMessage(name) {
  return (
    `The browser could not read ${name}: the app that shared it did not ` +
    `give access to its contents. Save it to the device first, then attach ` +
    `it from Files.`
  )
}
