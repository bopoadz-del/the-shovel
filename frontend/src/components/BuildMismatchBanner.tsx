import { useSyncExternalStore } from 'react'
import { MISMATCH_MESSAGE } from '../lib/buildCheck.js'
import { hasBuildMismatch, subscribeBuildMismatch } from '../lib/buildWatch'

/** Says plainly, once the page and the server name different builds, that the
 *  page must be reloaded. It does not reload by itself: a running answer
 *  stream or a half-typed question is the user's to keep or drop. */
export default function BuildMismatchBanner() {
  const show = useSyncExternalStore(subscribeBuildMismatch, hasBuildMismatch)
  if (!show) return null
  return (
    <div
      role="alert"
      style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 10000,
        display: 'flex', gap: 12, alignItems: 'center', justifyContent: 'center',
        padding: '8px 16px', background: '#7a4b00', color: '#fff', fontSize: 14,
      }}
    >
      <span>{MISMATCH_MESSAGE}</span>
      <button type="button" onClick={() => window.location.reload()}>
        Reload
      </button>
    </div>
  )
}
