/* WorkspaceShell — 3-column flex container for the workspace page.
 *
 * Quarry design 2026-06-21 — 250 left + flexible center + 360 right.
 * The shell exposes a ``rightExpanded`` prop that flips a data
 * attribute on the root; the WorkspaceShell.css promotes the right
 * panel to a full-width overlay covering main + left when set. The
 * RightPanel's expand button is the typical trigger but any caller
 * can drive the state.
 *
 * Mobile (<768px, spec 2026-07-24-mobile-ui-design):
 *   - the left panel is an off-canvas drawer; the shell owns its open
 *     state and stamps data-left-open, CSS slides it in over a backdrop
 *   - the right panel is hidden unless rightExpanded (full overlay);
 *     ``onRightOpen`` lets the page expand it from the mobile toolbar
 *   - the toolbar buttons and backdrop only exist visually inside the
 *     breakpoint — desktop layout is untouched
 */
import { useState, type ReactNode } from 'react'
import { Menu, PanelRight } from 'lucide-react'
import './WorkspaceShell.css'

interface Props {
  left: ReactNode
  main: ReactNode
  right: ReactNode
  /** Top bar — AppHeader with theme toggle slot. */
  header: ReactNode
  /** When true, the right panel covers the full body area as an overlay. */
  rightExpanded?: boolean
  /** Mobile toolbar's "Panel" button — expand the right panel overlay. */
  onRightOpen?: () => void
}

export default function WorkspaceShell({
  header, left, main, right, rightExpanded = false, onRightOpen,
}: Props) {
  const [leftOpen, setLeftOpen] = useState(false)

  return (
    <div
      className="workspace-shell"
      data-right-expanded={rightExpanded ? 'true' : 'false'}
      data-left-open={leftOpen ? 'true' : 'false'}
    >
      <div className="workspace-shell__header">{header}</div>
      <div className="workspace-shell__body">
        <div className="workspace-shell__mobilebar" role="toolbar" aria-label="Panels">
          <button
            type="button"
            className="workspace-shell__mobilebtn"
            onClick={() => setLeftOpen(true)}
            aria-label="Open project menu"
          >
            <Menu size={18} />
          </button>
          {onRightOpen && (
            <button
              type="button"
              className="workspace-shell__mobilebtn"
              onClick={onRightOpen}
              aria-label="Open sources panel"
            >
              <PanelRight size={18} />
            </button>
          )}
        </div>
        {leftOpen && (
          <div
            className="workspace-shell__backdrop"
            onClick={() => setLeftOpen(false)}
            aria-hidden="true"
          />
        )}
        <aside
          className="workspace-shell__left"
          aria-label="Project navigation"
          onClickCapture={(e) => {
            // Mobile drawer UX: choosing anything (project link, conversation,
            // document action) closes the drawer. No-op on desktop — leftOpen
            // only has a visual effect inside the mobile breakpoint.
            const el = e.target as HTMLElement
            if (leftOpen && el.closest('a, button')) setLeftOpen(false)
          }}
        >{left}</aside>
        <main className="workspace-shell__main">{main}</main>
        <aside className="workspace-shell__right" aria-label="Sources and documents">{right}</aside>
      </div>
    </div>
  )
}
