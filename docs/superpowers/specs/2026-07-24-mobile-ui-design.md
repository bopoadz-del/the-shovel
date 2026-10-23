# Mobile UI — responsive workspace (2026-07-24)

## Goal
Make the existing React app usable on phones. One codebase, one breakpoint
(<= 768px), no new dependencies, no separate mobile routes. Desktop layout is
untouched above the breakpoint.

## Current state
The 3-column WorkspaceShell (240px left / flexible main / 360px right)
"collapses" both side panels to 56px strips on mobile — full panel content
crushed into an unusable sliver. Chat bubbles cap at 78% width, the composer
font is below 16px (iOS zooms on focus), and the header brand + breadcrumb
overflow small screens. The projects grid is already responsive
(auto-fill minmax) and the viewport meta is correct.

## Approaches considered
1. Responsive CSS + drawer behavior in the existing shell (CHOSEN) — smallest
   diff, single source of truth, follows the existing data-attribute pattern
   (`data-right-expanded`).
2. Separate mobile components/routes — permanent duplication, drift risk.
3. PWA/native wrapper — out of pilot scope; nothing here precludes it later.

## Design (breakpoint: max-width 768px)

### WorkspaceShell
- Left panel becomes an off-canvas drawer: `position: absolute`, width
  `min(85vw, 320px)`, hidden via `translateX(-100%)`, slides in when the
  shell stamps `data-left-open="true"`. A tap on the backdrop closes it.
- Shell owns the drawer state internally (`useState`) — no prop drilling;
  desktop ignores it entirely (CSS scopes all of it under the breakpoint).
- Two mobile-only buttons render inside the shell body (hidden on desktop):
  "Menu" (opens the left drawer) and "Panel" (calls a new optional
  `onRightOpen` prop so the page can expand the right panel).
- Right panel is `display: none` on mobile EXCEPT in the existing
  `data-right-expanded="true"` overlay mode, which already covers the full
  body and has its own collapse button. `onRightOpen` is the mobile way in.
- The broken 56px-rail rule is deleted.

### Chat
- Bubbles: `max-width: 92%` on mobile (78% desktop unchanged).
- Markdown tables keep their existing `overflow-x: auto` wrapper (verified
  present) — no page-level horizontal scroll.
- Composer: textarea `font-size: 16px` on mobile (prevents iOS focus zoom),
  container gets `padding-bottom: env(safe-area-inset-bottom)`.
- Bubble action buttons (Word / Excel / export offers) get taller touch
  targets on mobile via padding.

### Header
- `brand-name` hides below 480px; breadcrumb segments truncate with
  ellipsis so the row never wraps.

### Out of scope
Arabic/RTL (separate deferred work item), PWA manifest/offline, per-page
mobile redesigns beyond the above (Login and Projects already stack).

## Testing
- `tsc --noEmit` + Vite production build must pass (CI docker build compiles
  the frontend).
- Drawer state logic is CSS + a two-state toggle; behavior verified by
  building and driving the app at a mobile viewport (Playwright browser at
  390x844) as a manual acceptance pass.
