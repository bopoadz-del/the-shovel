import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
    },
    rules: {
      // ── CORRECTNESS: stays an ERROR and blocks CI ────────────────────────
      //
      // This rule is why lint is a CI gate at all. On 2026-08-03 a `useMemo`
      // was added BELOW the wsState early returns in ProjectWorkspace.tsx.
      // `tsc -b && vite build` compiled it happily — Rules of Hooks is a
      // runtime contract, not a type error — so it shipped and crashed the
      // whole workspace into the ErrorBoundary with React #310 ("Rendered
      // more hooks than during the previous render"). Every user opening a
      // project saw "Something went wrong."
      //
      // Verified against a minimal repro before wiring CI:
      //   error  React Hook "useMemo" is called conditionally ... Did you
      //          accidentally call a React Hook after an early return?
      //          react-hooks/rules-of-hooks
      //
      // Do NOT downgrade this one.
      'react-hooks/rules-of-hooks': 'error',

      // ── ADVISORY: warnings, so CI can gate on correctness TODAY ──────────
      //
      // Both rules below flag WORKING code. Under react-hooks v7's recommended
      // set they were errors, which left 13 errors on a healthy build and made
      // lint unusable as a gate — which is exactly why it was never wired into
      // CI, and how the crash above reached production. A gate that cannot be
      // switched on protects nothing.
      //
      // Demoted deliberately, with the trade-off stated rather than hidden.
      // They stay visible as warnings and should be worked down.

      // React-Compiler optimisation hint, not a bug. 10 occurrences, all
      // ordinary bootstrap / data-fetch effects (auth bootstrap, project load,
      // document preview). Restructuring auth and workspace effects is a real
      // behavioural change and belongs in its own reviewed PR — not bundled
      // into a CI-enablement change, on a live client system.
      'react-hooks/set-state-in-effect': 'warn',

      // Fast-refresh developer-experience rule: AuthContext.tsx and
      // ThemeContext.tsx export their hook alongside their provider. Splitting
      // them is mechanical but touches 8 import sites for zero runtime gain.
      'react-refresh/only-export-components': 'warn',
    },
  },
])
