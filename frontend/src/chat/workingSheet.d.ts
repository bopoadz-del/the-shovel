export interface SheetGivenView { id: string; what: string; value: unknown; unit: string; source: string }
export interface SheetInputView { name: string; ref: string; value?: unknown; unit?: string; missing?: boolean }
export interface SheetStepView {
  id: string
  tool: string
  status: string
  inputs: SheetInputView[]
  result: { name?: string; value?: unknown; unit?: string } | null
  error: string
  needs: string[]
}
export interface SheetView {
  givens: SheetGivenView[]
  steps: SheetStepView[]
  missing: Array<{ for: string; label: string }>
  done: boolean
  unsourced: number
}
export declare function readSheet(raw: unknown): Record<string, unknown> | null
export declare function figureNumber(figure: unknown): number | null
export declare function sourceWord(source: unknown): string
export declare function givenSource(given: { value?: unknown }, provenance: unknown): string
export declare function entryWords(entry: unknown): string
export declare function workingView(sheet: unknown, provenance: unknown): SheetView | null
export declare function endingWords(ended: unknown): string
export declare function showNumber(v: unknown): string
