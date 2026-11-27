export interface SheetGivenView { key: string; what: string; value: unknown; unit: string; source: string }
export interface SheetInputView { name: string; from: string; value?: unknown; unit?: string; missing?: boolean }
export interface SheetStepView {
  key: string
  label: string
  tool: string
  status: string
  inputs: SheetInputView[]
  result: { name?: string; value?: unknown; unit?: string } | null
  needs: string[]
}
export interface SheetView {
  givens: SheetGivenView[]
  steps: SheetStepView[]
  missing: Array<{ for: string; label: string }>
  done: boolean
  unchecked: number
}
export declare const STEP_STATUSES: string[]
export declare function readSheet(raw: unknown): Record<string, unknown> | null
export declare function checkedWord(checked: unknown): string
export declare function givenSource(given: unknown): string
export declare function entryWords(entry: unknown): string
export declare function workingView(sheet: unknown): SheetView | null
export declare function endingWords(ended: unknown): string
export declare function showNumber(v: unknown): string
export declare function unitWords(unit: unknown): string
export declare function quantityText(value: unknown, unit: unknown): string
export declare function literalQuantity(ref: unknown): { value: unknown; unit: string }
