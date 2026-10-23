/* Visible source-class labels at the glass. Must stay aligned with
 * ``SOURCE_CLASS_LABELS`` in app/core/rag/source_class.py — the Python
 * glass tests assert both maps list the same four wordings.
 *
 * Do not re-classify here. The backend already tagged the chunk (#468);
 * this file only turns the machine class into the operator-facing label.
 * An unseen class is read as words (underscores become spaces). The
 * machine id is never the label.
 */
import { textLeaksInternalCode, wordsForIdentifier } from './userFacingText'

export const SOURCE_CLASS_LABELS: Record<string, string> = {
  project_corpus: 'this contract',
  master_corpus: 'master corpus',
  knowledge_base: 'knowledge base',
  template: 'template',
}

export function sourceClassLabel(sourceClass?: string | null): string {
  const key = (sourceClass || 'project_corpus').trim().toLowerCase()
  const mapped = SOURCE_CLASS_LABELS[key]
  if (mapped) return mapped
  return wordsForIdentifier(key) || SOURCE_CLASS_LABELS.project_corpus
}

/** The words on the Sources card. A label that is itself an id is not shown. */
export function visibleSourceClassLabel(
  sourceClass?: string | null,
  provided?: string | null,
): string {
  const given = (provided || '').trim()
  if (given && !textLeaksInternalCode(given)) return given
  return sourceClassLabel(sourceClass)
}
