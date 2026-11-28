/* Shared chat types — kept in their own file so chat/ components don't
 * have to round-trip through pages/ProjectWorkspace for the same shapes. */

export type MessageRole = 'user' | 'assistant'

export interface CitedSource {
  doc_id: string
  doc_name: string
  page_or_section: string
  score: number
  confidence: 'High' | 'Medium' | 'Low'
  /** Owning project of the cited file (workspace, Master Corpus, or GK). */
  project_id?: string
  layer?: string
  layer_label?: string
  /** #468 machine class — project_corpus | knowledge_base | template | master_corpus */
  source_class?: string
  /** Visible wording: this contract / knowledge base / template / master corpus */
  source_class_label?: string
}

/** A data-backed download offer the platform attaches to an answer that cites
 *  an exportable document (e.g. a priced BOQ → a formula-linked cost BOQ). The
 *  bubble renders one link per descriptor under the answer; clicking POSTs to
 *  `endpoint` with `payload` and downloads the returned file. No extra UI. */
export interface ExportDescriptor {
  label: string
  format: string
  method: string
  endpoint: string
  payload: Record<string, unknown>
}

/** One cited passage behind a numbered marker in the answer text (the answer
 *  contract, ``app/agents/answer_contract.py``). The text carries only the
 *  marker; document, section, passage and tier live in the Sources panel. */
export interface AnswerCitation {
  marker: number
  document: string
  section?: string
  passage?: string
  /** Visible tier wording copied from the turn's own evidence, or null. */
  tier?: string | null
  /** ``document`` or ``calculation`` (a calculation's passage is its inputs). */
  kind?: string
}

/** How much of the project was searchable for this answer. */
export interface AnswerCoverage {
  indexed: number
  total: number
}

export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  streaming?: boolean
  error?: boolean
  /** Transient tool-activity label shown while the agent is calling tools. */
  toolStatus?: string
  /** Top retrieved sources, populated by the SSE 'end' event when RAG fired. */
  sources?: CitedSource[]
  /** Data-backed download offers from the SSE 'end' event (e.g. cost BOQ). */
  exports?: ExportDescriptor[]
  /** Marker entries from the SSE 'end' event; shown in the Sources panel. */
  citations?: AnswerCitation[]
  /** Coverage from the SSE 'end' event; a Sources panel line, never answer text. */
  coverage?: AnswerCoverage
  /** The answer contract's rule 5: supporting text moved out of the lead,
   *  word for word. Shown under the answer, folded, never deleted. */
  details?: string
  /** The final working sheet of a catalogue-path turn (SSE 'end'); shown
   *  folded under the answer by ShowWorking, never merged into its text. */
  sheet?: unknown
  /** trace/1 ending of a catalogue-path turn: answer | asked_input | ceiling. */
  ended?: string
}
