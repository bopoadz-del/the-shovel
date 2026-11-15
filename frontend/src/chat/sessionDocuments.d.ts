export interface SessionDocumentRow {
  id: string
  name: string
  tag: 'Uploaded' | 'Produced'
}

export function withSessionId(form: FormData, sessionId: string | null | undefined): FormData
export function sessionDocumentTag(origin: unknown): 'Uploaded' | 'Produced'
export function sessionDocumentsFor(
  fetchedId: string | null | undefined,
  activeId: string | null | undefined,
  payload: unknown,
): SessionDocumentRow[] | null
