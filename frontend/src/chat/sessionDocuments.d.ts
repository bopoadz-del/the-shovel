export interface SessionDocumentRow {
  id: string
  name: string
  tag: 'Uploaded' | 'Produced'
}

export function withSessionId(form: FormData, sessionId: string | null | undefined): FormData
export function uploadSessionFor(
  target: 'project' | 'session',
  sessionId: string | null | undefined,
): string | null | undefined
export function belongsToProjectList(doc: unknown): boolean
export function sessionDocumentTag(origin: unknown): 'Uploaded' | 'Produced'
export function documentPath(
  projectId: string,
  docId: string,
  tail: string,
  sessionId?: string | null,
): string

export function sessionDocumentsFor(
  fetchedId: string | null | undefined,
  activeId: string | null | undefined,
  payload: unknown,
): SessionDocumentRow[] | null

export interface SessionDocumentsState {
  sid: string
  status: 'ready' | 'error'
  rows: SessionDocumentRow[]
}

export interface SessionDocumentsView {
  kind: 'loading' | 'error' | 'empty' | 'list'
  rows: SessionDocumentRow[]
}

export function sessionDocumentsLoaded(
  fetchedId: string | null | undefined,
  activeId: string | null | undefined,
  payload: unknown,
): SessionDocumentsState | null
export function sessionDocumentsFailed(
  fetchedId: string | null | undefined,
  activeId: string | null | undefined,
  previous: SessionDocumentsState | null | undefined,
): SessionDocumentsState | null
export function sessionDocumentsView(
  state: SessionDocumentsState | null | undefined,
  activeId: string | null | undefined,
): SessionDocumentsView
export function readSessionDocuments(
  getJson: (path: string) => Promise<unknown>,
  path: string,
  fetchedId: string,
  activeIdNow: () => string | null | undefined,
  previous: SessionDocumentsState | null | undefined,
): Promise<SessionDocumentsState | null>
