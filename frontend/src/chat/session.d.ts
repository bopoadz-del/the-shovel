export declare function legacyConversationId(projectId: string | null | undefined): string | null
export declare function freshSessionEntropy(): string
export declare function mintConversationId(
  projectId: string | null | undefined,
  entropy: string,
): string | null
export declare function openConversationId(projectId: string | null | undefined): string | null
export declare function historyBelongsToSession<T>(
  fetchedId: string | null | undefined,
  activeId: string | null | undefined,
  messages: T,
): T | null
export declare function orderSessionsNewestFirst<T extends { updated_at?: string | null }>(
  rows: T[],
): T[]
export declare function sessionListTitle(title: string | null | undefined): string
export declare function renameTitleError(title: string): string | null
export declare function sessionMemoryKey(projectId: string | null | undefined): string | null
export declare function restorableSessionId(
  projectId: string | null | undefined,
  stored: unknown,
): string | null
export declare function readRememberedSession(
  projectId: string | null | undefined,
  storage: Pick<Storage, 'getItem'> | null | undefined,
): string | null
export declare function rememberSession(
  projectId: string | null | undefined,
  conversationId: string | null | undefined,
  storage: Pick<Storage, 'setItem' | 'removeItem'> | null | undefined,
): void
export declare function afterSessionDeleted<T extends { id: string }>(
  rows: T[],
  deletedId: string,
  activeId: string | null | undefined,
): { rows: T[]; replaceActive: boolean }
export declare function deleteSessionQuestion(title: string | null | undefined): string
