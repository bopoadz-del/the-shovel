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
