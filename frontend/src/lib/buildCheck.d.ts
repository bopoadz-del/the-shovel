export declare const BUILD_HEADER: string
export declare const MISMATCH_MESSAGE: string
export declare function compareBuilds(
  clientBuild: string | null | undefined,
  serverBuild: string | null | undefined,
): 'same' | 'differs' | 'unknown'
export interface BuildWatcher {
  observe(response: Response): void
  check(serverBuild: string | null | undefined): void
}
export declare function createBuildWatcher(
  clientBuild: string | null | undefined,
  onMismatch: (serverBuild: string) => void,
): BuildWatcher
export declare function installFetchObserver(
  target: { fetch: typeof fetch },
  watcher: BuildWatcher,
): void
export declare function installVisibilityCheck(
  doc: Pick<Document, 'visibilityState' | 'addEventListener' | 'removeEventListener'>,
  target: { fetch: typeof fetch },
  url: string,
  isQuiet: () => boolean,
): () => void
