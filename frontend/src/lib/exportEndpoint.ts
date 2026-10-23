/** Stamp the active UI project onto a server export URL.

 * Chat remaps master_corpus → drive_archive for RAG. A download offer that
 * keeps that backing id 404s (`Project 'drive_archive' not found`). The
 * workspace URL is the project the user is in.
 */
export function exportEndpointForWorkspace(
  endpoint: string,
  workspaceProjectId: string | undefined,
): string {
  if (!workspaceProjectId || !endpoint) return endpoint
  return endpoint.replace(
    /^(\/v1\/projects\/)[^/?#]+/,
    `$1${workspaceProjectId}`,
  )
}
