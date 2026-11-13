import type { Project } from './ProjectCard'

// Split the listing into the user's own projects and the server-flagged
// system knowledge layers (`is_system`), which render apart and read-only.
export function partitionProjects(projects: Project[]): {
  userProjects: Project[]
  layers: Project[]
} {
  return {
    userProjects: projects.filter((p) => !p.is_system),
    layers: projects.filter((p) => p.is_system),
  }
}
