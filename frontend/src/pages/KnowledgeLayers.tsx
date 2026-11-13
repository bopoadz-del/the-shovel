import { Link } from 'react-router-dom'
import type { Project } from './ProjectCard'

// System knowledge layers (the server flags them `is_system`: configured
// general-knowledge corpora and the master corpus) are not user projects.
// They render as one read-only row: name and indexed chunk count, a link to
// open the layer, and no Delete or other project controls. The server
// refuses deletes of these ids as well.
function chunkLabel(count: number | null | undefined): string {
  if (typeof count !== 'number') return 'chunk count unavailable'
  return `${count.toLocaleString('en-US')} ${count === 1 ? 'chunk' : 'chunks'}`
}

export default function KnowledgeLayers({ layers }: { layers: Project[] }) {
  if (layers.length === 0) return null
  return (
    <section className="knowledge-layers" aria-labelledby="knowledge-layers-title">
      <h2 id="knowledge-layers-title" className="knowledge-layers__title">
        Knowledge layers
      </h2>
      <ul className="knowledge-layers__list">
        {layers.map((layer) => (
          <li key={layer.id} className="knowledge-layers__item" data-layer-id={layer.id}>
            <Link to={`/projects/${layer.id}`} className="knowledge-layers__name">
              {layer.name}
            </Link>
            <span className="knowledge-layers__count">{chunkLabel(layer.chunk_count)}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}
