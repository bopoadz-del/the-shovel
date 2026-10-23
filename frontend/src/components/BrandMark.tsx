/* BrandMark — the shovel logo from the Quarry standalone design
 * (2026-06-21). One component so the header tile and the login card can
 * never drift apart again. Colors come from the enclosing tile's
 * `color`; size is the SVG box the caller needs. */

interface Props {
  size?: number
}

export default function BrandMark({ size = 19 }: Props) {
  return (
    <svg viewBox="0 0 32 32" width={size} height={size} aria-hidden="true">
      <g transform="rotate(-30 16 16)">
        <ellipse cx="16" cy="6" rx="4" ry="3" fill="none" stroke="currentColor" strokeWidth="2.2" />
        <rect x="14.7" y="8" width="2.6" height="12" rx="0.4" fill="currentColor" />
        <path d="M10.2 19.5 Q9.2 24.6 16 28 Q22.8 24.6 21.8 19.5 Z" fill="currentColor" />
      </g>
    </svg>
  )
}
