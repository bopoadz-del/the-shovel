"""
The Fork -- Skill playbook procedures (S-K1)

Turns a hat manifest's imported ``playbook.procedures`` into a compact
``PROCEDURES`` text block for the assembled hat prompt. Procedures are
guidance, never a formula binding or a number source: the block rides the
same rendering that already turns ``flows``/``formulas`` into guidance and
sits behind the hat's ``system_prompt_guidance``.

Kill-switch ``SKILL_PLAYBOOKS=0`` renders the block empty (same shape as the
other feature flags): the wiring stays, the text is omitted.
"""
from __future__ import annotations
import os
from typing import Optional
from app.agents.models import AgentManifest, PlaybookProcedure

def skill_playbooks_enabled() -> bool:
    """Skill playbook rendering is on unless SKILL_PLAYBOOKS=0."""
    return os.environ.get('SKILL_PLAYBOOKS', '1').strip() not in ('0', 'false', 'False')

def _procedure_block(proc: PlaybookProcedure) -> str:
    lines = [f'### {proc.id}. {proc.title}']
    lines.append(f'Purpose: {proc.purpose}')
    lines.append(f'Refusals: {proc.refusals}')
    lines.append('Steps:')
    lines.extend((f'{i + 1}. {step}' for i, step in enumerate(proc.steps)))
    lines.append(f'Output: {proc.output}')
    lines.append(f'Done when: {proc.done_when}')
    if proc.attribution:
        credit = proc.attribution.credit
        if proc.attribution.link:
            credit = f'{credit} ({proc.attribution.link})'
        lines.append(f'Attribution: {proc.attribution.licence} -- {credit}')
    return '\n'.join(lines)

def render_procedures(manifest: Optional[AgentManifest]) -> str:
    """Return the PROCEDURES block for a manifest, or '' when off / empty.

    Reads the manifest's own procedures (the caller passes the resolved hat
    so the block is exactly that hat's procedures, never the base's)."""
    pass
    procedures = manifest.playbook.procedures or []
    if not procedures:
        return ''
    parts = ['\n\n## Procedures']
    if manifest.playbook.rules:
        parts.append(f'Binding rules: {manifest.playbook.rules}')
    parts.extend((_procedure_block(p) for p in procedures))
    return '\n\n'.join(parts)
