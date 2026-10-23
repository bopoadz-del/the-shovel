"""Cerebrum platform agents — AI assistants that talk to your users and call blocks as tools.

Each agent is declared in `app/agents/configs/<name>.md` (YAML frontmatter + system prompt).
The `AGENT_REGISTRY` dict mirrors `BLOCK_REGISTRY` and is rebuilt at startup from the configs.
"""

from .runtime import Agent, AGENT_REGISTRY, load_agents, get_agent  # noqa: F401

__all__ = ["Agent", "AGENT_REGISTRY", "load_agents", "get_agent"]
