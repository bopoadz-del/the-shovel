"""Containers — domain kit host (virgin Fork ships base only).

``ConstructionContainer`` is not imported at package load. It registers when
the construction kit is enabled (``CEREBRUM_DOMAIN_KITS=construction`` or
Block Store install).
"""

from .base import DomainContainer

__all__ = ["DomainContainer"]


def __getattr__(name: str):
    if name == "ConstructionContainer":
        from .construction import ConstructionContainer

        return ConstructionContainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
