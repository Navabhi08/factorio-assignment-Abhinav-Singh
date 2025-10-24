"""Compatibility layer that re-exports the factory solver API.

The implementation now lives in :mod:`factory.main` so that the entire solver is
contained in a single module.  This file simply re-exports the public surface so
existing imports keep working.
"""

from .main import FactoryModel, FactoryResult, Machine, Recipe, solve_from_json

__all__ = [
    "FactoryModel",
    "FactoryResult",
    "Machine",
    "Recipe",
    "solve_from_json",
]
