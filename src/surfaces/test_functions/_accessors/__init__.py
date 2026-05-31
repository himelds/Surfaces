"""Accessor classes for BaseTestFunction.

``spec`` and ``meta`` are intentionally NOT accessors: ``func.spec`` and
``func.meta`` return the resolved ``FunctionSpec`` / ``MetaSpec`` dataclasses
directly (single source of truth). The accessors below wrap genuinely
behavioral, mutable instance state.
"""

from ._callbacks import CallbackAccessor
from ._data import DataAccessor
from ._errors import ErrorAccessor
from ._memory import MemoryAccessor
from ._modifiers import ModifierAccessor

__all__ = [
    "CallbackAccessor",
    "DataAccessor",
    "ErrorAccessor",
    "MemoryAccessor",
    "ModifierAccessor",
]
