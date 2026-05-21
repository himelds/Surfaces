# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Utility functions for the collection module."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from surfaces.test_functions._function_spec import FunctionSpec, MetaSpec


def get_merged_spec(func_cls: Type) -> Dict[str, Any]:
    """Get resolved function spec for a class."""
    spec = getattr(func_cls, "_spec", FunctionSpec())
    if isinstance(spec, FunctionSpec):
        return spec.as_dict()
    return dict(spec)


def get_merged_meta(func_cls: Type) -> Dict[str, Any]:
    """Get resolved metadata for a class."""
    meta = getattr(func_cls, "_meta", MetaSpec())
    if isinstance(meta, MetaSpec):
        return meta.as_dict()
    return dict(meta)


def get_spec_value(func_cls: Type, key: str) -> Any:
    """Get a specific filter value from function spec or metadata."""
    spec = get_merged_spec(func_cls)
    if key in spec:
        return spec[key]
    return get_merged_meta(func_cls).get(key)


def get_n_dim(func_cls: Type) -> Optional[int]:
    """Get n_dim from spec or class attribute."""
    # Check spec first
    spec = get_merged_spec(func_cls)
    if "n_dim" in spec and spec["n_dim"] is not None:
        return spec["n_dim"]

    # Check class attribute (for fixed-dimension functions)
    if hasattr(func_cls, "n_dim") and func_cls.n_dim is not None:
        return func_cls.n_dim

    return None


def get_category(func_cls: Type) -> str:
    """Determine the category of a function class."""
    module = func_cls.__module__

    # Check more specific patterns first
    if ".constrained" in module:
        return "constrained"
    elif ".algebraic" in module:
        return "algebraic"
    elif ".bbob" in module:
        return "bbob"
    elif ".cec" in module:
        return "cec"
    elif ".machine_learning" in module:
        return "ml"
    elif ".simulation" in module:
        return "simulation"
    else:
        return "other"
