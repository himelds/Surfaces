"""Typed specifications for test function characteristics and metadata.

``FunctionSpec`` describes mathematical and benchmark-relevant properties.
``MetaSpec`` describes identity, documentation, and display metadata.

Concrete classes may still declare ``_spec`` and ``_meta`` as plain dicts
for compactness. ``BaseTestFunction.__init_subclass__`` resolves those
declarations into frozen dataclass instances once at class-definition time.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Type


@dataclass(frozen=True)
class FunctionSpec:
    """Immutable specification of a test function's mathematical properties.

    Every concrete test function class has exactly one resolved FunctionSpec
    stored as ``cls._spec``. The spec is resolved once at class-definition
    time by ``BaseTestFunction.__init_subclass__``: child classes may define
    ``_spec`` as a plain dict containing only the fields they want to
    override; the remaining fields are inherited from the parent's spec.

    Fields that vary per instance (``n_dim`` on scalable functions,
    ``f_global``/``x_global`` on functions with a known optimum) are declared
    here with a neutral class-level default. Their per-instance value is
    injected by the ``BaseTestFunction.spec`` property, which lifts the
    matching instance attribute via ``dataclasses.replace`` on every access.
    So ``cls._spec`` holds the static template and ``func.spec`` holds the
    instance-resolved view.

    Display/identity attributes (name, latex_formula, reference) belong
    to :class:`MetaSpec` and are exposed through ``func.meta``.

    Deferred cleanup (see project notes): ``func_id`` is currently kept here
    but is catalogue identity and should move to :class:`MetaSpec`; the
    top-level ``func.f_global``/``func.x_global`` attributes still exist as
    the declaration site and should eventually become namespace-only.
    """

    n_dim: Optional[int] = None
    n_objectives: int = 1

    default_bounds: Optional[Tuple[float, float]] = (-5.0, 5.0)

    func_id: Optional[int] = None

    continuous: Optional[bool] = True
    differentiable: Optional[bool] = True
    convex: Optional[bool] = False
    separable: Optional[bool] = False
    unimodal: Optional[bool] = False
    scalable: bool = False
    discrete: bool = False

    constrained: bool = False
    stochastic: bool = False
    simulation_based: bool = False
    expensive: bool = False
    ode_based: bool = False
    deceptive: bool = False
    multimodal: bool = False

    convex_front: Optional[bool] = None
    concave_front: Optional[bool] = None
    disconnected_front: Optional[bool] = None

    eval_cost: Optional[float] = None

    deprecated: bool = False

    # Global optimum. Per-instance for some functions (e.g. BBOB), so the
    # class-level default is None and the real value is lifted from the
    # instance by the ``spec`` property. ``x_global`` is excluded from
    # equality because an ndarray optimum cannot be compared with ``==``.
    f_global: Optional[float] = None
    x_global: Any = dataclasses.field(default=None, compare=False)


@dataclass(frozen=True)
class MetaSpec:
    """Immutable display and documentation metadata for a test function.

    Metadata is intentionally separate from ``FunctionSpec``. A function's
    name, formula string, source reference, and visualization hints are not
    mathematical properties of the optimization problem, but they are still
    important enough to be resolved and validated through a typed schema.
    """

    name: Optional[str] = None
    slug: Optional[str] = None

    tagline: Optional[str] = None
    reference: Optional[str] = None
    reference_url: Optional[str] = None

    latex_formula: Optional[str] = None
    pgfmath_formula: Optional[str] = None

    display_bounds: Any = None
    display_projection: Optional[dict[str, Any]] = None

    category: Optional[str] = None
    domain: Optional[str] = None


_SPEC_FIELD_NAMES = frozenset(FunctionSpec.__dataclass_fields__)

_META_FIELD_NAMES = frozenset(MetaSpec.__dataclass_fields__)

_LEGACY_META_ALIASES = {
    "_name_": "slug",
}

_CLASS_META_ATTRS = {
    "name": "name",
    "_name_": "slug",
    "tagline": "tagline",
    "reference": "reference",
    "reference_url": "reference_url",
    "latex_formula": "latex_formula",
    "pgfmath_formula": "pgfmath_formula",
    "display_bounds": "display_bounds",
    "display_projection": "display_projection",
    "category": "category",
    "domain": "domain",
}


def _as_plain_dict(raw: Any, attr_name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, (FunctionSpec, MetaSpec)):
        return dataclasses.asdict(raw)
    if isinstance(raw, Mapping):
        return dict(raw)
    raise TypeError(
        f"{attr_name} must be a dict, FunctionSpec, or MetaSpec; got {type(raw).__name__}"
    )


def _split_legacy_spec(raw_spec: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split old ``_spec`` declarations into function and metadata fields."""
    spec_values: dict[str, Any] = {}
    meta_values: dict[str, Any] = {}
    unknown: list[str] = []

    for key, value in _as_plain_dict(raw_spec, "_spec").items():
        normalized = _LEGACY_META_ALIASES.get(key, key)
        if key in _SPEC_FIELD_NAMES:
            spec_values[key] = value
        elif normalized in _META_FIELD_NAMES:
            meta_values[normalized] = value
        else:
            unknown.append(key)

    if unknown:
        keys = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown _spec key(s): {keys}")

    return spec_values, meta_values


def _parse_meta(raw_meta: Any) -> dict[str, Any]:
    meta_values: dict[str, Any] = {}
    unknown: list[str] = []

    for key, value in _as_plain_dict(raw_meta, "_meta").items():
        normalized = _LEGACY_META_ALIASES.get(key, key)
        if normalized in _META_FIELD_NAMES:
            meta_values[normalized] = value
        else:
            unknown.append(key)

    if unknown:
        keys = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown _meta key(s): {keys}")

    return meta_values


def resolve_parent_spec(cls: Type) -> FunctionSpec:
    """Walk MRO to find the nearest ancestor with a resolved FunctionSpec."""
    for parent in cls.__mro__[1:]:
        val = parent.__dict__.get("_spec")
        if isinstance(val, FunctionSpec):
            return val
    return FunctionSpec()


def resolve_parent_meta(cls: Type) -> MetaSpec:
    """Walk MRO to find the nearest ancestor with a resolved MetaSpec."""
    for parent in cls.__mro__[1:]:
        val = parent.__dict__.get("_meta")
        if isinstance(val, MetaSpec):
            return val
    return MetaSpec()


def resolve_function_spec(cls: Type, raw_spec: Any = None) -> FunctionSpec:
    """Resolve a class's function spec from its parent and local declaration."""
    parent_spec = resolve_parent_spec(cls)
    spec_values, _ = _split_legacy_spec(raw_spec)
    return dataclasses.replace(parent_spec, **spec_values)


def resolve_meta_spec(cls: Type, raw_spec: Any = None, raw_meta: Any = None) -> MetaSpec:
    """Resolve a class's metadata spec from parent, legacy spec, and attributes."""
    parent_meta = resolve_parent_meta(cls)
    _, meta_from_spec = _split_legacy_spec(raw_spec)
    meta_from_meta = _parse_meta(raw_meta)

    class_values = {
        field: cls.__dict__[attr]
        for attr, field in _CLASS_META_ATTRS.items()
        if attr in cls.__dict__
    }

    values = {
        **dataclasses.asdict(parent_meta),
        **meta_from_spec,
        **meta_from_meta,
        **class_values,
    }
    return MetaSpec(**values)
