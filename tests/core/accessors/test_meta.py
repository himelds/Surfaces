"""Tests for ``func.meta``: the resolved MetaSpec dataclass.

``func.meta`` returns the frozen class-level :class:`MetaSpec`. Metadata is
static (no per-instance fields today), so this is the same object the class
resolved at definition time. Attribute access only; no dict protocol.
"""

import dataclasses

import pytest

from surfaces.test_functions._function_spec import MetaSpec
from surfaces.test_functions.algebraic import SphereFunction


class TestMetaType:
    """func.meta is a MetaSpec dataclass."""

    def test_meta_returns_meta_spec(self):
        func = SphereFunction(n_dim=2)
        assert isinstance(func.meta, MetaSpec)

    def test_class_meta_is_meta_spec(self):
        assert isinstance(SphereFunction._meta, MetaSpec)

    def test_meta_is_class_level(self):
        """Metadata is static, so meta is the resolved class MetaSpec."""
        func = SphereFunction(n_dim=2)
        assert func.meta is type(func)._meta


class TestMetaAttributes:
    """Typed attribute access for metadata fields."""

    def test_name(self):
        assert SphereFunction(n_dim=2).meta.name == "Sphere Function"

    def test_slug(self):
        assert SphereFunction(n_dim=2).meta.slug == "sphere_function"

    def test_latex_formula(self):
        meta = SphereFunction(n_dim=2).meta
        assert meta.latex_formula is not None
        assert r"\sum" in meta.latex_formula

    def test_pgfmath_formula(self):
        assert SphereFunction(n_dim=2).meta.pgfmath_formula == "#1^2 + #2^2"

    def test_reference_is_none(self):
        assert SphereFunction(n_dim=2).meta.reference is None

    def test_reference_url(self):
        meta = SphereFunction(n_dim=2).meta
        assert meta.reference_url is not None
        assert "sfu.ca" in meta.reference_url

    def test_tagline(self):
        meta = SphereFunction(n_dim=2).meta
        assert isinstance(meta.tagline, str)

    def test_display_metadata(self):
        meta = SphereFunction(n_dim=2).meta
        assert meta.display_bounds == (-5.0, 5.0)
        assert meta.display_projection == {"fixed_value": 0.0}

    def test_func_id_not_on_meta(self):
        """func_id currently lives on spec, not meta (relocation deferred)."""
        assert not hasattr(SphereFunction(n_dim=2).meta, "func_id")


class TestMetaReadOnly:
    """The resolved meta is a frozen dataclass (FrozenInstanceError is an
    AttributeError subclass)."""

    def test_name_not_settable(self):
        func = SphereFunction(n_dim=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            func.meta.name = "other"

    def test_latex_formula_not_settable(self):
        func = SphereFunction(n_dim=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            func.meta.latex_formula = "other"

    def test_tagline_not_settable(self):
        func = SphereFunction(n_dim=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            func.meta.tagline = "other"


class TestNoDictProtocol:
    """The dict-emulation surface was removed; attribute access only."""

    def test_no_get_method(self):
        assert not hasattr(SphereFunction(n_dim=2).meta, "get")

    def test_not_subscriptable(self):
        with pytest.raises(TypeError):
            SphereFunction(n_dim=2).meta["name"]
