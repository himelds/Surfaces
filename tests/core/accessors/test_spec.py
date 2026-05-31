"""Tests for ``func.spec``: the resolved FunctionSpec dataclass.

``func.spec`` returns a frozen :class:`FunctionSpec` resolved for this
instance (no accessor wrapper). Attribute access is the only read path;
the old dict protocol (get/[]/in/as_dict) was removed.
"""

import dataclasses

import pytest

from surfaces.test_functions._function_spec import FunctionSpec
from surfaces.test_functions.algebraic import SphereFunction


class TestSpecType:
    """func.spec is a FunctionSpec dataclass, not an accessor."""

    def test_spec_returns_function_spec(self):
        func = SphereFunction(n_dim=2)
        assert isinstance(func.spec, FunctionSpec)

    def test_class_spec_is_function_spec(self):
        """Classes store the resolved static spec as FunctionSpec."""
        assert isinstance(SphereFunction._spec, FunctionSpec)

    def test_spec_is_frozen(self):
        """The resolved spec is immutable."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            func.spec.convex = False

    def test_repeated_access_is_equal(self):
        """Repeated access yields equal specs (resolved fresh each time)."""
        func = SphereFunction(n_dim=2)
        params = {key: 0.5 for key in func.search_space.keys()}
        spec1 = func.spec
        _ = func(params)
        assert func.spec == spec1


class TestSpecAttributes:
    """Typed attribute access for static fields."""

    def test_convex(self):
        assert SphereFunction(n_dim=2).spec.convex is True

    def test_unimodal(self):
        assert SphereFunction(n_dim=2).spec.unimodal is True

    def test_separable(self):
        assert SphereFunction(n_dim=2).spec.separable is True

    def test_continuous(self):
        assert SphereFunction(n_dim=2).spec.continuous is True

    def test_differentiable(self):
        assert SphereFunction(n_dim=2).spec.differentiable is True

    def test_scalable(self):
        assert SphereFunction(n_dim=2).spec.scalable is True

    def test_default_bounds(self):
        assert SphereFunction(n_dim=2).spec.default_bounds == (-5.0, 5.0)

    def test_func_id_none_for_algebraic(self):
        """func_id still lives on spec; it is None for non-catalogue functions."""
        assert SphereFunction(n_dim=2).spec.func_id is None


class TestSpecInstanceResolution:
    """Per-instance fields are lifted onto the resolved spec."""

    def test_n_dim_reflects_instance(self):
        assert SphereFunction(n_dim=5).spec.n_dim == 5

    def test_n_objectives(self):
        assert SphereFunction(n_dim=2).spec.n_objectives == 1

    def test_f_global(self):
        assert SphereFunction(n_dim=2).spec.f_global == 0.0

    def test_x_global(self):
        assert SphereFunction(n_dim=3).spec.x_global == (0.0, 0.0, 0.0)


class TestNoDictProtocol:
    """The dict-emulation surface was removed; attribute access only."""

    def test_no_get_method(self):
        assert not hasattr(SphereFunction(n_dim=2).spec, "get")

    def test_not_subscriptable(self):
        with pytest.raises(TypeError):
            SphereFunction(n_dim=2).spec["convex"]


class TestSpecResolution:
    """_spec declarations resolve through the MRO.

    SphereFunction overrides convex/unimodal/separable/scalable;
    AlgebraicFunction defines default_bounds/continuous/differentiable;
    BaseTestFunction defines the boolean defaults.
    """

    def test_child_overrides_parent(self):
        spec = SphereFunction(n_dim=2).spec
        assert spec.convex is True
        assert spec.unimodal is True
        assert spec.separable is True
        assert spec.scalable is True

    def test_intermediate_class_values_preserved(self):
        spec = SphereFunction(n_dim=2).spec
        assert spec.continuous is True
        assert spec.differentiable is True
