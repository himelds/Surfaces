"""Tests for func.spec (FunctionSpec) with ML functions (requires sklearn)."""

import dataclasses

from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction


class TestSpecWithMLFunctions:
    """Test the resolved FunctionSpec for machine learning functions."""

    def test_ml_spec_has_expected_keys(self):
        """ML function spec contains ML-specific keys."""
        func = KNeighborsClassifierFunction()
        spec = func.spec
        assert dataclasses.is_dataclass(spec)
        # ML functions have their own spec values
        assert func.spec.scalable is False

    def test_ml_spec_attribute_access(self):
        """Attribute access works on ML function specs."""
        func = KNeighborsClassifierFunction()
        assert hasattr(func.spec, "continuous")
        assert getattr(func.spec, "nonexistent", "default") == "default"

    def test_ml_f_global_via_spec(self):
        """f_global is accessible via spec for ML functions."""
        func = KNeighborsClassifierFunction()
        # ML functions may or may not have f_global set
        _ = func.spec.f_global  # Should not raise
