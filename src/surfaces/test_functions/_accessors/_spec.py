"""SpecAccessor: dict-like access to function characteristics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .._function_spec import FunctionSpec

if TYPE_CHECKING:
    import numpy as np

    from .._base_test_function import BaseTestFunction


class SpecAccessor:
    """Namespaced access to function specification/characteristics.

    Wraps the MRO-based _spec merging and provides dict-like access
    plus typed properties for common spec fields.

    Parameters
    ----------
    func : BaseTestFunction
        The test function instance.
    """

    def __init__(self, func: "BaseTestFunction") -> None:
        self._func = func

    @property
    def _spec(self) -> FunctionSpec:
        spec = getattr(type(self._func), "_spec", FunctionSpec())
        if isinstance(spec, FunctionSpec):
            return spec
        return FunctionSpec(**spec)

    def as_dict(self) -> dict:
        """Return the resolved static function spec as a plain dict."""
        return self._spec.as_dict()

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-compatible get."""
        return self.as_dict().get(key, default)

    # Dict-like protocol

    def __getitem__(self, key: str) -> Any:
        d = self.as_dict()
        if key not in d:
            raise KeyError(key)
        return d[key]

    def __contains__(self, key: str) -> bool:
        return key in self.as_dict()

    def __repr__(self) -> str:
        return f"SpecAccessor({self.as_dict()!r})"

    # Typed properties from _spec

    @property
    def n_dim(self) -> Optional[int]:
        return getattr(self._func, "n_dim", self._spec.n_dim)

    @property
    def n_objectives(self) -> int:
        # Prefer instance/class attribute (set by __init__ or class-level)
        # over static spec, so configurable n_objectives (DTLZ, WFG) works.
        return getattr(self._func, "n_objectives", self._spec.n_objectives)

    @property
    def default_bounds(self) -> Optional[tuple]:
        return self._spec.default_bounds

    @property
    def func_id(self) -> Optional[int]:
        return self._spec.func_id

    @property
    def convex(self) -> Optional[bool]:
        return self._spec.convex

    @property
    def unimodal(self) -> Optional[bool]:
        return self._spec.unimodal

    @property
    def separable(self) -> Optional[bool]:
        return self._spec.separable

    @property
    def continuous(self) -> Optional[bool]:
        return self._spec.continuous

    @property
    def differentiable(self) -> Optional[bool]:
        return self._spec.differentiable

    @property
    def scalable(self) -> bool:
        return self._spec.scalable

    @property
    def discrete(self) -> bool:
        return self._spec.discrete

    @property
    def constrained(self) -> bool:
        return self._spec.constrained

    @property
    def stochastic(self) -> bool:
        return self._spec.stochastic

    @property
    def simulation_based(self) -> bool:
        return self._spec.simulation_based

    @property
    def expensive(self) -> bool:
        return self._spec.expensive

    @property
    def ode_based(self) -> bool:
        return self._spec.ode_based

    @property
    def deceptive(self) -> bool:
        return self._spec.deceptive

    @property
    def multimodal(self) -> bool:
        return self._spec.multimodal

    @property
    def convex_front(self) -> Optional[bool]:
        return self._spec.convex_front

    @property
    def concave_front(self) -> Optional[bool]:
        return self._spec.concave_front

    @property
    def disconnected_front(self) -> Optional[bool]:
        return self._spec.disconnected_front

    # Global optimum (from class-level attrs on concrete functions)

    @property
    def eval_cost(self) -> Optional[float]:
        """Evaluation cost in Compute Units (CU)."""
        return self._spec.eval_cost

    @property
    def deprecated(self) -> bool:
        return self._spec.deprecated

    @property
    def f_global(self) -> Optional[float]:
        return getattr(self._func, "f_global", None)

    @property
    def x_global(self) -> "Optional[np.ndarray]":
        return getattr(self._func, "x_global", None)
