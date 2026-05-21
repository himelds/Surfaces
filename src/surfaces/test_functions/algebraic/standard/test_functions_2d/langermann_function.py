# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import math
from typing import Any, Callable, Dict, List, Optional, Union

from surfaces._array_utils import ArrayLike, get_array_namespace
from surfaces.modifiers import BaseModifier

from ..._base_algebraic_function import AlgebraicFunction


class LangermannFunction(AlgebraicFunction):
    """Langermann two-dimensional test function.

    A multimodal function with many unevenly distributed local minima.

    Parameters
    ----------
    objective : str, default="minimize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Attributes
    ----------
    n_dim : int
        Number of dimensions (always 2).
    c : ndarray
        Coefficient vector.
    m : int
        Number of terms in the summation.
    A : ndarray
        Matrix of center coordinates.

    Examples
    --------
    >>> from surfaces.test_functions import LangermannFunction
    >>> func = LangermannFunction()
    >>> result = func({"x0": 0.0, "x1": 0.0})
    """

    _spec = {
        "eval_cost": 0.6,
        "convex": False,
        "unimodal": False,
        "separable": False,
        "scalable": False,
        "default_bounds": (0.0, 10.0),
    }

    f_global = None  # Complex to determine analytically
    x_global = None

    n_dim = 2

    latex_formula = r"f(x, y) = \sum_{i=1}^{m} c_i \exp\left(-\frac{1}{\pi}\sum_{j=1}^{2}(x_j - A_{ji})^2\right) \cos\left(\pi\sum_{j=1}^{2}(x_j - A_{ji})^2\right)"
    pgfmath_formula = None  # Complex summation not expressible in pgfmath

    # Function sheet attributes
    tagline = (
        "Irregularly distributed local minima with varying depths. "
        "A challenging landscape without clear structure."
    )
    display_bounds = (0.0, 10.0)
    reference = None
    reference_url = "https://www.sfu.ca/~ssurjano/langer.html"

    c = (1, 2, 5, 2, 3)
    m = 5
    A = ((3, 5, 2, 1, 7), (5, 2, 1, 4, 9))

    def __init__(
        self,
        objective: str = "minimize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
    ) -> None:
        super().__init__(objective, modifiers, memory, collect_data, callbacks, catch_errors)
        self.n_dim = 2

    def _objective(self, params: Dict[str, Any]) -> float:
        x = (params["x0"], params["x1"])
        outer = 0.0

        for m in range(self.m):
            inner = 0.0
            for dim in range(self.n_dim):
                diff = x[dim] - self.A[dim][m]
                inner += diff**2

            outer += self.c[m] * math.exp(-inner / math.pi) * math.cos(math.pi * inner)

        return outer

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation.

        Parameters
        ----------
        X : ArrayLike
            Array of shape (n_points, 2).

        Returns
        -------
        ArrayLike
            Array of shape (n_points,).
        """
        xp = get_array_namespace(X)

        c = xp.asarray(self.c)
        centers = xp.asarray(self.A).T
        inner = xp.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

        return xp.sum(c * xp.exp(-inner / math.pi) * xp.cos(math.pi * inner), axis=1)

    def _search_space(
        self,
        min: float = 0,
        max: float = 10,
        value_types: str = "array",
        size: int = 10000,
    ) -> Dict[str, Any]:
        return super()._create_n_dim_search_space(
            min=min, max=max, size=size, value_types=value_types
        )
