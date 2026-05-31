# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""DTLZ7 multi-objective test function.

DTLZ7 produces a disconnected set of Pareto-optimal regions in
objective space. The first M-1 objectives are simply the position
parameters, while the last objective is a nonlinear function that
creates :math:`2^{M-1}` disconnected segments on the Pareto front.
"""

from itertools import product
from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from .._base_multi_objective import BaseMultiObjectiveTestFunction

_PARETO_POSITION_INTERVALS = (
    (0.0, 0.25141183608891715),
    (0.631626530700061, 0.8594008566446932),
)


def _phi(f_pos):
    return f_pos * (1 + np.sin(3 * np.pi * f_pos))


def _largest_remainder_counts(n_points, weights):
    if n_points < 0:
        raise ValueError(f"n_points must be >= 0, got {n_points}")

    counts = np.zeros(len(weights), dtype=int)
    if n_points == 0:
        return counts

    weights = np.asarray(weights, dtype=float)
    fractions = n_points * weights / np.sum(weights)
    counts += np.floor(fractions).astype(int)

    deficit = n_points - int(np.sum(counts))
    if deficit > 0:
        remainders = fractions - counts
        for idx in np.argsort(remainders)[::-1][:deficit]:
            counts[idx] += 1

    return counts


def _pareto_position_samples(n_points, n_position):
    if n_points < 0:
        raise ValueError(f"n_points must be >= 0, got {n_points}")
    if n_points == 0:
        return np.empty((0, n_position))

    regions = list(product(_PARETO_POSITION_INTERVALS, repeat=n_position))
    volumes = [np.prod([hi - lo for lo, hi in region]) for region in regions]
    counts = _largest_remainder_counts(n_points, volumes)

    rng = np.random.default_rng(42)
    samples = []
    for region, count in zip(regions, counts):
        if count == 0:
            continue

        if n_position == 1:
            lo, hi = region[0]
            samples.append(np.linspace(lo, hi, count).reshape(-1, 1))
            continue

        region_samples = np.zeros((count, n_position))
        for dim, (lo, hi) in enumerate(region):
            region_samples[:, dim] = rng.uniform(lo, hi, count)
        samples.append(region_samples)

    return np.vstack(samples)


class DTLZ7(BaseMultiObjectiveTestFunction):
    r"""DTLZ7 multi-objective test function.

    DTLZ7 is unique among the DTLZ family in that its Pareto front
    consists of :math:`2^{M-1}` disconnected regions. The first
    :math:`M-1` objectives are identity mappings of the position
    parameters, while the last objective creates the disconnected
    geometry through a sinusoidal term.

    The objectives are:

    .. math::

        f_i(\mathbf{x}) = x_i \quad i = 1, \ldots, M-1

        f_M(\mathbf{x}) = (1 + g(\mathbf{x}_d)) \cdot h(f_1, \ldots, f_{M-1}, g)

    where:

    .. math::

        g(\mathbf{x}_d) = 1 + \frac{9}{k} \sum_{i=M}^{n} x_i

        h = M - \sum_{i=1}^{M-1} \frac{f_i}{1+g}
            \left(1 + \sin(3\pi f_i)\right)

    Parameters
    ----------
    n_objectives : int, default=3
        Number of objectives :math:`M`.
    n_dim : int, optional
        Total number of decision variables. Defaults to
        ``(n_objectives - 1) + 20``.
    **kwargs
        Additional keyword arguments passed to
        :class:`BaseMultiObjectiveTestFunction`.

    Attributes
    ----------
    n_objectives : int
        Number of objectives.

    References
    ----------
    .. [1] Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2005).
       Scalable test problems for evolutionary multiobjective optimization.
       In Evolutionary Multiobjective Optimization (pp. 105-145). Springer.

    Examples
    --------
    >>> from surfaces.test_functions.algebraic.multi_objective.dtlz import DTLZ7
    >>> func = DTLZ7(n_objectives=3)
    >>> result = func(np.full(func.n_dim, 0.5))
    >>> result.shape
    (3,)
    """

    name = "DTLZ7"
    n_objectives = 3
    _k = 20
    _spec = {
        "eval_cost": 1.8,
        "continuous": True,
        "differentiable": True,
        "disconnected_front": True,
        "scalable": True,
        "default_bounds": (0.0, 1.0),
    }

    def __init__(self, n_objectives: int = 3, n_dim: int = None, **kwargs):
        if n_dim is None:
            n_dim = (n_objectives - 1) + self._k
        if n_dim < n_objectives:
            raise ValueError(
                f"n_dim must be >= n_objectives, got n_dim={n_dim}, n_objectives={n_objectives}"
            )
        super().__init__(n_dim, n_objectives=n_objectives, **kwargs)

    def _objective(self, params: Dict[str, Any]) -> np.ndarray:
        x = self._params_to_array(params)
        M = self.n_objectives
        x_dist = x[M - 1 :]

        k = len(x_dist)
        g = 1 + (9 / k) * np.sum(x_dist)

        f = np.zeros(M)
        for i in range(M - 1):
            f[i] = x[i]

        h = M - np.sum(f[: M - 1] / (1 + g) * (1 + np.sin(3 * np.pi * f[: M - 1])))
        f[M - 1] = (1 + g) * h

        return f

    def _pareto_front(self, n_points: int) -> np.ndarray:
        """Disconnected nondominated Pareto front regions with g=1."""
        M = self.n_objectives
        f_pos = _pareto_position_samples(n_points, M - 1)
        f_last = 2 * M - np.sum(_phi(f_pos), axis=1)

        return np.column_stack([f_pos, f_last])

    def _pareto_set(self, n_points: int) -> np.ndarray:
        """Distance params at 0.0, position params in nondominated intervals."""
        M = self.n_objectives
        x = np.zeros((n_points, self.n_dim))
        x[:, : M - 1] = _pareto_position_samples(n_points, M - 1)
        return x

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized DTLZ7 evaluation."""
        xp = get_array_namespace(X)
        M = self.n_objectives
        x_dist = X[:, M - 1 :]

        k = x_dist.shape[1]
        g = 1 + (9 / k) * xp.sum(x_dist, axis=1)

        f_pos = X[:, : M - 1]

        h = M - xp.sum(
            f_pos / (1 + g)[:, None] * (1 + xp.sin(3 * np.pi * f_pos)),
            axis=1,
        )
        f_last = (1 + g) * h

        return xp.concatenate([f_pos, f_last[:, None]], axis=1)
