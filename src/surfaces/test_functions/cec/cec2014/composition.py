# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2014 Composition Functions (F23-F30).

Composition functions create complex landscapes by combining multiple
basic functions with different optima locations.
"""

from typing import Any, Dict, List

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2014 import CEC2014Function

_SCHWEFEL_OPTIMUM_OFFSET = 4.189828872724338e2


def _hybrid_group_sizes(n_dim: int, proportions: List[float]) -> List[int]:
    """Calculate hybrid group sizes from report proportions."""
    sizes = []
    remaining = n_dim
    for p in proportions[:-1]:
        size = int(np.ceil(n_dim * p))
        sizes.append(size)
        remaining -= size
    sizes.append(remaining)
    return sizes


class _CompositionBase(CEC2014Function):
    """Base class for composition functions."""

    _spec = {
        "unimodal": False,
        "separable": False,
    }

    # To be defined by subclasses
    n_functions: int = 0
    sigmas: List[float] = []
    lambdas: List[float] = []
    biases: List[float] = []

    @property
    def x_global(self):
        """Global optimum location (first component's optimum)."""
        optima = self._get_composition_optima()
        return optima[0]

    def _get_composition_optima(self) -> np.ndarray:
        """Get optima locations for each component function."""
        data = self._load_data()
        key = f"shift_{self.func_id}"
        if key in data:
            stacked = data[key]
            return stacked[: self.n_functions, : self.n_dim]
        rng = np.random.default_rng(seed=self.func_id * 1000)
        return rng.uniform(-80, 80, size=(self.n_functions, self.n_dim))

    def _get_composition_rotation(self, component_idx: int) -> np.ndarray:
        """Get rotation matrix for a specific composition component."""
        data = self._load_data()
        key = f"rotation_{self.func_id}"
        if key in data:
            stacked = data[key]
            D = self.n_dim
            return stacked[component_idx * D : (component_idx + 1) * D, :]
        return np.eye(self.n_dim)

    def _get_composition_shuffle_indices(self, component_idx: int) -> np.ndarray:
        """Get shuffle indices for a hybrid composition component."""
        data = self._load_data()
        key = f"shuffle_{self.func_id}"
        if key in data:
            stacked = data[key]
            D = self.n_dim
            return stacked[component_idx * D : (component_idx + 1) * D]
        return np.arange(self.n_dim)

    def _compute_weights(self, x: np.ndarray, optima: np.ndarray) -> np.ndarray:
        """Compute weights for each component function."""
        weights = np.zeros(self.n_functions)
        dist_sq = np.zeros(self.n_functions)

        for i in range(self.n_functions):
            diff = x - optima[i]
            dist_sq[i] = np.sum(diff**2)

        exact = dist_sq == 0
        if np.any(exact):
            weights[exact] = 1.0 / np.count_nonzero(exact)
            return weights

        for i in range(self.n_functions):
            weights[i] = (1.0 / np.sqrt(dist_sq[i])) * np.exp(
                -dist_sq[i] / (2 * self.n_dim * self.sigmas[i] ** 2)
            )

        weight_sum = np.sum(weights)
        if weight_sum == 0:
            weights = np.ones(self.n_functions) / self.n_functions
        else:
            weights = weights / weight_sum

        return weights

    def _batch_compute_weights(self, X: ArrayLike, optima: ArrayLike) -> ArrayLike:
        """Compute weights for each component function (batch version).

        Parameters
        ----------
        X : ArrayLike
            Input batch of shape (n_points, n_dim).
        optima : ArrayLike
            Optima locations of shape (n_functions, n_dim).

        Returns
        -------
        ArrayLike
            Weights of shape (n_points, n_functions).
        """
        xp = get_array_namespace(X)

        diff = X[:, None, :] - optima[None, :, :]  # (n_points, n_functions, n_dim)
        dist_sq = xp.sum(diff**2, axis=2)  # (n_points, n_functions)

        sigmas = xp.asarray(self.sigmas, dtype=X.dtype)
        safe_dist_sq = xp.where(dist_sq == 0, 1.0, dist_sq)
        weights = (1.0 / xp.sqrt(safe_dist_sq)) * xp.exp(-dist_sq / (2 * self.n_dim * sigmas**2))

        weight_sum = xp.sum(weights, axis=1, keepdims=True)
        safe_weight_sum = xp.where(weight_sum == 0, 1, weight_sum)
        normalized = xp.where(weight_sum == 0, 1.0 / self.n_functions, weights / safe_weight_sum)

        exact = dist_sq == 0
        exact_count = xp.sum(exact, axis=1, keepdims=True)
        safe_exact_count = xp.where(exact_count == 0, 1, exact_count)
        exact_weights = xp.where(exact, 1.0 / safe_exact_count, 0.0)

        return xp.where(exact_count > 0, exact_weights, normalized)


# Basic functions for composition (same as hybrid but standalone)
def _sphere(z: np.ndarray) -> float:
    return np.sum(z**2)


def _high_conditioned_elliptic(z: np.ndarray) -> float:
    D = len(z)
    if D == 1:
        return z[0] ** 2
    result = 0.0
    for i in range(D):
        result += (10**6) ** (i / (D - 1)) * z[i] ** 2
    return result


def _bent_cigar(z: np.ndarray) -> float:
    if len(z) == 1:
        return z[0] ** 2
    return z[0] ** 2 + 10**6 * np.sum(z[1:] ** 2)


def _discus(z: np.ndarray) -> float:
    if len(z) == 1:
        return 10**6 * z[0] ** 2
    return 10**6 * z[0] ** 2 + np.sum(z[1:] ** 2)


def _rosenbrock(z: np.ndarray) -> float:
    z = z * 2.048 / 100 + 1
    result = 0.0
    for i in range(len(z) - 1):
        result += 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
    return result


def _ackley(z: np.ndarray) -> float:
    D = len(z)
    sum1 = np.sum(z**2)
    sum2 = np.sum(np.cos(2 * np.pi * z))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / D)) - np.exp(sum2 / D) + 20 + np.e


def _griewank(z: np.ndarray) -> float:
    z = z * 600 / 100
    D = len(z)
    sum_sq = np.sum(z**2) / 4000
    prod_cos = np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1))))
    return sum_sq - prod_cos + 1


def _rastrigin(z: np.ndarray) -> float:
    z = z * 5.12 / 100
    D = len(z)
    return 10 * D + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))


def _schwefel(z: np.ndarray) -> float:
    D = len(z)
    z = z * 1000 / 100 + 4.209687462275036e2
    result = 0.0
    for i in range(D):
        zi = z[i]
        if abs(zi) <= 500:
            result += zi * np.sin(np.sqrt(abs(zi)))
        elif zi > 500:
            result += (500 - zi % 500) * np.sin(np.sqrt(abs(500 - zi % 500))) - (zi - 500) ** 2 / (
                10000 * D
            )
        else:
            result += (abs(zi) % 500 - 500) * np.sin(np.sqrt(abs(abs(zi) % 500 - 500))) - (
                zi + 500
            ) ** 2 / (10000 * D)
    return _SCHWEFEL_OPTIMUM_OFFSET * D - result


def _weierstrass(z: np.ndarray) -> float:
    z = z * 0.5 / 100
    a, b, k_max = 0.5, 3, 20
    D = len(z)
    result = 0.0
    for i in range(D):
        for k in range(k_max + 1):
            result += a**k * np.cos(2 * np.pi * b**k * (z[i] + 0.5))
    offset = sum(a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max + 1))
    return result - D * offset


def _happycat(z: np.ndarray) -> float:
    z = z * 5 / 100 - 1
    D = len(z)
    alpha = 1.0 / 8.0
    sum_sq = np.sum(z**2)
    sum_z = np.sum(z)
    return abs(sum_sq - D) ** (2 * alpha) + (0.5 * sum_sq + sum_z) / D + 0.5


def _hgbat(z: np.ndarray) -> float:
    z = z * 5 / 100 - 1
    D = len(z)
    sum_sq = np.sum(z**2)
    sum_z = np.sum(z)
    return abs(sum_sq**2 - sum_z**2) ** 0.5 + (0.5 * sum_sq + sum_z) / D + 0.5


def _katsuura(z: np.ndarray) -> float:
    z = z * 5 / 100
    D = len(z)
    result = 1.0
    for i in range(D):
        inner_sum = 0.0
        for j in range(1, 33):
            inner_sum += abs(2**j * z[i] - round(2**j * z[i])) / (2**j)
        result *= (1 + (i + 1) * inner_sum) ** (10 / (D**1.2))
    return (10 / D**2) * result - (10 / D**2)


def _expanded_griewank_rosenbrock(z: np.ndarray) -> float:
    D = len(z)
    z = z * 5 / 100 + 1
    result = 0.0
    for i in range(D - 1):
        t = 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
        result += t**2 / 4000 - np.cos(t) + 1
    t = 100 * (z[-1] ** 2 - z[0]) ** 2 + (z[-1] - 1) ** 2
    result += t**2 / 4000 - np.cos(t) + 1
    return result


def _expanded_scaffer(z: np.ndarray) -> float:
    D = len(z)

    def schaffer_f6(x1, x2):
        t = x1**2 + x2**2
        return 0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2

    result = 0.0
    for i in range(D - 1):
        result += schaffer_f6(z[i], z[i + 1])
    result += schaffer_f6(z[-1], z[0])
    return result


def _batch_sphere(Z: ArrayLike) -> ArrayLike:
    """Vectorized Sphere: sum(z_i^2)."""
    xp = get_array_namespace(Z)
    return xp.sum(Z**2, axis=1)


def _batch_high_conditioned_elliptic(Z: ArrayLike) -> ArrayLike:
    """Vectorized High Conditioned Elliptic: sum(10^6^(i/(D-1)) * z_i^2)."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    if D == 1:
        return Z[:, 0] ** 2
    i = xp.arange(D, dtype=Z.dtype)
    coeffs = (10**6) ** (i / (D - 1))
    return xp.sum(coeffs * Z**2, axis=1)


def _batch_bent_cigar(Z: ArrayLike) -> ArrayLike:
    """Vectorized Bent Cigar: z_0^2 + 10^6 * sum(z_i^2, i>0)."""
    xp = get_array_namespace(Z)
    if Z.shape[1] == 1:
        return Z[:, 0] ** 2
    return Z[:, 0] ** 2 + 10**6 * xp.sum(Z[:, 1:] ** 2, axis=1)


def _batch_discus(Z: ArrayLike) -> ArrayLike:
    """Vectorized Discus: 10^6 * z_0^2 + sum(z_i^2, i>0)."""
    xp = get_array_namespace(Z)
    if Z.shape[1] == 1:
        return 10**6 * Z[:, 0] ** 2
    return 10**6 * Z[:, 0] ** 2 + xp.sum(Z[:, 1:] ** 2, axis=1)


def _batch_rosenbrock(Z: ArrayLike) -> ArrayLike:
    """Vectorized Rosenbrock: sum(100*(z_i^2 - z_{i+1})^2 + (z_i - 1)^2)."""
    xp = get_array_namespace(Z)
    Z_shifted = Z * 2.048 / 100 + 1
    z_i = Z_shifted[:, :-1]
    z_i1 = Z_shifted[:, 1:]
    return xp.sum(100 * (z_i**2 - z_i1) ** 2 + (z_i - 1) ** 2, axis=1)


def _batch_ackley(Z: ArrayLike) -> ArrayLike:
    """Vectorized Ackley function."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    sum1 = xp.sum(Z**2, axis=1)
    sum2 = xp.sum(xp.cos(2 * np.pi * Z), axis=1)
    return -20 * xp.exp(-0.2 * xp.sqrt(sum1 / D)) - xp.exp(sum2 / D) + 20 + np.e


def _batch_griewank(Z: ArrayLike) -> ArrayLike:
    """Vectorized Griewank function."""
    xp = get_array_namespace(Z)
    Z = Z * 600 / 100
    D = Z.shape[1]
    sum_sq = xp.sum(Z**2, axis=1) / 4000
    i = xp.arange(1, D + 1, dtype=Z.dtype)
    prod_cos = xp.prod(xp.cos(Z / xp.sqrt(i)), axis=1)
    return sum_sq - prod_cos + 1


def _batch_rastrigin(Z: ArrayLike) -> ArrayLike:
    """Vectorized Rastrigin: 10*D + sum(z_i^2 - 10*cos(2*pi*z_i))."""
    xp = get_array_namespace(Z)
    Z = Z * 5.12 / 100
    D = Z.shape[1]
    return 10 * D + xp.sum(Z**2 - 10 * xp.cos(2 * np.pi * Z), axis=1)


def _batch_schwefel(Z: ArrayLike) -> ArrayLike:
    """Vectorized Schwefel function with boundary handling."""
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    Z_shifted = Z * 1000 / 100 + 4.209687462275036e2

    abs_z = xp.abs(Z_shifted)

    # Case 1: |z| <= 500
    term1 = Z_shifted * xp.sin(xp.sqrt(abs_z))

    # Case 2: z > 500
    mod_pos = 500 - xp.mod(Z_shifted, 500)
    term2 = mod_pos * xp.sin(xp.sqrt(xp.abs(mod_pos))) - (Z_shifted - 500) ** 2 / (10000 * D)

    # Case 3: z < -500
    mod_neg = xp.mod(abs_z, 500) - 500
    term3 = mod_neg * xp.sin(xp.sqrt(xp.abs(mod_neg))) - (Z_shifted + 500) ** 2 / (10000 * D)

    result = xp.where(
        abs_z <= 500,
        term1,
        xp.where(Z_shifted > 500, term2, term3),
    )

    return _SCHWEFEL_OPTIMUM_OFFSET * D - xp.sum(result, axis=1)


def _batch_weierstrass(Z: ArrayLike) -> ArrayLike:
    """Vectorized Weierstrass function."""
    xp = get_array_namespace(Z)
    Z = Z * 0.5 / 100
    a, b, k_max = 0.5, 3, 20
    D = Z.shape[1]

    k = xp.arange(k_max + 1, dtype=Z.dtype)
    a_k = a**k
    b_k = b**k

    Z_expanded = Z[:, :, None]
    cos_terms = a_k * xp.cos(2 * np.pi * b_k * (Z_expanded + 0.5))
    result = xp.sum(cos_terms, axis=(1, 2))

    offset_k = a_k * xp.cos(2 * np.pi * b_k * 0.5)
    offset = D * xp.sum(offset_k)

    return result - offset


def _batch_happycat(Z: ArrayLike) -> ArrayLike:
    """Vectorized HappyCat function."""
    xp = get_array_namespace(Z)
    Z = Z * 5 / 100 - 1
    D = Z.shape[1]
    alpha = 1.0 / 8.0
    sum_sq = xp.sum(Z**2, axis=1)
    sum_z = xp.sum(Z, axis=1)
    return xp.abs(sum_sq - D) ** (2 * alpha) + (0.5 * sum_sq + sum_z) / D + 0.5


def _batch_hgbat(Z: ArrayLike) -> ArrayLike:
    """Vectorized HGBat function."""
    xp = get_array_namespace(Z)
    Z = Z * 5 / 100 - 1
    D = Z.shape[1]
    sum_sq = xp.sum(Z**2, axis=1)
    sum_z = xp.sum(Z, axis=1)
    return xp.abs(sum_sq**2 - sum_z**2) ** 0.5 + (0.5 * sum_sq + sum_z) / D + 0.5


def _batch_katsuura(Z: ArrayLike) -> ArrayLike:
    """Vectorized Katsuura function."""
    xp = get_array_namespace(Z)
    Z = Z * 5 / 100
    D = Z.shape[1]

    j = xp.arange(1, 33, dtype=Z.dtype)
    two_j = 2.0**j

    Z_expanded = Z[:, :, None]
    scaled = two_j * Z_expanded
    inner_sum = xp.sum(xp.abs(scaled - xp.round(scaled)) / two_j, axis=2)

    i = xp.arange(1, D + 1, dtype=Z.dtype)
    terms = (1 + i * inner_sum) ** (10 / (D**1.2))

    result = xp.prod(terms, axis=1)
    return (10 / D**2) * result - (10 / D**2)


def _batch_expanded_griewank_rosenbrock(Z: ArrayLike) -> ArrayLike:
    """Vectorized Expanded Griewank-Rosenbrock function."""
    xp = get_array_namespace(Z)
    Z_shifted = Z * 5 / 100 + 1

    z_i = Z_shifted[:, :-1]
    z_i1 = Z_shifted[:, 1:]

    t_main = 100 * (z_i**2 - z_i1) ** 2 + (z_i - 1) ** 2
    griewank_main = t_main**2 / 4000 - xp.cos(t_main) + 1

    t_wrap = 100 * (Z_shifted[:, -1] ** 2 - Z_shifted[:, 0]) ** 2 + (Z_shifted[:, -1] - 1) ** 2
    griewank_wrap = t_wrap**2 / 4000 - xp.cos(t_wrap) + 1

    return xp.sum(griewank_main, axis=1) + griewank_wrap


def _batch_expanded_scaffer(Z: ArrayLike) -> ArrayLike:
    """Vectorized Expanded Scaffer F6 function."""
    xp = get_array_namespace(Z)

    z_i = Z[:, :-1]
    z_i1 = Z[:, 1:]
    t_main = z_i**2 + z_i1**2
    schaffer_main = 0.5 + (xp.sin(xp.sqrt(t_main)) ** 2 - 0.5) / (1 + 0.001 * t_main) ** 2

    t_wrap = Z[:, -1] ** 2 + Z[:, 0] ** 2
    schaffer_wrap = 0.5 + (xp.sin(xp.sqrt(t_wrap)) ** 2 - 0.5) / (1 + 0.001 * t_wrap) ** 2

    return xp.sum(schaffer_main, axis=1) + schaffer_wrap


_SCALAR_COMPONENTS = {
    "ackley": _ackley,
    "bent_cigar": _bent_cigar,
    "discus": _discus,
    "expanded_griewank_rosenbrock": _expanded_griewank_rosenbrock,
    "expanded_scaffer": _expanded_scaffer,
    "griewank": _griewank,
    "happycat": _happycat,
    "hgbat": _hgbat,
    "high_conditioned_elliptic": _high_conditioned_elliptic,
    "katsuura": _katsuura,
    "rastrigin": _rastrigin,
    "rosenbrock": _rosenbrock,
    "schwefel": _schwefel,
    "sphere": _sphere,
    "weierstrass": _weierstrass,
}

_BATCH_COMPONENTS = {
    "ackley": _batch_ackley,
    "bent_cigar": _batch_bent_cigar,
    "discus": _batch_discus,
    "expanded_griewank_rosenbrock": _batch_expanded_griewank_rosenbrock,
    "expanded_scaffer": _batch_expanded_scaffer,
    "griewank": _batch_griewank,
    "happycat": _batch_happycat,
    "hgbat": _batch_hgbat,
    "high_conditioned_elliptic": _batch_high_conditioned_elliptic,
    "katsuura": _batch_katsuura,
    "rastrigin": _batch_rastrigin,
    "rosenbrock": _batch_rosenbrock,
    "schwefel": _batch_schwefel,
    "sphere": _batch_sphere,
    "weierstrass": _batch_weierstrass,
}


def _hybrid_component_value(
    z: np.ndarray,
    shuffle_idx: np.ndarray,
    proportions: List[float],
    component_names: tuple[str, ...],
) -> float:
    z_shuffled = z[shuffle_idx]
    sizes = _hybrid_group_sizes(len(z), proportions)

    result = 0.0
    start = 0
    for size, component_name in zip(sizes, component_names):
        group = z_shuffled[start : start + size]
        if len(group) > 0:
            result += _SCALAR_COMPONENTS[component_name](group)
        start += size
    return result


def _batch_hybrid_component_value(
    Z: ArrayLike,
    shuffle_idx: ArrayLike,
    proportions: List[float],
    component_names: tuple[str, ...],
) -> ArrayLike:
    xp = get_array_namespace(Z)
    Z_shuffled = Z[:, shuffle_idx]
    sizes = _hybrid_group_sizes(Z.shape[1], proportions)

    result = xp.zeros(Z.shape[0], dtype=Z.dtype)
    start = 0
    for size, component_name in zip(sizes, component_names):
        group = Z_shuffled[:, start : start + size]
        if group.shape[1] > 0:
            result = result + _BATCH_COMPONENTS[component_name](group)
        start += size
    return result


class CompositionFunction1(_CompositionBase):
    """F23: Composition Function 1.

    Combines 5 rotated functions with different optima.

    Components:
    1. Rosenbrock
    2. High Conditioned Elliptic
    3. Bent Cigar
    4. Discus
    5. High Conditioned Elliptic

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 5
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [1, 1e-6, 1e-26, 1e-6, 1e-6]
    biases = [0, 100, 200, 300, 400]
    component_names = (
        "rosenbrock",
        "high_conditioned_elliptic",
        "bent_cigar",
        "discus",
        "high_conditioned_elliptic",
    )

    _spec = {
        "eval_cost": 7.4,
        "name": "Composition Function 1",
        "func_id": 23,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        functions = [_SCALAR_COMPONENTS[name] for name in self.component_names]

        x = self._params_to_array(params)
        optima = self._get_composition_optima()
        weights = self._compute_weights(x, optima)

        result = 0.0
        for i in range(self.n_functions):
            M = self._get_composition_rotation(i)
            z = M @ (x - optima[i])
            f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
            result += weights[i] * f_val

        return result + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F23: Rosenbrock + HCE + Bent Cigar + Discus + HCE."""
        xp = get_array_namespace(X)
        optima = xp.asarray(self._get_composition_optima(), dtype=X.dtype)

        # Compute weights: (n_points, n_functions)
        weights = self._batch_compute_weights(X, optima)

        batch_funcs = [_BATCH_COMPONENTS[name] for name in self.component_names]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            M = xp.asarray(self._get_composition_rotation(i), dtype=X.dtype)
            Z = (X - optima[i]) @ M.T
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction2(_CompositionBase):
    """F24: Composition Function 2.

    Combines 3 functions with different properties.

    Components:
    1. Schwefel
    2. Rastrigin
    3. HGBat

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 3
    sigmas = [20, 20, 20]
    lambdas = [1, 1, 1]
    biases = [0, 100, 200]
    component_names = ("schwefel", "rastrigin", "hgbat")

    _spec = {
        "eval_cost": 5.6,
        "name": "Composition Function 2",
        "func_id": 24,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        functions = [_SCALAR_COMPONENTS[name] for name in self.component_names]

        x = self._params_to_array(params)
        optima = self._get_composition_optima()
        weights = self._compute_weights(x, optima)

        result = 0.0
        for i in range(self.n_functions):
            M = self._get_composition_rotation(i)
            z = M @ (x - optima[i])
            f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
            result += weights[i] * f_val

        return result + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F24: Schwefel + Rastrigin + HGBat."""
        xp = get_array_namespace(X)
        optima = xp.asarray(self._get_composition_optima(), dtype=X.dtype)

        weights = self._batch_compute_weights(X, optima)

        batch_funcs = [_BATCH_COMPONENTS[name] for name in self.component_names]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            M = xp.asarray(self._get_composition_rotation(i), dtype=X.dtype)
            Z = (X - optima[i]) @ M.T
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction3(_CompositionBase):
    """F25: Composition Function 3.

    Combines 3 functions.

    Components:
    1. Schwefel
    2. Rastrigin
    3. High Conditioned Elliptic

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 3
    sigmas = [10, 30, 50]
    lambdas = [0.25, 1, 1e-7]
    biases = [0, 100, 200]
    component_names = ("schwefel", "rastrigin", "high_conditioned_elliptic")

    _spec = {
        "eval_cost": 5.9,
        "name": "Composition Function 3",
        "func_id": 25,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        functions = [_SCALAR_COMPONENTS[name] for name in self.component_names]

        x = self._params_to_array(params)
        optima = self._get_composition_optima()
        weights = self._compute_weights(x, optima)

        result = 0.0
        for i in range(self.n_functions):
            M = self._get_composition_rotation(i)
            z = M @ (x - optima[i])
            f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
            result += weights[i] * f_val

        return result + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F25: Schwefel + Rastrigin + HCE."""
        xp = get_array_namespace(X)
        optima = xp.asarray(self._get_composition_optima(), dtype=X.dtype)

        weights = self._batch_compute_weights(X, optima)

        batch_funcs = [_BATCH_COMPONENTS[name] for name in self.component_names]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            M = xp.asarray(self._get_composition_rotation(i), dtype=X.dtype)
            Z = (X - optima[i]) @ M.T
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction4(_CompositionBase):
    """F26: Composition Function 4.

    Combines 5 functions.

    Components:
    1. Schwefel
    2. HappyCat
    3. High Conditioned Elliptic
    4. Weierstrass
    5. Griewank

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 5
    sigmas = [10, 10, 10, 10, 10]
    lambdas = [0.25, 1, 1e-7, 2.5, 10]
    biases = [0, 100, 200, 300, 400]
    component_names = (
        "schwefel",
        "happycat",
        "high_conditioned_elliptic",
        "weierstrass",
        "griewank",
    )

    _spec = {
        "eval_cost": 20.6,
        "name": "Composition Function 4",
        "func_id": 26,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        functions = [_SCALAR_COMPONENTS[name] for name in self.component_names]

        x = self._params_to_array(params)
        optima = self._get_composition_optima()
        weights = self._compute_weights(x, optima)

        result = 0.0
        for i in range(self.n_functions):
            M = self._get_composition_rotation(i)
            z = M @ (x - optima[i])
            f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
            result += weights[i] * f_val

        return result + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F26: Schwefel + HappyCat + HCE + Weierstrass + Griewank."""
        xp = get_array_namespace(X)
        optima = xp.asarray(self._get_composition_optima(), dtype=X.dtype)

        weights = self._batch_compute_weights(X, optima)

        batch_funcs = [_BATCH_COMPONENTS[name] for name in self.component_names]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            M = xp.asarray(self._get_composition_rotation(i), dtype=X.dtype)
            Z = (X - optima[i]) @ M.T
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction5(_CompositionBase):
    """F27: Composition Function 5.

    Combines 5 functions.

    Components:
    1. HGBat
    2. Rastrigin
    3. Schwefel
    4. Weierstrass
    5. High Conditioned Elliptic

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 5
    sigmas = [10, 10, 10, 20, 20]
    lambdas = [10, 10, 2.5, 25, 1e-6]
    biases = [0, 100, 200, 300, 400]
    component_names = (
        "hgbat",
        "rastrigin",
        "schwefel",
        "weierstrass",
        "high_conditioned_elliptic",
    )

    _spec = {
        "eval_cost": 20.4,
        "name": "Composition Function 5",
        "func_id": 27,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        functions = [_SCALAR_COMPONENTS[name] for name in self.component_names]

        x = self._params_to_array(params)
        optima = self._get_composition_optima()
        weights = self._compute_weights(x, optima)

        result = 0.0
        for i in range(self.n_functions):
            M = self._get_composition_rotation(i)
            z = M @ (x - optima[i])
            f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
            result += weights[i] * f_val

        return result + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F27: HGBat + Rastrigin + Schwefel + Weierstrass + HCE."""
        xp = get_array_namespace(X)
        optima = xp.asarray(self._get_composition_optima(), dtype=X.dtype)

        weights = self._batch_compute_weights(X, optima)

        batch_funcs = [_BATCH_COMPONENTS[name] for name in self.component_names]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            M = xp.asarray(self._get_composition_rotation(i), dtype=X.dtype)
            Z = (X - optima[i]) @ M.T
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction6(_CompositionBase):
    """F28: Composition Function 6.

    Combines 5 functions.

    Components:
    1. Expanded Griewank-Rosenbrock
    2. HappyCat
    3. Schwefel
    4. Expanded Scaffer
    5. High Conditioned Elliptic

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 5
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [2.5, 10, 2.5, 5e-4, 1e-6]
    biases = [0, 100, 200, 300, 400]
    component_names = (
        "expanded_griewank_rosenbrock",
        "happycat",
        "schwefel",
        "expanded_scaffer",
        "high_conditioned_elliptic",
    )

    _spec = {
        "eval_cost": 9.4,
        "name": "Composition Function 6",
        "func_id": 28,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        functions = [_SCALAR_COMPONENTS[name] for name in self.component_names]

        x = self._params_to_array(params)
        optima = self._get_composition_optima()
        weights = self._compute_weights(x, optima)

        result = 0.0
        for i in range(self.n_functions):
            M = self._get_composition_rotation(i)
            z = M @ (x - optima[i])
            f_val = self.lambdas[i] * functions[i](z) + self.biases[i]
            result += weights[i] * f_val

        return result + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F28: EGR + HappyCat + Schwefel + Expanded Scaffer + HCE."""
        xp = get_array_namespace(X)
        optima = xp.asarray(self._get_composition_optima(), dtype=X.dtype)

        weights = self._batch_compute_weights(X, optima)

        batch_funcs = [_BATCH_COMPONENTS[name] for name in self.component_names]

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            M = xp.asarray(self._get_composition_rotation(i), dtype=X.dtype)
            Z = (X - optima[i]) @ M.T
            f_vals = self.lambdas[i] * batch_funcs[i](Z) + self.biases[i]
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction7(_CompositionBase):
    """F29: Composition Function 7.

    Combines 3 hybrid functions.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 3
    sigmas = [10, 30, 50]
    lambdas = [1, 1, 1]
    biases = [0, 100, 200]
    component_names = ("hybrid1", "hybrid2", "hybrid3")
    hybrid_components = (
        (
            [0.3, 0.3, 0.4],
            ("schwefel", "rastrigin", "high_conditioned_elliptic"),
        ),
        (
            [0.3, 0.3, 0.4],
            ("bent_cigar", "hgbat", "rastrigin"),
        ),
        (
            [0.2, 0.2, 0.3, 0.3],
            ("griewank", "weierstrass", "rosenbrock", "expanded_scaffer"),
        ),
    )

    _spec = {
        "eval_cost": 16.1,
        "name": "Composition Function 7",
        "func_id": 29,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        optima = self._get_composition_optima()
        weights = self._compute_weights(x, optima)

        result = 0.0
        for i in range(self.n_functions):
            M = self._get_composition_rotation(i)
            z = M @ (x - optima[i])
            shuffle_idx = self._get_composition_shuffle_indices(i)
            proportions, component_names = self.hybrid_components[i]
            f_val = (
                self.lambdas[i]
                * _hybrid_component_value(z, shuffle_idx, proportions, component_names)
                + self.biases[i]
            )
            result += weights[i] * f_val

        return result + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F29: 3 hybrid functions."""
        xp = get_array_namespace(X)
        optima = xp.asarray(self._get_composition_optima(), dtype=X.dtype)
        weights = self._batch_compute_weights(X, optima)

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            M = xp.asarray(self._get_composition_rotation(i), dtype=X.dtype)
            Z = (X - optima[i]) @ M.T
            shuffle_idx = xp.asarray(self._get_composition_shuffle_indices(i))
            proportions, component_names = self.hybrid_components[i]
            f_vals = (
                self.lambdas[i]
                * _batch_hybrid_component_value(Z, shuffle_idx, proportions, component_names)
                + self.biases[i]
            )
            result = result + weights[:, i] * f_vals

        return result + self.f_global


class CompositionFunction8(_CompositionBase):
    """F30: Composition Function 8.

    Combines 3 hybrid functions (different from F29).

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    n_functions = 3
    sigmas = [10, 30, 50]
    lambdas = [1, 1, 1]
    biases = [0, 100, 200]
    component_names = ("hybrid4", "hybrid5", "hybrid6")
    hybrid_components = (
        (
            [0.2, 0.2, 0.3, 0.3],
            ("hgbat", "discus", "expanded_griewank_rosenbrock", "rastrigin"),
        ),
        (
            [0.1, 0.2, 0.2, 0.2, 0.3],
            (
                "expanded_scaffer",
                "hgbat",
                "rosenbrock",
                "schwefel",
                "high_conditioned_elliptic",
            ),
        ),
        (
            [0.1, 0.2, 0.2, 0.2, 0.3],
            ("katsuura", "happycat", "expanded_griewank_rosenbrock", "schwefel", "ackley"),
        ),
    )

    _spec = {
        "eval_cost": 19.0,
        "name": "Composition Function 8",
        "func_id": 30,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        optima = self._get_composition_optima()
        weights = self._compute_weights(x, optima)

        result = 0.0
        for i in range(self.n_functions):
            M = self._get_composition_rotation(i)
            z = M @ (x - optima[i])
            shuffle_idx = self._get_composition_shuffle_indices(i)
            proportions, component_names = self.hybrid_components[i]
            f_val = (
                self.lambdas[i]
                * _hybrid_component_value(z, shuffle_idx, proportions, component_names)
                + self.biases[i]
            )
            result += weights[i] * f_val

        return result + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized F30: 3 hybrid functions."""
        xp = get_array_namespace(X)
        optima = xp.asarray(self._get_composition_optima(), dtype=X.dtype)
        weights = self._batch_compute_weights(X, optima)

        result = xp.zeros(X.shape[0], dtype=X.dtype)
        for i in range(self.n_functions):
            M = xp.asarray(self._get_composition_rotation(i), dtype=X.dtype)
            Z = (X - optima[i]) @ M.T
            shuffle_idx = xp.asarray(self._get_composition_shuffle_indices(i))
            proportions, component_names = self.hybrid_components[i]
            f_vals = (
                self.lambdas[i]
                * _batch_hybrid_component_value(Z, shuffle_idx, proportions, component_names)
                + self.biases[i]
            )
            result = result + weights[:, i] * f_vals

        return result + self.f_global
