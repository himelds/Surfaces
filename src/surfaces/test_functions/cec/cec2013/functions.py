# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2013 Benchmark Functions (F1-F28)."""

import math
from typing import Any, Dict

import numpy as np

from surfaces._array_utils import ArrayLike, get_array_namespace

from ._base_cec2013 import CEC2013Function

_SCHWEFEL_OPTIMUM = 4.209687462275036e2
_SCHWEFEL_OFFSET = 4.189828872724338e2


def _lambda_scale_np(x: np.ndarray, alpha: float) -> np.ndarray:
    D = len(x)
    if D <= 1:
        return x.copy()
    idx = np.arange(D, dtype=float)
    return x * np.power(alpha, 0.5 * idx / (D - 1))


def _batch_lambda_scale_xp(X: ArrayLike, alpha: float) -> ArrayLike:
    xp = get_array_namespace(X)
    D = X.shape[1]
    if D <= 1:
        return X
    idx = xp.arange(D, dtype=X.dtype)
    return X * xp.power(float(alpha), 0.5 * idx / (D - 1))


def _osz_endpoints(x: np.ndarray) -> np.ndarray:
    z = x.copy()
    if len(z) == 0:
        return z
    for i in (0, len(z) - 1):
        xi = x[i]
        if xi == 0:
            z[i] = 0.0
            continue
        c1 = 10.0 if xi > 0 else 5.5
        c2 = 7.9 if xi > 0 else 3.1
        x_hat = np.log(abs(xi))
        z[i] = np.sign(xi) * np.exp(x_hat + 0.049 * (np.sin(c1 * x_hat) + np.sin(c2 * x_hat)))
    return z


def _batch_osz_endpoints(X: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(X)
    Z = X.copy()
    if X.shape[1] == 0:
        return Z

    cols = (0,) if X.shape[1] == 1 else (0, X.shape[1] - 1)
    for col in cols:
        values = X[:, col]
        nonzero = values != 0
        safe_abs = xp.where(nonzero, xp.abs(values), 1.0)
        x_hat = xp.where(nonzero, xp.log(safe_abs), 0.0)
        c1 = xp.where(values > 0, 10.0, 5.5)
        c2 = xp.where(values > 0, 7.9, 3.1)
        transformed = xp.sign(values) * xp.exp(
            x_hat + 0.049 * (xp.sin(c1 * x_hat) + xp.sin(c2 * x_hat))
        )
        Z[:, col] = xp.where(nonzero, transformed, 0.0)
    return Z


def _rotate_with_index(func: CEC2013Function, x: np.ndarray, index: int) -> np.ndarray:
    return func._get_rotation_matrix(index) @ x


def _batch_rotate_with_index(func: CEC2013Function, X: ArrayLike, index: int) -> ArrayLike:
    xp = get_array_namespace(X)
    matrix = xp.asarray(func._get_rotation_matrix(index), dtype=X.dtype)
    return X @ matrix.T


def _shift_with_index(func: CEC2013Function, x: np.ndarray, index: int) -> np.ndarray:
    return x - func._get_shift_vector(index)


def _batch_shift_with_index(func: CEC2013Function, X: ArrayLike, index: int) -> ArrayLike:
    xp = get_array_namespace(X)
    shift = xp.asarray(func._get_shift_vector(index), dtype=X.dtype)
    return X - shift


def _sphere_core(z: np.ndarray) -> float:
    return float(np.sum(z**2))


def _elliptic_core(z: np.ndarray) -> float:
    D = len(z)
    if D <= 1:
        return float(z[0] ** 2)
    idx = np.arange(D, dtype=float)
    coeffs = np.power(1e6, idx / (D - 1))
    return float(np.sum(coeffs * z**2))


def _different_powers_core(z: np.ndarray) -> float:
    D = len(z)
    if D <= 1:
        return float(abs(z[0]) ** 2)
    idx = np.arange(D, dtype=float)
    exponents = 2 + 4 * idx / (D - 1)
    return float(np.sqrt(np.sum(np.abs(z) ** exponents)))


def _bent_cigar_core(z: np.ndarray) -> float:
    if len(z) <= 1:
        return float(z[0] ** 2)
    return float(z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2))


def _discus_core(z: np.ndarray) -> float:
    if len(z) <= 1:
        return float(1e6 * z[0] ** 2)
    return float(1e6 * z[0] ** 2 + np.sum(z[1:] ** 2))


def _rosenbrock_core(z: np.ndarray) -> float:
    return float(np.sum(100 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1) ** 2))


def _schaffer_f7_core(z: np.ndarray) -> float:
    S = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
    result = np.sum(np.sqrt(S) * (np.sin(50 * S**0.2) ** 2 + 1))
    return float((result / (len(z) - 1)) ** 2)


def _ackley_core(z: np.ndarray) -> float:
    D = len(z)
    sum1 = np.sum(z**2)
    sum2 = np.sum(np.cos(2 * np.pi * z))
    return float(-20 * np.exp(-0.2 * np.sqrt(sum1 / D)) - np.exp(sum2 / D) + 20 + np.e)


def _weierstrass_core(z: np.ndarray) -> float:
    a = 0.5
    b = 3
    k_max = 20
    result = 0.0
    for zi in z:
        for k in range(k_max + 1):
            result += a**k * np.cos(2 * np.pi * b**k * (zi + 0.5))
    offset = len(z) * sum(a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max + 1))
    return float(result - offset)


def _griewank_core(z: np.ndarray) -> float:
    D = len(z)
    return float(np.sum(z**2) / 4000 - np.prod(np.cos(z / np.sqrt(np.arange(1, D + 1)))) + 1)


def _rastrigin_core(z: np.ndarray) -> float:
    return float(np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10))


def _schwefel_core(z: np.ndarray) -> float:
    D = len(z)
    total = 0.0
    for zi in z:
        if zi > 500:
            mod = 500 - zi % 500
            total -= mod * np.sin(np.sqrt(abs(mod)))
            total += ((zi - 500) / 100) ** 2 / D
        elif zi < -500:
            mod = abs(zi) % 500 - 500
            total -= mod * np.sin(np.sqrt(abs(mod)))
            total += ((zi + 500) / 100) ** 2 / D
        else:
            total -= zi * np.sin(np.sqrt(abs(zi)))
    return float(_SCHWEFEL_OFFSET * D + total)


def _katsuura_core(z: np.ndarray) -> float:
    D = len(z)
    result = 1.0
    for i, zi in enumerate(z):
        inner_sum = 0.0
        for j in range(1, 33):
            pow2 = 2**j
            inner_sum += abs(pow2 * zi - round(pow2 * zi)) / pow2
        result *= (1 + (i + 1) * inner_sum) ** (10 / (D**1.2))
    return float((10 / D**2) * result - (10 / D**2))


def _expanded_griewank_rosenbrock_core(z: np.ndarray) -> float:
    z_next = np.roll(z, -1)
    t = 100 * (z**2 - z_next) ** 2 + (z - 1) ** 2
    return float(np.sum(t**2 / 4000 - np.cos(t) + 1))


def _expanded_schaffer_f6_core(z: np.ndarray) -> float:
    z_next = np.roll(z, -1)
    t = z**2 + z_next**2
    return float(np.sum(0.5 + (np.sin(np.sqrt(t)) ** 2 - 0.5) / (1 + 0.001 * t) ** 2))


def _batch_sphere_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    return xp.sum(Z**2, axis=1)


def _batch_elliptic_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    if D <= 1:
        return Z[:, 0] ** 2
    idx = xp.arange(D, dtype=Z.dtype)
    coeffs = xp.power(1e6, idx / (D - 1))
    return xp.sum(coeffs * Z**2, axis=1)


def _batch_different_powers_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    if D <= 1:
        return xp.abs(Z[:, 0]) ** 2
    idx = xp.arange(D, dtype=Z.dtype)
    exponents = 2 + 4 * idx / (D - 1)
    return xp.sqrt(xp.sum(xp.abs(Z) ** exponents, axis=1))


def _batch_bent_cigar_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    if Z.shape[1] <= 1:
        return Z[:, 0] ** 2
    return Z[:, 0] ** 2 + 1e6 * xp.sum(Z[:, 1:] ** 2, axis=1)


def _batch_discus_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    if Z.shape[1] <= 1:
        return 1e6 * Z[:, 0] ** 2
    return 1e6 * Z[:, 0] ** 2 + xp.sum(Z[:, 1:] ** 2, axis=1)


def _batch_rosenbrock_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    return xp.sum(100 * (Z[:, :-1] ** 2 - Z[:, 1:]) ** 2 + (Z[:, :-1] - 1) ** 2, axis=1)


def _batch_schaffer_f7_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    S = xp.sqrt(Z[:, :-1] ** 2 + Z[:, 1:] ** 2)
    result = xp.sum(xp.sqrt(S) * (xp.sin(50 * S**0.2) ** 2 + 1), axis=1)
    return (result / (D - 1)) ** 2


def _batch_ackley_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    sum1 = xp.sum(Z**2, axis=1)
    sum2 = xp.sum(xp.cos(2 * math.pi * Z), axis=1)
    return -20 * xp.exp(-0.2 * xp.sqrt(sum1 / D)) - xp.exp(sum2 / D) + 20 + math.e


def _batch_weierstrass_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    a = 0.5
    b = 3
    k_max = 20
    D = Z.shape[1]
    k = xp.arange(k_max + 1, dtype=Z.dtype)
    a_k = a**k
    b_k = b**k
    cos_terms = a_k * xp.cos(2 * math.pi * b_k * (Z[:, :, None] + 0.5))
    offset = D * xp.sum(a_k * xp.cos(2 * math.pi * b_k * 0.5))
    return xp.sum(cos_terms, axis=(1, 2)) - offset


def _batch_griewank_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    idx = xp.sqrt(xp.arange(1, D + 1, dtype=Z.dtype))
    return xp.sum(Z**2, axis=1) / 4000 - xp.prod(xp.cos(Z / idx), axis=1) + 1


def _batch_rastrigin_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    return xp.sum(Z**2 - 10 * xp.cos(2 * math.pi * Z) + 10, axis=1)


def _batch_schwefel_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    term_inside = -Z * xp.sin(xp.sqrt(xp.abs(Z)))

    mod_pos = 500 - xp.mod(Z, 500)
    term_pos = -mod_pos * xp.sin(xp.sqrt(xp.abs(mod_pos))) + ((Z - 500) / 100) ** 2 / D

    mod_neg = xp.mod(xp.abs(Z), 500) - 500
    term_neg = -mod_neg * xp.sin(xp.sqrt(xp.abs(mod_neg))) + ((Z + 500) / 100) ** 2 / D

    terms = xp.where(Z > 500, term_pos, xp.where(Z < -500, term_neg, term_inside))
    return _SCHWEFEL_OFFSET * D + xp.sum(terms, axis=1)


def _batch_katsuura_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    D = Z.shape[1]
    j = xp.arange(1, 33, dtype=Z.dtype)
    pow2 = xp.power(2.0, j)
    scaled = Z[:, :, None] * pow2
    inner = xp.sum(xp.abs(scaled - xp.round(scaled)) / pow2, axis=2)
    idx = xp.arange(1, D + 1, dtype=Z.dtype)
    result = xp.prod((1 + idx * inner) ** (10 / (D**1.2)), axis=1)
    return (10 / D**2) * result - (10 / D**2)


def _batch_expanded_griewank_rosenbrock_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    Z_next = xp.roll(Z, -1, axis=1)
    T = 100 * (Z**2 - Z_next) ** 2 + (Z - 1) ** 2
    return xp.sum(T**2 / 4000 - xp.cos(T) + 1, axis=1)


def _batch_expanded_schaffer_f6_core(Z: ArrayLike) -> ArrayLike:
    xp = get_array_namespace(Z)
    Z_next = xp.roll(Z, -1, axis=1)
    T = Z**2 + Z_next**2
    return xp.sum(0.5 + (xp.sin(xp.sqrt(T)) ** 2 - 0.5) / (1 + 0.001 * T) ** 2, axis=1)


def _component_value(
    func: CEC2013Function,
    x: np.ndarray,
    component: str,
    shift_index: int,
    rotation_index: int = 1,
) -> float:
    z = _component_z(func, x, component, shift_index, rotation_index)
    return _COMPONENT_CORES[component](z)


def _batch_component_value(
    func: CEC2013Function,
    X: ArrayLike,
    component: str,
    shift_index: int,
    rotation_index: int = 1,
) -> ArrayLike:
    Z = _batch_component_z(func, X, component, shift_index, rotation_index)
    return _BATCH_COMPONENT_CORES[component](Z)


def _component_z(
    func: CEC2013Function,
    x: np.ndarray,
    component: str,
    shift_index: int,
    rotation_index: int = 1,
) -> np.ndarray:
    if component == "sphere":
        return _shift_with_index(func, x, shift_index)
    if component == "elliptic":
        z = _rotate_with_index(func, _shift_with_index(func, x, shift_index), rotation_index)
        return _osz_endpoints(z)
    if component == "bent_cigar":
        z = _rotate_with_index(func, _shift_with_index(func, x, shift_index), rotation_index)
        z = func._asymmetric(z, 0.5)
        return _rotate_with_index(func, z, rotation_index + 1)
    if component == "discus":
        z = _rotate_with_index(func, _shift_with_index(func, x, shift_index), rotation_index)
        return _osz_endpoints(z)
    if component == "different_powers":
        return _rotate_with_index(func, _shift_with_index(func, x, shift_index), rotation_index)
    if component == "rosenbrock":
        z = _rotate_with_index(
            func, _shift_with_index(func, x, shift_index) * 2.048 / 100, rotation_index
        )
        return z + 1
    if component == "schaffer_f7":
        z = _rotate_with_index(func, _shift_with_index(func, x, shift_index), rotation_index)
        z = func._asymmetric(z, 0.5)
        z = _lambda_scale_np(z, 10)
        return _rotate_with_index(func, z, rotation_index + 1)
    if component == "ackley":
        z = _rotate_with_index(func, _shift_with_index(func, x, shift_index), rotation_index)
        z = func._asymmetric(z, 0.5)
        z = _lambda_scale_np(z, 10)
        return _rotate_with_index(func, z, rotation_index + 1)
    if component == "weierstrass":
        z = _rotate_with_index(
            func, _shift_with_index(func, x, shift_index) * 0.5 / 100, rotation_index
        )
        z = func._asymmetric(z, 0.5)
        z = _lambda_scale_np(z, 10)
        return _rotate_with_index(func, z, rotation_index + 1)
    if component == "griewank":
        z = _rotate_with_index(
            func, _shift_with_index(func, x, shift_index) * 600 / 100, rotation_index
        )
        return _lambda_scale_np(z, 100)
    if component == "rastrigin":
        z = _shift_with_index(func, x, shift_index) * 5.12 / 100
        z = _osz_endpoints(z)
        z = func._asymmetric(z, 0.2)
        return _lambda_scale_np(z, 10)
    if component == "rotated_rastrigin":
        z = _rotate_with_index(
            func, _shift_with_index(func, x, shift_index) * 5.12 / 100, rotation_index
        )
        z = _osz_endpoints(z)
        z = func._asymmetric(z, 0.2)
        z = _rotate_with_index(func, z, rotation_index + 1)
        z = _lambda_scale_np(z, 10)
        return _rotate_with_index(func, z, rotation_index)
    if component == "step_rastrigin":
        z = _rotate_with_index(
            func, _shift_with_index(func, x, shift_index) * 5.12 / 100, rotation_index
        )
        z = np.where(np.abs(z) > 0.5, np.floor(2 * z + 0.5) / 2, z)
        z = _osz_endpoints(z)
        z = func._asymmetric(z, 0.2)
        z = _rotate_with_index(func, z, rotation_index + 1)
        z = _lambda_scale_np(z, 10)
        return _rotate_with_index(func, z, rotation_index)
    if component == "schwefel":
        z = _shift_with_index(func, x, shift_index) * 1000 / 100
        z = _lambda_scale_np(z, 10)
        return z + _SCHWEFEL_OPTIMUM
    if component == "rotated_schwefel":
        z = _rotate_with_index(
            func, _shift_with_index(func, x, shift_index) * 1000 / 100, rotation_index
        )
        z = _lambda_scale_np(z, 10)
        return z + _SCHWEFEL_OPTIMUM
    if component == "katsuura":
        z = _rotate_with_index(
            func, _shift_with_index(func, x, shift_index) * 5 / 100, rotation_index
        )
        z = _lambda_scale_np(z, 100)
        return _rotate_with_index(func, z, rotation_index + 1)
    if component == "expanded_griewank_rosenbrock":
        z = _rotate_with_index(
            func, _shift_with_index(func, x, shift_index) * 5 / 100, rotation_index
        )
        return z + 1
    if component == "expanded_schaffer_f6":
        z = _rotate_with_index(func, _shift_with_index(func, x, shift_index), rotation_index)
        z = func._asymmetric(z, 0.5)
        return _rotate_with_index(func, z, rotation_index + 1)
    raise ValueError(f"Unknown CEC2013 component: {component}")


def _batch_component_z(
    func: CEC2013Function,
    X: ArrayLike,
    component: str,
    shift_index: int,
    rotation_index: int = 1,
) -> ArrayLike:
    xp = get_array_namespace(X)
    if component == "sphere":
        return _batch_shift_with_index(func, X, shift_index)
    if component == "elliptic":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index), rotation_index
        )
        return _batch_osz_endpoints(Z)
    if component == "bent_cigar":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index), rotation_index
        )
        Z = func._batch_asymmetric(Z, 0.5)
        return _batch_rotate_with_index(func, Z, rotation_index + 1)
    if component == "discus":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index), rotation_index
        )
        return _batch_osz_endpoints(Z)
    if component == "different_powers":
        return _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index), rotation_index
        )
    if component == "rosenbrock":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index) * 2.048 / 100, rotation_index
        )
        return Z + 1
    if component == "schaffer_f7":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index), rotation_index
        )
        Z = func._batch_asymmetric(Z, 0.5)
        Z = _batch_lambda_scale_xp(Z, 10)
        return _batch_rotate_with_index(func, Z, rotation_index + 1)
    if component == "ackley":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index), rotation_index
        )
        Z = func._batch_asymmetric(Z, 0.5)
        Z = _batch_lambda_scale_xp(Z, 10)
        return _batch_rotate_with_index(func, Z, rotation_index + 1)
    if component == "weierstrass":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index) * 0.5 / 100, rotation_index
        )
        Z = func._batch_asymmetric(Z, 0.5)
        Z = _batch_lambda_scale_xp(Z, 10)
        return _batch_rotate_with_index(func, Z, rotation_index + 1)
    if component == "griewank":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index) * 600 / 100, rotation_index
        )
        return _batch_lambda_scale_xp(Z, 100)
    if component == "rastrigin":
        Z = _batch_shift_with_index(func, X, shift_index) * 5.12 / 100
        Z = _batch_osz_endpoints(Z)
        Z = func._batch_asymmetric(Z, 0.2)
        return _batch_lambda_scale_xp(Z, 10)
    if component == "rotated_rastrigin":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index) * 5.12 / 100, rotation_index
        )
        Z = _batch_osz_endpoints(Z)
        Z = func._batch_asymmetric(Z, 0.2)
        Z = _batch_rotate_with_index(func, Z, rotation_index + 1)
        Z = _batch_lambda_scale_xp(Z, 10)
        return _batch_rotate_with_index(func, Z, rotation_index)
    if component == "step_rastrigin":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index) * 5.12 / 100, rotation_index
        )
        Z = xp.where(xp.abs(Z) > 0.5, xp.floor(2 * Z + 0.5) / 2, Z)
        Z = _batch_osz_endpoints(Z)
        Z = func._batch_asymmetric(Z, 0.2)
        Z = _batch_rotate_with_index(func, Z, rotation_index + 1)
        Z = _batch_lambda_scale_xp(Z, 10)
        return _batch_rotate_with_index(func, Z, rotation_index)
    if component == "schwefel":
        Z = _batch_shift_with_index(func, X, shift_index) * 1000 / 100
        Z = _batch_lambda_scale_xp(Z, 10)
        return Z + _SCHWEFEL_OPTIMUM
    if component == "rotated_schwefel":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index) * 1000 / 100, rotation_index
        )
        Z = _batch_lambda_scale_xp(Z, 10)
        return Z + _SCHWEFEL_OPTIMUM
    if component == "katsuura":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index) * 5 / 100, rotation_index
        )
        Z = _batch_lambda_scale_xp(Z, 100)
        return _batch_rotate_with_index(func, Z, rotation_index + 1)
    if component == "expanded_griewank_rosenbrock":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index) * 5 / 100, rotation_index
        )
        return Z + 1
    if component == "expanded_schaffer_f6":
        Z = _batch_rotate_with_index(
            func, _batch_shift_with_index(func, X, shift_index), rotation_index
        )
        Z = func._batch_asymmetric(Z, 0.5)
        return _batch_rotate_with_index(func, Z, rotation_index + 1)
    raise ValueError(f"Unknown CEC2013 component: {component}")


_COMPONENT_CORES = {
    "sphere": _sphere_core,
    "elliptic": _elliptic_core,
    "bent_cigar": _bent_cigar_core,
    "discus": _discus_core,
    "different_powers": _different_powers_core,
    "rosenbrock": _rosenbrock_core,
    "schaffer_f7": _schaffer_f7_core,
    "ackley": _ackley_core,
    "weierstrass": _weierstrass_core,
    "griewank": _griewank_core,
    "rastrigin": _rastrigin_core,
    "rotated_rastrigin": _rastrigin_core,
    "step_rastrigin": _rastrigin_core,
    "schwefel": _schwefel_core,
    "rotated_schwefel": _schwefel_core,
    "katsuura": _katsuura_core,
    "expanded_griewank_rosenbrock": _expanded_griewank_rosenbrock_core,
    "expanded_schaffer_f6": _expanded_schaffer_f6_core,
}


_BATCH_COMPONENT_CORES = {
    "sphere": _batch_sphere_core,
    "elliptic": _batch_elliptic_core,
    "bent_cigar": _batch_bent_cigar_core,
    "discus": _batch_discus_core,
    "different_powers": _batch_different_powers_core,
    "rosenbrock": _batch_rosenbrock_core,
    "schaffer_f7": _batch_schaffer_f7_core,
    "ackley": _batch_ackley_core,
    "weierstrass": _batch_weierstrass_core,
    "griewank": _batch_griewank_core,
    "rastrigin": _batch_rastrigin_core,
    "rotated_rastrigin": _batch_rastrigin_core,
    "step_rastrigin": _batch_rastrigin_core,
    "schwefel": _batch_schwefel_core,
    "rotated_schwefel": _batch_schwefel_core,
    "katsuura": _batch_katsuura_core,
    "expanded_griewank_rosenbrock": _batch_expanded_griewank_rosenbrock_core,
    "expanded_schaffer_f6": _batch_expanded_schaffer_f6_core,
}


def _lunacek_bi_rastrigin_value(
    func: CEC2013Function,
    x: np.ndarray,
    shift_index: int,
    rotated: bool,
) -> float:
    D = len(x)
    mu0 = 2.5
    d = 1.0
    s = 1 - 1 / (2 * np.sqrt(D + 20) - 8.2)
    mu1 = -np.sqrt((mu0**2 - d) / s)
    shift = func._get_shift_vector(shift_index)

    y = (x - shift) * 10 / 100
    x_hat = 2 * np.where(shift < 0, -1.0, 1.0) * y + mu0
    z = x_hat - mu0
    if rotated:
        z = _rotate_with_index(func, z, 1)
        z = _lambda_scale_np(z, 100)
        z = _rotate_with_index(func, z, 2)
    else:
        z = _lambda_scale_np(z, 100)

    sum1 = np.sum((x_hat - mu0) ** 2)
    sum2 = np.sum((x_hat - mu1) ** 2)
    sum3 = np.sum(np.cos(2 * np.pi * z))
    return float(min(sum1, d * D + s * sum2) + 10 * (D - sum3))


def _batch_lunacek_bi_rastrigin_value(
    func: CEC2013Function,
    X: ArrayLike,
    shift_index: int,
    rotated: bool,
) -> ArrayLike:
    xp = get_array_namespace(X)
    D = X.shape[1]
    mu0 = 2.5
    d = 1.0
    s = 1 - 1 / (2 * math.sqrt(D + 20) - 8.2)
    mu1 = -math.sqrt((mu0**2 - d) / s)
    shift = xp.asarray(func._get_shift_vector(shift_index), dtype=X.dtype)

    Y = (X - shift) * 10 / 100
    X_hat = 2 * xp.where(shift < 0, -1.0, 1.0) * Y + mu0
    Z = X_hat - mu0
    if rotated:
        Z = _batch_rotate_with_index(func, Z, 1)
        Z = _batch_lambda_scale_xp(Z, 100)
        Z = _batch_rotate_with_index(func, Z, 2)
    else:
        Z = _batch_lambda_scale_xp(Z, 100)

    sum1 = xp.sum((X_hat - mu0) ** 2, axis=1)
    sum2 = xp.sum((X_hat - mu1) ** 2, axis=1)
    sum3 = xp.sum(xp.cos(2 * math.pi * Z), axis=1)
    return xp.minimum(sum1, d * D + s * sum2) + 10 * (D - sum3)


class Sphere(CEC2013Function):
    """F1: Sphere Function.

    Properties:
    - Unimodal
    - Separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = False

    _spec = {
        "eval_cost": 0.8,
        "name": "Sphere Function",
        "func_id": 1,
        "unimodal": True,
        "convex": True,
        "separable": True,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        z = self._shift(x)
        return np.sum(z**2) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)
        Z = self._batch_shift(X, self.shift_index)
        return xp.sum(Z**2, axis=1) + self.f_global


class RotatedHighConditionedElliptic(CEC2013Function):
    """F2: Rotated High Conditioned Elliptic Function.

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 2.6,
        "name": "Rotated High Conditioned Elliptic Function",
        "func_id": 2,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "elliptic", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return _batch_component_value(self, X, "elliptic", self.shift_index) + self.f_global


class RotatedBentCigar(CEC2013Function):
    """F3: Rotated Bent Cigar Function.

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 2.0,
        "name": "Rotated Bent Cigar Function",
        "func_id": 3,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "bent_cigar", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return _batch_component_value(self, X, "bent_cigar", self.shift_index) + self.f_global


class RotatedDiscus(CEC2013Function):
    """F4: Rotated Discus Function.

    Properties:
    - Unimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 2.9,
        "name": "Rotated Discus Function",
        "func_id": 4,
        "unimodal": True,
        "convex": True,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "discus", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return _batch_component_value(self, X, "discus", self.shift_index) + self.f_global


class DifferentPowers(CEC2013Function):
    """F5: Different Powers Function.

    Properties:
    - Unimodal
    - Separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = False

    _spec = {
        "eval_cost": 0.8,
        "name": "Different Powers Function",
        "func_id": 5,
        "unimodal": True,
        "separable": True,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        z = _shift_with_index(self, x, self.shift_index)
        return _different_powers_core(z) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        Z = _batch_shift_with_index(self, X, self.shift_index)
        return _batch_different_powers_core(Z) + self.f_global


class RotatedRosenbrock(CEC2013Function):
    """F6: Rotated Rosenbrock's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 1.6,
        "name": "Rotated Rosenbrock's Function",
        "func_id": 6,
        "unimodal": False,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "rosenbrock", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return _batch_component_value(self, X, "rosenbrock", self.shift_index) + self.f_global


class RotatedSchafferF7(CEC2013Function):
    """F7: Rotated Schaffer's F7 Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 3.0,
        "name": "Rotated Schaffer's F7 Function",
        "func_id": 7,
        "unimodal": False,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "schaffer_f7", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return _batch_component_value(self, X, "schaffer_f7", self.shift_index) + self.f_global


class RotatedAckley(CEC2013Function):
    """F8: Rotated Ackley's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 2.7,
        "name": "Rotated Ackley's Function",
        "func_id": 8,
        "unimodal": False,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "ackley", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return _batch_component_value(self, X, "ackley", self.shift_index) + self.f_global


class RotatedWeierstrass(CEC2013Function):
    """F9: Rotated Weierstrass Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    - Continuous but not differentiable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 15.8,
        "name": "Rotated Weierstrass Function",
        "func_id": 9,
        "unimodal": False,
        "separable": False,
        "differentiable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "weierstrass", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return _batch_component_value(self, X, "weierstrass", self.shift_index) + self.f_global


class RotatedGriewank(CEC2013Function):
    """F10: Rotated Griewank's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 1.8,
        "name": "Rotated Griewank's Function",
        "func_id": 10,
        "unimodal": False,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "griewank", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return _batch_component_value(self, X, "griewank", self.shift_index) + self.f_global


class Rastrigin(CEC2013Function):
    """F11: Rastrigin's Function (non-rotated).

    Properties:
    - Highly multimodal
    - Separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = False

    _spec = {
        "eval_cost": 3.4,
        "name": "Rastrigin's Function",
        "func_id": 11,
        "unimodal": False,
        "separable": True,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "rastrigin", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return _batch_component_value(self, X, "rastrigin", self.shift_index) + self.f_global


class RotatedRastrigin(CEC2013Function):
    """F12: Rotated Rastrigin's Function.

    Properties:
    - Highly multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 4.3,
        "name": "Rotated Rastrigin's Function",
        "func_id": 12,
        "unimodal": False,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "rotated_rastrigin", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return (
            _batch_component_value(self, X, "rotated_rastrigin", self.shift_index) + self.f_global
        )


class StepRastrigin(CEC2013Function):
    """F13: Non-Continuous Rotated Rastrigin's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Non-continuous
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 6.2,
        "name": "Non-Continuous Rotated Rastrigin's Function",
        "func_id": 13,
        "unimodal": False,
        "separable": False,
        "continuous": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "step_rastrigin", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return _batch_component_value(self, X, "step_rastrigin", self.shift_index) + self.f_global


class Schwefel(CEC2013Function):
    """F14: Schwefel's Function (non-rotated).

    Properties:
    - Multimodal
    - Separable
    - Deceptive
    """

    shift_index = 1
    uses_rotation = False

    _spec = {
        "eval_cost": 1.5,
        "name": "Schwefel's Function",
        "func_id": 14,
        "unimodal": False,
        "separable": True,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "schwefel", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return _batch_component_value(self, X, "schwefel", self.shift_index) + self.f_global


class RotatedSchwefel(CEC2013Function):
    """F15: Rotated Schwefel's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Deceptive
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 2.0,
        "name": "Rotated Schwefel's Function",
        "func_id": 15,
        "unimodal": False,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "rotated_schwefel", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return _batch_component_value(self, X, "rotated_schwefel", self.shift_index) + self.f_global


class RotatedKatsuura(CEC2013Function):
    """F16: Rotated Katsuura Function.

    Properties:
    - Multimodal
    - Non-separable
    - Non-differentiable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 42.3,
        "name": "Rotated Katsuura Function",
        "func_id": 16,
        "unimodal": False,
        "separable": False,
        "differentiable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "katsuura", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return _batch_component_value(self, X, "katsuura", self.shift_index) + self.f_global


class LunacekBiRastrigin(CEC2013Function):
    """F17: Lunacek Bi-Rastrigin Function (non-rotated).

    Properties:
    - Multimodal
    - Separable
    - Two global optima
    """

    shift_index = 1
    uses_rotation = False

    _spec = {
        "eval_cost": 2.3,
        "name": "Lunacek Bi-Rastrigin Function",
        "func_id": 17,
        "unimodal": False,
        "separable": True,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _lunacek_bi_rastrigin_value(self, x, self.shift_index, rotated=False) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return (
            _batch_lunacek_bi_rastrigin_value(self, X, self.shift_index, rotated=False)
            + self.f_global
        )


class RotatedLunacekBiRastrigin(CEC2013Function):
    """F18: Rotated Lunacek Bi-Rastrigin Function.

    Properties:
    - Multimodal
    - Non-separable
    - Two global optima
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 2.9,
        "name": "Rotated Lunacek Bi-Rastrigin Function",
        "func_id": 18,
        "unimodal": False,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _lunacek_bi_rastrigin_value(self, x, self.shift_index, rotated=True) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return (
            _batch_lunacek_bi_rastrigin_value(self, X, self.shift_index, rotated=True)
            + self.f_global
        )


class RotatedExpandedGriewankRosenbrock(CEC2013Function):
    """F19: Rotated Expanded Griewank's plus Rosenbrock's Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 2.0,
        "name": "Rotated Expanded Griewank's plus Rosenbrock's Function",
        "func_id": 19,
        "unimodal": False,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        z = self._shift_rotate(x)
        z = z * 5 / 100 + 1

        D = self.n_dim
        result = 0.0
        for i in range(D - 1):
            t = 100 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
            result += t**2 / 4000 - np.cos(t) + 1

        t = 100 * (z[-1] ** 2 - z[0]) ** 2 + (z[-1] - 1) ** 2
        result += t**2 / 4000 - np.cos(t) + 1

        return result + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        xp = get_array_namespace(X)

        Z = self._batch_shift_rotate(X)
        Z = Z * 5 / 100 + 1

        # z_next = roll(z, -1) for wrap-around: z_next[-1] = z[0]
        Z_next = xp.roll(Z, -1, axis=1)

        # t = 100 * (z^2 - z_next)^2 + (z - 1)^2
        T = 100 * (Z**2 - Z_next) ** 2 + (Z - 1) ** 2

        # Griewank transformation: t^2/4000 - cos(t) + 1
        result = xp.sum(T**2 / 4000 - xp.cos(T) + 1, axis=1)

        return result + self.f_global


class RotatedExpandedScafferF6(CEC2013Function):
    """F20: Rotated Expanded Scaffer's F6 Function.

    Properties:
    - Multimodal
    - Non-separable
    - Scalable
    """

    shift_index = 1
    uses_rotation = True

    _spec = {
        "eval_cost": 2.9,
        "name": "Rotated Expanded Scaffer's F6 Function",
        "func_id": 20,
        "unimodal": False,
        "separable": False,
    }

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        return _component_value(self, x, "expanded_schaffer_f6", self.shift_index) + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        """Vectorized batch evaluation."""
        return (
            _batch_component_value(self, X, "expanded_schaffer_f6", self.shift_index)
            + self.f_global
        )


class _CompositionBase(CEC2013Function):
    """Base class for CEC 2013 composition functions."""

    _spec = {
        "unimodal": False,
        "separable": False,
    }

    n_functions: int = 0
    sigmas: list = []
    lambdas: list = []
    biases: list = []
    component_functions: list = []

    @property
    def x_global(self) -> np.ndarray:
        """Global optimum location, defined by the lowest-bias component."""
        return self._get_shift_vector(1)

    def _component_optima(self) -> np.ndarray:
        return np.stack([self._get_shift_vector(i + 1) for i in range(self.n_functions)])

    def _compute_weights(self, x: np.ndarray) -> np.ndarray:
        """Compute report-specified composition weights."""
        optima = self._component_optima()
        diff = x[None, :] - optima
        dist_sq = np.sum(diff**2, axis=1)
        exact = np.flatnonzero(dist_sq == 0)
        if len(exact) > 0:
            weights = np.zeros(self.n_functions)
            weights[exact[0]] = 1.0
            return weights

        sigmas = np.asarray(self.sigmas, dtype=float)
        weights = (1.0 / np.sqrt(dist_sq)) * np.exp(-dist_sq / (2 * self.n_dim * sigmas**2))
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return np.ones(self.n_functions) / self.n_functions
        return weights / weight_sum

    def _batch_compute_weights(self, X: ArrayLike, optima: ArrayLike = None) -> ArrayLike:
        """Compute report-specified composition weights for a batch."""
        xp = get_array_namespace(X)
        if optima is None:
            optima = xp.asarray(self._component_optima(), dtype=X.dtype)

        diff = X[:, None, :] - optima[None, :, :]
        dist_sq = xp.sum(diff**2, axis=2)
        exact = dist_sq == 0
        has_exact = xp.any(exact, axis=1, keepdims=True)

        sigmas = xp.asarray(self.sigmas, dtype=X.dtype)
        safe_dist_sq = xp.where(exact, 1.0, dist_sq)
        weights = (1.0 / xp.sqrt(safe_dist_sq)) * xp.exp(-dist_sq / (2 * self.n_dim * sigmas**2))
        weight_sum = xp.sum(weights, axis=1, keepdims=True)
        normalized = xp.where(weight_sum == 0, 1.0 / self.n_functions, weights / weight_sum)

        exact_sum = xp.sum(exact, axis=1, keepdims=True)
        exact_sum = xp.where(has_exact, exact_sum, 1)
        one_hot = exact / exact_sum
        return xp.where(has_exact, one_hot, normalized)

    def _objective(self, params: Dict[str, Any]) -> float:
        x = self._params_to_array(params)
        weights = self._compute_weights(x)
        result = 0.0
        for i, component in enumerate(self.component_functions):
            value = _component_value(self, x, component, i + 1, i + 1)
            result += weights[i] * (self.lambdas[i] * value + self.biases[i])
        return result + self.f_global

    def _batch_objective(self, X: ArrayLike) -> ArrayLike:
        xp = get_array_namespace(X)
        optima = xp.asarray(self._component_optima(), dtype=X.dtype)
        weights = self._batch_compute_weights(X, optima)
        result = xp.zeros(X.shape[0], dtype=X.dtype)
        lambdas = xp.asarray(self.lambdas, dtype=X.dtype)
        biases = xp.asarray(self.biases, dtype=X.dtype)
        for i, component in enumerate(self.component_functions):
            values = _batch_component_value(self, X, component, i + 1, i + 1)
            result = result + weights[:, i] * (lambdas[i] * values + biases[i])
        return result + self.f_global


class CompositionFunction1(_CompositionBase):
    """F21: Composition Function 1."""

    _spec = {
        "eval_cost": 7.7,
        "name": "Composition Function 1",
        "func_id": 21,
    }

    n_functions = 5
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [1, 1e-6, 1e-26, 1e-6, 0.1]
    biases = [0, 100, 200, 300, 400]
    component_functions = ["rosenbrock", "different_powers", "bent_cigar", "discus", "sphere"]


class CompositionFunction2(_CompositionBase):
    """F22: Composition Function 2."""

    _spec = {
        "eval_cost": 5.0,
        "name": "Composition Function 2",
        "func_id": 22,
    }

    n_functions = 3
    sigmas = [20, 20, 20]
    lambdas = [1, 1, 1]
    biases = [0, 100, 200]
    component_functions = ["schwefel", "schwefel", "schwefel"]


class CompositionFunction3(_CompositionBase):
    """F23: Composition Function 3."""

    _spec = {
        "eval_cost": 7.0,
        "name": "Composition Function 3",
        "func_id": 23,
    }

    n_functions = 3
    sigmas = [20, 20, 20]
    lambdas = [1, 1, 1]
    biases = [0, 100, 200]
    component_functions = ["rotated_schwefel", "rotated_schwefel", "rotated_schwefel"]


class CompositionFunction4(_CompositionBase):
    """F24: Composition Function 4."""

    _spec = {
        "eval_cost": 18.7,
        "name": "Composition Function 4",
        "func_id": 24,
    }

    n_functions = 3
    sigmas = [20, 20, 20]
    lambdas = [0.25, 1, 2.5]
    biases = [0, 100, 200]
    component_functions = ["rotated_schwefel", "rotated_rastrigin", "weierstrass"]


class CompositionFunction5(_CompositionBase):
    """F25: Composition Function 5."""

    _spec = {
        "eval_cost": 6.8,
        "name": "Composition Function 5",
        "func_id": 25,
    }

    n_functions = 3
    sigmas = [10, 30, 50]
    lambdas = [0.25, 1, 2.5]
    biases = [0, 100, 200]
    component_functions = ["rotated_schwefel", "rotated_rastrigin", "weierstrass"]


class CompositionFunction6(_CompositionBase):
    """F26: Composition Function 6."""

    _spec = {
        "eval_cost": 21.7,
        "name": "Composition Function 6",
        "func_id": 26,
    }

    n_functions = 5
    sigmas = [10, 10, 10, 10, 10]
    lambdas = [0.25, 1, 1e-7, 2.5, 10]
    biases = [0, 100, 200, 300, 400]
    component_functions = [
        "rotated_schwefel",
        "rotated_rastrigin",
        "elliptic",
        "weierstrass",
        "griewank",
    ]


class CompositionFunction7(_CompositionBase):
    """F27: Composition Function 7."""

    _spec = {
        "eval_cost": 21.7,
        "name": "Composition Function 7",
        "func_id": 27,
    }

    n_functions = 5
    sigmas = [10, 10, 10, 20, 20]
    lambdas = [100, 10, 2.5, 25, 0.1]
    biases = [0, 100, 200, 300, 400]
    component_functions = [
        "griewank",
        "rotated_rastrigin",
        "rotated_schwefel",
        "weierstrass",
        "sphere",
    ]


class CompositionFunction8(_CompositionBase):
    """F28: Composition Function 8."""

    _spec = {
        "eval_cost": 9.9,
        "name": "Composition Function 8",
        "func_id": 28,
    }

    n_functions = 5
    sigmas = [10, 20, 30, 40, 50]
    lambdas = [2.5, 2.5e-3, 2.5, 5e-4, 0.1]
    biases = [0, 100, 200, 300, 400]
    component_functions = [
        "expanded_griewank_rosenbrock",
        "schaffer_f7",
        "rotated_schwefel",
        "expanded_schaffer_f6",
        "sphere",
    ]
