"""Numerical gradient utilities for checking analytical backward passes.

Numerical differentiation is intentionally slow: each input coordinate needs
two forward evaluations. These helpers are therefore debugging tools for small
arrays, not optimization algorithms for training models.
"""

from collections.abc import Callable

import numpy as np


def _as_floating_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return a finite real-valued array copied into floating-point storage."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.astype(np.float64, copy=True)


def _validate_step(step: float) -> float:
    """Return a finite, positive numerical-difference step."""
    if isinstance(step, (bool, np.bool_)) or not isinstance(
        step, (int, float, np.number)
    ):
        raise TypeError("step must be numeric")
    step = float(step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    return step


def _scalar_function_value(value: object) -> float:
    """Validate and return one finite scalar function value."""
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError("f must return a scalar")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError("f must return a real numeric scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError("f must return a finite scalar")
    return result


def eval_numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    *,
    step: float = 1e-5,
) -> np.ndarray:
    """Estimate the gradient of a scalar-valued function at ``x``.

    Each coordinate uses the centered-difference approximation
    ``(f(x + h) - f(x - h)) / (2h)`` while every other coordinate remains
    fixed. Calculations use a floating-point copy, so ``x`` is not mutated.

    Args:
        f: Function that accepts an array shaped like ``x`` and returns one
            finite scalar.
        x: Point at which to estimate the gradient. It may have any shape.
        step: Finite, positive perturbation size ``h``.

    Returns:
        Numerical gradient with the same shape as ``x`` and dtype ``float64``.
    """
    if not callable(f):
        raise TypeError("f must be callable")
    working = _as_floating_array(x, name="x")
    step = _validate_step(step)
    gradient = np.empty_like(working)

    for index in np.ndindex(working.shape):
        original = working[index]
        working[index] = original + step
        value_plus = _scalar_function_value(f(working))
        working[index] = original - step
        value_minus = _scalar_function_value(f(working))
        working[index] = original
        gradient[index] = (value_plus - value_minus) / (2 * step)

    return gradient


def eval_numerical_gradient_array(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    upstream_gradient: np.ndarray,
    *,
    step: float = 1e-5,
) -> np.ndarray:
    """Estimate an input gradient for an array-valued function.

    If ``out = f(x)`` and ``upstream_gradient = dL/dout``, this function
    numerically estimates ``dL/dx``. For each input coordinate it computes the
    centered change in every output coordinate and takes its elementwise inner
    product with the upstream gradient.

    Args:
        f: Function mapping an array shaped like ``x`` to a finite real array.
        x: Input point whose gradient is being checked.
        upstream_gradient: Analytical upstream gradient with the same shape as
            ``f(x)``.
        step: Finite, positive perturbation size ``h``.

    Returns:
        Numerical input gradient with the shape of ``x`` and dtype ``float64``.
    """
    if not callable(f):
        raise TypeError("f must be callable")
    working = _as_floating_array(x, name="x")
    upstream = _as_floating_array(upstream_gradient, name="upstream_gradient")
    step = _validate_step(step)

    initial_output = _as_floating_array(f(working), name="f(x)")
    if initial_output.shape != upstream.shape:
        raise ValueError("upstream_gradient must have the same shape as f(x)")

    gradient = np.empty_like(working)
    for index in np.ndindex(working.shape):
        original = working[index]
        working[index] = original + step
        output_plus = _as_floating_array(f(working), name="f(x + step)")
        working[index] = original - step
        output_minus = _as_floating_array(f(working), name="f(x - step)")
        working[index] = original

        if output_plus.shape != upstream.shape or output_minus.shape != upstream.shape:
            raise ValueError("f must return the same shape for every perturbation")
        local_change = (output_plus - output_minus) / (2 * step)
        gradient[index] = np.sum(local_change * upstream)

    return gradient


def relative_error(
    analytical: np.ndarray,
    numerical: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> float:
    """Return the maximum scale-aware difference between two gradients.

    The denominator is ``max(epsilon, |analytical| + |numerical|)`` for each
    coordinate. ``epsilon`` keeps two values near zero from causing division
    by zero or an unhelpfully large relative error.
    """
    analytical_array = _as_floating_array(analytical, name="analytical")
    numerical_array = _as_floating_array(numerical, name="numerical")
    if analytical_array.shape != numerical_array.shape:
        raise ValueError("analytical and numerical gradients must have equal shapes")
    if analytical_array.size == 0:
        raise ValueError("gradients must not be empty")
    epsilon = _validate_step(epsilon)

    numerator = np.abs(analytical_array - numerical_array)
    denominator = np.maximum(
        epsilon,
        np.abs(analytical_array) + np.abs(numerical_array),
    )
    return float(np.max(numerator / denominator))
