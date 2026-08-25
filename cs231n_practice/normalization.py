"""Reusable NumPy batch-normalization and dropout layers.

Batch normalization in this module operates on feature matrices with shape
``(N, D)``. Dropout is elementwise and therefore accepts arrays of any shape.
Forward functions return caches consumed by their matching backward functions.
"""

from typing import Any

import numpy as np

BatchNormCache = tuple[np.ndarray, np.ndarray, np.ndarray]
DropoutCache = tuple[str, np.ndarray | None]


def _as_real_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return a nonempty, finite, real numeric NumPy array."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _real_scalar(value: float, *, name: str) -> float:
    """Return a finite real scalar as a Python float."""
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real number")
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not np.isfinite(probability):
        raise ValueError(f"{name} must be finite")
    return probability


def batchnorm_forward(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    state: dict[str, Any],
) -> tuple[np.ndarray, BatchNormCache | None]:
    """Normalize a feature matrix and apply a learned scale and shift.

    Args:
        x: Activations with shape ``(N, D)``.
        gamma: Learned per-feature scales with shape ``(D,)``.
        beta: Learned per-feature shifts with shape ``(D,)``.
        state: Mutable configuration containing ``mode`` (``"train"`` or
            ``"test"``) and optional ``epsilon``, ``momentum``,
            ``running_mean``, and ``running_variance`` values.

    Returns:
        The normalized output and a training cache. The cache is ``None`` in
        test mode because no backward pass should be needed for inference.

    Notes:
        Training mutates ``state`` by updating its running mean and variance.
    """
    x = _as_real_array(x, name="x")
    gamma = _as_real_array(gamma, name="gamma")
    beta = _as_real_array(beta, name="beta")
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional")
    if gamma.ndim != 1 or beta.ndim != 1:
        raise ValueError("gamma and beta must be one-dimensional")
    if gamma.shape != (x.shape[1],) or beta.shape != (x.shape[1],):
        raise ValueError("gamma and beta must contain one value per feature")
    if not isinstance(state, dict):
        raise TypeError("state must be a dictionary")

    mode = state.get("mode", "train")
    epsilon = _real_scalar(state.get("epsilon", 1e-5), name="epsilon")
    momentum = _real_scalar(state.get("momentum", 0.9), name="momentum")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must be in [0, 1]")
    calculation_dtype = np.result_type(
        x.dtype, gamma.dtype, beta.dtype, np.float32
    )
    x = x.astype(calculation_dtype, copy=False)
    gamma = gamma.astype(calculation_dtype, copy=False)
    beta = beta.astype(calculation_dtype, copy=False)

    default_statistics = np.zeros(x.shape[1], dtype=calculation_dtype)
    running_mean = _as_real_array(
        state.get("running_mean", default_statistics), name="running_mean"
    ).astype(calculation_dtype, copy=False)
    running_variance = _as_real_array(
        state.get("running_variance", default_statistics),
        name="running_variance",
    ).astype(calculation_dtype, copy=False)
    if running_mean.shape != (x.shape[1],) or running_variance.shape != (
        x.shape[1],
    ):
        raise ValueError("running statistics must contain one value per feature")
    if np.any(running_variance < 0):
        raise ValueError("running_variance must be nonnegative")

    if mode == "train":
        mean = x.mean(axis=0)
        variance = x.var(axis=0)
        inverse_std = 1.0 / np.sqrt(variance + epsilon)
        normalized = (x - mean) * inverse_std
        output = gamma * normalized + beta
        state["running_mean"] = momentum * running_mean + (1.0 - momentum) * mean
        state["running_variance"] = (
            momentum * running_variance + (1.0 - momentum) * variance
        )
        return output, (normalized, gamma, inverse_std)

    if mode == "test":
        normalized = (x - running_mean) / np.sqrt(running_variance + epsilon)
        return gamma * normalized + beta, None

    raise ValueError("mode must be 'train' or 'test'")


def batchnorm_backward(
    dout: np.ndarray,
    cache: BatchNormCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through training-mode batch normalization."""
    if not isinstance(cache, tuple) or len(cache) != 3:
        raise TypeError("cache must be the tuple returned in training mode")
    normalized = _as_real_array(cache[0], name="cached normalized")
    gamma = _as_real_array(cache[1], name="cached gamma")
    inverse_std = _as_real_array(cache[2], name="cached inverse_std")
    dout = _as_real_array(dout, name="dout")
    if normalized.ndim != 2 or dout.shape != normalized.shape:
        raise ValueError("dout must match the cached normalized shape")
    if gamma.shape != (normalized.shape[1],) or inverse_std.shape != (
        normalized.shape[1],
    ):
        raise ValueError("cached feature values have incompatible shapes")

    gradient_dtype = np.result_type(
        dout.dtype, normalized.dtype, gamma.dtype, inverse_std.dtype, np.float32
    )
    dout = dout.astype(gradient_dtype, copy=False)
    normalized = normalized.astype(gradient_dtype, copy=False)
    gamma = gamma.astype(gradient_dtype, copy=False)
    inverse_std = inverse_std.astype(gradient_dtype, copy=False)

    num_examples = dout.shape[0]
    dbeta = dout.sum(axis=0)
    dgamma = (dout * normalized).sum(axis=0)
    dx = (gamma * inverse_std / num_examples) * (
        num_examples * dout
        - dout.sum(axis=0)
        - normalized * (dout * normalized).sum(axis=0)
    )
    return dx, dgamma, dbeta


def dropout_forward(
    x: np.ndarray,
    *,
    keep_probability: float,
    mode: str,
    generator: np.random.Generator,
) -> tuple[np.ndarray, DropoutCache]:
    """Apply inverted dropout while preserving input shape and expectation."""
    x = _as_real_array(x, name="x")
    keep_probability = _real_scalar(keep_probability, name="keep_probability")
    if not 0.0 < keep_probability <= 1.0:
        raise ValueError("keep_probability must be in (0, 1]")
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be a NumPy random Generator")
    calculation_dtype = np.result_type(x.dtype, np.float32)
    x = x.astype(calculation_dtype, copy=False)

    if mode == "train":
        mask = (generator.random(x.shape) < keep_probability).astype(
            calculation_dtype
        ) / keep_probability
        return x * mask, (mode, mask)
    if mode == "test":
        return x.copy(), (mode, None)
    raise ValueError("mode must be 'train' or 'test'")


def dropout_backward(dout: np.ndarray, cache: DropoutCache) -> np.ndarray:
    """Backpropagate through the cached inverted-dropout operation."""
    dout = _as_real_array(dout, name="dout")
    if not isinstance(cache, tuple) or len(cache) != 2:
        raise TypeError("cache must be the tuple returned by dropout_forward")
    mode, mask = cache
    if mode == "train":
        if not isinstance(mask, np.ndarray) or mask.shape != dout.shape:
            raise ValueError("cached mask must have the same shape as dout")
        return dout * mask
    if mode == "test":
        if mask is not None:
            raise ValueError("test-mode cache must not contain a mask")
        return dout.copy()
    raise ValueError("cached mode must be 'train' or 'test'")
