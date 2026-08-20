"""Reusable NumPy layers for fully connected neural networks.

Each forward function returns both its output and a cache. The corresponding
backward function consumes that cache together with an upstream gradient and
returns gradients shaped like the forward inputs.

This module uses the neural-layer weight convention: inputs have shape
``(N, D)``, weights have shape ``(D, M)``, and affine outputs have shape
``(N, M)``.
"""

import numpy as np

AffineCache = tuple[np.ndarray, np.ndarray]
ReLUCache = np.ndarray
AffineReLUCache = tuple[AffineCache, ReLUCache]


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


def affine_forward(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, AffineCache]:
    """Compute a batched affine transformation ``x @ weights + bias``.

    Args:
        x: Input matrix with shape ``(N, D)``.
        weights: Weight matrix with shape ``(D, M)``.
        bias: Output bias with shape ``(M,)``.

    Returns:
        A pair containing the output with shape ``(N, M)`` and a cache for
        :func:`affine_backward`.
    """
    x = _as_real_array(x, name="x")
    weights = _as_real_array(weights, name="weights")
    bias = _as_real_array(bias, name="bias")

    if x.ndim != 2 or weights.ndim != 2:
        raise ValueError("x and weights must be two-dimensional")
    if bias.ndim != 1:
        raise ValueError("bias must be one-dimensional")
    if x.shape[1] != weights.shape[0]:
        raise ValueError("x and weights have incompatible feature dimensions")
    if weights.shape[1] != bias.shape[0]:
        raise ValueError("weights and bias have incompatible output dimensions")

    calculation_dtype = np.result_type(
        x.dtype,
        weights.dtype,
        bias.dtype,
        np.float32,
    )
    x = x.astype(calculation_dtype, copy=False)
    weights = weights.astype(calculation_dtype, copy=False)
    bias = bias.astype(calculation_dtype, copy=False)

    output = x @ weights + bias
    cache = (x, weights)
    return output, cache


def affine_backward(
    dout: np.ndarray,
    cache: AffineCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through a batched affine transformation.

    Args:
        dout: Upstream gradient with shape ``(N, M)``.
        cache: The input and weights returned by :func:`affine_forward`.

    Returns:
        Gradients ``(dx, dweights, dbias)`` with shapes ``(N, D)``, ``(D, M)``,
        and ``(M,)`` respectively.
    """
    if not isinstance(cache, tuple) or len(cache) != 2:
        raise TypeError("cache must be the pair returned by affine_forward")
    x = _as_real_array(cache[0], name="cached x")
    weights = _as_real_array(cache[1], name="cached weights")
    dout = _as_real_array(dout, name="dout")

    if x.ndim != 2 or weights.ndim != 2 or dout.ndim != 2:
        raise ValueError("cached x, cached weights, and dout must be two-dimensional")
    expected_shape = (x.shape[0], weights.shape[1])
    if x.shape[1] != weights.shape[0] or dout.shape != expected_shape:
        raise ValueError("dout shape is incompatible with the affine cache")

    calculation_dtype = np.result_type(x.dtype, weights.dtype, dout.dtype, np.float32)
    x = x.astype(calculation_dtype, copy=False)
    weights = weights.astype(calculation_dtype, copy=False)
    dout = dout.astype(calculation_dtype, copy=False)

    dx = dout @ weights.T
    dweights = x.T @ dout
    dbias = dout.sum(axis=0)
    return dx, dweights, dbias


def relu_forward(x: np.ndarray) -> tuple[np.ndarray, ReLUCache]:
    """Apply ReLU elementwise and cache its pre-activation input."""
    x = _as_real_array(x, name="x")
    calculation_dtype = np.result_type(x.dtype, np.float32)
    x = x.astype(calculation_dtype, copy=False)
    return np.maximum(0, x), x


def relu_backward(dout: np.ndarray, cache: ReLUCache) -> np.ndarray:
    """Pass gradients where the cached ReLU input is strictly positive.

    The derivative at exactly zero follows the project convention of zero.
    Neither ``dout`` nor the cached pre-activation is mutated.
    """
    dout = _as_real_array(dout, name="dout")
    pre_activation = _as_real_array(cache, name="ReLU cache")
    if dout.shape != pre_activation.shape:
        raise ValueError("dout and the ReLU cache must have the same shape")

    calculation_dtype = np.result_type(dout.dtype, pre_activation.dtype, np.float32)
    dout = dout.astype(calculation_dtype, copy=False)
    return dout * (pre_activation > 0)


def affine_relu_forward(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, AffineReLUCache]:
    """Compute an affine transformation followed by elementwise ReLU."""
    pre_activation, affine_cache = affine_forward(x, weights, bias)
    output, relu_cache = relu_forward(pre_activation)
    return output, (affine_cache, relu_cache)


def affine_relu_backward(
    dout: np.ndarray,
    cache: AffineReLUCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through ReLU and then through its affine input layer."""
    if not isinstance(cache, tuple) or len(cache) != 2:
        raise TypeError("cache must be the pair returned by affine_relu_forward")
    affine_cache, relu_cache = cache
    dpre_activation = relu_backward(dout, relu_cache)
    return affine_backward(dpre_activation, affine_cache)


def softmax_loss(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return stable mean cross-entropy and its score gradient.

    Args:
        scores: Class scores with shape ``(N, C)``.
        labels: Integer target indices with shape ``(N,)``.

    Returns:
        Mean data loss and ``dL/dscores`` with shape ``(N, C)``.
    """
    scores = _as_real_array(scores, name="scores")
    labels = np.asarray(labels)
    if scores.ndim != 2:
        raise ValueError("scores must be two-dimensional")
    if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
        raise ValueError("labels must provide one target per score row")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels must be integers")
    if np.any((labels < 0) | (labels >= scores.shape[1])):
        raise ValueError("labels must be valid class indices")

    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    sums = exp_scores.sum(axis=1, keepdims=True)
    probabilities = exp_scores / sums
    correct_shifted = shifted[np.arange(scores.shape[0]), labels]
    loss = np.mean(np.log(sums[:, 0]) - correct_shifted)

    dscores = probabilities.copy()
    dscores[np.arange(scores.shape[0]), labels] -= 1.0
    dscores /= scores.shape[0]
    return float(loss), dscores
