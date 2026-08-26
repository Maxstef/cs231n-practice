"""Reusable residual blocks built from the project's affine layers.

The block in this module is deliberately small and educational. It applies an
``affine -> ReLU -> affine`` residual branch and adds an identity shortcut. It
operates on feature matrices rather than convolutional feature maps.
"""

import numpy as np

from cs231n_practice.layers import (
    AffineCache,
    AffineReLUCache,
    affine_backward,
    affine_forward,
    affine_relu_backward,
    affine_relu_forward,
)

ResidualCache = tuple[AffineReLUCache, AffineCache]


def residual_forward(
    x: np.ndarray,
    weights1: np.ndarray,
    bias1: np.ndarray,
    weights2: np.ndarray,
    bias2: np.ndarray,
) -> tuple[np.ndarray, ResidualCache]:
    """Apply an affine-ReLU-affine branch plus an identity shortcut.

    The residual branch computes

    ``residual = ReLU(x @ weights1 + bias1) @ weights2 + bias2``

    and the block returns ``x + residual``. Consequently, the second affine
    layer must produce the same feature dimension as ``x``.

    Args:
        x: Input feature matrix with shape ``(N, D)``.
        weights1: First-layer weights with shape ``(D, H)``.
        bias1: First-layer bias with shape ``(H,)``.
        weights2: Second-layer weights with shape ``(H, D)``.
        bias2: Second-layer bias with shape ``(D,)``.

    Returns:
        The block output with shape ``(N, D)`` and a cache for
        :func:`residual_backward`.

    Raises:
        ValueError: If the residual branch and identity shortcut shapes differ.
    """
    hidden, hidden_cache = affine_relu_forward(x, weights1, bias1)
    residual, residual_cache = affine_forward(hidden, weights2, bias2)
    if residual.shape != np.asarray(x).shape:
        raise ValueError(
            "residual branch output must have the same shape as the "
            "identity shortcut"
        )
    return x + residual, (hidden_cache, residual_cache)


def residual_backward(
    dout: np.ndarray,
    cache: ResidualCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through the residual branch and identity shortcut.

    The upstream gradient follows both paths. Backpropagation through the
    learned branch produces ``dx_branch``, while the identity shortcut
    contributes ``dout`` directly. The input gradient is therefore
    ``dx_branch + dout``.

    Returns:
        ``(dx, dweights1, dbias1, dweights2, dbias2)`` with shapes matching
        their corresponding forward inputs.
    """
    if not isinstance(cache, tuple) or len(cache) != 2:
        raise TypeError("cache must be the pair returned by residual_forward")
    hidden_cache, residual_cache = cache
    dhidden, dweights2, dbias2 = affine_backward(dout, residual_cache)
    dx_branch, dweights1, dbias1 = affine_relu_backward(
        dhidden, hidden_cache
    )
    dx = dx_branch + np.asarray(dout)
    return dx, dweights1, dbias1, dweights2, dbias2

