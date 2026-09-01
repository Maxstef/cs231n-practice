"""Reusable NumPy operations for scaled dot-product attention."""

import numpy as np

AttentionCache = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray | None,
]


def _as_real_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return a nonempty, finite array containing real numbers."""
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


def _broadcast_attention_mask(
    mask: np.ndarray,
    score_shape: tuple[int, ...],
) -> np.ndarray:
    """Return a Boolean mask broadcast to an attention score shape."""
    mask = np.asarray(mask)
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError("mask must contain Boolean values")
    try:
        broadcast_mask = np.broadcast_to(mask, score_shape)
    except ValueError as error:
        raise ValueError(f"mask is not broadcastable to score shape {score_shape}") from error
    if not np.all(np.any(broadcast_mask, axis=-1)):
        raise ValueError("every query must have at least one allowed key")
    return broadcast_mask


def _masked_softmax_with_mask(
    scores: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Calculate last-axis softmax and return its broadcast mask."""
    broadcast_mask = None
    if mask is not None:
        broadcast_mask = _broadcast_attention_mask(mask, scores.shape)
        scores = np.where(broadcast_mask, scores, -np.inf)

    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    weights = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    return weights, broadcast_mask


def masked_softmax(
    scores: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply stable softmax across keys, optionally excluding masked entries.

    Args:
        scores: Attention scores with shape ``(..., L_q, L_k)``.
        mask: Boolean array broadcastable to ``scores``. ``True`` marks an
            allowed query-key pair and ``False`` marks a forbidden pair.

    Returns:
        Nonnegative weights with the same shape as ``scores``. Each row along
        the final key axis sums to one, and masked entries are exactly zero.
    """
    scores = _as_real_array(scores, name="scores")
    if scores.ndim < 2:
        raise ValueError("scores must have shape (..., L_q, L_k)")
    weights, _ = _masked_softmax_with_mask(scores, mask)
    return weights


def scaled_dot_product_attention_forward(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, AttentionCache]:
    """Compute stable scaled dot-product attention.

    Shapes are ``query (..., L_q, D_k)``, ``key (..., L_k, D_k)``, and
    ``value (..., L_k, D_v)``. Leading dimensions must match. This supports
    both batched tensors ``(N, L, D)`` and future multi-head tensors
    ``(N, H_heads, L, D_head)``.

    Returns:
        Output with shape ``(..., L_q, D_v)``, attention weights with shape
        ``(..., L_q, L_k)``, and a cache for the matching backward function.
    """
    query = _as_real_array(query, name="query")
    key = _as_real_array(key, name="key")
    value = _as_real_array(value, name="value")
    if query.ndim < 2 or key.ndim != query.ndim or value.ndim != query.ndim:
        raise ValueError("query, key, and value must have matching rank >= 2")
    if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
        raise ValueError("query, key, and value leading dimensions must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key feature dimensions must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value sequence lengths must match")

    key_dim = query.shape[-1]
    scale = float(np.sqrt(key_dim))
    scores = (query @ np.swapaxes(key, -1, -2)) / scale
    weights, broadcast_mask = _masked_softmax_with_mask(scores, mask)
    output = weights @ value
    cache = (query, key, value, weights, scale, broadcast_mask)
    return output, weights, cache


def scaled_dot_product_attention_backward(
    dout: np.ndarray,
    cache: AttentionCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gradients with respect to query, key, and value."""
    query, key, value, weights, scale, mask = cache
    dout = _as_real_array(dout, name="dout")
    expected_shape = (*query.shape[:-2], query.shape[-2], value.shape[-1])
    if dout.shape != expected_shape:
        raise ValueError(f"dout must have shape {expected_shape}")

    dweights = dout @ np.swapaxes(value, -1, -2)
    dvalue = np.swapaxes(weights, -1, -2) @ dout

    softmax_correction = np.sum(
        dweights * weights,
        axis=-1,
        keepdims=True,
    )
    dscores = weights * (dweights - softmax_correction)
    if mask is not None:
        dscores = np.where(mask, dscores, 0.0)

    dquery = (dscores @ key) / scale
    dkey = (np.swapaxes(dscores, -1, -2) @ query) / scale
    return dquery, dkey, dvalue
