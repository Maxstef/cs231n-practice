"""Reusable NumPy layers that connect token sequences to sequence models.

Token IDs use shape ``(N, T)``. Embeddings use ``(N, T, D)``, hidden states
use ``(N, T, H)``, and vocabulary scores use ``(N, T, V)``.
"""

import numpy as np

EmbeddingCache = tuple[np.ndarray, tuple[int, int]]
TemporalAffineCache = tuple[np.ndarray, np.ndarray]


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


def _as_token_ids(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return a nonempty two-dimensional array of integer token IDs."""
    token_ids = np.asarray(value)
    if not np.issubdtype(token_ids.dtype, np.integer) or np.issubdtype(
        token_ids.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain integer token IDs")
    if token_ids.ndim != 2 or token_ids.size == 0:
        raise ValueError(f"{name} must have nonempty shape (N, T)")
    return token_ids


def embedding_forward(
    token_ids: np.ndarray,
    embedding_matrix: np.ndarray,
) -> tuple[np.ndarray, EmbeddingCache]:
    """Select one learned embedding vector for every token occurrence.

    Args:
        token_ids: Integer vocabulary indexes with shape ``(N, T)``.
        embedding_matrix: Learned vectors with shape ``(V, D)``.

    Returns:
        Embeddings with shape ``(N, T, D)`` and a backward cache.
    """
    token_ids = _as_token_ids(token_ids, name="token_ids")
    embedding_matrix = _as_real_array(embedding_matrix, name="embedding_matrix")
    if embedding_matrix.ndim != 2:
        raise ValueError("embedding_matrix must have shape (V, D)")

    vocabulary_size = embedding_matrix.shape[0]
    if np.any(token_ids < 0) or np.any(token_ids >= vocabulary_size):
        raise ValueError("token_ids contain an index outside the vocabulary")

    output = embedding_matrix[token_ids]
    cache = (token_ids, embedding_matrix.shape)
    return output, cache


def embedding_backward(dout: np.ndarray, cache: EmbeddingCache) -> np.ndarray:
    """Accumulate upstream gradients into the selected embedding rows."""
    token_ids, embedding_shape = cache
    dout = _as_real_array(dout, name="dout")
    expected_shape = (*token_ids.shape, embedding_shape[1])
    if dout.shape != expected_shape:
        raise ValueError(f"dout must have shape {expected_shape}")

    calculation_dtype = np.result_type(dout.dtype, np.float32)
    dembedding = np.zeros(embedding_shape, dtype=calculation_dtype)
    for n in range(token_ids.shape[0]):
        for t in range(token_ids.shape[1]):
            dembedding[token_ids[n, t]] += dout[n, t]
    return dembedding


def temporal_affine_forward(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, TemporalAffineCache]:
    """Apply one shared affine projection at every sequence position."""
    x = _as_real_array(x, name="x")
    weights = _as_real_array(weights, name="weights")
    bias = _as_real_array(bias, name="bias")
    if x.ndim != 3:
        raise ValueError("x must have shape (N, T, H)")
    if weights.ndim != 2 or bias.ndim != 1:
        raise ValueError("weights and bias must have shapes (H, V) and (V,)")

    num_examples, num_steps, hidden_dim = x.shape
    if weights.shape[0] != hidden_dim or bias.shape[0] != weights.shape[1]:
        raise ValueError("x, weights, and bias have incompatible dimensions")

    vocabulary_size = weights.shape[1]
    x_flat = x.reshape(num_examples * num_steps, hidden_dim)
    scores = (x_flat @ weights + bias).reshape(
        num_examples, num_steps, vocabulary_size
    )
    return scores, (x, weights)


def temporal_affine_backward(
    dout: np.ndarray,
    cache: TemporalAffineCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through a shared temporal affine projection."""
    x, weights = cache
    dout = _as_real_array(dout, name="dout")
    num_examples, num_steps, hidden_dim = x.shape
    vocabulary_size = weights.shape[1]
    expected_shape = (num_examples, num_steps, vocabulary_size)
    if dout.shape != expected_shape:
        raise ValueError(f"dout must have shape {expected_shape}")

    x_flat = x.reshape(num_examples * num_steps, hidden_dim)
    dout_flat = dout.reshape(num_examples * num_steps, vocabulary_size)
    dx = (dout_flat @ weights.T).reshape(x.shape)
    dweights = x_flat.T @ dout_flat
    dbias = np.sum(dout_flat, axis=0)
    return dx, dweights, dbias


def temporal_softmax_loss(
    scores: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return stable next-token loss and score gradients, averaged over ``N``.

    ``mask[n, t]`` is true for a real target and false for padding. Masked
    positions contribute neither loss nor gradient.
    """
    scores = _as_real_array(scores, name="scores")
    targets = _as_token_ids(targets, name="targets")
    mask = np.asarray(mask)
    if scores.ndim != 3:
        raise ValueError("scores must have shape (N, T, V)")
    if targets.shape != scores.shape[:2] or mask.shape != targets.shape:
        raise ValueError("targets and mask must match the (N, T) score axes")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError("mask must contain Boolean values")

    num_examples, num_steps, vocabulary_size = scores.shape
    if np.any(targets < 0) or np.any(targets >= vocabulary_size):
        raise ValueError("targets contain an index outside the vocabulary")

    scores_flat = scores.reshape(num_examples * num_steps, vocabulary_size)
    targets_flat = targets.reshape(num_examples * num_steps)
    mask_flat = mask.reshape(num_examples * num_steps)
    shifted = scores_flat - np.max(scores_flat, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    sums = np.sum(exp_scores, axis=1, keepdims=True)
    probabilities = exp_scores / sums

    rows = np.arange(scores_flat.shape[0])
    losses = np.log(sums[:, 0]) - shifted[rows, targets_flat]
    loss = np.sum(mask_flat * losses) / num_examples

    dscores_flat = probabilities
    dscores_flat[rows, targets_flat] -= 1.0
    dscores_flat *= mask_flat[:, None]
    dscores_flat /= num_examples
    return float(loss), dscores_flat.reshape(scores.shape)
