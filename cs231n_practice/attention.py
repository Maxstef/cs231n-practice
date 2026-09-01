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
MultiHeadAttentionCache = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    AttentionCache,
    int,
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


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """Convert a sequence from ``(N, L, D)`` to ``(N, H, L, D_h)``.

    This operation only rearranges values. It does not calculate the learned
    query, key, or value projections that normally precede the split.
    """
    x = _as_real_array(x, name="x")
    if x.ndim != 3:
        raise ValueError("x must have shape (N, L, D)")
    if isinstance(num_heads, (bool, np.bool_)) or not isinstance(
        num_heads, (int, np.integer)
    ):
        raise TypeError("num_heads must be an integer")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if x.shape[-1] % num_heads != 0:
        raise ValueError("feature dimension must be divisible by num_heads")

    batch_size, sequence_length, model_dim = x.shape
    head_dim = model_dim // num_heads
    return x.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(
        0, 2, 1, 3
    )


def merge_heads(x: np.ndarray) -> np.ndarray:
    """Convert ``(N, H, L, D_h)`` back to ``(N, L, H * D_h)``."""
    x = _as_real_array(x, name="x")
    if x.ndim != 4:
        raise ValueError("x must have shape (N, H, L, D_h)")
    batch_size, num_heads, sequence_length, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(
        batch_size,
        sequence_length,
        num_heads * head_dim,
    )


def multi_head_attention_forward(
    query_input: np.ndarray,
    key_input: np.ndarray,
    value_input: np.ndarray,
    query_weights: np.ndarray,
    key_weights: np.ndarray,
    value_weights: np.ndarray,
    output_weights: np.ndarray,
    query_bias: np.ndarray,
    key_bias: np.ndarray,
    value_bias: np.ndarray,
    output_bias: np.ndarray,
    num_heads: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, MultiHeadAttentionCache]:
    """Compute general multi-head attention with learned projections.

    The three inputs have shapes ``(N, L_q, D_q)``, ``(N, L_k, D_k_in)``,
    and ``(N, L_k, D_v_in)``. Their projection matrices must all produce the
    same model dimension ``D``, which is divided equally among the heads.
    ``output_weights`` may map that concatenated dimension to any output
    dimension. Passing the same array as all three inputs gives self-attention.

    Returns:
        Output ``(N, L_q, D_out)``, weights ``(N, H, L_q, L_k)``, and a
        cache for :func:`multi_head_attention_backward`.
    """
    query_input = _as_real_array(query_input, name="query_input")
    key_input = _as_real_array(key_input, name="key_input")
    value_input = _as_real_array(value_input, name="value_input")
    query_weights = _as_real_array(query_weights, name="query_weights")
    key_weights = _as_real_array(key_weights, name="key_weights")
    value_weights = _as_real_array(value_weights, name="value_weights")
    output_weights = _as_real_array(output_weights, name="output_weights")
    query_bias = _as_real_array(query_bias, name="query_bias")
    key_bias = _as_real_array(key_bias, name="key_bias")
    value_bias = _as_real_array(value_bias, name="value_bias")
    output_bias = _as_real_array(output_bias, name="output_bias")

    inputs = (query_input, key_input, value_input)
    projections = (query_weights, key_weights, value_weights, output_weights)
    biases = (query_bias, key_bias, value_bias, output_bias)
    if any(array.ndim != 3 for array in inputs):
        raise ValueError("attention inputs must be three-dimensional")
    if any(array.ndim != 2 for array in projections):
        raise ValueError("projection weights must be two-dimensional")
    if any(array.ndim != 1 for array in biases):
        raise ValueError("projection biases must be one-dimensional")
    if not (query_input.shape[0] == key_input.shape[0] == value_input.shape[0]):
        raise ValueError("attention inputs must have the same batch size")
    if key_input.shape[1] != value_input.shape[1]:
        raise ValueError("key and value sequence lengths must match")

    model_dim = query_weights.shape[1]
    if (
        query_weights.shape[0] != query_input.shape[2]
        or key_weights.shape != (key_input.shape[2], model_dim)
        or value_weights.shape != (value_input.shape[2], model_dim)
    ):
        raise ValueError("input and Q/K/V projection shapes are incompatible")
    if (
        query_bias.shape != (model_dim,)
        or key_bias.shape != (model_dim,)
        or value_bias.shape != (model_dim,)
    ):
        raise ValueError("Q/K/V biases must match the projected model dimension")
    if output_weights.shape[0] != model_dim:
        raise ValueError("output projection input must match the model dimension")
    if output_bias.shape != (output_weights.shape[1],):
        raise ValueError("output bias must match the output projection")

    query = query_input @ query_weights + query_bias
    key = key_input @ key_weights + key_bias
    value = value_input @ value_weights + value_bias
    query_heads = split_heads(query, num_heads)
    key_heads = split_heads(key, num_heads)
    value_heads = split_heads(value, num_heads)

    head_output, attention_weights, attention_cache = (
        scaled_dot_product_attention_forward(
            query_heads,
            key_heads,
            value_heads,
            mask,
        )
    )
    merged_output = merge_heads(head_output)
    output = merged_output @ output_weights + output_bias
    cache = (
        query_input,
        key_input,
        value_input,
        query_weights,
        key_weights,
        value_weights,
        output_weights,
        merged_output,
        attention_cache,
        num_heads,
    )
    return output, attention_weights, cache


def multi_head_attention_backward(
    dout: np.ndarray,
    cache: MultiHeadAttentionCache,
) -> tuple[np.ndarray, ...]:
    """Backpropagate through projections and multi-head attention.

    Returns gradients in the order ``dquery_input``, ``dkey_input``,
    ``dvalue_input``, ``dquery_weights``, ``dkey_weights``,
    ``dvalue_weights``, ``doutput_weights``, ``dquery_bias``, ``dkey_bias``,
    ``dvalue_bias``, and ``doutput_bias``. For self-attention, add the first
    three gradients because they are paths to the same original input.
    """
    (
        query_input,
        key_input,
        value_input,
        query_weights,
        key_weights,
        value_weights,
        output_weights,
        merged_output,
        attention_cache,
        num_heads,
    ) = cache
    dout = _as_real_array(dout, name="dout")
    expected_shape = (
        query_input.shape[0],
        query_input.shape[1],
        output_weights.shape[1],
    )
    if dout.shape != expected_shape:
        raise ValueError(f"dout must have shape {expected_shape}")

    doutput_weights = (
        merged_output.reshape(-1, merged_output.shape[-1]).T
        @ dout.reshape(-1, dout.shape[-1])
    )
    doutput_bias = dout.sum(axis=(0, 1))
    dmerged_output = dout @ output_weights.T
    dhead_output = split_heads(dmerged_output, num_heads)
    dquery_heads, dkey_heads, dvalue_heads = (
        scaled_dot_product_attention_backward(dhead_output, attention_cache)
    )
    dquery = merge_heads(dquery_heads)
    dkey = merge_heads(dkey_heads)
    dvalue = merge_heads(dvalue_heads)

    dquery_weights = (
        query_input.reshape(-1, query_input.shape[-1]).T
        @ dquery.reshape(-1, dquery.shape[-1])
    )
    dkey_weights = (
        key_input.reshape(-1, key_input.shape[-1]).T
        @ dkey.reshape(-1, dkey.shape[-1])
    )
    dvalue_weights = (
        value_input.reshape(-1, value_input.shape[-1]).T
        @ dvalue.reshape(-1, dvalue.shape[-1])
    )
    dquery_bias = dquery.sum(axis=(0, 1))
    dkey_bias = dkey.sum(axis=(0, 1))
    dvalue_bias = dvalue.sum(axis=(0, 1))
    dquery_input = dquery @ query_weights.T
    dkey_input = dkey @ key_weights.T
    dvalue_input = dvalue @ value_weights.T

    return (
        dquery_input,
        dkey_input,
        dvalue_input,
        dquery_weights,
        dkey_weights,
        dvalue_weights,
        doutput_weights,
        dquery_bias,
        dkey_bias,
        dvalue_bias,
        doutput_bias,
    )
