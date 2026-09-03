"""Reusable NumPy building blocks for pre-norm Transformer encoders."""

import numpy as np

from cs231n_practice.attention import (
    MultiHeadAttentionCache,
    multi_head_attention_backward,
    multi_head_attention_forward,
)
from cs231n_practice.normalization import (
    LayerNormCache,
    layernorm_backward,
    layernorm_forward,
)

FeedForwardCache = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
TransformerEncoderCache = tuple[
    LayerNormCache,
    MultiHeadAttentionCache,
    LayerNormCache,
    FeedForwardCache,
]


def _as_real_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return a nonempty, finite array containing real numeric values."""
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


def feed_forward_forward(
    x: np.ndarray,
    weights1: np.ndarray,
    bias1: np.ndarray,
    weights2: np.ndarray,
    bias2: np.ndarray,
) -> tuple[np.ndarray, FeedForwardCache]:
    """Apply an affine-ReLU-affine network independently to every token."""
    x = _as_real_array(x, name="x")
    weights1 = _as_real_array(weights1, name="weights1")
    bias1 = _as_real_array(bias1, name="bias1")
    weights2 = _as_real_array(weights2, name="weights2")
    bias2 = _as_real_array(bias2, name="bias2")
    if x.ndim != 3:
        raise ValueError("x must have shape (N, L, D)")
    if weights1.ndim != 2 or weights2.ndim != 2:
        raise ValueError("feed-forward weights must be two-dimensional")
    if bias1.ndim != 1 or bias2.ndim != 1:
        raise ValueError("feed-forward biases must be one-dimensional")
    if weights1.shape[0] != x.shape[-1] or bias1.shape != (weights1.shape[1],):
        raise ValueError("first feed-forward projection has incompatible shapes")
    if weights2.shape != (weights1.shape[1], x.shape[-1]):
        raise ValueError("second feed-forward projection must return model width")
    if bias2.shape != (x.shape[-1],):
        raise ValueError("second feed-forward bias must match model width")

    calculation_dtype = np.result_type(
        x.dtype,
        weights1.dtype,
        bias1.dtype,
        weights2.dtype,
        bias2.dtype,
        np.float32,
    )
    x = x.astype(calculation_dtype, copy=False)
    weights1 = weights1.astype(calculation_dtype, copy=False)
    bias1 = bias1.astype(calculation_dtype, copy=False)
    weights2 = weights2.astype(calculation_dtype, copy=False)
    bias2 = bias2.astype(calculation_dtype, copy=False)
    pre_activation = x @ weights1 + bias1
    hidden = np.maximum(0.0, pre_activation)
    output = hidden @ weights2 + bias2
    return output, (x, weights1, pre_activation, weights2)


def feed_forward_backward(
    dout: np.ndarray,
    cache: FeedForwardCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return gradients for the position-wise feed-forward network."""
    x, weights1, pre_activation, weights2 = cache
    dout = _as_real_array(dout, name="dout")
    if dout.shape != x.shape:
        raise ValueError("dout must match the feed-forward output shape")

    hidden = np.maximum(0.0, pre_activation)
    flat_dout = dout.reshape(-1, dout.shape[-1])
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    dweights2 = flat_hidden.T @ flat_dout
    dbias2 = dout.sum(axis=(0, 1))
    dhidden = dout @ weights2.T
    dpre_activation = dhidden * (pre_activation > 0.0)
    flat_x = x.reshape(-1, x.shape[-1])
    flat_dpre = dpre_activation.reshape(-1, dpre_activation.shape[-1])
    dweights1 = flat_x.T @ flat_dpre
    dbias1 = dpre_activation.sum(axis=(0, 1))
    dx = dpre_activation @ weights1.T
    return dx, dweights1, dbias1, dweights2, dbias2


def residual_add(x: np.ndarray, branch_output: np.ndarray) -> np.ndarray:
    """Add a same-shaped sublayer update to its residual stream."""
    x = _as_real_array(x, name="x")
    branch_output = _as_real_array(branch_output, name="branch_output")
    if x.shape != branch_output.shape:
        raise ValueError("x and branch_output must have the same shape")
    return x + branch_output


def transformer_encoder_block_forward(
    x: np.ndarray,
    parameters: dict[str, np.ndarray],
    num_heads: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, TransformerEncoderCache]:
    """Calculate one pre-norm Transformer encoder block.

    Positional information should already be present in ``x``. The block uses
    layer-normalization/self-attention/residual followed by
    layer-normalization/feed-forward/residual.
    """
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dictionary")
    required = {
        "gamma1", "beta1", "W_query", "W_key", "W_value", "W_output",
        "b_query", "b_key", "b_value", "b_output", "gamma2", "beta2",
        "W1", "b1", "W2", "b2",
    }
    missing = required - parameters.keys()
    if missing:
        raise ValueError(f"missing Transformer parameters: {sorted(missing)}")

    normalized1, layernorm1_cache = layernorm_forward(
        x, parameters["gamma1"], parameters["beta1"]
    )
    attention_update, attention_weights, attention_cache = (
        multi_head_attention_forward(
            normalized1,
            normalized1,
            normalized1,
            parameters["W_query"],
            parameters["W_key"],
            parameters["W_value"],
            parameters["W_output"],
            parameters["b_query"],
            parameters["b_key"],
            parameters["b_value"],
            parameters["b_output"],
            num_heads,
            mask,
        )
    )
    after_attention = residual_add(x, attention_update)
    normalized2, layernorm2_cache = layernorm_forward(
        after_attention, parameters["gamma2"], parameters["beta2"]
    )
    feed_forward_update, feed_forward_cache = feed_forward_forward(
        normalized2,
        parameters["W1"],
        parameters["b1"],
        parameters["W2"],
        parameters["b2"],
    )
    output = residual_add(after_attention, feed_forward_update)
    cache = (
        layernorm1_cache,
        attention_cache,
        layernorm2_cache,
        feed_forward_cache,
    )
    return output, attention_weights, cache


def transformer_encoder_block_backward(
    dout: np.ndarray,
    cache: TransformerEncoderCache,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Backpropagate through one pre-norm Transformer encoder block."""
    layernorm1_cache, attention_cache, layernorm2_cache, feed_forward_cache = cache
    dout = _as_real_array(dout, name="dout")

    # Second residual: one path is direct and one enters the FFN branch.
    dafter_attention = dout.copy()
    (
        dnormalized2,
        dW1,
        db1,
        dW2,
        db2,
    ) = feed_forward_backward(dout, feed_forward_cache)
    dlayernorm2_input, dgamma2, dbeta2 = layernorm_backward(
        dnormalized2, layernorm2_cache
    )
    dafter_attention += dlayernorm2_input

    # First residual: again preserve the direct path and backpropagate through
    # all three self-attention roles of the shared normalized input.
    dx = dafter_attention.copy()
    attention_gradients = multi_head_attention_backward(
        dafter_attention, attention_cache
    )
    (
        dquery_input,
        dkey_input,
        dvalue_input,
        dW_query,
        dW_key,
        dW_value,
        dW_output,
        db_query,
        db_key,
        db_value,
        db_output,
    ) = attention_gradients
    dnormalized1 = dquery_input + dkey_input + dvalue_input
    dlayernorm1_input, dgamma1, dbeta1 = layernorm_backward(
        dnormalized1, layernorm1_cache
    )
    dx += dlayernorm1_input

    gradients = {
        "gamma1": dgamma1,
        "beta1": dbeta1,
        "W_query": dW_query,
        "W_key": dW_key,
        "W_value": dW_value,
        "W_output": dW_output,
        "b_query": db_query,
        "b_key": db_key,
        "b_value": db_value,
        "b_output": db_output,
        "gamma2": dgamma2,
        "beta2": dbeta2,
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }
    return dx, gradients
