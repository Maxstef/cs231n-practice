"""Reusable positional representations for sequence models."""

import numpy as np

LearnedPositionCache = tuple[tuple[int, int, int], tuple[int, int]]


def _positive_integer(value: int, *, name: str) -> int:
    """Validate and return a positive Python-compatible integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


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


def sinusoidal_positional_encoding(
    sequence_length: int,
    model_dim: int,
) -> np.ndarray:
    """Return fixed sine/cosine encodings with shape ``(L, D)``.

    Even dimensions contain sine values and their following odd dimensions
    contain cosine values at the same frequency. The implementation supports
    both even and odd model dimensions.
    """
    sequence_length = _positive_integer(sequence_length, name="sequence_length")
    model_dim = _positive_integer(model_dim, name="model_dim")

    positions = np.arange(sequence_length, dtype=np.float64)[:, None]
    even_dimensions = np.arange(0, model_dim, 2, dtype=np.float64)
    angles = positions / np.power(10000.0, even_dimensions / model_dim)

    encoding = np.zeros((sequence_length, model_dim), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles)
    number_of_odd_dimensions = encoding[:, 1::2].shape[1]
    encoding[:, 1::2] = np.cos(angles[:, :number_of_odd_dimensions])
    return encoding


def add_positional_encoding(
    x: np.ndarray,
    encoding: np.ndarray | None = None,
) -> np.ndarray:
    """Add one positional vector to every batch example at each position.

    Args:
        x: Sequence representations with shape ``(N, L, D)``.
        encoding: Optional fixed encoding with shape ``(L, D)``. When omitted,
            a sinusoidal encoding is generated for the input shape.

    Returns:
        The element-wise sum with shape ``(N, L, D)``.
    """
    x = _as_real_array(x, name="x")
    if x.ndim != 3:
        raise ValueError("x must have shape (N, L, D)")
    if encoding is None:
        encoding = sinusoidal_positional_encoding(x.shape[1], x.shape[2])
    else:
        encoding = _as_real_array(encoding, name="encoding")
        if encoding.shape != x.shape[1:]:
            raise ValueError(f"encoding must have shape {x.shape[1:]}")

    calculation_dtype = np.result_type(x.dtype, encoding.dtype, np.float32)
    return x.astype(calculation_dtype, copy=False) + encoding.astype(
        calculation_dtype,
        copy=False,
    )


def learned_positional_embedding_forward(
    x: np.ndarray,
    position_table: np.ndarray,
) -> tuple[np.ndarray, LearnedPositionCache]:
    """Add the first ``L`` rows of a learned position table to ``x``.

    ``x`` has shape ``(N, L, D)`` and ``position_table`` has shape
    ``(L_max, D)``. A table may be longer than the current sequence, but it
    must contain at least ``L`` rows.
    """
    x = _as_real_array(x, name="x")
    position_table = _as_real_array(position_table, name="position_table")
    if x.ndim != 3:
        raise ValueError("x must have shape (N, L, D)")
    if position_table.ndim != 2:
        raise ValueError("position_table must have shape (L_max, D)")
    if position_table.shape[1] != x.shape[2]:
        raise ValueError("x and position_table feature dimensions must match")
    if position_table.shape[0] < x.shape[1]:
        raise ValueError("position_table is shorter than the input sequence")

    calculation_dtype = np.result_type(x.dtype, position_table.dtype, np.float32)
    x = x.astype(calculation_dtype, copy=False)
    position_table = position_table.astype(calculation_dtype, copy=False)
    output = x + position_table[: x.shape[1]]
    cache = (x.shape, position_table.shape)
    return output, cache


def learned_positional_embedding_backward(
    dout: np.ndarray,
    cache: LearnedPositionCache,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gradients for the input and learned position table.

    A position-table row is shared by every example in a batch, so its
    gradient is the sum of upstream gradients over the batch dimension. Rows
    beyond the current sequence length receive zero gradient.
    """
    dout = _as_real_array(dout, name="dout")
    if not isinstance(cache, tuple) or len(cache) != 2:
        raise TypeError("cache must be returned by learned_positional_embedding_forward")
    input_shape, table_shape = cache
    if dout.shape != input_shape:
        raise ValueError(f"dout must have shape {input_shape}")

    dx = dout.copy()
    dposition_table = np.zeros(table_shape, dtype=dout.dtype)
    dposition_table[: input_shape[1]] = dout.sum(axis=0)
    return dx, dposition_table

