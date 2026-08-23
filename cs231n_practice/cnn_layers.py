"""Naive NumPy layers for convolutional neural networks.

The explicit loops in this module prioritize transparent shape and gradient
reasoning over speed. Inputs use the channels-first ``(N, C, H, W)`` convention,
and filter banks use ``(F, C, HH, WW)``.
"""

import numpy as np

ConvCache = tuple[np.ndarray, np.ndarray, np.ndarray, int, int]
MaxPoolCache = tuple[np.ndarray, int, int, int]


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


def _validate_spatial_parameter(value: int, *, name: str, minimum: int) -> int:
    """Return an integer spatial parameter satisfying the given minimum."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        requirement = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be {requirement}")
    return value


def convolution_output_shape(
    input_height: int,
    input_width: int,
    filter_height: int,
    filter_width: int,
    *,
    stride: int = 1,
    padding: int = 0,
) -> tuple[int, int]:
    """Return spatial output dimensions for an exactly tiled convolution."""
    dimensions = {
        "input_height": input_height,
        "input_width": input_width,
        "filter_height": filter_height,
        "filter_width": filter_width,
    }
    for name, value in dimensions.items():
        dimensions[name] = _validate_spatial_parameter(value, name=name, minimum=1)
    stride = _validate_spatial_parameter(stride, name="stride", minimum=1)
    padding = _validate_spatial_parameter(padding, name="padding", minimum=0)

    height_travel = (
        dimensions["input_height"]
        + 2 * padding
        - dimensions["filter_height"]
    )
    width_travel = (
        dimensions["input_width"]
        + 2 * padding
        - dimensions["filter_width"]
    )
    if height_travel < 0 or width_travel < 0:
        raise ValueError("filters cannot be larger than the padded input")
    if height_travel % stride != 0 or width_travel % stride != 0:
        raise ValueError("filters must tile the padded input exactly")

    return 1 + height_travel // stride, 1 + width_travel // stride


def conv_forward_naive(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    stride: int = 1,
    padding: int = 0,
) -> tuple[np.ndarray, ConvCache]:
    """Compute a naive batched convolution forward pass.

    Args:
        x: Input with shape ``(N, C, H, W)``.
        weights: Filter bank with shape ``(F, C, HH, WW)``.
        bias: One bias per filter with shape ``(F,)``.
        stride: Spatial step between neighboring filter positions.
        padding: Zero-valued border added to every side of each image.

    Returns:
        Output with shape ``(N, F, H_out, W_out)`` and a cache for
        :func:`conv_backward_naive`.
    """
    x = _as_real_array(x, name="x")
    weights = _as_real_array(weights, name="weights")
    bias = _as_real_array(bias, name="bias")
    if x.ndim != 4 or weights.ndim != 4:
        raise ValueError("x and weights must be four-dimensional")
    if bias.ndim != 1:
        raise ValueError("bias must be one-dimensional")

    num_examples, input_channels, input_height, input_width = x.shape
    num_filters, filter_channels, filter_height, filter_width = weights.shape
    if input_channels != filter_channels:
        raise ValueError("filters must have the same channel count as x")
    if bias.shape[0] != num_filters:
        raise ValueError("bias must contain one value per filter")

    output_height, output_width = convolution_output_shape(
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride=stride,
        padding=padding,
    )
    stride = int(stride)
    padding = int(padding)
    calculation_dtype = np.result_type(
        x.dtype, weights.dtype, bias.dtype, np.float32
    )
    x = x.astype(calculation_dtype, copy=False)
    weights = weights.astype(calculation_dtype, copy=False)
    bias = bias.astype(calculation_dtype, copy=False)
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
    )
    output = np.empty(
        (num_examples, num_filters, output_height, output_width),
        dtype=calculation_dtype,
    )

    for example_index in range(num_examples):
        for filter_index in range(num_filters):
            for output_row in range(output_height):
                height_start = output_row * stride
                height_end = height_start + filter_height
                for output_col in range(output_width):
                    width_start = output_col * stride
                    width_end = width_start + filter_width
                    patch = x_padded[
                        example_index,
                        :,
                        height_start:height_end,
                        width_start:width_end,
                    ]
                    output[example_index, filter_index, output_row, output_col] = (
                        np.sum(patch * weights[filter_index]) + bias[filter_index]
                    )

    return output, (x, weights, bias, stride, padding)


def conv_backward_naive(
    dout: np.ndarray,
    cache: ConvCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through a naive batched convolution.

    Args:
        dout: Upstream gradient with shape ``(N, F, H_out, W_out)``.
        cache: Values returned by :func:`conv_forward_naive`.

    Returns:
        ``(dx, dweights, dbias)`` with the shapes of the corresponding forward
        inputs.
    """
    if not isinstance(cache, tuple) or len(cache) != 5:
        raise TypeError("cache must be the tuple returned by conv_forward_naive")
    x = _as_real_array(cache[0], name="cached x")
    weights = _as_real_array(cache[1], name="cached weights")
    bias = _as_real_array(cache[2], name="cached bias")
    stride = _validate_spatial_parameter(cache[3], name="cached stride", minimum=1)
    padding = _validate_spatial_parameter(
        cache[4], name="cached padding", minimum=0
    )
    dout = _as_real_array(dout, name="dout")

    num_examples, _, input_height, input_width = x.shape
    num_filters, _, filter_height, filter_width = weights.shape
    output_height, output_width = convolution_output_shape(
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride=stride,
        padding=padding,
    )
    expected_dout_shape = (
        num_examples,
        num_filters,
        output_height,
        output_width,
    )
    if dout.shape != expected_dout_shape:
        raise ValueError(f"dout must have shape {expected_dout_shape}")

    gradient_dtype = np.result_type(
        x.dtype, weights.dtype, bias.dtype, dout.dtype, np.float32
    )
    x = x.astype(gradient_dtype, copy=False)
    weights = weights.astype(gradient_dtype, copy=False)
    dout = dout.astype(gradient_dtype, copy=False)
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
    )
    dx_padded = np.zeros_like(x_padded, dtype=gradient_dtype)
    dweights = np.zeros_like(weights, dtype=gradient_dtype)
    dbias = np.zeros_like(bias, dtype=gradient_dtype)

    for example_index in range(num_examples):
        for filter_index in range(num_filters):
            for output_row in range(output_height):
                height_start = output_row * stride
                height_end = height_start + filter_height
                for output_col in range(output_width):
                    width_start = output_col * stride
                    width_end = width_start + filter_width
                    patch = x_padded[
                        example_index,
                        :,
                        height_start:height_end,
                        width_start:width_end,
                    ]
                    upstream = dout[
                        example_index, filter_index, output_row, output_col
                    ]
                    dx_padded[
                        example_index,
                        :,
                        height_start:height_end,
                        width_start:width_end,
                    ] += upstream * weights[filter_index]
                    dweights[filter_index] += upstream * patch
                    dbias[filter_index] += upstream

    if padding == 0:
        dx = dx_padded
    else:
        dx = dx_padded[
            :, :, padding:-padding, padding:-padding
        ].copy()
    return dx, dweights, dbias


def pooling_output_shape(
    input_height: int,
    input_width: int,
    pool_height: int,
    pool_width: int,
    *,
    stride: int,
) -> tuple[int, int]:
    """Return spatial output dimensions for exactly tiled pooling windows.

    A stride larger than the pooling window is valid when the final window
    still aligns with the input boundary. In that case, some input positions
    lie in the gaps between windows and do not affect the output.
    """
    dimensions = {
        "input_height": input_height,
        "input_width": input_width,
        "pool_height": pool_height,
        "pool_width": pool_width,
    }
    for name, value in dimensions.items():
        dimensions[name] = _validate_spatial_parameter(value, name=name, minimum=1)
    stride = _validate_spatial_parameter(stride, name="stride", minimum=1)

    height_travel = dimensions["input_height"] - dimensions["pool_height"]
    width_travel = dimensions["input_width"] - dimensions["pool_width"]
    if height_travel < 0 or width_travel < 0:
        raise ValueError("pooling windows cannot be larger than the input")
    if height_travel % stride != 0 or width_travel % stride != 0:
        raise ValueError("pooling windows must tile the input exactly")

    return 1 + height_travel // stride, 1 + width_travel // stride


def max_pool_forward_naive(
    x: np.ndarray,
    *,
    pool_height: int,
    pool_width: int,
    stride: int,
) -> tuple[np.ndarray, MaxPoolCache]:
    """Compute max pooling independently in each NCHW input channel.

    Args:
        x: Input with shape ``(N, C, H, W)``.
        pool_height: Height of each spatial pooling window.
        pool_width: Width of each spatial pooling window.
        stride: Spatial step between neighboring windows.

    Returns:
        Output with shape ``(N, C, H_out, W_out)`` and a cache for
        :func:`max_pool_backward_naive`.
    """
    x = _as_real_array(x, name="x")
    if x.ndim != 4:
        raise ValueError("x must be four-dimensional")
    num_examples, num_channels, input_height, input_width = x.shape
    output_height, output_width = pooling_output_shape(
        input_height,
        input_width,
        pool_height,
        pool_width,
        stride=stride,
    )
    pool_height = int(pool_height)
    pool_width = int(pool_width)
    stride = int(stride)
    calculation_dtype = np.result_type(x.dtype, np.float32)
    x = x.astype(calculation_dtype, copy=False)
    output = np.empty(
        (num_examples, num_channels, output_height, output_width),
        dtype=calculation_dtype,
    )

    for example_index in range(num_examples):
        for channel_index in range(num_channels):
            for output_row in range(output_height):
                height_start = output_row * stride
                height_end = height_start + pool_height
                for output_col in range(output_width):
                    width_start = output_col * stride
                    width_end = width_start + pool_width
                    window = x[
                        example_index,
                        channel_index,
                        height_start:height_end,
                        width_start:width_end,
                    ]
                    output[
                        example_index, channel_index, output_row, output_col
                    ] = np.max(window)

    return output, (x, pool_height, pool_width, stride)


def max_pool_backward_naive(
    dout: np.ndarray,
    cache: MaxPoolCache,
) -> np.ndarray:
    """Backpropagate through max pooling using a first-maximum tie rule.

    ``np.argmax`` selects the first maximum in flattened row-major order. The
    complete upstream gradient is routed to that position. Contributions use
    accumulation because pooling windows may overlap.
    """
    if not isinstance(cache, tuple) or len(cache) != 4:
        raise TypeError("cache must be the tuple returned by max_pool_forward_naive")
    x = _as_real_array(cache[0], name="cached x")
    pool_height = _validate_spatial_parameter(
        cache[1], name="cached pool_height", minimum=1
    )
    pool_width = _validate_spatial_parameter(
        cache[2], name="cached pool_width", minimum=1
    )
    stride = _validate_spatial_parameter(cache[3], name="cached stride", minimum=1)
    dout = _as_real_array(dout, name="dout")

    num_examples, num_channels, input_height, input_width = x.shape
    output_height, output_width = pooling_output_shape(
        input_height,
        input_width,
        pool_height,
        pool_width,
        stride=stride,
    )
    expected_dout_shape = (
        num_examples,
        num_channels,
        output_height,
        output_width,
    )
    if dout.shape != expected_dout_shape:
        raise ValueError(f"dout must have shape {expected_dout_shape}")

    gradient_dtype = np.result_type(x.dtype, dout.dtype, np.float32)
    x = x.astype(gradient_dtype, copy=False)
    dout = dout.astype(gradient_dtype, copy=False)
    dx = np.zeros_like(x, dtype=gradient_dtype)

    for example_index in range(num_examples):
        for channel_index in range(num_channels):
            for output_row in range(output_height):
                height_start = output_row * stride
                height_end = height_start + pool_height
                for output_col in range(output_width):
                    width_start = output_col * stride
                    width_end = width_start + pool_width
                    window = x[
                        example_index,
                        channel_index,
                        height_start:height_end,
                        width_start:width_end,
                    ]
                    flat_index = int(np.argmax(window))
                    local_row, local_col = np.unravel_index(
                        flat_index, window.shape
                    )
                    dx[
                        example_index,
                        channel_index,
                        height_start + local_row,
                        width_start + local_col,
                    ] += dout[
                        example_index, channel_index, output_row, output_col
                    ]

    return dx
