"""Small reusable helpers for video clips and temporal fusion."""

from numbers import Integral
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def _integer(value: int, name: str, *, minimum: int) -> int:
    """Validate an integer argument without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _video_tensor(value: torch.Tensor, name: str, ndim: int) -> torch.Tensor:
    """Validate a single video or batch with nonempty axes."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if value.ndim != ndim or any(size == 0 for size in value.shape):
        expected = "(T, C, H, W)" if ndim == 4 else "(N, T, C, H, W)"
        raise ValueError(f"{name} must have nonempty shape {expected}")
    return value


def _triple(
    value: int | Sequence[int], name: str, *, minimum: int
) -> tuple[int, int, int]:
    """Return one integer repeated three times or validate a length-3 value."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        item = _integer(value, name, minimum=minimum)
        return item, item, item
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an integer or a sequence of three integers")
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return tuple(
        _integer(item, f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def sample_clip_indices(
    total_frames: int,
    start: int,
    length: int,
    stride: int = 1,
) -> list[int]:
    """Return ordered frame indices for one valid strided clip.

    A clip selects ``start + k * stride`` for ``k = 0, ..., length - 1``.
    The final index must be less than ``total_frames``.
    """
    total_frames = _integer(total_frames, "total_frames", minimum=1)
    start = _integer(start, "start", minimum=0)
    length = _integer(length, "length", minimum=1)
    stride = _integer(stride, "stride", minimum=1)
    last = start + (length - 1) * stride
    if last >= total_frames:
        raise ValueError("clip extends beyond the video")
    return [start + k * stride for k in range(length)]


def sample_video_clip(
    video: torch.Tensor,
    start: int,
    length: int,
    stride: int = 1,
) -> torch.Tensor:
    """Select one ``(L, C, H, W)`` clip from a ``(T, C, H, W)`` video."""
    video = _video_tensor(video, "video", 4)
    indices = sample_clip_indices(video.shape[0], start, length, stride)
    index_tensor = torch.tensor(indices, dtype=torch.long, device=video.device)
    return video.index_select(0, index_tensor)


def video_to_conv3d_batch(videos: torch.Tensor) -> torch.Tensor:
    """Convert ``(N, T, C, H, W)`` to ``(N, C, T, H, W)``."""
    videos = _video_tensor(videos, "videos", 5)
    return videos.permute(0, 2, 1, 3, 4)


def stack_frames_as_channels(videos: torch.Tensor) -> torch.Tensor:
    """Convert ``(N, T, C, H, W)`` to early-fusion ``(N, T*C, H, W)``.

    Time and channels are adjacent in the input, so each frame's ``C`` channels
    stay together and frames keep their original order.
    """
    videos = _video_tensor(videos, "videos", 5)
    number_of_videos, number_of_frames, channels, height, width = videos.shape
    return videos.reshape(
        number_of_videos, number_of_frames * channels, height, width
    )


def temporal_difference(videos: torch.Tensor) -> torch.Tensor:
    """Return consecutive next-frame minus current-frame differences.

    The input has shape ``(N, T, C, H, W)`` and the result has shape
    ``(N, T - 1, C, H, W)``. Floating-point input is required so negative
    changes are represented correctly instead of wrapping as unsigned values.
    """
    videos = _video_tensor(videos, "videos", 5)
    if videos.shape[1] < 2:
        raise ValueError("videos must contain at least two frames")
    if not videos.is_floating_point():
        raise TypeError("videos must contain floating-point values")
    return videos[:, 1:] - videos[:, :-1]


def stack_temporal_differences(videos: torch.Tensor) -> torch.Tensor:
    """Stack consecutive frame differences along the channel axis.

    This converts ``(N, T, C, H, W)`` videos to motion inputs shaped
    ``(N, (T - 1) * C, H, W)``. Within the stacked channels, all ``C``
    channels from one time step remain together and time order is preserved.
    """
    differences = temporal_difference(videos)
    number_of_videos, steps, channels, height, width = differences.shape
    return differences.reshape(
        number_of_videos, steps * channels, height, width
    )


def video_features_to_tokens(features: torch.Tensor) -> torch.Tensor:
    """Convert ``(N, C, T, H, W)`` features to ``(N, T*H*W, C)`` tokens."""
    if not isinstance(features, torch.Tensor):
        raise TypeError("features must be a PyTorch tensor")
    if features.ndim != 5 or any(size == 0 for size in features.shape):
        raise ValueError("features must have nonempty shape (N, C, T, H, W)")
    n, channels, time, height, width = features.shape
    return features.permute(0, 2, 3, 4, 1).reshape(
        n, time * height * width, channels
    )


def tokens_to_video_features(
    tokens: torch.Tensor,
    time: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert ``(N, T*H*W, C)`` tokens to ``(N, C, T, H, W)`` features."""
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("tokens must be a PyTorch tensor")
    if tokens.ndim != 3 or any(size == 0 for size in tokens.shape):
        raise ValueError("tokens must have nonempty shape (N, T*H*W, C)")
    time = _integer(time, "time", minimum=1)
    height = _integer(height, "height", minimum=1)
    width = _integer(width, "width", minimum=1)
    n, positions, channels = tokens.shape
    if positions != time * height * width:
        raise ValueError("token count must equal time * height * width")
    return tokens.reshape(n, time, height, width, channels).permute(
        0, 4, 1, 2, 3
    )


def attention_entry_counts(
    time: int,
    height: int,
    width: int,
) -> tuple[int, int]:
    """Return joint and factorized space-time attention matrix entry counts.

    Joint attention uses one ``(T*H*W)`` squared matrix. Factorized attention
    counts spatial attention within every frame plus temporal attention at
    every spatial position: ``T*(H*W)^2 + H*W*T^2``.
    """
    time = _integer(time, "time", minimum=1)
    height = _integer(height, "height", minimum=1)
    width = _integer(width, "width", minimum=1)
    spatial_positions = height * width
    joint = (time * spatial_positions) ** 2
    factorized = time * spatial_positions**2 + spatial_positions * time**2
    return joint, factorized


class NonLocalBlock3D(nn.Module):
    """Apply residual global self-attention to ``(N, C, T, H, W)`` features.

    Pointwise 3D convolutions project channels into query, key, and value
    features. Attention then connects every space-time position to every
    other position. The learnable residual scale starts at zero, so a newly
    created block initially returns its input exactly.
    """

    def __init__(
        self,
        channels: int,
        attention_channels: int | None = None,
    ) -> None:
        super().__init__()
        channels = _integer(channels, "channels", minimum=1)
        if attention_channels is None:
            attention_channels = max(1, channels // 2)
        attention_channels = _integer(
            attention_channels, "attention_channels", minimum=1
        )
        self.channels = channels
        self.attention_channels = attention_channels
        self.query = nn.Conv3d(channels, attention_channels, kernel_size=1)
        self.key = nn.Conv3d(channels, attention_channels, kernel_size=1)
        self.value = nn.Conv3d(channels, attention_channels, kernel_size=1)
        self.output = nn.Conv3d(attention_channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input plus its learned non-local attention update."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a PyTorch tensor")
        if x.ndim != 5 or any(size == 0 for size in x.shape):
            raise ValueError("x must have nonempty shape (N, C, T, H, W)")
        if x.shape[1] != self.channels:
            raise ValueError(f"x must contain {self.channels} channels")
        if not x.is_floating_point():
            raise TypeError("x must contain floating-point values")

        n, _, time, height, width = x.shape
        query = self.query(x).flatten(2).transpose(1, 2)
        key = self.key(x).flatten(2)
        value = self.value(x).flatten(2).transpose(1, 2)
        scores = query @ key / self.attention_channels**0.5
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ value
        attended_features = attended.transpose(1, 2).reshape(
            n, self.attention_channels, time, height, width
        )
        projected = self.output(attended_features)
        return x + self.gamma * projected


def average_clip_scores(scores: torch.Tensor) -> torch.Tensor:
    """Average ``(N, K, classes)`` scores across ``K`` clips per video."""
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a PyTorch tensor")
    if scores.ndim != 3 or any(size == 0 for size in scores.shape):
        raise ValueError("scores must have nonempty shape (N, K, classes)")
    if not scores.is_floating_point():
        raise TypeError("scores must contain floating-point values")
    return scores.mean(dim=1)


def conv3d_output_shape(
    input_shape: Sequence[int],
    kernel_size: int | Sequence[int],
    padding: int | Sequence[int] = 0,
    stride: int | Sequence[int] = 1,
) -> tuple[int, int, int]:
    """Return ``(T_out, H_out, W_out)`` for a 3D convolution.

    ``input_shape`` contains ``(T, H, W)``. Kernel size, padding, and stride
    may each be one integer or separate temporal and spatial values.
    """
    input_t, input_h, input_w = _triple(input_shape, "input_shape", minimum=1)
    kernel_t, kernel_h, kernel_w = _triple(
        kernel_size, "kernel_size", minimum=1
    )
    padding_t, padding_h, padding_w = _triple(
        padding, "padding", minimum=0
    )
    stride_t, stride_h, stride_w = _triple(stride, "stride", minimum=1)

    output = tuple(
        (input_size + 2 * pad - kernel) // step + 1
        for input_size, kernel, pad, step in zip(
            (input_t, input_h, input_w),
            (kernel_t, kernel_h, kernel_w),
            (padding_t, padding_h, padding_w),
            (stride_t, stride_h, stride_w),
        )
    )
    if any(size <= 0 for size in output):
        raise ValueError("kernel does not fit the padded input")
    return output  # type: ignore[return-value]


def conv3d_forward_naive(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
) -> np.ndarray:
    """Compute a direct 3D cross-correlation for educational verification.

    Args:
        x: Input with shape ``(N, C_in, T, H, W)``.
        weights: Filter bank with shape
            ``(C_out, C_in, K_t, K_h, K_w)``.
        bias: One bias per output channel, shape ``(C_out,)``.
        stride: Integer or ``(S_t, S_h, S_w)``.
        padding: Integer or ``(P_t, P_h, P_w)``.
    """
    x = np.asarray(x)
    weights = np.asarray(weights)
    bias = np.asarray(bias)
    for value, name in ((x, "x"), (weights, "weights"), (bias, "bias")):
        if not np.issubdtype(value.dtype, np.number) or np.issubdtype(
            value.dtype, np.complexfloating
        ):
            raise TypeError(f"{name} must contain real numeric values")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain finite values")
    if x.ndim != 5 or any(size == 0 for size in x.shape):
        raise ValueError("x must have nonempty shape (N, C_in, T, H, W)")
    if weights.ndim != 5 or any(size == 0 for size in weights.shape):
        raise ValueError(
            "weights must have nonempty shape (C_out, C_in, K_t, K_h, K_w)"
        )
    if x.shape[1] != weights.shape[1]:
        raise ValueError("x and weights must have the same input-channel count")
    if bias.shape != (weights.shape[0],):
        raise ValueError("bias must contain one value per output channel")

    stride_t, stride_h, stride_w = _triple(stride, "stride", minimum=1)
    padding_t, padding_h, padding_w = _triple(
        padding, "padding", minimum=0
    )
    output_t, output_h, output_w = conv3d_output_shape(
        x.shape[2:], weights.shape[2:], padding, stride
    )
    padded = np.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (padding_t, padding_t),
            (padding_h, padding_h),
            (padding_w, padding_w),
        ),
    )
    calculation_dtype = np.result_type(x.dtype, weights.dtype, bias.dtype, np.float32)
    output = np.empty(
        (x.shape[0], weights.shape[0], output_t, output_h, output_w),
        dtype=calculation_dtype,
    )
    kernel_t, kernel_h, kernel_w = weights.shape[2:]

    for n in range(x.shape[0]):
        for f in range(weights.shape[0]):
            for t in range(output_t):
                t_start = t * stride_t
                for row in range(output_h):
                    row_start = row * stride_h
                    for column in range(output_w):
                        column_start = column * stride_w
                        patch = padded[
                            n,
                            :,
                            t_start : t_start + kernel_t,
                            row_start : row_start + kernel_h,
                            column_start : column_start + kernel_w,
                        ]
                        output[n, f, t, row, column] = (
                            np.sum(patch * weights[f]) + bias[f]
                        )
    return output


def inflate_conv2d_weights(
    weights: np.ndarray, temporal_kernel_size: int
) -> np.ndarray:
    """Repeat 2D filters through time and preserve their summed scale."""
    weights = np.asarray(weights)
    temporal_kernel_size = _integer(
        temporal_kernel_size, "temporal_kernel_size", minimum=1
    )
    if weights.ndim != 4 or any(size == 0 for size in weights.shape):
        raise ValueError("weights must have nonempty shape (C_out, C_in, K_h, K_w)")
    if not np.issubdtype(weights.dtype, np.number) or np.issubdtype(
        weights.dtype, np.complexfloating
    ):
        raise TypeError("weights must contain real numeric values")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain finite values")

    calculation_dtype = np.result_type(weights.dtype, np.float32)
    weights = weights.astype(calculation_dtype, copy=False)
    return np.repeat(
        weights[:, :, None, :, :], temporal_kernel_size, axis=2
    ) / temporal_kernel_size


def spatiotemporal_receptive_field(
    kernel_sizes: Sequence[int | Sequence[int]],
    strides: int | Sequence[int | Sequence[int]] = 1,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return receptive field and jump for a stack of 3D layers.

    Values are ordered as ``(time, height, width)``. Dilation is assumed to be
    one. ``strides`` may be one shared integer or one value per layer.
    """
    kernels = list(kernel_sizes)
    if not kernels:
        raise ValueError("kernel_sizes must contain at least one layer")
    kernel_triples = [
        _triple(kernel, f"kernel_sizes[{index}]", minimum=1)
        for index, kernel in enumerate(kernels)
    ]
    if isinstance(strides, Integral) and not isinstance(strides, bool):
        stride_triples = [_triple(strides, "strides", minimum=1)] * len(kernels)
    else:
        if isinstance(strides, (str, bytes)) or not isinstance(strides, Sequence):
            raise TypeError("strides must be an integer or one value per layer")
        if len(strides) != len(kernels):
            raise ValueError("strides must contain one value per layer")
        stride_triples = [
            _triple(step, f"strides[{index}]", minimum=1)
            for index, step in enumerate(strides)
        ]

    receptive_field = [1, 1, 1]
    jump = [1, 1, 1]
    for kernel, step in zip(kernel_triples, stride_triples):
        for axis in range(3):
            receptive_field[axis] += (kernel[axis] - 1) * jump[axis]
            jump[axis] *= step[axis]
    return tuple(receptive_field), tuple(jump)  # type: ignore[return-value]
