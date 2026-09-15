"""Small reusable helpers for video clips and temporal fusion."""

from numbers import Integral

import torch


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


def average_clip_scores(scores: torch.Tensor) -> torch.Tensor:
    """Average ``(N, K, classes)`` scores across ``K`` clips per video."""
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a PyTorch tensor")
    if scores.ndim != 3 or any(size == 0 for size in scores.shape):
        raise ValueError("scores must have nonempty shape (N, K, classes)")
    if not scores.is_floating_point():
        raise TypeError("scores must contain floating-point values")
    return scores.mean(dim=1)
