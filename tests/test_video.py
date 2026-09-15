import pytest
import torch

from cs231n_practice.video import (
    average_clip_scores,
    sample_clip_indices,
    sample_video_clip,
    stack_frames_as_channels,
    video_to_conv3d_batch,
)


def test_sample_clip_indices_keeps_order_and_stride() -> None:
    assert sample_clip_indices(8, start=1, length=3, stride=2) == [1, 3, 5]
    assert sample_clip_indices(8, start=0, length=4) == [0, 1, 2, 3]
    assert sample_clip_indices(8, start=7, length=1) == [7]


@pytest.mark.parametrize(
    ("total_frames", "start", "length", "stride"),
    [(8, 7, 2, 1), (8, 8, 1, 1), (8, 0, 5, 2)],
)
def test_sample_clip_indices_rejects_out_of_range_clip(
    total_frames: int, start: int, length: int, stride: int
) -> None:
    with pytest.raises(ValueError, match="beyond"):
        sample_clip_indices(total_frames, start, length, stride)


@pytest.mark.parametrize(
    ("total_frames", "start", "length", "stride", "error"),
    [
        (0, 0, 1, 1, ValueError),
        (8, -1, 1, 1, ValueError),
        (8, 0, 0, 1, ValueError),
        (8, 0, 1, 0, ValueError),
        (8, True, 1, 1, TypeError),
        (8, 0, 1.5, 1, TypeError),
    ],
)
def test_sample_clip_indices_rejects_invalid_arguments(
    total_frames: object,
    start: object,
    length: object,
    stride: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        sample_clip_indices(total_frames, start, length, stride)  # type: ignore[arg-type]


def test_sample_video_clip_selects_requested_frames() -> None:
    video = torch.arange(8 * 3 * 2 * 4).reshape(8, 3, 2, 4)

    clip = sample_video_clip(video, start=1, length=3, stride=2)

    assert clip.shape == (3, 3, 2, 4)
    torch.testing.assert_close(clip, video[[1, 3, 5]])


def test_video_to_conv3d_batch_preserves_all_values() -> None:
    videos = torch.arange(2 * 4 * 3 * 2 * 5).reshape(2, 4, 3, 2, 5)

    output = video_to_conv3d_batch(videos)

    assert output.shape == (2, 3, 4, 2, 5)
    torch.testing.assert_close(output.permute(0, 2, 1, 3, 4), videos)


def test_stack_frames_as_channels_keeps_frame_channel_order() -> None:
    videos = torch.arange(2 * 4 * 3 * 2 * 5).reshape(2, 4, 3, 2, 5)

    output = stack_frames_as_channels(videos)

    assert output.shape == (2, 12, 2, 5)
    for t in range(4):
        torch.testing.assert_close(output[:, 3 * t : 3 * (t + 1)], videos[:, t])


def test_average_clip_scores_reduces_only_clip_axis() -> None:
    scores = torch.tensor(
        [[[1.0, 3.0], [3.0, 5.0]], [[7.0, 2.0], [9.0, 4.0]]]
    )

    averaged = average_clip_scores(scores)

    assert averaged.shape == (2, 2)
    torch.testing.assert_close(averaged, torch.tensor([[2.0, 4.0], [8.0, 3.0]]))


def test_video_helpers_reject_wrong_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        sample_video_clip(torch.zeros(2, 3, 4), start=0, length=1)
    with pytest.raises(ValueError, match="shape"):
        video_to_conv3d_batch(torch.zeros(2, 3, 4, 5))
    with pytest.raises(ValueError, match="shape"):
        average_clip_scores(torch.zeros(2, 3))
