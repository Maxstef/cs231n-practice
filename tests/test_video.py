import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cs231n_practice.video import (
    average_clip_scores,
    conv3d_forward_naive,
    conv3d_output_shape,
    inflate_conv2d_weights,
    sample_clip_indices,
    sample_video_clip,
    spatiotemporal_receptive_field,
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


def test_conv3d_output_shape_supports_independent_dimensions() -> None:
    output = conv3d_output_shape(
        (8, 16, 20), kernel_size=(3, 3, 5), padding=(1, 1, 2), stride=2
    )

    assert output == (4, 8, 10)


def test_conv3d_forward_naive_matches_pytorch() -> None:
    generator = np.random.default_rng(11)
    x = generator.normal(size=(2, 2, 4, 5, 6))
    weights = generator.normal(size=(3, 2, 2, 3, 3))
    bias = generator.normal(size=(3,))

    output = conv3d_forward_naive(
        x, weights, bias, stride=(2, 1, 2), padding=(1, 1, 1)
    )
    expected = F.conv3d(
        torch.from_numpy(x),
        torch.from_numpy(weights),
        torch.from_numpy(bias),
        stride=(2, 1, 2),
        padding=(1, 1, 1),
    ).numpy()

    np.testing.assert_allclose(output, expected, rtol=1e-12, atol=1e-12)


def test_conv3d_forward_naive_includes_bias_and_all_input_channels() -> None:
    x = np.ones((1, 2, 2, 1, 1))
    weights = np.array([[[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]])

    output = conv3d_forward_naive(x, weights, np.array([5.0]))

    assert output.shape == (1, 1, 1, 1, 1)
    np.testing.assert_allclose(output.item(), 1 + 2 + 3 + 4 + 5)


def test_inflate_conv2d_weights_preserves_temporal_sum() -> None:
    weights = np.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)

    inflated = inflate_conv2d_weights(weights, temporal_kernel_size=3)

    assert inflated.shape == (2, 3, 3, 2, 2)
    np.testing.assert_allclose(inflated.sum(axis=2), weights)
    np.testing.assert_allclose(inflated[:, :, 0], weights / 3)


def test_spatiotemporal_receptive_field_tracks_jump() -> None:
    receptive_field, jump = spatiotemporal_receptive_field(
        [(3, 3, 3), (3, 3, 3)], strides=[1, (2, 2, 2)]
    )

    assert receptive_field == (5, 5, 5)
    assert jump == (2, 2, 2)


def test_receptive_field_uses_previous_layer_jump() -> None:
    receptive_field, jump = spatiotemporal_receptive_field(
        [(3, 3, 3), (3, 3, 3)], strides=[2, 1]
    )

    assert receptive_field == (7, 7, 7)
    assert jump == (2, 2, 2)


def test_new_video_utilities_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="fit"):
        conv3d_output_shape((2, 4, 4), kernel_size=(3, 1, 1))
    with pytest.raises(ValueError, match="input-channel"):
        conv3d_forward_naive(
            np.zeros((1, 2, 3, 3, 3)),
            np.zeros((1, 1, 1, 1, 1)),
            np.zeros(1),
        )
    with pytest.raises(ValueError, match="shape"):
        inflate_conv2d_weights(np.zeros((2, 3, 3)), 3)
    with pytest.raises(ValueError, match="one value per layer"):
        spatiotemporal_receptive_field([3, 3], strides=[1])
