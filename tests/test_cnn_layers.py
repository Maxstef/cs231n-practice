import numpy as np
import pytest

from cs231n_practice.cnn_layers import (
    conv_backward_naive,
    conv_forward_naive,
    convolution_output_shape,
    max_pool_backward_naive,
    max_pool_forward_naive,
    pooling_output_shape,
)
from cs231n_practice.gradient_check import eval_numerical_gradient_array


def test_convolution_output_shape_handles_padding_and_stride() -> None:
    assert convolution_output_shape(5, 5, 3, 3) == (3, 3)
    assert convolution_output_shape(5, 5, 3, 3, padding=1) == (5, 5)
    assert convolution_output_shape(7, 9, 3, 3, padding=1, stride=2) == (4, 5)


def test_conv_forward_matches_hand_computed_values() -> None:
    x = np.arange(1, 17, dtype=np.float64).reshape(1, 1, 4, 4)
    weights = np.array(
        [
            [[[1, 0], [0, -1]]],
            [[[1, 1], [1, 1]]],
        ],
        dtype=np.float64,
    )
    bias = np.array([0.0, 0.5])

    output, _ = conv_forward_naive(x, weights, bias)

    expected = np.array(
        [
            [
                [[-5, -5, -5], [-5, -5, -5], [-5, -5, -5]],
                [
                    [14.5, 18.5, 22.5],
                    [30.5, 34.5, 38.5],
                    [46.5, 50.5, 54.5],
                ],
            ]
        ]
    )
    np.testing.assert_allclose(output, expected)


def test_conv_backward_matches_hand_computed_overlap_gradients() -> None:
    x = np.arange(9, dtype=np.float64).reshape(1, 1, 3, 3)
    weights = np.ones((1, 1, 2, 2))
    bias = np.zeros(1)
    output, cache = conv_forward_naive(x, weights, bias)

    dx, dweights, dbias = conv_backward_naive(np.ones_like(output), cache)

    np.testing.assert_array_equal(
        dx[0, 0],
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
    )
    np.testing.assert_array_equal(dweights, [[[[8, 12], [20, 24]]]])
    np.testing.assert_array_equal(dbias, [4])
    assert dx.shape == x.shape
    assert dweights.shape == weights.shape
    assert dbias.shape == bias.shape


def test_conv_gradients_match_numerical_gradients_with_padding() -> None:
    generator = np.random.default_rng(42)
    x = generator.normal(size=(1, 2, 4, 4))
    weights = generator.normal(size=(2, 2, 3, 3))
    bias = generator.normal(size=2)
    output, cache = conv_forward_naive(x, weights, bias, padding=1)
    dout = generator.normal(size=output.shape)

    dx, dweights, dbias = conv_backward_naive(dout, cache)
    numerical_dx = eval_numerical_gradient_array(
        lambda candidate: conv_forward_naive(
            candidate, weights, bias, padding=1
        )[0],
        x,
        dout,
    )
    numerical_dweights = eval_numerical_gradient_array(
        lambda candidate: conv_forward_naive(
            x, candidate, bias, padding=1
        )[0],
        weights,
        dout,
    )
    numerical_dbias = eval_numerical_gradient_array(
        lambda candidate: conv_forward_naive(
            x, weights, candidate, padding=1
        )[0],
        bias,
        dout,
    )

    np.testing.assert_allclose(dx, numerical_dx, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(
        dweights, numerical_dweights, rtol=1e-8, atol=1e-9
    )
    np.testing.assert_allclose(dbias, numerical_dbias, rtol=1e-8, atol=1e-9)


def test_conv_forward_promotes_integer_inputs_to_floating_point() -> None:
    output, cache = conv_forward_naive(
        np.ones((1, 1, 3, 3), dtype=np.int16),
        np.ones((1, 1, 2, 2), dtype=np.int16),
        np.zeros(1, dtype=np.int16),
    )
    dx, dweights, dbias = conv_backward_naive(np.full(output.shape, 0.5), cache)

    assert np.issubdtype(output.dtype, np.floating)
    assert np.issubdtype(dx.dtype, np.floating)
    assert np.issubdtype(dweights.dtype, np.floating)
    assert np.issubdtype(dbias.dtype, np.floating)
    assert np.any(dx % 1 != 0)


def test_conv_forward_does_not_mutate_inputs() -> None:
    x = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4)
    weights = np.ones((1, 1, 3, 3))
    bias = np.array([0.25])
    originals = (x.copy(), weights.copy(), bias.copy())

    conv_forward_naive(x, weights, bias, padding=1)

    np.testing.assert_array_equal(x, originals[0])
    np.testing.assert_array_equal(weights, originals[1])
    np.testing.assert_array_equal(bias, originals[2])


def test_conv_forward_rejects_incompatible_channels_and_bias() -> None:
    with pytest.raises(ValueError, match="same channel count"):
        conv_forward_naive(
            np.ones((1, 3, 5, 5)),
            np.ones((2, 2, 3, 3)),
            np.zeros(2),
        )
    with pytest.raises(ValueError, match="one value per filter"):
        conv_forward_naive(
            np.ones((1, 3, 5, 5)),
            np.ones((2, 3, 3, 3)),
            np.zeros(3),
        )


def test_convolution_rejects_invalid_spatial_configuration() -> None:
    with pytest.raises(ValueError, match="tile"):
        convolution_output_shape(6, 6, 3, 3, stride=2)
    with pytest.raises(TypeError, match="stride must be an integer"):
        convolution_output_shape(5, 5, 3, 3, stride=1.5)
    with pytest.raises(ValueError, match="padding must be nonnegative"):
        convolution_output_shape(5, 5, 3, 3, padding=-1)


def test_conv_backward_rejects_wrong_upstream_shape() -> None:
    _, cache = conv_forward_naive(
        np.ones((2, 3, 5, 5)),
        np.ones((4, 3, 3, 3)),
        np.zeros(4),
        padding=1,
    )

    with pytest.raises(ValueError, match="dout must have shape"):
        conv_backward_naive(np.ones((2, 4, 4, 4)), cache)


def test_pooling_output_shape_supports_overlap_and_gaps() -> None:
    assert pooling_output_shape(8, 8, 2, 2, stride=2) == (4, 4)
    assert pooling_output_shape(5, 5, 3, 3, stride=1) == (3, 3)
    assert pooling_output_shape(2, 2, 2, 2, stride=1) == (1, 1)
    assert pooling_output_shape(8, 8, 2, 2, stride=3) == (3, 3)


def test_max_pool_forward_processes_channels_independently() -> None:
    first_channel = np.array(
        [[1, 3, 2, 0], [4, 6, 5, 1], [7, 2, 9, 8], [3, 5, 4, 2]],
        dtype=np.float64,
    )
    x = np.stack([first_channel, first_channel + 100], axis=0)[None, ...]

    output, _ = max_pool_forward_naive(
        x, pool_height=2, pool_width=2, stride=2
    )

    expected = np.array([[6, 5], [7, 9]])
    assert output.shape == (1, 2, 2, 2)
    np.testing.assert_array_equal(output[0, 0], expected)
    np.testing.assert_array_equal(output[0, 1], expected + 100)


def test_max_pool_backward_accumulates_overlapping_windows() -> None:
    x = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, 9.0, 6.0], [7.0, 8.0, 5.0]]]]
    )
    output, cache = max_pool_forward_naive(
        x, pool_height=2, pool_width=2, stride=1
    )

    dx = max_pool_backward_naive(np.ones_like(output), cache)

    expected = np.zeros_like(x)
    expected[0, 0, 1, 1] = 4.0
    np.testing.assert_array_equal(dx, expected)


def test_max_pool_backward_uses_first_maximum_when_values_tie() -> None:
    x = np.array([[[[5.0, 5.0], [1.0, 0.0]]]])
    output, cache = max_pool_forward_naive(
        x, pool_height=2, pool_width=2, stride=2
    )

    dx = max_pool_backward_naive(np.array([[[[3.0]]]]), cache)

    np.testing.assert_array_equal(dx, [[[[3.0, 0.0], [0.0, 0.0]]]])


def test_max_pool_backward_matches_numerical_gradient_away_from_ties() -> None:
    generator = np.random.default_rng(19)
    x = generator.normal(size=(2, 2, 4, 4))
    output, cache = max_pool_forward_naive(
        x, pool_height=2, pool_width=2, stride=2
    )
    dout = generator.normal(size=output.shape)

    dx = max_pool_backward_naive(dout, cache)
    numerical_dx = eval_numerical_gradient_array(
        lambda candidate: max_pool_forward_naive(
            candidate, pool_height=2, pool_width=2, stride=2
        )[0],
        x,
        dout,
    )

    np.testing.assert_allclose(dx, numerical_dx, rtol=1e-8, atol=1e-9)


def test_max_pool_forward_with_gaps_skips_uncovered_values() -> None:
    x = np.zeros((1, 1, 8, 8), dtype=np.float64)
    x[0, 0, 2, 2] = 100.0  # Between windows for pool size 2 and stride 3.

    output, cache = max_pool_forward_naive(
        x, pool_height=2, pool_width=2, stride=3
    )
    dx = max_pool_backward_naive(np.ones_like(output), cache)

    assert output.shape == (1, 1, 3, 3)
    assert np.max(output) == 0.0
    assert dx[0, 0, 2, 2] == 0.0


def test_max_pool_rejects_invalid_shapes_and_parameters() -> None:
    with pytest.raises(ValueError, match="four-dimensional"):
        max_pool_forward_naive(
            np.ones((3, 4, 4)), pool_height=2, pool_width=2, stride=2
        )
    with pytest.raises(ValueError, match="larger than the input"):
        pooling_output_shape(2, 2, 3, 2, stride=1)
    with pytest.raises(ValueError, match="tile"):
        pooling_output_shape(6, 6, 3, 3, stride=2)
    with pytest.raises(TypeError, match="stride must be an integer"):
        pooling_output_shape(5, 5, 3, 3, stride=True)


def test_max_pool_backward_rejects_wrong_upstream_shape() -> None:
    _, cache = max_pool_forward_naive(
        np.ones((2, 3, 4, 4)), pool_height=2, pool_width=2, stride=2
    )

    with pytest.raises(ValueError, match="dout must have shape"):
        max_pool_backward_naive(np.ones((2, 3, 3, 3)), cache)
