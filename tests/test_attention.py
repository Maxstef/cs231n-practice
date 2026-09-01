import numpy as np
import pytest

from cs231n_practice.attention import (
    masked_softmax,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)
from cs231n_practice.gradient_check import eval_numerical_gradient_array


def test_masked_softmax_normalizes_valid_keys_and_zeros_masked_keys() -> None:
    scores = np.array([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]])
    mask = np.array([[[True, False, True]]])

    weights = masked_softmax(scores, mask)

    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)
    np.testing.assert_array_equal(weights[..., 1], 0.0)
    np.testing.assert_allclose(weights[0, 1, [0, 2]], 0.5)


def test_uniform_attention_returns_mean_value() -> None:
    query = np.zeros((2, 3, 4))
    key = np.ones((2, 5, 4))
    value = np.arange(40, dtype=float).reshape(2, 5, 4)

    output, weights, _ = scaled_dot_product_attention_forward(query, key, value)

    np.testing.assert_allclose(weights, 1.0 / 5.0)
    expected = np.repeat(value.mean(axis=1, keepdims=True), 3, axis=1)
    np.testing.assert_allclose(output, expected)


@pytest.mark.parametrize("use_mask", [False, True])
def test_attention_backward_matches_numerical_gradients(use_mask: bool) -> None:
    generator = np.random.default_rng(101)
    arguments = [
        generator.normal(size=(2, 3, 4)),
        generator.normal(size=(2, 5, 4)),
        generator.normal(size=(2, 5, 3)),
    ]
    mask = None
    if use_mask:
        mask = np.array(
            [
                [[True, True, False, True, False]],
                [[True, False, True, True, True]],
            ]
        )

    output, _, cache = scaled_dot_product_attention_forward(*arguments, mask)
    dout = generator.normal(size=output.shape)
    analytical = scaled_dot_product_attention_backward(dout, cache)

    for index, (value, gradient) in enumerate(zip(arguments, analytical)):
        def forward(candidate: np.ndarray) -> np.ndarray:
            current = arguments.copy()
            current[index] = candidate
            return scaled_dot_product_attention_forward(*current, mask)[0]

        numerical = eval_numerical_gradient_array(forward, value, dout)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-8)


def test_attention_supports_multi_head_leading_dimensions() -> None:
    generator = np.random.default_rng(103)
    query = generator.normal(size=(2, 3, 4, 5))
    key = generator.normal(size=(2, 3, 6, 5))
    value = generator.normal(size=(2, 3, 6, 7))
    mask = np.ones((2, 1, 4, 6), dtype=bool)
    mask[:, :, :, -1] = False

    output, weights, _ = scaled_dot_product_attention_forward(
        query, key, value, mask
    )

    assert output.shape == (2, 3, 4, 7)
    assert weights.shape == (2, 3, 4, 6)
    np.testing.assert_array_equal(weights[..., -1], 0.0)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)


def test_masked_softmax_rejects_an_entirely_masked_row() -> None:
    with pytest.raises(ValueError, match="at least one allowed key"):
        masked_softmax(
            np.zeros((1, 2, 3)),
            np.array([[[True, True, True], [False, False, False]]]),
        )


@pytest.mark.parametrize(
    ("query_shape", "key_shape", "value_shape"),
    [
        ((2, 3, 4), (2, 5, 6), (2, 5, 3)),
        ((2, 3, 4), (2, 5, 4), (2, 6, 3)),
        ((2, 3, 4), (3, 5, 4), (3, 5, 3)),
    ],
)
def test_attention_rejects_incompatible_shapes(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        scaled_dot_product_attention_forward(
            np.zeros(query_shape),
            np.zeros(key_shape),
            np.zeros(value_shape),
        )


def test_attention_backward_rejects_wrong_upstream_shape() -> None:
    query = np.zeros((2, 3, 4))
    key = np.zeros((2, 5, 4))
    value = np.zeros((2, 5, 6))
    _, _, cache = scaled_dot_product_attention_forward(query, key, value)

    with pytest.raises(ValueError, match="dout must have shape"):
        scaled_dot_product_attention_backward(np.zeros((2, 3, 5)), cache)
