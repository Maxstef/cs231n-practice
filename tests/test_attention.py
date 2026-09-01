import numpy as np
import pytest

from cs231n_practice.attention import (
    masked_softmax,
    merge_heads,
    multi_head_attention_backward,
    multi_head_attention_forward,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
    split_heads,
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


def test_split_and_merge_heads_are_exact_inverses() -> None:
    x = np.arange(2 * 3 * 12).reshape(2, 3, 12)

    heads = split_heads(x, num_heads=3)
    reconstructed = merge_heads(heads)

    assert heads.shape == (2, 3, 3, 4)
    np.testing.assert_array_equal(reconstructed, x)


def _multi_head_arguments(generator: np.random.Generator) -> list[np.ndarray]:
    """Return small cross-attention inputs and parameters for gradient tests."""
    return [
        generator.normal(size=(1, 2, 3)),  # query input
        generator.normal(size=(1, 3, 4)),  # key input
        generator.normal(size=(1, 3, 5)),  # value input
        generator.normal(scale=0.2, size=(3, 4)),  # W_query
        generator.normal(scale=0.2, size=(4, 4)),  # W_key
        generator.normal(scale=0.2, size=(5, 4)),  # W_value
        generator.normal(scale=0.2, size=(4, 3)),  # W_output
        generator.normal(scale=0.1, size=4),  # b_query
        generator.normal(scale=0.1, size=4),  # b_key
        generator.normal(scale=0.1, size=4),  # b_value
        generator.normal(scale=0.1, size=3),  # b_output
    ]


def test_multi_head_attention_supports_masked_cross_attention() -> None:
    generator = np.random.default_rng(107)
    arguments = _multi_head_arguments(generator)
    mask = np.array([[[[True, True, False], [True, False, True]]]])

    output, weights, _ = multi_head_attention_forward(
        *arguments,
        num_heads=2,
        mask=mask,
    )

    assert output.shape == (1, 2, 3)
    assert weights.shape == (1, 2, 2, 3)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)
    np.testing.assert_array_equal(weights[~np.broadcast_to(mask, weights.shape)], 0.0)


def test_multi_head_attention_backward_matches_numerical_gradients() -> None:
    generator = np.random.default_rng(109)
    arguments = _multi_head_arguments(generator)
    mask = np.array([[[[True, True, False], [True, False, True]]]])
    output, _, cache = multi_head_attention_forward(
        *arguments,
        num_heads=2,
        mask=mask,
    )
    dout = generator.normal(size=output.shape)
    analytical = multi_head_attention_backward(dout, cache)

    for index, (value, gradient) in enumerate(zip(arguments, analytical)):
        def forward(candidate: np.ndarray) -> np.ndarray:
            current = arguments.copy()
            current[index] = candidate
            return multi_head_attention_forward(
                *current,
                num_heads=2,
                mask=mask,
            )[0]

        numerical = eval_numerical_gradient_array(forward, value, dout)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=1e-8)


def test_self_attention_adds_three_input_gradient_paths() -> None:
    generator = np.random.default_rng(113)
    x = generator.normal(size=(1, 2, 3))
    model_dim = 4
    parameters = [
        generator.normal(scale=0.2, size=(3, model_dim)),
        generator.normal(scale=0.2, size=(3, model_dim)),
        generator.normal(scale=0.2, size=(3, model_dim)),
        generator.normal(scale=0.2, size=(model_dim, 2)),
        np.zeros(model_dim),
        np.zeros(model_dim),
        np.zeros(model_dim),
        np.zeros(2),
    ]
    output, _, cache = multi_head_attention_forward(
        x,
        x,
        x,
        *parameters,
        num_heads=2,
    )
    dout = generator.normal(size=output.shape)
    dx_query, dx_key, dx_value, *_ = multi_head_attention_backward(dout, cache)

    def forward(candidate: np.ndarray) -> np.ndarray:
        return multi_head_attention_forward(
            candidate,
            candidate,
            candidate,
            *parameters,
            num_heads=2,
        )[0]

    numerical = eval_numerical_gradient_array(forward, x, dout)
    np.testing.assert_allclose(
        dx_query + dx_key + dx_value,
        numerical,
        rtol=2e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize("num_heads", [0, 3, True])
def test_split_heads_rejects_invalid_head_count(num_heads: object) -> None:
    expected_error = TypeError if isinstance(num_heads, bool) else ValueError
    with pytest.raises(expected_error):
        split_heads(np.zeros((2, 3, 4)), num_heads)  # type: ignore[arg-type]
