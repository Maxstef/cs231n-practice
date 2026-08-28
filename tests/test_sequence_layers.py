import numpy as np
import pytest

from cs231n_practice.gradient_check import (
    eval_numerical_gradient,
    eval_numerical_gradient_array,
)
from cs231n_practice.sequence_layers import (
    embedding_backward,
    embedding_forward,
    temporal_affine_backward,
    temporal_affine_forward,
    temporal_softmax_loss,
)


def test_embedding_forward_selects_expected_rows() -> None:
    matrix = np.arange(15, dtype=float).reshape(5, 3)
    token_ids = np.array([[2, 0], [4, 2]])

    output, _ = embedding_forward(token_ids, matrix)

    np.testing.assert_array_equal(output, matrix[token_ids])


def test_embedding_backward_accumulates_repeated_tokens() -> None:
    token_ids = np.array([[1, 2, 1], [0, 2, 1]])
    matrix = np.zeros((4, 3))
    _, cache = embedding_forward(token_ids, matrix)

    dembedding = embedding_backward(np.ones((2, 3, 3)), cache)

    np.testing.assert_array_equal(
        dembedding,
        np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0],
                  [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]]),
    )


def test_embedding_backward_matches_numerical_gradient() -> None:
    generator = np.random.default_rng(61)
    token_ids = np.array([[0, 2, 0], [1, 2, 3]])
    matrix = generator.normal(size=(4, 3))
    output, cache = embedding_forward(token_ids, matrix)
    dout = generator.normal(size=output.shape)

    analytical = embedding_backward(dout, cache)
    numerical = eval_numerical_gradient_array(
        lambda value: embedding_forward(token_ids, value)[0], matrix, dout
    )

    np.testing.assert_allclose(analytical, numerical, rtol=1e-7, atol=1e-8)


def test_temporal_affine_backward_matches_numerical_gradients() -> None:
    generator = np.random.default_rng(67)
    arguments = [
        generator.normal(size=(2, 3, 4)),
        generator.normal(size=(4, 5)),
        generator.normal(size=5),
    ]
    output, cache = temporal_affine_forward(*arguments)
    dout = generator.normal(size=output.shape)
    analytical = temporal_affine_backward(dout, cache)

    for index, (value, gradient) in enumerate(zip(arguments, analytical)):
        def forward(candidate: np.ndarray) -> np.ndarray:
            current = arguments.copy()
            current[index] = candidate
            return temporal_affine_forward(*current)[0]

        numerical = eval_numerical_gradient_array(forward, value, dout)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-8)


def test_temporal_softmax_matches_numerical_gradient_and_masks_padding() -> None:
    generator = np.random.default_rng(71)
    scores = generator.normal(size=(2, 3, 4))
    targets = np.array([[1, 2, 0], [3, 0, 1]])
    mask = np.array([[True, True, False], [True, False, False]])

    loss, analytical = temporal_softmax_loss(scores, targets, mask)
    numerical = eval_numerical_gradient(
        lambda value: temporal_softmax_loss(value, targets, mask)[0], scores
    )

    assert np.isfinite(loss)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(analytical[~mask], 0.0)


def test_temporal_softmax_is_stable_for_large_scores() -> None:
    scores = np.array([[[10_000.0, 10_001.0, 9_999.0]]])
    targets = np.array([[1]])
    mask = np.array([[True]])

    loss, dscores = temporal_softmax_loss(scores, targets, mask)

    assert np.isfinite(loss)
    assert np.all(np.isfinite(dscores))


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (embedding_forward, (np.array([[4]]), np.zeros((4, 2)))),
        (
            temporal_affine_forward,
            (np.zeros((2, 3, 4)), np.zeros((3, 5)), np.zeros(5)),
        ),
        (
            temporal_softmax_loss,
            (np.zeros((2, 3, 4)), np.zeros((2, 2), dtype=int),
             np.ones((2, 2), dtype=bool)),
        ),
    ],
)
def test_sequence_layers_reject_incompatible_inputs(function, arguments) -> None:
    with pytest.raises(ValueError):
        function(*arguments)
