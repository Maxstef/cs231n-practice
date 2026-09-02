import numpy as np
import pytest

from cs231n_practice.gradient_check import eval_numerical_gradient_array
from cs231n_practice.positional_encoding import (
    add_positional_encoding,
    learned_positional_embedding_backward,
    learned_positional_embedding_forward,
    sinusoidal_positional_encoding,
)


@pytest.mark.parametrize("model_dim", [1, 5, 6])
def test_sinusoidal_encoding_supports_odd_and_even_dimensions(
    model_dim: int,
) -> None:
    encoding = sinusoidal_positional_encoding(4, model_dim)

    assert encoding.shape == (4, model_dim)
    np.testing.assert_allclose(encoding[0, 0::2], 0.0)
    np.testing.assert_allclose(encoding[0, 1::2], 1.0)
    np.testing.assert_allclose(encoding[:, 0], np.sin(np.arange(4)))


def test_sinusoidal_encoding_uses_expected_first_frequency_pair() -> None:
    encoding = sinusoidal_positional_encoding(5, 8)
    positions = np.arange(5)

    np.testing.assert_allclose(encoding[:, 0], np.sin(positions))
    np.testing.assert_allclose(encoding[:, 1], np.cos(positions))
    np.testing.assert_allclose(
        encoding[:, 2],
        np.sin(positions / np.power(10000.0, 2 / 8)),
    )


def test_add_positional_encoding_broadcasts_across_batch() -> None:
    x = np.zeros((3, 4, 6))
    encoding = np.arange(24, dtype=float).reshape(4, 6)

    output = add_positional_encoding(x, encoding)

    assert output.shape == x.shape
    for batch_index in range(x.shape[0]):
        np.testing.assert_array_equal(output[batch_index], encoding)


def test_add_positional_encoding_can_generate_sinusoidal_values() -> None:
    x = np.ones((2, 3, 4))

    output = add_positional_encoding(x)

    expected = x + sinusoidal_positional_encoding(3, 4)
    np.testing.assert_allclose(output, expected)


def test_learned_position_forward_uses_only_current_sequence_rows() -> None:
    x = np.zeros((2, 3, 4))
    table = np.arange(6 * 4, dtype=float).reshape(6, 4)

    output, _ = learned_positional_embedding_forward(x, table)

    assert output.shape == x.shape
    np.testing.assert_array_equal(output[0], table[:3])
    np.testing.assert_array_equal(output[1], table[:3])


def test_learned_position_backward_sums_shared_table_gradients() -> None:
    x = np.zeros((2, 3, 4))
    table = np.zeros((5, 4))
    _, cache = learned_positional_embedding_forward(x, table)
    dout = np.arange(24, dtype=float).reshape(2, 3, 4)

    dx, dtable = learned_positional_embedding_backward(dout, cache)

    np.testing.assert_array_equal(dx, dout)
    np.testing.assert_array_equal(dtable[:3], dout.sum(axis=0))
    np.testing.assert_array_equal(dtable[3:], 0.0)


def test_learned_position_backward_matches_numerical_gradients() -> None:
    generator = np.random.default_rng(127)
    x = generator.normal(size=(2, 3, 4))
    table = generator.normal(size=(5, 4))
    output, cache = learned_positional_embedding_forward(x, table)
    dout = generator.normal(size=output.shape)
    dx, dtable = learned_positional_embedding_backward(dout, cache)

    numerical_dx = eval_numerical_gradient_array(
        lambda candidate: learned_positional_embedding_forward(candidate, table)[0],
        x,
        dout,
    )
    numerical_dtable = eval_numerical_gradient_array(
        lambda candidate: learned_positional_embedding_forward(x, candidate)[0],
        table,
        dout,
    )

    np.testing.assert_allclose(dx, numerical_dx, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(dtable, numerical_dtable, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize(
    ("sequence_length", "model_dim", "expected_error"),
    [
        (0, 4, ValueError),
        (3, -1, ValueError),
        (True, 4, TypeError),
        (3.0, 4, TypeError),
    ],
)
def test_sinusoidal_encoding_rejects_invalid_dimensions(
    sequence_length: object,
    model_dim: object,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        sinusoidal_positional_encoding(  # type: ignore[arg-type]
            sequence_length,
            model_dim,
        )


def test_learned_position_rejects_a_short_table() -> None:
    with pytest.raises(ValueError, match="shorter"):
        learned_positional_embedding_forward(
            np.zeros((2, 4, 3)),
            np.zeros((3, 3)),
        )
